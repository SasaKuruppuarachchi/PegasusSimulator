#!/usr/bin/env python
"""
| File: 8_camera_vehicle.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto and Filip Stec. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation
with a single vehicle equipped with a camera, producing rgb and camera info ROS2 topics.
"""

# Imports to start Isaac Sim from this script
import carb
from omni.isaac.kit import SimulationApp
import time

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

# Enable/disable ROS bridge extensions to keep only ROS2 Bridge
disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, FLAT_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphs import ROS2Camera

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

# Import Isaac Sim Action Graph components
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims

# ROS 2 imports
import rclpy
from utils.drone_location_pub import DroneLocationPublisher
# lidar
from omni.isaac.sensor import IMUSensor
import omni
import omni.kit.viewport.utility
import omni.replicator.core as rep
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import nucleus, stage
from pxr import Gf


GRAVITY = 9.81



class AgipixApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()
        self.assets_root_path = nucleus.get_assets_root_path()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.phy_dt = 300.0
        self.pub_dt = 100.0 # HZ = 1/dt
        self.rendering_dt = 30.0
        self.pg._world_settings = {"physics_dt": 1.0 / self.phy_dt, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / self.rendering_dt}
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(FLAT_ENVIRONMENTS["Hospital"]) #
        #self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [MavlinkBackend(mavlink_config)]

        # Create camera graph for the existing Camera prim on the Iris model, which can be found 
        # at the prim path `/World/quadrotor/body/Camera`. The camera prim path is the local path from the vehicle's prim path
        # to the camera prim, to which this graph will be connected. All ROS2 topics published by this graph will have 
        # namespace `quadrotor` and frame_id `Camera` followed by the selected camera types (`rgb`, `camera_info`).
        config_multirotor.graphs = [ROS2Camera("body/Camera", config={"types": ['rgb', 'camera_info'],"tf_frame_id": "camera"})]

        self.drone = Multirotor(
            "/World/drone0",
            ROBOTS['Agipix v2'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()
        self.stage = omni.usd.get_context().get_stage()
        self.drone_prim = self.stage.GetPrimAtPath(self.drone._stage_prefix + "/body")

        
        # Initialize ROS 2
        rclpy.init()
        self.node = DroneLocationPublisher()
        self.create_rtx_lidar()
        self.create_imu_sensor()

        # Initialize the Action Graph to publish drone odometry
        #self.init_action_graph()
        #self.init_pub_time_graph()
        self.stop_sim = False
        self.sim_elapsed_time=None
        self.real_elapsed_time=None
        self.world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        self.physics_stp_cnt = 0

        #self.setup_post_load()
        
    async def setup_post_load(self):
        pass    # Auxiliar variable for the timeline callback example
        
    def create_imu_sensor(self):
        self.isaac_imu = IMUSensor(
            prim_path=self.drone._stage_prefix + "/body" + "/Imu",
            name="imu",
            frequency=100,
            translation=np.array([0, 0, 0]),
            orientation=np.array([0, -1, 0, 0]), # [w,x,y,z]
            linear_acceleration_filter_size = 10,
            angular_velocity_filter_size = 10,
            orientation_filter_size = 10,
        )
        
    def create_rtx_lidar(self):
        # Create the lidar sensor that generates data into "RtxSensorCpu"
        # Sensor needs to be rotated 90 degrees about X so that its Z up

        # Possible options are Example_Rotary and Example_Solid_State
        # drive sim applies 0.5,-0.5,-0.5,w(-0.5), we have to apply the reverse
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/sensor",
            parent=self.drone._stage_prefix + "/body",
            config="approx_mid_360",
            translation=(self.node.lidar_trans[0] , self.node.lidar_trans[1] ,self.node.lidar_trans[2] ),
            orientation=Gf.Quatd(self.node.lidar_ori[0] , self.node.lidar_ori[1], self.node.lidar_ori[2], self.node.lidar_ori[3]),  # Gf.Quatd is w,i,j,k
        )

        # RTX sensors are cameras and must be assigned to their own render product
        hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")

        self.simulation_context = SimulationContext(physics_dt=1.0 / self.phy_dt, rendering_dt=1.0 / self.rendering_dt, stage_units_in_meters=1.0)
        simulation_app.update()

        # Create Point cloud publisher pipeline in the post process graph
        writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        writer.initialize(topicName="point_cloud", frameId="drone0/lidar_link")
        writer.attach([hydra_texture])

        # Create the debug draw pipeline in the post process graph
        #writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
        #writer.attach([hydra_texture])


        # Create LaserScan publisher pipeline in the post process graph
        #writer = rep.writers.get("RtxLidar" + "ROS2PublishLaserScan")
        #writer.initialize(topicName="laser_scan", frameId="drone0/base_link")
        #writer.attach([hydra_texture])

        simulation_app.update()

    def init_action_graph(self):
        keys = og.Controller.Keys

        (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/action_graph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("tick", "omni.graph.action.OnTick"),
                    ("read_times", "omni.isaac.core_nodes.IsaacReadTimes"),
                    ("compute_odometry", "omni.isaac.core_nodes.IsaacComputeOdometry"),
                    ("publish_clock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ("publish_odometry", "omni.isaac.ros2_bridge.ROS2PublishOdometry")
                ],
                keys.SET_VALUES: [
                    ("compute_odometry.inputs:chassisPrim", "/World/drone0"),
                    ("publish_clock.inputs:topicName", "clock"),
                    #("publish_odometry.inputs:nodeNamespace", "drone0"),
                    ("publish_odometry.inputs:topicName", "drone0/gt"),
                    ("publish_odometry.inputs:odomFrameId", "world"),
                    ("publish_odometry.inputs:chassisFrameId", "drone0/base_link")
                ],
                keys.CONNECT: [
                    ("tick.outputs:tick", "read_times.inputs:execIn"),
                    ("read_times.outputs:execOut", "compute_odometry.inputs:execIn"),
                    ("read_times.outputs:execOut", "publish_clock.inputs:execIn"),
                    ("read_times.outputs:systemTime", "publish_clock.inputs:timeStamp"),
                    ("read_times.outputs:systemTime", "publish_odometry.inputs:timeStamp"),
                    ("compute_odometry.outputs:execOut", "publish_odometry.inputs:execIn"),
                    ("compute_odometry.outputs:angularVelocity", "publish_odometry.inputs:angularVelocity"),
                    ("compute_odometry.outputs:linearVelocity", "publish_odometry.inputs:linearVelocity"),
                    ("compute_odometry.outputs:orientation", "publish_odometry.inputs:orientation"),
                    ("compute_odometry.outputs:position", "publish_odometry.inputs:position")
                ],
            },
        )

    def init_pub_time_graph(self):
        keys = og.Controller.Keys

        (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/time_graph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("tick", "omni.graph.action.OnTick"),
                    ("read_times", "omni.isaac.core_nodes.IsaacReadTimes"),
                    ("publish_clock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    
                ],
                keys.SET_VALUES: [
                    ("publish_clock.inputs:topicName", "clock")
                ],
                keys.CONNECT: [
                    ("tick.outputs:tick", "read_times.inputs:execIn"),
                    ("read_times.outputs:execOut", "publish_clock.inputs:execIn"),
                    ("read_times.outputs:simulationTime", "publish_clock.inputs:timeStamp")
                ],
            },
        )

    def physics_step(self, dt: float):
        # publish clock
        current_sim_time = self.simulation_context.current_time
        current_time = time.time()
        self.node.publish_clock(current_sim_time)
        
        if self.physics_stp_cnt:
            # do something in between publishes
            pass
            
        elif self.sim_elapsed_time is None:
            self.sim_elapsed_time = current_sim_time
            self.real_elapsed_time = current_time
        else:
            self.sim_dt = current_sim_time - self.sim_elapsed_time
            self.real_dt = current_time - self.real_elapsed_time
            #print(self.sim_dt,self.real_dt)
            self.sim_elapsed_time = current_sim_time
            self.real_elapsed_time = current_time
            self.node.publish_rtf(self.real_dt,self.sim_dt)
            
            state = self.drone._state 
            #position = self.drone_prim.GetAttribute('xformOp:translate')
            #orientation = self.drone_prim.GetAttribute('xformOp:orient')
            self.node.publish_gt(state,current_sim_time)
            
            # publish gt_imu
            self.node.publish_gt_imu(current_sim_time, state)
            
            # [prop_force0,..,prop_force3, sum_force (N),rolling moment (Nm)]
            self.node.publish_gt_forces(self.drone.forces,self.drone.rolling_moment) 
            # weight of the system 1.657 Kg . Total Thrust at hover = 16.256 N, Mass Normalised = 9.81 N/Kg
            
            # publish own_imu
            imu_frame = self.isaac_imu.get_current_frame()
            self.node.publish_self_imu(imu_frame)
            
            
        if self.physics_stp_cnt >= self.phy_dt/self.pub_dt-1:
            self.physics_stp_cnt = 0
        else:
            self.physics_stp_cnt += 1
        return

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        #self.timeline.play()
        self.simulation_context.play()
        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            simulation_app.update()
        
        # Cleanup and stop
        self.drone.stop()
        carb.log_warn("Agipix Simulation App is closing.")
        self.simulation_context.stop()
        #self.timeline.stop()
        simulation_app.close()

def main():
    # Instantiate the template app
    pg_app = AgipixApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
