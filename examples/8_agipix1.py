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
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphs import ROS2Camera

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

# Import Isaac Sim Action Graph components
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims

# ROS 2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

#lidar
import asyncio                                                  # Used to run sample asynchronously to not block rendering thread
from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor
from pxr import UsdGeom, Gf, UsdPhysics 


class DroneLocationPublisher(Node):
    def __init__(self):
        super().__init__('drone_location_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'drone0/gt_pose', 10)
        self.rtf_publisher_ = self.create_publisher(Float32, 'real_Time_factor', 10)

    def publish_location(self, position, orientation):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'vicon_map'
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        # Assuming orientation is an instance of Gf.Quatd
        real_part = orientation.GetReal()
        imaginary_part = orientation.GetImaginary()

        msg.pose.orientation.w = real_part
        msg.pose.orientation.x = imaginary_part[0]
        msg.pose.orientation.y = imaginary_part[1]
        msg.pose.orientation.z = imaginary_part[2]

        self.publisher_.publish(msg)

    def publish_rtf(self, real_elapsed, sim_elapsed):
        msg= Float32()
        msg.data = sim_elapsed/real_elapsed
        if(msg.data>0.01):
            self.rtf_publisher_.publish(msg)

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

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
        config_multirotor.graphs = [ROS2Camera("body/Camera", config={"types": ['rgb', 'camera_info']})]

        self.drone = Multirotor(
            "/World/drone0",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        self.stage = omni.usd.get_context().get_stage()
        self.drone_prim = self.stage.GetPrimAtPath("/World/drone0/body")
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR

        # These commands are the Python-equivalent of the first half of this tutorial
        #omni.kit.commands.execute('AddPhysicsSceneCommand',stage = self.stage, path='/World/PhysicsScene')
        self.lidarPath = "/mid360"
        result, prim = omni.kit.commands.execute(
                    "RangeSensorCreateLidar",
                    path=self.lidarPath,
                    parent=self.drone._stage_prefix + "/body",
                    min_range=0.4,
                    max_range=100.0,
                    draw_points=True,
                    draw_lines=False,
                    horizontal_fov=360.0,
                    vertical_fov=59.0,
                    horizontal_resolution=0.4,
                    vertical_resolution=4.0,
                    rotation_rate=10.0,
                    high_lod=True,
                    yaw_offset=0.0,
                    enable_semantics=False
                )
        
        # Create a cube, sphere, add collision and different semantic labels
        primType = ["Cube", "Sphere"]
        for i in range(2):
            prim = self.stage.DefinePrim("/World/"+primType[i], primType[i])
            UsdGeom.XformCommonAPI(prim).SetTranslate((-2.0, -2.0 + i * 4.0, 0.0))
            UsdGeom.XformCommonAPI(prim).SetScale((1, 1, 1))
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)

        #Initialize ROS 2
        rclpy.init()

        # Create ROS 2 publisher node
        self.node = DroneLocationPublisher()

        # Initialize the Action Graph to publish drone odometry
        #self.init_action_graph()
        self.init_pub_time_graph()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False
        
        self.sim_elapsed_time=0.0
        self.real_elapsed_time=0.0

    async def get_lidar_param(self):                                    # Function to retrieve data from the LIDAR
        await omni.kit.app.get_app().next_update_async()            # wait one frame for data
        #self.timeline.pause()                                            # Pause the simulation to populate the LIDAR's depth buffers
        depth = self.lidarInterface.get_linear_depth_data("/World"+self.lidarPath)
        zenith = self.lidarInterface.get_zenith_data("/World"+self.lidarPath)
        azimuth = self.lidarInterface.get_azimuth_data("/World"+self.lidarPath)
        print("depth")                                       # Print the data
        #print("zenith", zenith)
        #print("azimuth", azimuth)

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


    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()
        asyncio.ensure_future(self.get_lidar_param())  

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
            # Get drone position and orientation
            position = self.drone_prim.GetAttribute('xformOp:translate')
            orientation = self.drone_prim.GetAttribute('xformOp:orient')
            
            # Publish drone location to ROS 2
            self.node.publish_location(position.Get(), orientation.Get())

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()