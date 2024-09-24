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
from rclpy.parameter import Parameter
# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation

# Import Isaac Sim Action Graph components
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims

# ROS 2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32
from rosgraph_msgs.msg import Clock

# lidar
from omni.isaac.sensor import IMUSensor
import omni
import omni.kit.viewport.utility
import omni.replicator.core as rep
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import nucleus, stage
from pxr import Gf
import math

class DroneLocationPublisher(Node):
    def __init__(self):
        super().__init__('drone_location_publisher')
        #self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.publisher_ = self.create_publisher(PoseStamped, 'drone0/gt_pose', 10)
        self.rtf_publisher_ = self.create_publisher(Float32, 'real_Time_factor', 10)
        #self.imu_publisher1_ = self.create_publisher(Imu, 'drone0/gt_imu1', 10)
        self.imu_publisher_ = self.create_publisher(Imu, 'drone0/gt_imu', 10)
        self.time_publisher = self.create_publisher(Clock, 'clock', 10)
        self.pre_pose_pos = None
        self.pre_pose_ori = None
        self.u = None
        #self.timer = self.create_timer(1.0, self.check_clock_topic)  # Check every 1 second

    def publish_location(self, position, orientation,sim_time):
        msg = PoseStamped()
        sim_time_ = Time()
        sim_time_.sec = math.floor(sim_time)
        sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
        msg.header.stamp = sim_time_ # self.get_clock().now().to_msg()
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
        
    def quattovec(self,quatf):
        vec = [] # x,y,z,w
        real_part = quatf.GetReal()
        imaginary_part = quatf.GetImaginary()

        vec.append(imaginary_part[0])
        vec.append(imaginary_part[1])
        vec.append(imaginary_part[2])
        vec.append(real_part)
        return vec
    def rotate_vector_by_quaternion(self,vector, quat):
        # Convert the quaternion to a Rotation object
        vector = np.array(vector)
        r = Rotation.from_quat(quat)
        
        # Rotate the vector using the quaternion
        rotated_vector = r.apply(vector)
        
        return rotated_vector
    
    def publish_clock(self, sim_time):
        time_msg = Clock()
        time_msg.clock.sec = math.floor(sim_time)
        time_msg.clock.nanosec = int((sim_time - time_msg.clock.sec) * 1e9)
        self.time_publisher.publish(time_msg)
        
    def publish_self_imu(self,self_imu):
        msg = Imu()
        sim_time_ = Time()
        sim_time_.sec = math.floor(self_imu['time'])
        sim_time_.nanosec = int((self_imu['time'] - sim_time_.sec) * 1e9)
        msg.header.stamp = sim_time_
        msg.header.frame_id = 'drone0/base_link'
        
        msg.linear_acceleration.x = float((self_imu['lin_acc'])[0])
        msg.linear_acceleration.y = float((self_imu['lin_acc'])[1])
        msg.linear_acceleration.z = float((self_imu['lin_acc'])[2])
        
        msg.orientation.w = float((self_imu['orientation'])[0])
        msg.orientation.x = float((self_imu['orientation'])[1])
        msg.orientation.y = float((self_imu['orientation'])[2])
        msg.orientation.z = float((self_imu['orientation'])[3])
        
        msg.angular_velocity.x = float((self_imu['ang_vel'])[0])
        msg.angular_velocity.y = float((self_imu['ang_vel'])[1])
        msg.angular_velocity.z = float((self_imu['ang_vel'])[2])
        
        self.imu_publisher_.publish(msg)
    
    def publish_gt_imu(self, position, orientation, sim_time, dt):
        if self.pre_pose_pos is None:
            # First time initialization
            self.pre_pose_pos = position
            self.pre_pose_ori = self.quattovec(orientation)
            return  # No IMU data to publish on first call

        # Initialize velocity if not available yet
        if self.u is None:
            self.u = (position - self.pre_pose_pos) / dt
            self.pre_pose_pos = position
            self.pre_pose_ori = self.quattovec(orientation)
            return

        if dt:
            msg = Imu()
            # Prepare the timestamp
            sim_time_ = Time()
            sim_time_.sec = math.floor(sim_time)
            sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
            msg.header.stamp = sim_time_
            msg.header.frame_id = 'drone0/base_link'

            # Calculate linear acceleration
            new_velocity = (position - self.pre_pose_pos) / dt
            lin_acc = (new_velocity - self.u) / dt - [0.0, 0.0, 9.822]  # Gravity compensation
            lin_acc = self.rotate_vector_by_quaternion(lin_acc, self.quattovec(orientation))

            msg.linear_acceleration.x = lin_acc[0]
            msg.linear_acceleration.y = lin_acc[1]
            msg.linear_acceleration.z = lin_acc[2]

            # Compute angular velocity using quaternion derivatives
            r_prev = Rotation.from_quat(self.pre_pose_ori)
            r_next = Rotation.from_quat(self.quattovec(orientation))
            r_rel = r_next * r_prev.inv()
            angular_velocity = r_rel.as_rotvec() / dt

            msg.angular_velocity.x = angular_velocity[0]
            msg.angular_velocity.y = angular_velocity[1]
            msg.angular_velocity.z = angular_velocity[2]

            # Set orientation
            q = orientation  # Assuming it's already a quaternion (w, x, y, z)
            msg.orientation.w = q.GetReal()
            msg.orientation.x = q.GetImaginary()[0]
            msg.orientation.y = q.GetImaginary()[1]
            msg.orientation.z = q.GetImaginary()[2]

            # Publish the IMU message
            self.imu_publisher1_.publish(msg)

            # Update previous state for next iteration
            self.u = new_velocity
            self.pre_pose_pos = position
            self.pre_pose_ori = self.quattovec(orientation)


    def publish_rtf(self, real_dt, sim_dt):
        msg= Float32()
        if(real_dt):
            msg.data = sim_dt/real_dt
            self.rtf_publisher_.publish(msg)
            
    def check_clock_topic(self):
        # Get the list of topics and types currently available
        topics_and_types = self.get_topic_names_and_types()

        # Check if /clock topic is available
        clock_topic_exists = any(topic == '/clock' for topic, _ in topics_and_types)

        # Set use_sim_time to True if /clock topic is present, otherwise keep it False
        if clock_topic_exists:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
            self.timer.cancel()
        print("counter")
        self.get_logger().info('/clock topic not available. use_sim_time remains False.')

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
        self.assets_root_path = nucleus.get_assets_root_path()

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

        self.create_rtx_lidar()
        self.create_imu_sensor()
        # Initialize ROS 2
        rclpy.init()

        # Create ROS 2 publisher node
        self.node = DroneLocationPublisher()

        # Initialize the Action Graph to publish drone odometry
        #self.init_action_graph()
        #self.init_pub_time_graph()
        self.stop_sim = False
        
        self.sim_elapsed_time=None
        self.real_elapsed_time=None
        self.world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        #self.setup_post_load()
        
    async def setup_post_load(self):
        pass

        

        # Auxiliar variable for the timeline callback example
        
    

    def create_imu_sensor(self):
        self.isaac_imu = IMUSensor(
            prim_path="/World/drone0/body/Imu",
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
            translation=(0, 0, 1.0),
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
        )

        # RTX sensors are cameras and must be assigned to their own render product
        hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")

        self.simulation_context = SimulationContext(physics_dt=1.0 / 250.0, rendering_dt=1.0 / 100.0, stage_units_in_meters=1.0)
        simulation_app.update()

        # Create Point cloud publisher pipeline in the post process graph
        writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        writer.initialize(topicName="point_cloud", frameId="drone0/base_link")
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

    def physics_step(self, step_size):
        # publish clock
        current_sim_time = self.simulation_context.current_time
        self.node.publish_clock(current_sim_time)
        if self.sim_elapsed_time is None:
            self.sim_elapsed_time = current_sim_time
            self.real_elapsed_time = time.time()
        else:
            self.sim_dt = current_sim_time - self.sim_elapsed_time
            self.real_dt = time.time() - self.real_elapsed_time
            #print(self.sim_dt,self.real_dt)
            self.sim_elapsed_time = current_sim_time
            self.real_elapsed_time = time.time()
            self.node.publish_rtf(self.real_dt,self.sim_dt)
            
            position = self.drone_prim.GetAttribute('xformOp:translate')
            orientation = self.drone_prim.GetAttribute('xformOp:orient')
            self.node.publish_location(position.Get(), orientation.Get(),current_sim_time)
            #self.node.publish_gt_imu(position.Get(), orientation.Get(),current_sim_time,self.sim_dt)
            
            imu_frame = self.isaac_imu.get_current_frame()
            #print(imu_frame)
            self.node.publish_self_imu(imu_frame)
            # publish own_imu
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
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.simulation_context.stop()
        #self.timeline.stop()
        simulation_app.close()

def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
