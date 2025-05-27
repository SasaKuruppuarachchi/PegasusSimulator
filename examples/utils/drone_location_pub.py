#!/usr/bin/env python

import math
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32, Float32MultiArray
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation
from pegasus.simulator.logic.state import State
from rclpy.parameter import Parameter

GRAVITY = 9.81

class DroneLocationPublisher(Node):
    def __init__(self):
        super().__init__('drone_location_publisher')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        #self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.gt_publisher_ = self.create_publisher(PoseStamped, 'drone0/gt_pose', qos_profile)
        self.rtf_publisher_ = self.create_publisher(Float32, 'real_Time_factor', qos_profile)
        self.self_imu_publisher_ = self.create_publisher(Imu, 'drone0/self_imu', qos_profile)
        self.imu_publisher_ = self.create_publisher(Imu, 'drone0/gt_imu', qos_profile)
        self.forces_publisher = self.create_publisher(Float32MultiArray, 'drone0/gt_forces',qos_profile)
        self.time_publisher = self.create_publisher(Clock, 'clock', qos_profile)
        self.pre_pose_pos = None
        self.pre_pose_ori = None
        self.u = None
        #self.timer = self.create_timer(1.0, self.check_clock_topic)  # Check every 1 second
        # Create a static transform broadcaster for lidar_link and base_link
        self.lidar_trans = [0.0795, 0.0, 0.0323]
        self.lidar_ori = [0.9238795, 0.0, 0.3826834, 0.0,]
        lidar_frame_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform('drone0/lidar_link','drone0/base_link',self.lidar_trans, self.lidar_ori, lidar_frame_broadcaster)
        
    def publish_static_transform(self, ch_frame, pr_frame, translation_xyz, orient_wxyz,broadcaster):
        # Define the static transform
        static_transform_stamped = TransformStamped()

        static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        static_transform_stamped.header.frame_id = pr_frame #'drone0/base_link'
        static_transform_stamped.child_frame_id = ch_frame #'drone0/lidar_link'

        # Set translation (in meters)
        static_transform_stamped.transform.translation.x = translation_xyz[0]
        static_transform_stamped.transform.translation.y = translation_xyz[1]
        static_transform_stamped.transform.translation.z = translation_xyz[2]

        # Set rotation (as a quaternion)
        static_transform_stamped.transform.rotation.w = orient_wxyz[0]
        static_transform_stamped.transform.rotation.x = orient_wxyz[1]
        static_transform_stamped.transform.rotation.y = orient_wxyz[2]
        static_transform_stamped.transform.rotation.z = orient_wxyz[3]

        # Broadcast the static transform
        broadcaster.sendTransform(static_transform_stamped)
        #self.get_logger().info('Publishing static transform from drone0/base_link to drone0/lidar_link')

    
        
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
        
        self.self_imu_publisher_.publish(msg)
        
    def publish_gt(self, state:State, sim_time):
        msg = PoseStamped()
        sim_time_ = Time()
        sim_time_.sec = math.floor(sim_time)
        sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
        msg.header.stamp = sim_time_ # self.get_clock().now().to_msg()
        msg.header.frame_id = 'vicon_map'
        position = state.position
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        # Assuming orientation is an instance of Gf.Quatd
        
        orientation = state.attitude
        msg.pose.orientation.w = orientation[3]
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]

        self.gt_publisher_.publish(msg)
    
    def publish_gt_imu(self, sim_time, state: State):
        #  @audit get pose in front frame check
        msg = Imu()
        # Prepare the timestamp
        sim_time_ = Time()
        sim_time_.sec = math.floor(sim_time)
        sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
        msg.header.stamp = sim_time_
        msg.header.frame_id = 'drone0/base_link'

        linear_acceleration = state.get_linear_body_velocity_ned_frd()
        msg.linear_acceleration.x = linear_acceleration[0]
        msg.linear_acceleration.y = linear_acceleration[1]
        msg.linear_acceleration.z = linear_acceleration[2] - GRAVITY
        
        angular_velocity = state.get_angular_velocity_frd()
        msg.angular_velocity.x = angular_velocity[0]
        msg.angular_velocity.y = angular_velocity[1]
        msg.angular_velocity.z = angular_velocity[2]

        # Set orientation
        attitude = state.get_attitude_ned_frd()
        msg.orientation.w = attitude[3]
        msg.orientation.x = attitude[0]
        msg.orientation.y = attitude[1]
        msg.orientation.z = attitude[2]

        # Publish the IMU message
        self.imu_publisher_.publish(msg)

        # Update previous state for next iteration
        
    def publish_gt_forces(self, prop_forces, rolling_torque):
        forces = Float32MultiArray()
        prop_forces = np.array(prop_forces)
        forces.data= [prop_forces[0], prop_forces[1], prop_forces[2], prop_forces[3],np.sum(prop_forces),rolling_torque]
        self.forces_publisher.publish(forces)


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