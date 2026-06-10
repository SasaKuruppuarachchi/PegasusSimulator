#!/usr/bin/env python
"""
Agipix PX4 + ROS2 standalone example.
"""

import argparse
import os
import sys
import time

import numpy as np
from isaacsim import SimulationApp
from scipy.spatial.transform import Rotation


# SimulationApp must be created immediately after import in Isaac Sim standalone scripts.
simulation_app = SimulationApp({"headless": False})


import carb
import omni
import omni.timeline
import isaacsim.storage.native as nucleus

from isaacsim.core.api import SimulationContext
from isaacsim.core.api.world import World
from isaacsim.core.utils.extensions import disable_extension, enable_extension
from isaacsim.sensors.physics import IMUSensor


# Keep only ROS2 bridge extension enabled.
disable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import FLAT_ENVIRONMENTS, ROBOTS

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.dirname(__file__))
from drone_location_pub import DroneLocationPublisher


DEFAULT_PHYSICS_HZ = 250.0
DEFAULT_PUBLISH_HZ = 100.0
DEFAULT_RENDER_HZ = 30.0


class AgipixApp:
    def __init__(self, namespace="drone", vehicle_id=0):
        self.namespace = namespace
        self.id = vehicle_id
        self.vehicle_name = f"{self.namespace}{self.id}"

        self.timeline = omni.timeline.get_timeline_interface()
        self.assets_root_path = nucleus.get_assets_root_path()

        self.phy_dt = DEFAULT_PHYSICS_HZ
        self.pub_dt = DEFAULT_PUBLISH_HZ
        self.rendering_dt = DEFAULT_RENDER_HZ

        self.lidar_trans = [0.0795, 0.0, 0.0323]
        self.lidar_ori = [0.9238795, 0.0, 0.3826834, 0.0]

        self.stop_sim = False
        self.sim_elapsed_time = None
        self.real_elapsed_time = None
        self.physics_stp_cnt = 0

        self._setup_world()
        self._spawn_vehicle()
        self._setup_publishers_and_sensors()

        self.world.add_physics_callback("sim_step", callback_fn=self.physics_step)

    def _setup_world(self):
        self.pg = PegasusInterface()
        self.pg._world_settings = {
            "physics_dt": 1.0 / self.phy_dt,
            "stage_units_in_meters": 1.0,
            "rendering_dt": 1.0 / self.rendering_dt,
        }
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(FLAT_ENVIRONMENTS["AKW_C"])

        self.simulation_context = SimulationContext(
            physics_dt=1.0 / self.phy_dt,
            rendering_dt=1.0 / self.rendering_dt,
            stage_units_in_meters=1.0,
        )

    def _build_vehicle_config(self):
        config_multirotor = MultirotorConfig()

        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": self.id,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )

        config_multirotor.backends = [
            PX4MavlinkBackend(mavlink_config),
            ROS2Backend(
                vehicle_id=self.id,
                config={
                    "namespace": self.namespace,
                    "pub_sensors": False,
                    "pub_graphical_sensors": True,
                    "pub_lidar_laserscan": False,
                    "pub_state": True,
                    "sub_control": False,
                },
            ),
        ]

        config_multirotor.graphical_sensors = [
            MonocularCamera(
                "cam0",
                config={
                    "depth": True,
                    "f_stop": 0.0,
                    "focus_distance": 0.6,
                    "position": np.array([0.13, 0.15, -0.022]),
                    "orientation": np.array([180.0, -180.0, 0.0]),
                    "intrinsics": np.array([
                        [606.3120727539062, 0.0, 314.6913146972656], [0.0, 605.92626953125, 252.1909942626953], [0.0, 0.0, 1.0]
                    ]),
                    "resolution": (640, 400),
                    "frequency": 30,
                },
            ),
            MonocularCamera(
                "cam1",
                config={
                    "depth": True,
                    "f_stop": 0.0,
                    "focus_distance": 0.6,
                    "position": np.array([0.13, -0.15, -0.022]),
                    "orientation": np.array([180.0, -180.0, 0.0]),
                    "intrinsics": np.array([
                        [606.3120727539062, 0.0, 314.6913146972656], [0.0, 605.92626953125, 252.1909942626953], [0.0, 0.0, 1.0]
                    ]),
                    "resolution": (640, 400),
                    "frequency": 30,
                },
            ),
            Lidar(
                "livox",
                config={
                    "position": np.array(self.lidar_trans),
                    "orientation": np.array(self.lidar_ori),
                    "sensor_configuration": "Mid_360",
                    "frame_id": "lidar_link",
                    "show_render": False,
                    "frequency": 10,
                },
            ),
        ]

        return config_multirotor

    def _spawn_vehicle(self):
        config_multirotor = self._build_vehicle_config()

        self.drone = Multirotor(
            f"/World/{self.vehicle_name}",
            ROBOTS["Agipix v2"],
            self.id,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

    def _setup_publishers_and_sensors(self):
        self.node = DroneLocationPublisher(
            namespace=self.namespace,
            vehicle_id=self.id,
            lidar_trans=self.lidar_trans,
            lidar_ori=self.lidar_ori,
        )

        simulation_app.update()
        self.create_imu_sensor()
        self.world.reset()

        self.stage = omni.usd.get_context().get_stage()
        self.drone_prim = self.stage.GetPrimAtPath(self.drone._stage_prefix + "/body")

    def create_imu_sensor(self):
        self.isaac_imu = IMUSensor(
            prim_path=self.drone._stage_prefix + "/body/Imu",
            name="imu",
            frequency=100,
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
        print("IMU sensor created")

    def physics_step(self, dt: float):
        current_sim_time = self.simulation_context.current_time
        current_time = time.time()
        self.node.publish_clock(current_sim_time)

        if self.physics_stp_cnt == 0:
            if self.sim_elapsed_time is None:
                self.sim_elapsed_time = current_sim_time
                self.real_elapsed_time = current_time
            else:
                sim_dt = current_sim_time - self.sim_elapsed_time
                real_dt = current_time - self.real_elapsed_time
                self.sim_elapsed_time = current_sim_time
                self.real_elapsed_time = current_time

                self.node.publish_rtf(real_dt, sim_dt)

                state = self.drone._state
                self.node.publish_gt(state, current_sim_time)
                self.node.publish_gt_imu(current_sim_time, state)
                self.node.publish_gt_forces(self.drone.forces, self.drone.rolling_moment)

                imu_frame = self.isaac_imu.get_current_frame()
                self.node.publish_self_imu(imu_frame)

        if self.physics_stp_cnt >= self.phy_dt / self.pub_dt - 1:
            self.physics_stp_cnt = 0
        else:
            self.physics_stp_cnt += 1

    def run(self):
        self.simulation_context.play()

        while simulation_app.is_running() and not self.stop_sim:
            simulation_app.update()

        self.drone.stop()
        carb.log_warn("Agipix Simulation App is closing.")
        self.simulation_context.stop()
        simulation_app.close()


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run the Agipix standalone simulation example.")
    parser.add_argument(
        "--namespace",
        default="drone",
        help="Base namespace prefix used for ROS topics and frame IDs.",
    )
    parser.add_argument(
        "--id",
        dest="vehicle_id",
        type=int,
        default=0,
        help="Vehicle identifier appended to the namespace.",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_cli_args()
    app = AgipixApp(namespace=args.namespace, vehicle_id=args.vehicle_id)
    app.run()


if __name__ == "__main__":
    main()
