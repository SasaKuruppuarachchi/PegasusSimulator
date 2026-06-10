#!/usr/bin/env python
"""
| File: 9_people.py
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation
| where people move around in the world.
"""

# Imports to start Isaac Sim from this script
import carb

from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from isaacsim.core.utils.extensions import enable_extension, disable_extension

# Enable/disable ROS bridge extensions to keep only ROS2 Bridge
disable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.ros2.bridge")

enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.graph.ui")
enable_extension("omni.anim.people")
enable_extension("isaacsim.replicator.agent.core")

# Update the simulation app with the new extensions
simulation_app.update()

from pegasus.simulator.params import FLAT_ENVIRONMENTS, ROBOTS

# -------------------------------------------------------------------------------------------------
# These lines are needed to restart the USD stage and make sure that the people extension is loaded
# -------------------------------------------------------------------------------------------------
import omni.usd
omni.usd.get_context().new_stage()

import numpy as np

# Drone start exclusion zone — people must not enter or spawn here
_DRONE_EXCLUSION_CENTER = np.array([0.0, 0.0, 0.0])
_DRONE_EXCLUSION_RADIUS = 2.0

def _clamp_away_from_zone(target, zone_center=_DRONE_EXCLUSION_CENTER, zone_radius=_DRONE_EXCLUSION_RADIUS):
    """Push target to the nearest point on the exclusion zone boundary if it falls inside."""
    target = np.array(target, dtype=float)
    delta = target[:2] - zone_center[:2]
    dist = np.linalg.norm(delta)
    if dist < zone_radius:
        if dist < 1e-6:
            delta = np.array([zone_radius, 0.0])
        else:
            delta = delta / dist * zone_radius
        target = target.copy()
        target[0] = zone_center[0] + delta[0]
        target[1] = zone_center[1] + delta[1]
    return target


def _apply_exclusion_zone(person, zone_center=_DRONE_EXCLUSION_CENTER, zone_radius=_DRONE_EXCLUSION_RADIUS):
    """Check the actor's current position and force an escape target if inside the zone.
    Returns True if the person was inside the zone; callers should skip normal update logic."""
    pos = person.state.position
    delta = pos[:2] - zone_center[:2]
    dist = np.linalg.norm(delta)
    if dist < zone_radius:
        if dist < 1e-6:
            escape_dir = np.array([1.0, 0.0])
        else:
            escape_dir = delta / dist
        escape_point = zone_center[:2] + escape_dir * (zone_radius + 1.0)
        person.update_target_position([escape_point[0], escape_point[1], pos[2]], 2.0)
        return True
    return False


def _detour_around_zone(pos, target, zone_center=_DRONE_EXCLUSION_CENTER, zone_radius=_DRONE_EXCLUSION_RADIUS, margin=0.5):
    """If the straight-line path from pos to target passes through the exclusion zone,
    return a tangent boundary waypoint that routes around it. Otherwise returns target unchanged."""
    p = np.array(pos[:2], dtype=float)
    t = np.array(target[:2], dtype=float)
    c = np.array(zone_center[:2], dtype=float)
    r = zone_radius + margin

    direction = t - p
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array(target, dtype=float)
    dir_unit = direction / length

    proj = np.clip(np.dot(c - p, dir_unit), 0.0, length)
    closest = p + proj * dir_unit
    dist_closest = np.linalg.norm(closest - c)

    if dist_closest >= r:
        return np.array(target, dtype=float)

    # Path intersects: steer to one of the two tangent points on the zone boundary
    perp = np.array([-dir_unit[1], dir_unit[0]])
    detour_left = c + perp * r
    detour_right = c - perp * r

    def total_dist(mid):
        return np.linalg.norm(mid - p) + np.linalg.norm(t - mid)

    detour_2d = detour_left if total_dist(detour_left) <= total_dist(detour_right) else detour_right
    z = float(np.asarray(target, dtype=float).flat[2]) if np.asarray(target).size > 2 else 0.0
    return np.array([detour_2d[0], detour_2d[1], z], dtype=float)

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

class WaypointPatrolController(PersonController):
    """Patrols through a list of 3D waypoints in sequence, looping back to start.
    Slows down when near a waypoint before snapping to the next one.
    """

    def __init__(self, waypoints, speed=1.2, arrival_radius=0.8):
        super().__init__()
        self._waypoints = [np.array(wp) for wp in waypoints]
        self._speed = speed
        self._arrival_radius = arrival_radius
        self._current_wp_idx = 0

    def update(self, dt: float):
        if _apply_exclusion_zone(self._person):
            return
        target = self._waypoints[self._current_wp_idx]
        pos = self._person.state.position
        dist = np.linalg.norm(target[:2] - pos[:2])
        if dist < self._arrival_radius:
            self._current_wp_idx = (self._current_wp_idx + 1) % len(self._waypoints)
            target = self._waypoints[self._current_wp_idx]
        target = _clamp_away_from_zone(target)
        immediate = _detour_around_zone(pos, target)
        speed = self._speed * min(1.0, dist / (self._arrival_radius * 2.0))
        speed = max(speed, 0.3)
        self._person.update_target_position(immediate.tolist(), speed)


class LemniscateController(PersonController):
    """Person traces a figure-8 (lemniscate of Bernoulli) path around a center point.
    The path is parameterized so the person moves at approximately constant speed.
    """

    def __init__(self, center, scale=4.0, speed=0.35):
        """
        Args:
            center: [x, y, z] center of the figure-8
            scale: half-axis length in meters
            speed: angular parameter rate (rad/s, controls traversal speed)
        """
        super().__init__()
        self._center = np.array(center)
        self._scale = scale
        self._gamma = 0.0
        self._gamma_dot = speed

    def update(self, dt: float):
        if _apply_exclusion_zone(self._person):
            return
        self._gamma += self._gamma_dot * dt
        denom = 1 + np.sin(self._gamma) ** 2
        x = self._center[0] + self._scale * np.cos(self._gamma) / denom
        y = self._center[1] + self._scale * np.sin(self._gamma) * np.cos(self._gamma) / denom
        z = self._center[2]
        target = _clamp_away_from_zone([x, y, z])
        immediate = _detour_around_zone(self._person.state.position, target)
        self._person.update_target_position(immediate.tolist(), 1.0)


class ReactivePersonController(PersonController):
    """Person behavior reacts to the drone state shared via a mutable dict.
    - If the drone is within flee_radius: person flees away at high speed.
    - If the drone is between flee_radius and approach_radius: person idles.
    - If the drone is beyond approach_radius: person slowly approaches the drone.
    
    Args:
        shared_state: a dict with key "drone_position" -> np.array([x, y, z])
        initial_pos: starting/idle position
        flee_radius: distance (m) at which the person starts fleeing
        approach_radius: distance (m) beyond which person starts approaching
        safe_retreat_point: fallback position when fleeing
    """

    IDLE = "idle"
    FLEE = "flee"
    FOLLOW = "follow"

    def __init__(self, shared_state, initial_pos, flee_radius=4.0, approach_radius=10.0):
        super().__init__()
        self._shared_state = shared_state
        self._initial_pos = np.array(initial_pos)
        self._flee_radius = flee_radius
        self._approach_radius = approach_radius
        self._mode = self.IDLE

    def update(self, dt: float):
        if _apply_exclusion_zone(self._person):
            return
        drone_pos = self._shared_state.get("drone_position", None)
        my_pos = self._person.state.position

        if drone_pos is None:
            safe_idle = _clamp_away_from_zone(self._initial_pos)
            self._person.update_target_position(safe_idle.tolist(), 0.5)
            return

        drone_pos = np.array(drone_pos)
        dist = np.linalg.norm(drone_pos[:2] - my_pos[:2])

        if dist < self._flee_radius:
            self._mode = self.FLEE
            flee_dir = my_pos[:2] - drone_pos[:2]
            norm = np.linalg.norm(flee_dir)
            if norm > 0.01:
                flee_dir /= norm
            flee_target = my_pos[:2] + flee_dir * 6.0
            flee = _clamp_away_from_zone([flee_target[0], flee_target[1], my_pos[2]])
            flee = _detour_around_zone(my_pos, flee)
            self._person.update_target_position(flee.tolist(), 1.8)
        elif dist > self._approach_radius:
            self._mode = self.FOLLOW
            approach_target = drone_pos.copy()
            approach_target[2] = my_pos[2]
            approach_target = _clamp_away_from_zone(approach_target)
            approach_target = _detour_around_zone(my_pos, approach_target)
            self._person.update_target_position(approach_target.tolist(), 0.8)
        else:
            self._mode = self.IDLE
            safe_idle = _clamp_away_from_zone(self._initial_pos)
            self._person.update_target_position(safe_idle.tolist(), 0.4)


# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

# -------------------------------------------------------------------------------------------------
# Define the PegasusApp class where the simulation will be run
# -------------------------------------------------------------------------------------------------
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
        #self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self.pg.load_asset(FLAT_ENVIRONMENTS["Full Warehouse"], "/World/layout")

        # Check the available assets for people
        people_assets_list = Person.get_character_asset_list()
        for person in people_assets_list:
            print(person)

        self._shared_drone_state = {"drone_position": None}

        patrol_waypoints = [
            [5.0, 5.0, 0.0],
            [5.0, -5.0, 0.0],
            [-5.0, -5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
        p1 = Person(
            "person1",
            "original_male_adult_construction_05",
            init_pos=[5.0, 5.0, 0.0],
            init_yaw=0.0,
            controller=WaypointPatrolController(patrol_waypoints, speed=1.2, arrival_radius=0.8),
        )

        p2 = Person(
            "person2",
            "original_female_adult_business_02",
            init_pos=[4.0, 8.0, 0.0],
            init_yaw=0.0,
            controller=LemniscateController(center=[0.0, 8.0, 0.0], scale=4.0, speed=0.4),
        )

        p3 = Person(
            "person3",
            "original_male_adult_construction_05",
            init_pos=[-3.0, 3.0, 0.0],
            init_yaw=1.0,
            controller=ReactivePersonController(
                shared_state=self._shared_drone_state,
                initial_pos=[-3.0, 3.0, 0.0],
                flee_radius=4.0,
                approach_radius=10.0,
            ),
        )

        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": "/home/sasa/PX4-Autopilot"
        })

        config_multirotor.backends = [
            PX4MavlinkBackend(mavlink_config),
            ROS2Backend(vehicle_id=1, 
                config={
                    "namespace": 'drone', 
                    "pub_sensors": False,
                    "pub_graphical_sensors": True,
                    "pub_state": True,
                    "pub_tf": False,
                    "sub_control": False,})]
        
        config_multirotor.graphical_sensors = [MonocularCamera("camera", config={"update_rate": 60.0})]
        
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Set the camera of the viewport to a nice position
        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

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