"""
| File: vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the Vehicle class which is used as the base for all the vehicles.
"""

# Numerical computations
import numpy as np
from scipy.spatial.transform import Rotation

# Low level APIs
import carb
from pxr import Usd, Gf

# High level Isaac sim APIs
import omni.usd
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from omni.usd import get_stage_next_free_path
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.experimental.prims import RigidPrim

# Extension APIs
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicle_manager import VehicleManager


def get_world_transform_xform(prim: Usd.Prim):
    """
    Get the local transformation of a prim using omni.usd.get_world_transform_matrix().
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.get_world_transform_matrix.html
    Args:
        prim (Usd.Prim): The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    return rotation


class Vehicle(Robot):
    
    def __init__(
        self,
        stage_prefix: str,
        usd_path: str = None,
        init_pos=[0.0, 0.0, 0.0],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        sensors=[],
        graphical_sensors=[],
        graphs=[],
        backends=[]
    ):
        """
        Class that initializes a vehicle in the isaac sim's curent stage

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_path (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.0].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
        """

        # Get the current world at which we want to spawn the vehicle
        self._world = PegasusInterface().world
        self._current_stage = self._world.stage

        # Save the name with which the vehicle will appear in the stage
        # and the name of the .usd file that contains its description
        self._stage_prefix = get_stage_next_free_path(self._current_stage, stage_prefix, False)
        self._usd_file = usd_path

        # Get the vehicle name by taking the last part of vehicle stage prefix
        self._vehicle_name = self._stage_prefix.rpartition("/")[-1]

        # Spawn the vehicle primitive in the world's stage
        self._prim = define_prim(self._stage_prefix, "Xform")
        self._prim = get_prim_at_path(self._stage_prefix)
        self._prim.GetReferences().AddReference(self._usd_file)

        # Initialize the "Robot" class
        # Note: we need to change the rotation to have qw first, because NVidia
        # does not keep a standard of quaternions inside its own libraries (not good, but okay)
        super().__init__(
            prim_path=self._stage_prefix,
            name=self._stage_prefix,
            position=init_pos,
            orientation=[init_orientation[3], init_orientation[0], init_orientation[1], init_orientation[2]],
            articulation_controller=None,
        )

        # Add this object for the world to track, so that if we clear the world, this object is deleted from memory and
        # as a consequence, from the VehicleManager as well
        self._world.scene.add(self)

        # Add the current vehicle to the vehicle manager, so that it knows
        # that a vehicle was instantiated
        VehicleManager.get_vehicle_manager().add_vehicle(self._stage_prefix, self)

        # Variable that will hold the current state of the vehicle
        self._state = State()

        # Add a callback to the physics engine to update the current state of the system
        self._world.add_physics_callback(self._stage_prefix + "/state", self.update_state)

        # Add the update method to the physics callback if the world was received
        # so that we can apply forces and torques to the vehicle. Note, this method should        # be implemented in classes that inherit the vehicle object
        self._world.add_physics_callback(self._stage_prefix + "/update", self.update)

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        # Cache for RigidBodyPrim instances, initialized on first physics step
        self._rigid_body_prims = {}
        self._physics_sim_view = None

        # Flag to track whether Robot.initialize() (ArticulationController) has been called
        self._articulation_initialized = False

        # Flag set by the UI delegate to suppress backend start/stop during the initialization
        # reset cycle (world.reset_async() + world.stop_async() called right after spawning).
        # Without this, reset_async fires a 'playing' event that launches PX4 prematurely.
        self._initializing = False
        # --------------------------------------------------------------------
        self._sensors = sensors
        
        for sensor in self._sensors:
            sensor.initialize(self, PegasusInterface().latitude, PegasusInterface().longitude, PegasusInterface().altitude)

        # Add callbacks to the physics engine to update each sensor at every timestep
        # and let the sensor decide depending on its internal update rate whether to generate new data
        self._world.add_physics_callback(self._stage_prefix + "/Sensors", self.update_sensors)

        # --------------------------------------------------------------------
        # -------------------- Add the graphical sensors to the vehicle ------
        # --------------------------------------------------------------------
        self._graphical_sensors = graphical_sensors

        for graphical_sensor in self._graphical_sensors:
            graphical_sensor.initialize(self)

        # Add callbacks to the rendering engine to update each graphical sensor at every timestep of the rendering engine
        self._world.add_render_callback(self._stage_prefix + "/GraphicalSensors", self.update_graphical_sensors)


        # --------------------------------------------------------------------
        # -------------------- Add the graphs to the vehicle -----------------
        # --------------------------------------------------------------------
        self._graphs = graphs

        for graph in self._graphs:
            graph.initialize(self)
        
        # --------------------------------------------------------------------
        # ---- Add (communication/control) backends to the vehicle -----------
        # --------------------------------------------------------------------
        self._backends = backends

        # Initialize the backends
        for backend in self._backends:
            backend.initialize(self)

        # Add a callbacks for the
        self._world.add_physics_callback(self._stage_prefix + "/mav_state", self.update_sim_state)


    def __del__(self):
        """
        Method that is invoked when a vehicle object gets destroyed. When this happens, we also invoke the 
        'remove_vehicle' from the VehicleManager in order to remove the vehicle from the list of active vehicles.
        """

        # Remove this object from the vehicleHandler
        VehicleManager.get_vehicle_manager().remove_vehicle(self._stage_prefix)

    """
    Properties
    """

    @property
    def state(self):
        """The state of the vehicle.

        Returns:
            State: The current state of the vehicle, i.e., position, orientation, linear and angular velocities...
        """
        return self._state
    
    @property
    def vehicle_name(self) -> str:
        """Vehicle name.

        Returns:
            Vehicle name (str): last prim name in vehicle prim path
        """
        return self._vehicle_name

    """
    Operations
    """

    def initialize(self, physics_sim_view=None):
        """Called by world.reset() via scene._finalize() to initialize the Robot
        and its ArticulationController with the physics simulation view.
        In Isaac Sim 6.0, this must be triggered by world.reset(), not manually,
        because the physics_sim_view is only valid during/after world.reset().
        """
        super().initialize(physics_sim_view=physics_sim_view)
        self._articulation_initialized = True

        # world.reset() (called during the init cycle AND when Play is pressed in UI mode)
        # clears callbacks before calling scene._finalize() -> vehicle.initialize().
        # The init-cycle reset clears only timeline callbacks; the Play-button reset clears ALL.
        # Re-register ALL callbacks here using remove-then-add to be idempotent in both cases.
        for name, callback in [
            (self._stage_prefix + "/state", self.update_state),
            (self._stage_prefix + "/update", self.update),
            (self._stage_prefix + "/Sensors", self.update_sensors),
            (self._stage_prefix + "/mav_state", self.update_sim_state),
        ]:
            try:
                self._world.remove_physics_callback(name)
            except Exception:
                pass
            self._world.add_physics_callback(name, callback)

        try:
            self._world.remove_render_callback(self._stage_prefix + "/GraphicalSensors")
        except Exception:
            pass
        self._world.add_render_callback(self._stage_prefix + "/GraphicalSensors", self.update_graphical_sensors)

        try:
            self._world.remove_timeline_callback(self._stage_prefix + "/start_stop_sim")
        except Exception:
            pass
        self._world.add_timeline_callback(self._stage_prefix + "/start_stop_sim", self.sim_start_stop)

    def sim_start_stop(self, event):
        """
        Callback that is called every time there is a timeline event such as starting/stoping the simulation.

        Args:
            event: A timeline event generated from Isaac Sim, such as starting or stoping the simulation.
        """

        # If the start/stop button was pressed, then call the start and stop methods accordingly.
        # Guard with _initializing: when world.reset_async() is called right after vehicle spawn
        # (UI flow), it briefly fires a 'playing' event. We must not start backends then, because
        # that would launch PX4 prematurely before the user presses Play.
        if self._world.is_playing() and self._sim_running == False and not self._initializing:
            # Re-register physics/render callbacks before starting. In Isaac Sim 6.0,
            # world.stop_async() clears physics and render callbacks without calling
            # initialize() again. Re-registering here (remove-then-add) ensures they
            # are always live when physics actually begins, regardless of what
            # stop_async() or any intermediate reset cleared.
            for name, callback in [
                (self._stage_prefix + "/state", self.update_state),
                (self._stage_prefix + "/update", self.update),
                (self._stage_prefix + "/Sensors", self.update_sensors),
                (self._stage_prefix + "/mav_state", self.update_sim_state),
            ]:
                try:
                    self._world.remove_physics_callback(name)
                except Exception:
                    pass
                self._world.add_physics_callback(name, callback)

            try:
                self._world.remove_render_callback(self._stage_prefix + "/GraphicalSensors")
            except Exception:
                pass
            self._world.add_render_callback(self._stage_prefix + "/GraphicalSensors", self.update_graphical_sensors)

            self._sim_running = True

            # Initialize the sensors
            for sensor in self._sensors:
                sensor.start()

            # Initialize the graphical sensors
            for graphical_sensor in self._graphical_sensors:
                graphical_sensor.start()

            # Intializes the communication with all the backends. This method is invoked automatically when the simulation starts
            for backend in self._backends:
                backend.start()

            # Invoke the start method of the vehicle (if it exists)
            self.start()

        if self._world.is_stopped() and self._sim_running == True:
            self._sim_running = False

            # Stop the sensors
            for sensor in self._sensors:
                sensor.stop()

            # Stop the graphical sensors
            for graphical_sensor in self._graphical_sensors:
                graphical_sensor.stop()

            # Signal all the backends that the simulation has stoped. This method is invoked automatically when the simulation stops
            for backend in self._backends:
                backend.stop()

            # Reset the rigid body prim cache on stop (stale RigidPrim objects become invalid after stop)
            # Do NOT reset _articulation_initialized here - it was set by world.reset() calling
            # scene._finalize() -> vehicle.initialize(). Resetting it causes update_state to
            # permanently skip after the first stop, since world.play() (not world.reset()) is
            # called when the user presses Play again.
            self._rigid_body_prims = {}
            self._articulation_initialized = False

            self.stop()

    def _get_rigid_body_prim(self, body_part: str):
        """
        Returns a cached RigidPrim for the given body_part.
        Uses isaacsim.core.experimental.prims.RigidPrim which handles
        physics initialization internally (no create_simulation_view needed).
        """
        if body_part not in self._rigid_body_prims:
            self._rigid_body_prims[body_part] = RigidPrim(self._stage_prefix + body_part)
        return self._rigid_body_prims[body_part]

    def apply_force(self, force, pos=[0.0, 0.0, 0.0], body_part="/body"):
        """
        Method that will apply a force on the rigidbody, on the part specified in the 'body_part' at its relative position
        given by 'pos' (following a FLU) convention. 

        Args:
            force (list): A 3-dimensional vector of floats with the force [Fx, Fy, Fz] on the body axis of the vehicle according to a FLU convention.
            pos (list): _description_. Defaults to [0.0, 0.0, 0.0].
            body_part (str): . Defaults to "/body".
        """

        rb = self._get_rigid_body_prim(body_part)
        if not rb.is_physics_tensor_entity_valid():
            return
        rb.apply_forces_and_torques_at_pos(
            forces=np.array([force]),
            positions=np.array([pos]),
            local_frame=True
        )

    def apply_torque(self, torque, body_part="/body"):
        """
        Method that when invoked applies a given torque vector to /<rigid_body_name>/"body" or to /<rigid_body_name>/<body_part>.

        Args:
            torque (list): A 3-dimensional vector of floats with the force [Tx, Ty, Tz] on the body axis of the vehicle according to a FLU convention.
            body_part (str): . Defaults to "/body".
        """

        rb = self._get_rigid_body_prim(body_part)
        if not rb.is_physics_tensor_entity_valid():
            return
        rb.apply_forces_and_torques_at_pos(
            torques=np.array([torque]),
            local_frame=True
        )

    def update_state(self, dt: float):
        """
        Method that is called at every physics step to retrieve and update the current state of the vehicle, i.e., get
        the current position, orientation, linear and angular velocities and acceleration of the vehicle.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # In Isaac Sim 6.0, the ArticulationController is initialized by world.reset()
        # via scene._finalize(), which calls Vehicle.initialize() with a valid physics_sim_view.
        # If world.reset() has not been called yet, skip state updates until it is.
        if not self._articulation_initialized:
            return

        rb = self._get_rigid_body_prim("/body")

        # Always get attitude via USD world transform (available immediately)
        prim = self._world.stage.GetPrimAtPath(self._stage_prefix + "/body")
        rotation_quat = get_world_transform_xform(prim).GetQuaternion()
        rotation_quat_real = rotation_quat.GetReal()
        rotation_quat_img = rotation_quat.GetImaginary()

        if rb.is_physics_tensor_entity_valid():
            # Get position and velocities from the physics tensor API
            positions, _ = rb.get_world_poses()
            position = np.array(positions[0])
            linear_vels, angular_vels = rb.get_velocities()
            linear_vel = np.array(linear_vels[0])
            ang_vel = np.array(angular_vels[0])
        else:
            # Physics tensor not ready yet — fall back to USD world transform for position, zero velocities
            world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
            position = np.array(world_transform.ExtractTranslation())
            linear_vel = np.zeros(3)
            ang_vel = np.zeros(3)

        # Get the linear acceleration of the body relative to the inertial frame, expressed in the inertial frame
        # Note: we must do this approximation, since the Isaac sim does not output the acceleration of the rigid body directly
        linear_acceleration = (np.array(linear_vel) - self._state.linear_velocity) / dt

        # Update the state variable X = [x,y,z]
        self._state.position = np.array(position)

        # Get the quaternion according in the [qx,qy,qz,qw] standard
        self._state.attitude = np.array(
            [rotation_quat_img[0], rotation_quat_img[1], rotation_quat_img[2], rotation_quat_real]
        )

        # Express the velocity of the vehicle in the inertial frame X_dot = [x_dot, y_dot, z_dot]
        self._state.linear_velocity = np.array(linear_vel)

        # The linear velocity V =[u,v,w] of the vehicle's body frame expressed in the body frame of reference
        # Note that: x_dot = Rot * V
        self._state.linear_body_velocity = (
            Rotation.from_quat(self._state.attitude).inv().apply(self._state.linear_velocity)
        )

        # omega = [p,q,r]
        self._state.angular_velocity = Rotation.from_quat(self._state.attitude).inv().apply(np.array(ang_vel))

        # The acceleration of the vehicle expressed in the inertial frame X_ddot = [x_ddot, y_ddot, z_ddot]
        self._state.linear_acceleration = linear_acceleration

    def start(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    def stop(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in
        simulation based on the motor speed. This method must be implemented
        by a class that inherits this type and it's called periodically by the physics engine.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        pass

    def update_sensors(self, dt: float):
        """Callback that is called at every physics steps and will call the sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_sensor method will also be called for
        every backend. For example, if new data is generated for an IMU and we have a PX4MavlinkBackend, then the update_sensor
        method will be called for that backend so that this data can latter be sent thorugh mavlink.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._sensors:
            sensor_data = sensor.update(self._state, dt)

            # If some data was updated and we have a mavlink backend or ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_sensor(sensor.sensor_type, sensor_data)

    def update_graphical_sensors(self, event):
        """Callback that is called at every rendering steps and will call the graphical_sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_graphical_sensor method will also be called for
        every backend. For example, if new data is generated for a monocular camera and we have a ROS2Backend, then the update_graphical_sensor
        method will be called for that backend so that this data can latter be sent through a ROS2 topic.

        Args:
            event (float): The timer event that contains the time elapsed between the previous and current function calls (s).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._graphical_sensors:
            sensor_data = sensor.update(self._state, event.payload['dt'])

            # If some data was updated and we have a ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_graphical_sensor(sensor.sensor_type, sensor_data)

    def update_sim_state(self, dt: float):
        """
        Callback that is used to "send" the current state for each backend being used to control the vehicle. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        for backend in self._backends:
            backend.update_state(self._state)