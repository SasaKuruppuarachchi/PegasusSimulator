"""
| File: monocular_camera.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Simulates a monocular camera attached to the vehicle
"""
__all__ = ["MonocularCamera"]

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.graphical_sensors import GraphicalSensor
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from isaacsim.sensors.camera import Camera
from omni.usd import get_stage_next_free_path

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation


class MonocularCamera(GraphicalSensor):
    """
    The class that implements a monocular camera sensor. This class inherits the base class GraphicalSensor.
    """

    def __init__(self, camera_name, config={}):
        """
        Initialize the MonocularCamera class
        
        Check the oficial documentation for the Camera class in Isaac Sim: 
        https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html#isaac-sim-sensors-camera

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the MonocularCamera - it can be empty or only have some of the parameters used by the MonocularCamera.

        Examples:
            The dictionary default parameters are

            >>> {"depth": True,
            >>> "position": np.array([0.30, 0.0, 0.0]),
            >>> "orientation": np.array([0.0, 0.0, 0.0]),
            >>> "resolution": (1920, 1200),
            >>> "frequency": 30,
            >>> "intrinsics": np.array([[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]),
            >>> "distortion_coefficients": np.array([0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]),
            >>> "diagonal_fov": 140.0}
            
            From https://docs.isaacsim.omniverse.nvidia.com/5.0.0/sensors/isaacsim_sensors_camera.html
            
            # Desired image resolution, camera intrinsics matrix, and distortion coefficients
            width, height = 1920, 1200
            camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]
            distortion_coefficients = [0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]

            # Camera sensor size and optical path parameters. These parameters are not the part of the
            # OpenCV camera model, but they are nessesary to simulate the depth of field effect.
            #
            # Note: To disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
            # Set pixel size (microns)
            pixel_size = 3
            # Set f-number, the ratio of the lens focal length to the diameter of the entrance pupil (unitless)
            f_stop = 1.8
            # Set focus distance (meters) - chosen as distance from camera to cube
            focus_distance = 1.5
            
            camera.initialize()

            # Calculate the focal length and aperture size from the camera matrix
            ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix  # fx, fy are in pixels, cx, cy are in pixels
            horizontal_aperture = pixel_size * width * 1e-6  # convert to meters
            vertical_aperture = pixel_size * height * 1e-6  # convert to meters
            focal_length_x = pixel_size * fx * 1e-6  # convert to meters
            focal_length_y = pixel_size * fy * 1e-6  # convert to meters
            focal_length = (focal_length_x + focal_length_y) / 2  # convert to meters

            # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
            camera.set_focal_length(focal_length)
            camera.set_focus_distance(focus_distance)
            camera.set_lens_aperture(f_stop)
            camera.set_horizontal_aperture(horizontal_aperture)
            camera.set_vertical_aperture(vertical_aperture)

            camera.set_clipping_range(0.05, 1.0e5)

            # Set the distortion coefficients
            camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=distortion_coefficients)
        """

        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="MonocularCamera", update_rate=config.get("frequency", 60.0))        
        
        # Setup the name of the camera primitive path
        self._camera_name = camera_name
        self._stage_prim_path = ""

        # Configurations of the camera
        
        self._depth = config.get("depth", True)
        self._mode = config.get("raw_calib_mode", False) 
        self._pixel_size = config.get("pixel_size", 3)
        # Set f-number, the ratio of the lens focal length to the diameter of the entrance pupil (unitless)
        self._f_stop = config.get("f_stop", 1.8)
        # Set focus distance (meters) - chosen as distance from camera to cube
        self._focus_distance = config.get("focus_distance", 1.5)
        self._position = config.get("position", np.array([0.30, 0.0, 0.0]))
        self._orientation = config.get("orientation", np.array([0.0, 0.0, 180.0]))
        self._resolution = config.get("resolution", (1920, 1200))
        self._frequency = config.get("frequency", 30)
        self._intrinsics = config.get("intrinsics", [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]])
        self._distortion_coefficients = config.get("distortion_coefficients", None)
        self._diagonal_fov = config.get("diagonal_fov", 140.0)
        
        self.focal_length = config.get("focal_length", 0.0036)  # meters
        self.horizontal_aperture = config.get("horizontal_aperture", 0.0032)  # meters
        self.vertical_aperture = config.get("vertical_aperture", 0.0024)  # meters

        # Setup an empty camera output dictionary
        self._state = {}
        self._camera_full_set = False

        self.counter = 0


    def initialize(self, vehicle):
        
        # Initialize the Super class "object" attributes
        super().initialize(vehicle)

        # Get the complete stage prefix for the camera
        self._stage_prim_path = get_stage_next_free_path(PegasusInterface().world.stage, self._vehicle.prim_path + "/body/" + self._camera_name, False)

        # Get the camera name that was actually created (and update the camera name)
        self._camera_name = self._stage_prim_path.rpartition("/")[-1]

        # Create the camera object attached to the rigid body vehicle
        self._camera = Camera(
            prim_path=self._stage_prim_path,
            frequency=self._frequency,
            resolution=self._resolution)
        
        # Set the camera position locally with respect to the drone
        self._camera.set_local_pose(np.array(self._position), Rotation.from_euler("ZYX", self._orientation, degrees=True).as_quat())
        
    def start(self):

        # Start the camera
        self._camera.initialize()

        # Calculate the focal length and aperture size from the camera matrix
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = self._intrinsics  # fx, fy are in pixels, cx, cy are in pixels
        (width, height) = self._resolution
        horizontal_aperture = self._pixel_size * width * 1e-6  # convert to meters
        vertical_aperture = self._pixel_size * height * 1e-6  # convert to meters
        focal_length_x = self._pixel_size * fx * 1e-6  # convert to meters
        focal_length_y = self._pixel_size * fy * 1e-6  # convert to meters
        focal_length = (focal_length_x + focal_length_y) / 2  # convert to meters
        
        if self._mode:
            focal_length = self.focal_length
            horizontal_aperture = self.horizontal_aperture
            vertical_aperture = self.vertical_aperture

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        self._camera.set_focal_length(focal_length)
        self._camera.set_focus_distance(self._focus_distance)
        self._camera.set_lens_aperture(self._f_stop)
        self._camera.set_horizontal_aperture(horizontal_aperture)
        self._camera.set_vertical_aperture(vertical_aperture)

        self._camera.set_clipping_range(0.01, 1.0e6)

        # Set the distortion coefficients

        #self._camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=self._distortion_coefficients)
        # Isaac Sim's camera API internally does `coefficients or []` which triggers a truth-value
        # evaluation on numpy arrays producing: ValueError: The truth value of an array with more than one element is ambiguous.
        # Therefore ensure we pass a plain (possibly flattened) python list.
        try:
            if self._distortion_coefficients is None:
                coeffs = []
            else:
                # Flatten in case user provided as matrix/column vector
                if isinstance(self._distortion_coefficients, np.ndarray):
                    coeffs = self._distortion_coefficients.reshape(-1).tolist()
                else:
                    # Convert any iterable (tuple, list) to list
                    coeffs = list(self._distortion_coefficients)
            self._camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=coeffs)
        except Exception as e:
            # Fallback: disable distortion if something unexpected occurs
            print(f"[MonocularCamera] Warning: could not set distortion coefficients ({e}). Proceeding without lens distortion.")
            self._camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=[])
            
        # Check if depth is enabled, if so, set the depth properties
        if self._depth:
            self._camera.add_distance_to_image_plane_to_frame()

        # Signal that the camera is fully set
        self._camera_full_set = True

    def stop(self):
        self._camera_full_set = False

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state


    @GraphicalSensor.update_at_rate
    def update(self, state: State, dt: float):
        """Method that gets the current RGB image from the camera and returns it as a dictionary.

        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).

        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor)
        """

        while self.counter < 100:
            self.counter += 1
            return

        # If all the camera properties are not set yet, return None
        if not self._camera_full_set:
            return None

        # Get the data from the camera
        # TODO: Fix this feature later
        try:
            self._state = {}
            self._state["camera_name"] = self._camera_name
            self._state["stage_prim_path"] = self._stage_prim_path
            #self._state["image"] = self._camera.get_rgba()[:, :, :3]
            self._state["height"] = self._resolution[1]
            self._state["width"] = self._resolution[0]
            self._state["frequency"] = self._frequency
            self._state["camera"] = self._camera

            # Check if we want to get the depth image
            #if self._depth:
            #    self._state["depth"] = self._camera.get_depth()

            if self._camera.get_lens_distortion_model() == "pinhole":
                self._state["intrinsics"] = self._camera.get_intrinsics_matrix()
            
        # If something goes wrong during the data acquisition, just return None
        except:
            self._state = None

        return self._state
