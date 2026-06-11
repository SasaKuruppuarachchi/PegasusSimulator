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

import omni.replicator.core as rep
from pxr import UsdGeom
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
        """

        # Initialize the Super class "object" attributes
        super().__init__(sensor_type="MonocularCamera", update_rate=config.get("frequency", 60.0))        
        
        # Setup the name of the camera primitive path
        self._camera_name = camera_name
        self._stage_prim_path = ""

        # Configurations of the camera
        self._depth = config.get("depth", True)
        self._position = config.get("position", np.array([0.30, 0.0, 0.0]))
        self._orientation = config.get("orientation", np.array([0.0, 0.0, 180.0]))
        self._resolution = config.get("resolution", (1920, 1200))
        self._frequency = config.get("frequency", 30)
        self._intrinsics = config.get("intrinsics", np.array([[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]))
        self._distortion_coefficients = config.get("distortion_coefficients", np.array([0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]))
        self._diagonal_fov = config.get("diagonal_fov", 140.0)

        # Setup an empty camera output dictionary
        self._state = {}
        self._camera_full_set = False
        self._render_product = None
        self._render_product_path = None
        self._camera_prim = None

        self.counter = 0


    def initialize(self, vehicle):
        
        # Initialize the Super class "object" attributes
        super().initialize(vehicle)

        # Get the complete stage prefix for the camera
        self._stage_prim_path = get_stage_next_free_path(PegasusInterface().world.stage, self._vehicle.prim_path + "/body/" + self._camera_name, False)

        # Get the camera name that was actually created (and update the camera name)
        self._camera_name = self._stage_prim_path.rpartition("/")[-1]

        # Create a camera prim using USD (standard USD camera)
        stage = PegasusInterface().world.stage
        camera_prim = stage.DefinePrim(self._stage_prim_path, "Camera")
        self._camera_prim = camera_prim

        # Set camera position/orientation using USD attributes
        from pxr import Gf
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(self._position[0]), float(self._position[1]), float(self._position[2])))
        orient_op = xformable.AddOrientOp()
        orient_scipy = Rotation.from_euler("ZYX", self._orientation, degrees=True).as_quat()  # [x, y, z, w]
        orient_op.Set(Gf.Quatf(float(orient_scipy[3]), float(orient_scipy[0]), float(orient_scipy[1]), float(orient_scipy[2])))
        
    def start(self):

        # Set the camera intrinsics
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = self._intrinsics

        # Create render product for the camera
        self._render_product = rep.create.render_product(self._stage_prim_path, self._resolution)
        self._render_product_path = self._render_product.path

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
            self._state["render_product_path"] = self._render_product_path
            #self._state["image"] = self._camera.get_rgba()[:, :, :3]
            self._state["height"] = self._resolution[1]
            self._state["width"] = self._resolution[0]
            self._state["frequency"] = self._frequency

            # Check if we want to get the depth image
            #if self._depth:
            #    self._state["depth"] = self._camera.get_depth()

            camera_prim = PegasusInterface().world.stage.GetPrimAtPath(self._stage_prim_path)
            projection = camera_prim.GetAttribute("projection").Get()
            if projection == "perspective":
                focal_length = camera_prim.GetAttribute("focalLength").Get()
                h_aperture = camera_prim.GetAttribute("horizontalAperture").Get()
                v_aperture = camera_prim.GetAttribute("verticalAperture").Get()
                if focal_length and h_aperture and v_aperture:
                    fx = focal_length * self._resolution[0] / h_aperture
                    fy = focal_length * self._resolution[1] / v_aperture
                    cx = self._resolution[0] / 2.0
                    cy = self._resolution[1] / 2.0
                    self._state["intrinsics"] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
        # If something goes wrong during the data acquisition, just return None
        except:
            self._state = None

        return self._state
