"""
Legacy snippets archived from 8_agipix.py.

This module is intentionally non-executable and only stores reference snippets
that were previously inline as comments.
"""

LEGACY_NOTES = """
1) Legacy RTX lidar helper that used manual replicator writer setup:

    def create_rtx_lidar(self):
        # Guard against re-initialization (e.g., script re-run in same Kit session)
        if getattr(self, "_lidar_initialized", False):
            carb.log_warn("RTX Lidar already initialized; skipping duplicate creation.")
            return

        # Clear potential stale Replicator pipeline that can introduce cycle warnings
        try:
            import omni.usd
            stage_ctx = omni.usd.get_context()
            stage = stage_ctx.get_stage()
            pipeline_path = "/Render/PostProcess/SDGPipeline"
            if stage.GetPrimAtPath(pipeline_path):
                if rep.orchestrator.get_is_started():
                    rep.orchestrator.stop()
                rep.orchestrator.clear()
        except Exception as e:
            carb.log_warn(f"Could not clear existing replicator pipeline: {e}")

        sensor_prim_path = self.drone._stage_prefix + "/body/lidar_sensor"
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            old = stage.GetPrimAtPath(sensor_prim_path)
            if old and old.IsValid():
                omni.kit.commands.execute("DeletePrims", paths=[sensor_prim_path])
        except Exception as e:
            carb.log_warn(f"Could not remove stale lidar prim: {e}")

        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=sensor_prim_path,
            parent=None,
            config="Mid_360",
            translation=(self.node.lidar_trans[0], self.node.lidar_trans[1], self.node.lidar_trans[2]),
            orientation=Gf.Quatd(self.node.lidar_ori[0], self.node.lidar_ori[1], self.node.lidar_ori[2], self.node.lidar_ori[3]),
            force_camera_prim=False,
        )

        hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
        simulation_app.update()

        try:
            pc_writer = rep.writers.get("RtxLidarROS2PublishPointCloud")
            pc_writer.initialize(topicName=f"{self.vehicle_name}/livox/lidar", frameId=f"{self.vehicle_name}/lidar_link")
            pc_writer.attach([hydra_texture])
        except Exception as e:
            carb.log_error(f"Failed to init lidar point cloud writer: {e}")

        simulation_app.update()
        self._lidar_initialized = True

2) Legacy environment alternatives:

    self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
    self.pg.load_environment(FLAT_ENVIRONMENTS["Hospital"])

3) Legacy high-detail camera profile:

    MonocularCamera(
        "Camera",
        config={
            "depth": True,
            "pixel_size": 3,
            "f_stop": 1.8,
            "focus_distance": 15,
            "position": np.array([0.30, 0.0, 0.0]),
            "orientation": np.array([180.0, -180.0, 0.0]),
            "resolution": (1920, 1200),
            "frequency": 30,
            "intrinsics": np.array([
                [958.8, 0.0, 957.8],
                [0.0, 956.7, 589.5],
                [0.0, 0.0, 1.0],
            ]),
            "distortion_coefficients": np.array([
                0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017
            ]),
            "diagonal_fov": 140.0,
        },
    )
"""
