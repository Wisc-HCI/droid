from copy import deepcopy

import cv2
import numpy as np

from droid.misc.parameters import hand_camera_id
from droid.misc.time import time_ms

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    print("WARNING: You have not setup the RealSense cameras, and currently cannot use them")


def gather_realsense_cameras():
    all_realsense_cameras = []
    try:
        devices = rs.context().query_devices()
    except NameError:
        return []

    for device in devices:
        all_realsense_cameras.append(RealSenseCamera(device))

    return all_realsense_cameras


resize_func_map = {"cv2": cv2.resize, None: None}

standard_params = dict(width=640, height=480, fps=30)
advanced_params = dict(width=1280, height=720, fps=15)


class RealSenseCamera:
    def __init__(self, camera):
        self.serial_number = str(camera.get_info(rs.camera_info.serial_number))
        self.camera_name = camera.get_info(rs.camera_info.name)
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = None
        self._current_params = None

        print("Opening RealSense: ", self.serial_number, self.camera_name)

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    ### Camera Modes ###
    def set_calibration_mode(self):
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.resizer_resolution = (0, 0)

        params = advanced_params if self.high_res_calibration else standard_params
        if self._current_params != params:
            self._configure_camera(params)
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.stream_resolution = self.traj_resolution
            self.resizer_resolution = (0, 0)
        else:
            self.stream_resolution = (0, 0)
            self.resizer_resolution = self.traj_resolution

        params = deepcopy(standard_params)
        if self.stream_resolution != (0, 0):
            params["width"], params["height"] = self.stream_resolution

        if self._current_params != params:
            self._configure_camera(params)
        self.current_mode = "trajectory"

    def _configure_camera(self, init_params):
        self.disable_camera()

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(self.serial_number)
        self._config.enable_stream(
            rs.stream.color,
            init_params["width"],
            init_params["height"],
            rs.format.bgr8,
            init_params["fps"],
        )
        if self.depth or self.pointcloud:
            self._config.enable_stream(
                rs.stream.depth,
                init_params["width"],
                init_params["height"],
                rs.format.z16,
                init_params["fps"],
            )
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        self._profile = self._pipeline.start(self._config)
        self._current_params = deepcopy(init_params)
        self.latency = int(2.5 * (1e3 / init_params["fps"]))

        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsics()
        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(color_intrinsics),
        }

    ### Calibration Utilities ###
    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]])
        intrinsics["distCoeffs"] = np.array(list(params.coeffs))
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    ### Recording Utilities ###
    def start_recording(self, filename):
        raise NotImplementedError("RealSense recording is not implemented for DROID trajectories")

    def stop_recording(self):
        raise NotImplementedError("RealSense recording is not implemented for DROID trajectories")

    ### Basic Camera Utilities ###
    def _process_frame(self, frame):
        frame = np.asanyarray(frame.get_data()).copy()
        if self.resizer_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resizer_resolution)

    def read_camera(self):
        if self.skip_reading:
            return {}, {}

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        color_frame = frames.get_color_frame()
        if not color_frame:
            return {}, timestamp_dict

        frame_received = int(color_frame.get_timestamp())
        timestamp_dict[self.serial_number + "_frame_received"] = frame_received
        timestamp_dict[self.serial_number + "_estimated_capture"] = frame_received

        data_dict = {}
        if self.image:
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(color_frame),
            }

        if self.depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                data_dict["depth"] = {
                    self.serial_number + "_left": self._process_frame(depth_frame),
                }

        return data_dict, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if hasattr(self, "_pipeline"):
            self._pipeline.stop()
            del self._pipeline
            self._current_params = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
