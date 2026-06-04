import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from queue import Empty, Queue

import h5py
import imageio
import numpy as np
import cv2

from droid.misc.subprocess_utils import run_threaded_command


def write_dict_to_hdf5(hdf5_file, data_dict, keys_to_ignore=["image", "depth", "pointcloud"]):
    for key in data_dict.keys():
        key = str(key)

        # Pass Over Specified Keys #
        if key in keys_to_ignore:
            continue

        # Examine Data #
        curr_data = data_dict[key]
        if type(curr_data) == list:
            curr_data = np.array(curr_data)
        if isinstance(curr_data, np.generic):
            curr_data = curr_data.item()
        dtype = type(curr_data)

        # Unwrap If Dictionary #
        if dtype == dict:
            if key not in hdf5_file:
                hdf5_file.create_group(key)
            write_dict_to_hdf5(hdf5_file[key], curr_data)
            continue

        # Make Room For Data #
        if key not in hdf5_file:
            if dtype != np.ndarray:
                dshape = ()
            else:
                dtype, dshape = curr_data.dtype, curr_data.shape
            if isinstance(curr_data, (str, bytes)):
                dtype = h5py.string_dtype(encoding="utf-8")
            hdf5_file.create_dataset(key, (1, *dshape), maxshape=(None, *dshape), dtype=dtype)
        else:
            hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

        # Save Data #
        hdf5_file[key][-1] = curr_data


class TrajectoryWriter:
    def __init__(self, filepath, metadata=None, exists_ok=False, save_images=True):
        assert (not os.path.isfile(filepath)) or exists_ok
        self._filepath = filepath
        self._save_images = save_images
        self._hdf5_file = h5py.File(filepath, "w")
        self._queue_dict = defaultdict(Queue)
        self._video_writers = {}
        self._video_files = {}
        self._open = True
        self._write_errors = []

        # Add Metadata #
        if metadata is not None:
            self._update_metadata(metadata)

        # Start HDF5 Writer Thread #
        def hdf5_writer(data):
            return write_dict_to_hdf5(self._hdf5_file, data)

        run_threaded_command(self._write_from_queue, args=(hdf5_writer, self._queue_dict["hdf5"]))

    def write_timestep(self, timestep):
        if self._save_images:
            self._update_video_files(timestep)
        self._queue_dict["hdf5"].put(timestep)

    def _update_metadata(self, metadata):
        for key in metadata:
            self._hdf5_file.attrs[key] = deepcopy(metadata[key])

    def _write_from_queue(self, writer, queue):
        while self._open:
            try:
                data = queue.get(timeout=1)
            except Empty:
                continue
            try:
                writer(data)
            except Exception as err:
                self._write_errors.append(err)
                print(f"TrajectoryWriter failed to write timestep: {err}", flush=True)
            finally:
                queue.task_done()

    def _update_video_files(self, timestep):
        image_dict = timestep["observations"]["image"]

        for video_id in image_dict:
            # Get Frame #
            img = image_dict[video_id]
            del image_dict[video_id]

            # Create Writer And Buffer #
            if video_id not in self._video_buffers:
                filename = self.create_video_file(video_id, ".mp4")
                self._video_writers[video_id] = imageio.get_writer(filename, macro_block_size=1)
                run_threaded_command(
                    self._write_from_queue, args=(self._video_writers[video_id].append_data, self._queue_dict[video_id])
                )

            # Add Image To Queue #
            self._queue_dict[video_id].put(img)

        del timestep["observations"]["image"]

    def create_video_file(self, video_id, suffix):
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix)
        self._video_files[video_id] = temp_file
        return temp_file.name

    def close(self, metadata=None):
        # Add Metadata #
        if metadata is not None:
            self._update_metadata(metadata)

        # Finish Remaining Jobs #
        [queue.join() for queue in self._queue_dict.values()]
        write_error = self._write_errors[0] if self._write_errors else None

        # Close Video Writers #
        for video_id in self._video_writers:
            self._video_writers[video_id].close()

        # Save Serialized Videos #
        for video_id in self._video_files:
            # Create Folder #
            if "videos" not in self._hdf5_file["observations"]:
                self._hdf5_file["observations"].create_group("videos")

            # Get Serialized Video #
            self._video_files[video_id].seek(0)
            serialized_video = np.asarray(self._video_files[video_id].read())

            # Save Data #
            self._hdf5_file["observations"]["videos"].create_dataset(video_id, data=serialized_video)
            self._video_files[video_id].close()

        # Close File #
        self._hdf5_file.close()
        self._open = False
        if write_error is not None:
            raise RuntimeError(f"TrajectoryWriter failed with {len(self._write_errors)} write error(s)") from write_error


class MP4TrajectoryRecorder:
    def __init__(self, folderpath, fps=15):
        self._folderpath = folderpath
        self._fps = fps
        self._writers = {}
        self._frame_shapes = {}
        os.makedirs(folderpath, exist_ok=True)

    def write_images(self, image_dict):
        for cam_id, image in sorted(image_dict.items()):
            video_id = self._video_id(cam_id)
            frame = self._prepare_frame(image)

            if video_id not in self._writers:
                height, width = frame.shape[:2]
                filepath = os.path.join(self._folderpath, video_id + ".mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writers[video_id] = cv2.VideoWriter(filepath, fourcc, self._fps, (width, height))
                self._frame_shapes[video_id] = (height, width)

            if frame.shape[:2] != self._frame_shapes[video_id]:
                height, width = self._frame_shapes[video_id]
                frame = cv2.resize(frame, (width, height))

            self._writers[video_id].write(frame)

    @staticmethod
    def _video_id(cam_id):
        return cam_id[:-5] if cam_id.endswith("_left") else cam_id

    @staticmethod
    def _prepare_frame(image):
        frame = np.asarray(image)
        if frame.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape {frame.shape}")
        frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def close(self):
        for writer in self._writers.values():
            writer.release()
        self._writers = {}
