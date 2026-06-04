# ruff: noqa

import contextlib
import csv
import dataclasses
import datetime
import faulthandler
import os
import signal
import socket
import sys
import time
from typing import Optional

import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
from droid.misc.parameters import hand_camera_id, varied_camera_1_id
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # RealSense camera serials.
    external_camera_id: str = varied_camera_1_id  # D435 external camera.
    wrist_camera_id: str = hand_camera_id  # D435I wrist camera.

    # Kept for compatibility with OpenPI's DROID command line. With this
    # two-RealSense setup, only "left" is available.
    external_camera: str = "left"

    max_timesteps: int = 600
    open_loop_horizon: int = 8

    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    save_video: bool = True
    results_dir: str = "results"

    enable_tesollo_gripper: bool = False
    openteach_root: str = "/home/kindred/Desktop/repo/Open-Teach"
    tesollo_topic_prefix: Optional[str] = None
    tesollo_controller_name: str = "delto_3f_controller"
    tesollo_gripper_min: float = 0.0
    tesollo_gripper_max: float = 2.0


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    if args.external_camera != "left":
        raise ValueError(
            "This RealSense setup has one external camera. Run with --external_camera=left "
            "or omit the flag."
        )

    print("Checking policy server at {}:{}...".format(args.remote_host, args.remote_port))
    _check_policy_server_reachable(args.remote_host, args.remote_port)
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print("Connected to the OpenPI policy server.")

    tesollo_gripper = _create_tesollo_gripper(args)

    camera_kwargs = dict(
        hand_camera=dict(image=True, depth=False, pointcloud=False, resolution=(320, 240), resize_func="cv2"),
        varied_camera=dict(image=True, depth=False, pointcloud=False, resolution=(320, 240), resize_func="cv2"),
    )

    env = RobotEnv(
        action_space="joint_velocity",
        gripper_action_space="position",
        camera_kwargs=camera_kwargs,
    )
    print("Created the DROID env with RealSense cameras.")

    results = []

    while True:
        instruction = input("Enter instruction: ")
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        video = []
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, env.get_observation(), save_to_disk=t_step == 0)
                video.append(curr_obs["external_image"])

                if (
                    pred_action_chunk is None
                    or actions_from_chunk_completed >= min(args.open_loop_horizon, len(pred_action_chunk))
                ):
                    actions_from_chunk_completed = 0
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs["external_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    with prevent_keyboard_interrupt():
                        raw_action_chunk = policy_client.infer(request_data)["actions"]
                    pred_action_chunk = _normalize_action_chunk(raw_action_chunk)
                    print("Received action chunk with shape {}.".format(pred_action_chunk.shape))

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                raw_gripper_action = float(action[-1].item())
                if tesollo_gripper is not None:
                    tesollo_gripper.move_normalized(raw_gripper_action)

                gripper_action = np.ones((1,)) if raw_gripper_action > 0.5 else np.zeros((1,))
                action = np.concatenate([action[:-1], gripper_action])
                action = np.clip(action, -1, 1)
                env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        os.makedirs(args.results_dir, exist_ok=True)
        video_filename = ""
        if args.save_video and video:
            video_filename = os.path.join(args.results_dir, "video_" + timestamp + ".mp4")
            _write_video(video_filename, video, fps=10)

        success = _prompt_success()
        results.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": video_filename,
                "instruction": instruction,
            }
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    csv_filename = os.path.join(args.results_dir, "eval_" + timestamp + ".csv")
    _write_results(csv_filename, results)
    print("Results saved to " + csv_filename)
    if tesollo_gripper is not None:
        tesollo_gripper.shutdown()


def _check_policy_server_reachable(host, port, timeout=3.0):
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
    except OSError as exc:
        raise RuntimeError(
            "Cannot reach OpenPI policy server at {}:{} before launching the robot. "
            "Start the policy server on the GPU desktop, make sure it is listening on "
            "0.0.0.0 or the desktop LAN IP, and verify that this control laptop can "
            "reach that IP/port. Original error: {}".format(host, port, exc)
        )


def _normalize_action_chunk(actions):
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]

    if actions.ndim != 2:
        raise RuntimeError("Expected policy actions with shape (N, 7) or (N, 8), got {}".format(actions.shape))

    if actions.shape[1] == 7:
        dummy_gripper = np.zeros((actions.shape[0], 1), dtype=actions.dtype)
        actions = np.concatenate([actions, dummy_gripper], axis=1)
    elif actions.shape[1] != 8:
        raise RuntimeError("Expected policy action dim 7 or 8, got shape {}".format(actions.shape))

    return actions


def _create_tesollo_gripper(args):
    if not args.enable_tesollo_gripper:
        return None
    return TesolloGripperAdapter(
        openteach_root=args.openteach_root,
        topic_prefix=args.tesollo_topic_prefix,
        controller_name=args.tesollo_controller_name,
        gripper_min=args.tesollo_gripper_min,
        gripper_max=args.tesollo_gripper_max,
    )


class TesolloGripperAdapter:
    def __init__(
        self,
        *,
        openteach_root,
        topic_prefix,
        controller_name,
        gripper_min,
        gripper_max,
    ):
        if openteach_root not in sys.path:
            sys.path.insert(0, openteach_root)

        from openteach.ros_links.tesollo_control import DexArmControl

        self._controller = DexArmControl(
            topic_prefix=topic_prefix,
            controller_name=controller_name,
            hand_type="right",
        )
        self._gripper_min = float(gripper_min)
        self._gripper_max = float(gripper_max)
        if self._gripper_max <= self._gripper_min:
            raise ValueError("tesollo_gripper_max must be greater than tesollo_gripper_min")

        self._last_command = None
        print(
            "Created Tesollo gripper adapter with normalized range [{:.4f}, {:.4f}].".format(
                self._gripper_min, self._gripper_max
            )
        )

    def move_normalized(self, value):
        normalized_value = float(np.clip(value, 0.0, 1.0))
        bend = self._gripper_min + normalized_value * (self._gripper_max - self._gripper_min)
        command = self._single_dof_to_tesollo_joints(bend)

        if self._last_command is not None and np.max(np.abs(command - self._last_command)) < 1e-3:
            return

        self._controller.move_hand(command)
        self._last_command = command

    def _single_dof_to_tesollo_joints(self, bend):
        command = np.zeros(12, dtype=np.float32)
        for offset in (0, 4, 8):
            command[offset] = 0.0
            command[offset + 1] = 0.0
            command[offset + 2] = bend
            command[offset + 3] = bend
        return command

    def shutdown(self):
        if hasattr(self._controller, "shutdown"):
            self._controller.shutdown()


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    external_image = None
    wrist_image = None

    for key, image in image_observations.items():
        # The suffix is "_left" to match DROID/OpenPI's left stereo image key.
        # For RealSense this is the single RGB stream, not a stereo-left image.
        if args.external_camera_id in key and "left" in key:
            external_image = image
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image

    missing = []
    if external_image is None:
        missing.append("external camera " + args.external_camera_id)
    if wrist_image is None:
        missing.append("wrist camera " + args.wrist_camera_id)
    if missing:
        raise RuntimeError(
            "Missing image observations for "
            + ", ".join(missing)
            + ". Available image keys: "
            + ", ".join(sorted(image_observations.keys()))
        )

    external_image = external_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([external_image, wrist_image], axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "external_image": external_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def _write_video(filename, frames, fps):
    first_frame = frames[0]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer for " + filename)
    try:
        for frame in frames:
            writer.write(frame[..., ::-1])
    finally:
        writer.release()


def _prompt_success() -> float:
    success: Optional[float] = None
    while success is None:
        value = input(
            "Did the rollout succeed? Enter y for 100%, n for 0%, "
            "or a numeric value 0-100 based on the evaluation spec: "
        )
        if value == "y":
            success = 1.0
        elif value == "n":
            success = 0.0
        else:
            success = float(value) / 100

        if not (0 <= success <= 1):
            print("Success must be a number in [0, 100].")
            success = None
    return success


def _write_results(filename, rows):
    fieldnames = ["success", "duration", "video_filename", "instruction"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(tyro.cli(Args))
