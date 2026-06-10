# The DROID Robot Platform

This repository contains the code for setting up your DROID robot platform and using it to collect teleoperated demonstration data. This platform was used to collect the [DROID dataset](https://droid-dataset.github.io), a large, in-the-wild dataset of robot manipulations.

If you are interested in using the DROID dataset for training robot policies, please check out our [policy learning repo](https://github.com/droid-dataset/droid_policy_learning).
For more information about DROID, please see the following links:

[**[Homepage]**](https://droid-dataset.github.io) &ensp; [**[Documentation]**](https://droid-dataset.github.io/droid) &ensp; [**[Paper]**](https://arxiv.org/abs/2403.12945) &ensp; [**[Dataset Visualizer]**](https://droid-dataset.github.io/dataset.html).

![](https://droid-dataset.github.io/droid/assets/index/droid_teaser.jpg)

---

## Setup Guide

We assembled a step-by-step guide for setting up the DROID robot platform in our [developer documentation](https://droid-dataset.github.io/droid).
This guide has been used to set up 18 DROID robot platforms over the course of the DROID dataset collection. Please refer to the steps in this guide for setting up your own robot. Specifically, you can follow these key steps:

1. [Hardware Assembly and Setup](https://droid-dataset.github.io/droid/docs/hardware-setup)
2. [Software Installation and Setup](https://droid-dataset.github.io/droid/docs/software-setup)
3. [Example Workflows to collect data or calibrate cameras](https://droid-dataset.github.io/droid/docs/example-workflows)

If you encounter issues during setup, please raise them as issues in this github repo.

---

## (Kindred's note) Openpi - Control laptop - NUC Pipeline:

### IP Configuration:

Openpi desktop (Lab's 4090): 192.168.4.5

Control laptop: 192.168.4.6 (You should set it up on your own control laptop)

NUC (Lab's RT-kernel laptop): 192.168.4.4

Left Franka arm: 192.168.4.3

### Step 1: Start a policy server

Since the DROID control laptop does not have a powerful GPU, we will start a remote policy server on a different machine with a more powerful GPU and then query it from the DROID control laptop during inference.

1. On a machine with a powerful GPU (~NVIDIA 4090), clone and install the `openpi` repository following the instructions in the [README](https://github.com/Physical-Intelligence/openpi). I have set it up in the lab's 4090 desktop.
2. Start the OpenPI server via the following command:

```bash
cd ~/repo/openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

If you want to run a fine-tuned policy, change the `policy.dir` to your policy

### Step 2: Run the DROID robot

1. Follow the instrution [here](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-nuc) to setup the NUC. I have set it up on our lab's laptop with Real-time kernel.
2. On NUC, run:

```bash
cd Desktop/repo/droid
conda activate polymetis-local
python scripts/server/run_server.py
```

3. Follow the instrution [here](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-laptopworkstation) to setup the control laptop, which should be your own laptop, since we are using realsense camera instead of ZED camera, we need to install `pyrealsense2` package:

```
pip install pyrealsense2
```

4. Clone the openpi repo and install the openpi client, which we will use to connect to the policy server (this has very few dependencies and should be very fast to install): with the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`.
5. Install `tyro`, which we will use for command line parsing: `pip install tyro`.
6. ~~Copy the `main.py` file from this directory to the `$DROID_ROOT/scripts` directory.~~
7. ~~Replace the camera IDs in the `main.py` file with the IDs of your cameras (you can find the camera IDs by running `ZED_Explorer` in the command line, which will open a tool that shows you all connected cameras and their IDs -- you can also use it to make sure that the cameras are well-positioned to see the scene you want the robot to interact with).~~ (You don't have to do this because I have already set it up for you.)
8. Run the `main.py` file. Make sure to point the IP and host address to the policy server. (To make sure the server machine is reachable from the DROID laptop, you can run `ping 192.168.4.5` from the DROID laptop.) Also make sure to specify the external camera to use for the policy (we only input one external camera), choose from ["left", "right"].

```bash
python3 scripts/main.py --remote_host=192.168.4.5 --remote_port=8000 --external_camera="left" --max_timesteps=800

```

### Finetune instructions

1. First, log in to CHTC with your netID:

```
ssh <netID>@ap2002.chtc.wisc.edu
```

2. Clone openpi repo:

```
git clone https://github.com/Wisc-HCI/openpi.git
git checkout pi05-droid-finetune
```

3. Submit the job:

```
condor_submit pi05_droid_finetune.sub
```

You can check the status of the job by running `condor_q` and you will receive a unique job submission number. Check the log and error messages by running:

```
tail -f ~/logs/pi05_droid_<sub number>.err # or pi05_droid_<sub number>.log
```

Once it's completed, transfer the checkpoint to our 4090 desktop:

```
rsync -avh --progress \
  <netID>@ap2002.chtc.wisc.edu:~/openpi/checkpoints/pi05_droid_finetune/realsense_droid/ \
  ~/Desktop/repo/openpi/checkpoints/pi05_droid_finetune/realsense_droid/
```

---

## (Callie's note) Dockerized control-laptop setup with openpi

This section documents the **Docker-based** control-laptop workflow on an
Arch/CachyOS host with **Intel RealSense** cameras. It supersedes the manual
host-side install in steps 4–7 of _Kindred's note_ above: the openpi client is
now **baked into the laptop Docker image**, and `scripts/main.py` is already
adapted for RealSense, so there is nothing to `pip install` by hand.

### Why Docker (and why the host OS doesn't matter)

The DROID robot stack (Polymetis, RealSense SDK, the `droid` package) and the
"Ubuntu-only" requirement in the upstream docs all live **inside** the
`ghcr.io/droid-dataset/droid_laptop:panda` image, which is built on
`nvidia/cuda:12.1.0-devel-ubuntu22.04`. The host being Arch/CachyOS is therefore
irrelevant — Docker provides the Ubuntu userland. The GPU is exposed to the
container through the NVIDIA Container Toolkit (`nvidia-ctk`).

> Arch host prerequisites (the official `scripts/setup/laptop_setup.sh` is
> Ubuntu/`apt`-only, so install these with `pacman` instead):
> `nvidia-container-toolkit`, `git-lfs`, `android-tools`, then
> `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

### What is baked into the image

`.docker/laptop/Dockerfile.laptop` installs the openpi client into the
container's `robot` conda env (Python 3.7) with these lines:

```dockerfile
RUN git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git /opt/openpi && \
    pip install msgpack websockets pillow dm-tree tyro && \
    pip install -e /opt/openpi/packages/openpi-client --no-deps
```

- **`--no-deps` is mandatory.** `openpi-client` pins `numpy>=1.22.4`, but the
  `robot` env is Python 3.7, which caps numpy at `1.21.6`. A normal
  `pip install -e .` fails on that constraint. The client's actual runtime deps
  (`msgpack`, `websockets`, `pillow`, `dm-tree`) are installed explicitly above
  and all have 3.7-compatible builds; the client code does not need numpy ≥1.22.
- `tyro` (CLI parsing for `main.py`) is installed in the same step.
- `cv2`, `tqdm`, and `PIL` — also used by `main.py` — are already present in the
  `robot` env, so nothing else is required.

`scripts/main.py` is committed and already adapted for the two-RealSense setup
(no ZED, no camera-ID editing needed):

- external camera = `varied_camera_1_id`, wrist camera = `hand_camera_id`
  (read from `droid/misc/parameters.py`),
- single RGB stream matched on the `_left` key, BGR→RGB conversion,
- a policy-server reachability check before the robot is created,
- `cv2`/`csv` for video/results (not `moviepy`/`pandas`),
- an optional Tesollo gripper hook (`--enable_tesollo_gripper`).

Two quality-of-life changes are also included:

- **`.dockerignore`** — keeps `.git`, `docs`, runtime `data/`/`cache/`/`results/`
  out of the build context (cuts it from ~1.4 GB to ~0.5 GB).
- **`scripts/main.py` is bind-mounted** in `docker-compose-laptop.yaml`, so you
  can edit it on the host and re-run **without rebuilding the image**.

> Note: because `Dockerfile.laptop` lives inside the `COPY . /app` context and
> cannot be `.dockerignore`d (the entrypoint script lives next to it), **editing
> the Dockerfile forces a full image rebuild**. Editing `main.py` does not.

### Build the laptop image (Arch host)

The upstream `laptop_setup.sh` exports build variables from
`droid/misc/parameters.py` and then runs `docker compose build`. On Arch you can
do the build step directly:

```bash
cd ~/Documents/GitHub/droid
export ROOT_DIR="$PWD" ROBOT_TYPE=panda LIBFRANKA_VERSION=0.9.0 \
       NUC_IP=192.168.4.4 ROBOT_IP=192.168.4.3 LAPTOP_IP=192.168.4.6
docker compose -f .docker/laptop/docker-compose-laptop.yaml build
```

### Run inference — full startup order

The pieces connect to each other on startup, so bring them up **in this order**:

1. **Robot.** Power on the Franka, open Franka Desk at `https://192.168.4.3`,
   unlock the joints, and **enable FCI**. (Polymetis connects to the FCI on
   startup, so this must be live first.)
2. **NUC (`192.168.4.4`).** Start the Polymetis control server (see _Kindred's
   note_ step 2). The laptop's `RobotEnv` connects to it over the LAN using
   `nuc_ip` from `parameters.py`.
3. **Policy server (4090 desktop, `192.168.4.5`).** In the openpi repo on the
   desktop:
   ```bash
   uv run scripts/serve_policy.py policy:checkpoint \
     --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
   # serves on 0.0.0.0:8000  (use your fine-tuned --policy.dir for the custom policy)
   ```
   You do **not** need a direct laptop↔desktop cable — both just need to be on
   the shared `192.168.4.x` LAN. Verify reachability from the laptop first:
   `ping 192.168.4.5`. (A wired laptop connection is recommended to reduce
   inference latency; 0.5–1 s per action chunk is normal.)
4. **Control laptop (this machine).** Start the container and run `main.py`
   inside it, pointing at the policy server:
   ```bash
   python3 scripts/main.py --remote_host=192.168.4.5 --remote_port=8000 \
       --external_camera="left" --max_timesteps=800
   ```
   `main.py` is interactive — it prompts for a free-form language instruction and,
   after each rollout, for a success score. Because of this, prefer launching it
   in an interactive shell inside the container rather than relying on the
   compose `command:` (which currently runs `python /app/scripts/main.py` with no
   `--remote_host`, i.e. it would default to `0.0.0.0`). To wire the server IP
   into the one-shot `docker compose up` path instead, change that `command:` to
   include `--remote_host=192.168.4.5 --remote_port=8000`.

### Troubleshooting

| Issue                                          | Fix                                                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Cannot reach OpenPI policy server` on startup | Confirm the server is running on the 4090 and listening on `0.0.0.0:8000`; `ping 192.168.4.5` from the laptop; check both are on the `192.168.4.x` LAN. |
| `RobotEnv` hangs/fails to create               | The NUC Polymetis server (step 2) isn't up, or the robot FCI (step 1) isn't enabled.                                                                    |
| Missing image observations for a camera        | RealSense serials in `droid/misc/parameters.py` don't match the connected cameras; the error message lists the available image keys.                    |
| Editing the Dockerfile triggers a long rebuild | Expected — see the note above. Edit `main.py` (bind-mounted) instead when possible.                                                                     |
