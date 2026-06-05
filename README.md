# The DROID Robot Platform

This repository contains the code for setting up your DROID robot platform and using it to collect teleoperated demonstration data. This platform was used to collect the [DROID dataset](https://droid-dataset.github.io), a large, in-the-wild dataset of robot manipulations.

If you are interested in using the DROID dataset for training robot policies, please check out our [policy learning repo](https://github.com/droid-dataset/droid_policy_learning).
For more information about DROID, please see the following links: 

[**[Homepage]**](https://droid-dataset.github.io) &ensp; [**[Documentation]**](https://droid-dataset.github.io/droid) &ensp; [**[Paper]**](https://arxiv.org/abs/2403.12945) &ensp; [**[Dataset Visualizer]**](https://droid-dataset.github.io/dataset.html).

![](https://droid-dataset.github.io/droid/assets/index/droid_teaser.jpg)

---------
## Setup Guide

We assembled a step-by-step guide for setting up the DROID robot platform in our [developer documentation](https://droid-dataset.github.io/droid).
This guide has been used to set up 18 DROID robot platforms over the course of the DROID dataset collection. Please refer to the steps in this guide for setting up your own robot. Specifically, you can follow these key steps:

1. [Hardware Assembly and Setup](https://droid-dataset.github.io/droid/docs/hardware-setup)
2. [Software Installation and Setup](https://droid-dataset.github.io/droid/docs/software-setup)
3. [Example Workflows to collect data or calibrate cameras](https://droid-dataset.github.io/droid/docs/example-workflows)

If you encounter issues during setup, please raise them as issues in this github repo.

----------
## (Kindred's note) Openpi - Control laptop - NUC Pipeline:

### Step 1: Start a policy server

Since the DROID control laptop does not have a powerful GPU, we will start a remote policy server on a different machine with a more powerful GPU and then query it from the DROID control laptop during inference.

1. On a machine with a powerful GPU (~NVIDIA 4090), clone and install the `openpi` repository following the instructions in the [README](https://github.com/Physical-Intelligence/openpi). I have set it up in the lab's 4090 desktop.
2. Start the OpenPI server via the following command:
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```
If you want to run a fine-tuned policy, change the policy.dir to your policy

### Step 2: Run the DROID robot

1. Follow the instrution [here](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-nuc) to setup the NUC. I have set it up on our lab's laptop with Real-time kernel.
2. On NUC, run:
```bash
cd Desktop/github/droid
conda activate polymetis-local
python scripts/server/run_server.py
```
3. Follow the instrution [here](https://droid-dataset.github.io/droid/software-setup/host-installation.html#configuring-the-laptopworkstation) to setup the control laptop, which should be your own laptop, since we are using realsense camera instead of ZED camera, we need to install `pyrealsense2` package:
```
pip install pyrealsense2
```
4. Clone the openpi repo and install the openpi client, which we will use to connect to the policy server (this has very few dependencies and should be very fast to install): with the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`.
5. Install `tyro`, which we will use for command line parsing: `pip install tyro`.
~~6. Copy the `main.py` file from this directory to the `$DROID_ROOT/scripts` directory.~~
~~7. Replace the camera IDs in the `main.py` file with the IDs of your cameras (you can find the camera IDs by running `ZED_Explorer` in the command line, which will open a tool that shows you all connected cameras and their IDs -- you can also use it to make sure that the cameras are well-positioned to see the scene you want the robot to interact with).~~ (You don't have to do this because I have already set it up for you.)
8. Run the `main.py` file. Make sure to point the IP and host address to the policy server. (To make sure the server machine is reachable from the DROID laptop, you can run `ping 192.168.4.5` from the DROID laptop.) Also make sure to specify the external camera to use for the policy (we only input one external camera), choose from ["left", "right"].

```bash
python3 scripts/main.py --remote_host=192.168.4.5 --remote_port=8000 --external_camera="left"
```

