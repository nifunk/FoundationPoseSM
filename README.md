# FoundationPoseROSSM: Multi-Object Pose Estimation and Tracking of Novel Objects

This repo is forked from [FoundationPoseROS2](https://github.com/ammar-n-abbas/FoundationPoseROS2). The only difference compared to the original repo is that the interface to query the pose estimation pipeline has been changed from ROS2 to python multiprocessing, making the framework
more modular and better compatible e.g. also with ROS 1.

<p align="center">
  <img src="assets/demo.gif" alt="Demo Video" width="330">
  <img src="assets/demo_robot.gif" alt="Robot Demo Video" width="434"><br>
</p>

FoundationPoseSM is a python shared memory-integrated system for 6D object pose estimation and tracking, based on the FoundationPose architecture. It uses RealSense2 with the Segment Anything Model 2 (SAM2) framework for end-to-end, model-based, real-time pose estimation and tracking of novel objects.

It is built on top of [FoundationPose](https://github.com/NVlabs/FoundationPose) and [live-pose](https://github.com/Kaivalya192/live-pose).

The main advantages to the previous repositories and [isaac_ros_foundationpose](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_foundationpose):
1. shared--memory based real-time framework that works with 8GB GPU, unlike [isaac_ros_foundationpose](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_foundationpose) which requires more than 64GB GPU.
2. SAM2-based automatic segmentation of the objects
3. Multi-object pose estimation and tracking
4. End-to-end assignment of object models with the segmented masks

Furthermore, it provides an interactive GUI for object model-to-mask assignment for end-to-end multi-pose estimation and tracking.

## Prerequisites

- **Ubuntu**
- **Minimum 8GB NVIDIA GPU**
- **CUDA 12.x**
- **Intel RealSense Camera**


## Dependencies

```bash
# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
```

## Env setup: conda 

```bash
# Clone repository
git clone https://github.com/ammar-n-abbas/FoundationPoseROS2.git
```

```bash
# Create conda environment
conda create -n foundationpose_ros python=3.10 -y

# Activate conda environment
conda activate foundationpose_ros
```

```bash
# Installing the correct cuda drivers inside the conda environment
# https://stackoverflow.com/questions/72684130/how-to-set-the-cuda-path-in-the-conda-environment
conda env config vars set CUDA_HOME=""
conda activate foundationpose_ros

# Install CUDA (e.g. 12.1) 
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
```

```bash
# Build extensions
cd FoundationPoseROS2 && export PATH=/usr/local/<YOUR_cuda-12.X_VERSION>/bin${PATH:+:${PATH}}~ && bash build_all_conda.sh
```
> [!IMPORTANT]
> In the `setup.py` file located at `/FoundationPose/bundlesdf/mycuda/`, the C++ flags should be updated from **C++14** to **C++17** for compatibility with newer Nvidia GPUs. It is located at lines 18 and 19 in the file's `nvcc_flags` and c_flags section.


## Run model-based demo

```bash
# Run camera node
source /opt/ros/<ROS_DISTRO>/setup.bash && ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true pointcloud.enable:=true
```

In a separate terminal

```bash
# Run foundationpose_ros_multi
conda activate foundationpose_ros && source /opt/ros/<ROS_DISTRO>/setup.bash && export PATH=/usr/local/<YOUR_cuda-12.X_VERSION>/bin${PATH:+:${PATH}}~ && python ./FoundationPoseROS2/foundationpose_ros_multi.py
```
> [!NOTE]
> Check the camera topics and edit accordingly in the Python file.

<div align="center">
  <video src="https://github.com/user-attachments/assets/4ef1f4cf-8900-451d-b006-47942b9f4606" width="200" />
</div>


## Run model-based demo through Rosbag2

### 1. Download the Recorded rosbag

First, download the recorded rosbag from the following link:

[Download Recorded rosbag](https://drive.google.com/drive/folders/17FRQog_fpqVxo_T85ItfpEa6fUo5fhZ7?usp=sharing)

### 2. Play the Recorded rosbag

Once you've downloaded the rosbag file, navigate to the directory where it's located, and play it back with the following command:

```bash
# Play the downloaded rosbag
source /opt/ros/<ROS_DISTRO>/setup.bash && ros2 bag play cube_demo_data_rosbag2/cube_demo_data_rosbag2.db3
```

Replace `<path_to_your_rosbag_file>` with the path to the `.db3` file you downloaded.

### 3. Run FoundationPoseROS2

In a separate terminal, activate your conda environment, export the correct CUDA version path and run the FoundationPoseROS2 script to start object pose estimation and tracking:

```bash
# Activate the conda environment and run foundationpose_ros_multi
conda activate foundationpose_ros && source /opt/ros/<ROS_DISTRO>/setup.bash && export PATH=/usr/local/<YOUR_cuda-12.X_VERSION>/bin${PATH:+:${PATH}}~ && python ./FoundationPoseROS2/foundationpose_ros_multi.py
```

<p align="center">
  <a href="https://youtu.be/p-QBK7yJKWs">
    <img src="https://img.youtube.com/vi/p-QBK7yJKWs/0.jpg" alt="Demo data rosbag screen recording">
  </a>
</p>

[Demo data rosbag terminal log](https://github.com/ammar-n-abbas/FoundationPoseROS2/blob/main/_misc/demo_data_terminal_log.txt)

## Run on novel objects

Add the mesh file in .obj or .stl format to the folder:
```bash
"./FoundationPoseROS2/demo_data/object_name/<OBJECT_MESH>.obj"
```

```bash
# Run
conda activate foundationpose_ros && source /opt/ros/<ROS_DISTRO>/setup.bash && export PATH=/usr/local/<YOUR_cuda-12.X_VERSION>/bin${PATH:+:${PATH}}~ && python ./FoundationPoseROS2/foundationpose_ros_multi.py
```

> [!NOTE]
> Check the camera topics and edit accordingly in the Python file.


## Features

- **Object Selection GUI**: Choose and reorder object files (.obj, .stl) using a simple Tkinter GUI.
- **Segmentation and Tracking**: SAM2 is used for object segmentation in real-time colour and depth images from a camera.
- **Pose Estimation**: Calculates and publishes the pose of detected objects based on camera images.
- **3D Visualization**: Visualize the objects’ pose with bounding boxes and axes.


## Pipeline

<p align="center">
    <img src="assets/pipeline.svg" alt="Algorithm Pipeline" style="width: 30%; height: auto;"/>
</p>

[![Star History Chart](https://api.star-history.com/svg?repos=ammar-n-abbas/FoundationPoseROS2&type=Date)](https://star-history.com/#ammar-n-abbas/FoundationPoseROS2&Date)

## Acknowledgement
This project has emanated from research conducted with the financial support of the EU Commission Recovery and Resilience Facility under the Science Foundation Ireland Future Digital Challenge Grant Number 22/NCF/FD/10929.
