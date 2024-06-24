
# galbot_description Package

[![RViz](https://img.shields.io/badge/RViz-Passing-brightgreen.svg)](link-to-your-windows-test)&nbsp;[![Sapien](https://img.shields.io/badge/Sapien-Partial%20Pass-orange.svg)](link-to-your-windows-test)&nbsp;[![Gazebo](https://img.shields.io/badge/Gazebo-Testing-yellow.svg)](link-to-your-windows-test)&nbsp;[![Isaac](https://img.shields.io/badge/Isaac-Testing-yellow.svg)](link-to-your-windows-test)&nbsp;[![Pybullet](https://img.shields.io/badge/Pybullet-Testing-yellow.svg)](link-to-your-windows-test)

```python
# Description: Readme for the `galbot_description` package
# Version: 1.4.2
# Date: 2023-12-22
# Author: Herman Ye@Galbot
# 
# Revision History:
# Date       Version  Author       Description
# ---------- -------- ------------ ------------
# 2023-10-13 1.0.0    Herman Ye    Initial version.
# 2023-10-20 1.1.0    Herman Ye    Add pybullet support.
# 2023-11-07 1.2.0    Herman Ye    Add robot head, camera intrinsic parameters.
# 2023-11-30 1.3.0    Herman Ye    Update for new robot hardware.
# 2023-12-06 1.3.1    Herman Ye    Fix bugs for base and head camera.
# 2023-12-21 1.4.0    Herman Ye    Add long sucker, rm75_6f, and curobo basic urdf.
# 2023-12-22 1.4.1    Herman Ye    Add long sucker sucked state tcp.
# 2023-12-22 1.4.2    Herman Ye    Fix rm75-6f arm link7 range limit bug.

```

## Overview

The `galbot_description` package is a ROS1 Noetic package that provides the robot description for the Galbot robot. This package includes configuration files, launch files, meshes, URDF (Unified Robot Description Format) files, RViz configuration, and other resources necessary for visualizing and simulating the Galbot robot in a ROS environment.

## Contents

```bash
galbot_description/
│
├── config/
│   └── (Configuration files for robot parameters)
│
├── launch/
│   └── (Launch files for visualizing hardware components)
│
├── meshes/
│   └── (3D mesh files of Galbot and its components)
│
├── rviz/
│   └── (Configuration files for RViz visualization)
|
├── simulators/
│   └── (Robot description configuration files for different simulators)
│
└── urdf/
    └── (xacro files defining kinematic and dynamic properties)
```

The `galbot_description` package includes the following directories:

- **config**: This directory contains configuration files that may be used for parameter settings or other configuration needs of the robot.

- **launch**: The launch directory contains launch files that allow you to visualize hardware components of the Galbot robot.

- **meshes**: In this directory, you will find 3D mesh files of the Galbot robot and its components. These mesh files are used for visualization and simulation purposes.

- **rviz**: The RViz directory contains configuration files for RViz, which is a 3D visualization tool for ROS. These configurations allow you to set up RViz to view the Galbot robot effectively.

- **urdf**: The URDF directory contains `xacro` files that define the kinematic and dynamic properties of the Galbot robot. These files are essential for robot modeling and simulation.

## Usage

To use the `galbot_description` package, please follow these steps:

### Build your ROS workspace

```bash
# Build the workspace
cd <your_ros1_ws>
catkin_make
```

### View single components of the Galbot robot

```bash
# View components of the Galbot robot
roslaunch galbot_description view_<component_name>.launch
```

For example, to view the camera component:

```bash
roslaunch galbot_description view_camera_d415_model.launch
```

### View all components of the Galbot robot

```bash
# View all components of the Galbot robot
roslaunch galbot_description view_<galbot_robot_name>.launch
```

For example, to view Galbot Zero:

```bash
roslaunch galbot_description view_galbot_zero.launch
```

### Transform Xacro files to URDF (if needed)

Xacro files are used instead of URDF to enhance maintainability and flexibility in robot description, and they can be transformed to URDF easily using the following command:

```bash
# Transform Xacro to URDF if needed
roscd galbot_description/urdf/
xacro galbot_zero.xacro -o galbot_zero.urdf
```

### Load description in launch files

For more information about how to use the `galbot_description` package in your code, please refer to the `galbot_description/launch/view_galbot_zero.launch`

```bash
# Check the launch file
roscd galbot_description/launch/
cat view_galbot_zero.launch
```

### Get camera intrinsic parameters for simulation

You can get the camera intrinsic parameters for simulation by checking the `galbot_description/config/camera_intrinsic_parameters.md` file.

### Test in Pybullet

- Test 1 Display only

```bash
cd galbot_sim/pybullet/demos
python3 display.py
```

- Test 2 GUI Control

```bash
cd galbot_sim/pybullet/demos
python3 display_with_control.py
```

### Test in Sapien

```bash
cd galbot_sim/sapien/demos
# Read the README.md file for more information
cat readme.md
```
### Test in Isaac Sim
galbot_description v1.4.0 provides basic curobo urdf support for Isaac Sim.
```bash
galbot_description/simulators/isaac_sim/galbot_zero.urdf
```

## Todo

There are a few things that are planned for future development:

- Enhance the robot model's visual aspect by adding a simple texture in the .dae format
- Streamline the collision part of the robot model
- Improve the performance of the robot model's visual elements with textures
- Resolve the Pybullet self-collision issue.
- Implement support for Gazebo Simulation
- Extend robot URDF support for Sapien to cover the entire robot, not just the upper half
- Provide support for ROS2 Humble

## Notes

For any issues, questions, or contributions related to this package, please contact Herman Ye@Galbot.

## License

```python
# Copyright (c) 2023 Galbot. All Rights Reserved.

# This software contains confidential and proprietary information of 
# Galbot, Inc. ("Confidential Information"). You shall not disclose such 
# Confidential Information and shall use it only in accordance with the 
# terms of the license agreement you entered into with Galbot, Inc.

# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY 
# PORTION OR DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE 
# RECEIVED THIS SOFTWARE IN ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY 
# AND DELETE IT FROM YOUR SYSTEM.
```
