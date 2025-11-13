# ROS 2 Cooperative Object Detection

A ROS 2 workspace for lidar-based object detection and cooperative perception using multiple models built on the OpenCOOD framework and single-vehicle object detection models. This project provides both single-vehicle object detection and multi-vehicle cooperative perception capabilities for V2X (Vehicle-to-Everything) applications. This project was developed from a fork of [ragibarnab's ros2-lidar-object-detection](https://github.com/ragibarnab/ros2-lidar-object-detection)

![](./demo/demo.gif)

## Overview

This workspace integrates the [PyTorch PointPillars implementation](https://github.com/zhulf0804/PointPillars) into ROS 2, enabling real-time 3D object detection from LiDAR point clouds. The system supports both standalone detection and cooperative perception scenarios where multiple connected autonomous vehicles (CAVs) share encoded features for improved detection accuracy.

## Features

- **Single-Vehicle Object Detection**: Real-time 3D object detection from LiDAR point clouds using PointPillars
- **Cooperative Perception**: Multi-vehicle feature encoding and fusion for enhanced V2X perception
- **ROS 2 Integration**: Native ROS 2 nodes with standard message interfaces
- **Visualization**: 3D object visualization tools for debugging and demonstration
- **Modular Architecture**: Separate packages for detection, messaging, and visualization

## Package Structure

This workspace contains the following ROS 2 packages:

- **`lidar_object_detection`**: ROS 2 wrapper around the PyTorch PointPillars implementation for single-vehicle object detection
- **`cooperative_perception`**: Encoder and fusion nodes for cooperative V2X perception
- **`cooperative_perception_msgs`**: Custom message definitions for cooperative perception (e.g., `EncodedFeature`)
- **`object_detection_msgs`**: Message definitions for object detection results
- **`object_visualization`**: Visualization nodes for 3D object detection results
- **`ros2_numpy`**: Utility package for ROS 2 and NumPy integration

## Prerequisites

- **ROS 2** (tested with ROS 2 Foxy)
- **Python 3.8+**
- **PyTorch** with CUDA support (for GPU acceleration)
- **NumPy**
- **OpenCOOD** dependencies (for cooperative perception)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ros2-object-detection-cooperative
```

2. Install dependencies:
```bash
# Install ROS 2 dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the workspace:
```bash
colcon build --symlink-install
```

4. Build the CUDA packages in the build space. 
```bash
cp -r src/lidar_object_detection/lidar_object_detection/ops/iou3d build/lidar_object_detection/lidar_object_detection/ops/
cp -r src/lidar_object_detection/lidar_object_detection/ops/voxelization build/lidar_object_detection/lidar_object_detection/ops/
cd build/lidar_object_detection/lidar_object_detection/ops
python setup.py build_ext --inplace
```

5. Source the workspace:
```bash
source install/setup.bash
```

## Usage

### Single-Vehicle Object Detection

Launch the basic object detection pipeline:

```bash
ros2 launch lidar_object_detection main.launch.py
```

This launches:
- `lidar_object_detector_node`: Processes LiDAR point clouds and performs object detection
- `lidar_publisher_node`: Publishes LiDAR data (if using simulated data)
- `object3d_visualizer_node`: Visualizes detected objects in RViz

### Cooperative Perception

Launch the cooperative perception system with multiple CAVs:

```bash
ros2 launch cooperative_perception cooperative_perception.launch.py \
    config_path:=/path/to/config.yaml \
    checkpoint_path:=/path/to/checkpoint.pth \
    data_root:=/path/to/data \
    cav_indices:=0,1,2 \
    session_id:=my_session
```

**Launch Parameters:**
- `config_path`: Path to the model configuration YAML file
- `checkpoint_path`: Path to the trained model checkpoint
- `data_root`: Root directory for dataset
- `cav_indices`: Comma-separated list of CAV indices (e.g., "0,1,2")
- `session_id`: Unique session identifier

The cooperative perception system includes:
- **Encoder Nodes**: One per CAV, encodes LiDAR features for transmission
- **Fusion Node**: Aggregates encoded features from multiple CAVs and performs cooperative fusion

### Verification

Use the provided verification script to check system status:

```bash
./verify_cooperative_perception.sh
```

This script checks:
- Active ROS 2 nodes
- Active topics
- Message rates on key topics (`/cooperative/encoded_features`, `/cooperative/detections`)

## Architecture

### Single-Vehicle Detection Flow

```
LiDAR Point Cloud → lidar_object_detector_node → Detected Objects → object3d_visualizer_node
```

### Cooperative Perception Flow

```
CAV 1: LiDAR → encoder_node → EncodedFeature → fusion_node → Detections
CAV 2: LiDAR → encoder_node → EncodedFeature ↗
CAV 3: LiDAR → encoder_node → EncodedFeature ↗
```

## Topics

### Key Topics

- `/cooperative/encoded_features`: Encoded features from CAV encoders (type: `cooperative_perception_msgs/EncodedFeature`)
- `/cooperative/detections`: Fused detection results (type: `object_detection_msgs/Detections`)
- `/lidar/points`: Input LiDAR point cloud (type: `sensor_msgs/PointCloud2`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PointPillars implementation: [zhulf0804/PointPillars](https://github.com/zhulf0804/PointPillars)
- OpenCOOD framework for cooperative perception

## Maintainers

- **Deveshwar Hariharan** (dhariha@ncsu.edu) - Lidar Object Detection
- **OpenCOOD** - Cooperative Perception
