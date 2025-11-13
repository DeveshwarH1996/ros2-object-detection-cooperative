#!/bin/bash
# Verification script for cooperative perception ROS 2 nodes

echo "=== Cooperative Perception Verification ==="
echo ""

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencood38
source /opt/ros/foxy/setup.bash
source ~/ros2_ws_cooper/ros2-object-detection-cooperative/install/setup.bash

echo "1. Checking active ROS 2 nodes..."
ros2 node list
echo ""

echo "2. Checking active topics..."
ros2 topic list
echo ""

echo "3. Checking EncodedFeature topic info..."
ros2 topic info /cooperative/encoded_features
echo ""

echo "4. Checking detections topic info..."
ros2 topic info /cooperative/detections
echo ""

echo "5. Monitoring EncodedFeature message rate (5 seconds)..."
timeout 5 ros2 topic hz /cooperative/encoded_features || echo "No messages on /cooperative/encoded_features"
echo ""

echo "6. Monitoring detections message rate (5 seconds)..."
timeout 5 ros2 topic hz /cooperative/detections || echo "No messages on /cooperative/detections"
echo ""

echo "7. Showing one detection message..."
timeout 3 ros2 topic echo /cooperative/detections --once || echo "No detection messages received"
echo ""

echo "=== Verification Complete ==="

