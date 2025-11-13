import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import torch

# Ensure imports resolve to the OpenCOOD workspace even when this package is a symlink.
OPENCOOD_ROOT = Path("/home/ecoprt/cooperative_pers/OpenCOOD").resolve()
if str(OPENCOOD_ROOT) not in sys.path:
    sys.path.append(str(OPENCOOD_ROOT))

from cooperative_perception_msgs.msg import EncodedFeature
from deployment.point_pillar_v2xvit_split import PointPillarSplitEncoder
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml


class CooperativeEncoderNode(Node):
    """ROS 2 node that encodes LiDAR point clouds for a single cooperative agent."""

    def __init__(self) -> None:
        super().__init__("cooperative_encoder_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("data_root", "")
        self.declare_parameter("split", "validate")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("timer_period", 0.5)
        self.declare_parameter("cav_index", 0)
        self.declare_parameter("session_id", "default_session")
        self.declare_parameter("sample_limit", -1)

        config_path = Path(self.get_parameter("config_path").get_parameter_value().string_value).expanduser()
        checkpoint_path = (
            Path(self.get_parameter("checkpoint_path").get_parameter_value().string_value).expanduser()
        )
        data_root_param = self.get_parameter("data_root").get_parameter_value().string_value
        split = self.get_parameter("split").get_parameter_value().string_value
        device_str = self.get_parameter("device").get_parameter_value().string_value
        timer_period = self.get_parameter("timer_period").get_parameter_value().double_value
        self.cav_index = self.get_parameter("cav_index").get_parameter_value().integer_value
        self.session_id = self.get_parameter("session_id").get_parameter_value().string_value
        self.sample_limit = self.get_parameter("sample_limit").get_parameter_value().integer_value

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if device_str.lower().startswith("cpu"):
            raise ValueError("CPU device is not supported for cooperative encoding.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required but not available.")

        self.device = torch.device(device_str)

        cfg = load_yaml(str(config_path))
        data_root = Path(data_root_param).expanduser() if data_root_param else None
        if data_root:
            cfg["root_dir"] = str((data_root / "train").resolve())
            cfg["validate_dir"] = str((data_root / "validate").resolve())

        self.max_cav = int(cfg.get("train_params", {}).get("max_cav", 5))

        train_flag = split == "train"
        self.dataset = build_dataset(cfg, visualize=False, train=train_flag)
        self.dataset_length = len(self.dataset)

        self.encoder = PointPillarSplitEncoder(device=device_str, config_path=str(config_path))
        state_dict = torch.load(str(checkpoint_path), map_location=self.encoder.device)
        weights = state_dict.get("state_dict", state_dict)
        self.encoder.load_state_dict(weights)
        self.encoder.encoder.eval()

        self.publisher = self.create_publisher(EncodedFeature, "encoded_features", 10)
        self.timer = self.create_timer(timer_period, self._on_timer)
        self.sample_index = 0

        self.get_logger().info(
            f"Encoder node initialized for CAV index {self.cav_index} with dataset size {self.dataset_length}."
        )

    def _build_header(self) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f"cav_{self.cav_index}"
        return header

    def _prepare_voxel_inputs(self, processed_lidar: dict) -> Optional[dict]:
        """Extract voxelized pillars for the configured CAV."""
        voxel_features_list = processed_lidar["voxel_features"]
        if self.cav_index >= len(voxel_features_list):
            return None

        voxel_features = torch.from_numpy(voxel_features_list[self.cav_index]).float().to(self.encoder.device)

        coords_np = processed_lidar["voxel_coords"][self.cav_index]
        if coords_np.shape[1] == 3:
            coords_np = np.pad(coords_np, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        voxel_coords = torch.from_numpy(coords_np).int().to(self.encoder.device)

        voxel_num_points = torch.from_numpy(processed_lidar["voxel_num_points"][self.cav_index]).int().to(
            self.encoder.device
        )

        return {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
        }

    def _on_timer(self) -> None:
        if self.sample_limit > 0 and self.sample_index >= self.sample_limit:
            return

        sample = self.dataset[self.sample_index % self.dataset_length]
        ego_dict = sample["ego"]
        cav_count = int(ego_dict["cav_num"])

        voxel_inputs = self._prepare_voxel_inputs(ego_dict["processed_lidar"])
        if voxel_inputs is None:
            self.get_logger().debug(
                f"Sample {self.sample_index}: CAV index {self.cav_index} not present (cav_count={cav_count})."
            )
            self.sample_index += 1
            return

        with torch.no_grad():
            spatial_feature = self.encoder.encoder(voxel_inputs)
        spatial_feature_cpu = spatial_feature.detach().cpu().float()
        # self.get_logger().info(f"spatial_feature_cpu.shape: {spatial_feature_cpu.shape}")

        # Remove batch dimension if present: (1, C, H, W) -> (C, H, W)
        if spatial_feature_cpu.dim() == 4:
            spatial_feature_cpu = spatial_feature_cpu.squeeze(0)
        feature_channels, feature_height, feature_width = spatial_feature_cpu.shape
        feature_flat = spatial_feature_cpu.numpy().ravel().astype(np.float32)

        metadata_velocity = float(ego_dict["velocity"][self.cav_index])
        metadata_time_delay = float(ego_dict["time_delay"][self.cav_index])
        metadata_infra = float(ego_dict["infra"][self.cav_index])
        spatial_matrix = ego_dict["spatial_correction_matrix"][self.cav_index].astype(np.float32).reshape(-1)

        msg = EncodedFeature()
        msg.header = self._build_header()
        msg.session_id = self.session_id
        msg.sample_idx = int(self.sample_index)
        msg.cav_id = int(ego_dict["object_ids"][self.cav_index]) if "object_ids" in ego_dict else self.cav_index
        msg.cav_index = int(self.cav_index)
        msg.cav_count = cav_count
        msg.max_cav = self.max_cav
        msg.feature_channels = feature_channels
        msg.feature_height = feature_height
        msg.feature_width = feature_width
        msg.feature = feature_flat.tolist()
        msg.velocity = metadata_velocity
        msg.time_delay = metadata_time_delay
        msg.infra = metadata_infra
        msg.spatial_correction = spatial_matrix.tolist()

        self.publisher.publish(msg)
        self.get_logger().info(
            f"CAV {self.cav_index}: Published sample {self.sample_index}, feature shape: ({feature_channels}, {feature_height}, {feature_width})"
        )

        self.sample_index += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CooperativeEncoderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

