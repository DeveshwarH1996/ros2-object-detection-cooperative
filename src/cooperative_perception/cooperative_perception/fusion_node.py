from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
import torch

OPENCOOD_ROOT = Path("/home/ecoprt/cooperative_pers/OpenCOOD").resolve()
if str(OPENCOOD_ROOT) not in sys.path:
    sys.path.append(str(OPENCOOD_ROOT))

from cooperative_perception_msgs.msg import EncodedFeature
from deployment.point_pillar_v2xvit_split import PointPillarSplitFusion
from object_detection_msgs.msg import Object3d, Object3dArray
from geometry_msgs.msg import Point
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import eval_utils


@dataclass
class BufferedFeature:
    feature: torch.Tensor
    cav_index: int
    velocity: float
    time_delay: float
    infra: float
    spatial_correction: np.ndarray


@dataclass
class SampleBuffer:
    expected_agents: int
    max_cav: int
    entries: Dict[int, BufferedFeature] = field(default_factory=dict)


class CooperativeFusionNode(Node):
    """ROS 2 node that aggregates encoded features and performs cooperative fusion."""

    def __init__(self) -> None:
        super().__init__("cooperative_fusion_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("data_root", "")
        self.declare_parameter("split", "validate")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("sample_limit", -1)
        self.declare_parameter("session_id", "default_session")
        self.declare_parameter("publish_visualization", True)

        config_path = Path(self.get_parameter("config_path").get_parameter_value().string_value).expanduser()
        checkpoint_path = Path(self.get_parameter("checkpoint_path").get_parameter_value().string_value).expanduser()
        data_root_param = self.get_parameter("data_root").get_parameter_value().string_value
        split = self.get_parameter("split").get_parameter_value().string_value
        device_str = self.get_parameter("device").get_parameter_value().string_value
        self.sample_limit = self.get_parameter("sample_limit").get_parameter_value().integer_value
        self.session_id = self.get_parameter("session_id").get_parameter_value().string_value
        self.publish_visualization = self.get_parameter("publish_visualization").get_parameter_value().bool_value

        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if device_str.lower().startswith("cpu"):
            raise ValueError("CPU device is not supported for cooperative fusion.")
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

        self.fusion = PointPillarSplitFusion(device=device_str, config_path=str(config_path))
        state_dict = torch.load(str(checkpoint_path), map_location=self.fusion.device)
        weights = state_dict.get("state_dict", state_dict)
        self.fusion.load_state_dict(weights)
        self.fusion.fusion.eval()

        self.buffers: Dict[int, SampleBuffer] = {}
        self.result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }
        self.processed_samples = 0

        self.subscription = self.create_subscription(
            EncodedFeature, "encoded_features", self._feature_callback, 50
        )
        self.detections_publisher = self.create_publisher(Object3dArray, "object_detections_3d", 10)

        self.get_logger().info("Cooperative fusion node initialized.")

    def _feature_callback(self, msg: EncodedFeature) -> None:
        if msg.session_id != self.session_id:
            return

        buffer = self.buffers.get(msg.sample_idx)
        if buffer is None:
            buffer = SampleBuffer(expected_agents=msg.cav_count, max_cav=msg.max_cav)
            self.buffers[msg.sample_idx] = buffer

        if len(msg.feature) != msg.feature_channels * msg.feature_height * msg.feature_width:
            self.get_logger().error(
                f"Feature size mismatch for sample {msg.sample_idx} (received {len(msg.feature)} floats)."
            )
            return

        feature_np = np.array(msg.feature, dtype=np.float32).reshape(
            msg.feature_channels, msg.feature_height, msg.feature_width
        )
        feature_tensor = torch.from_numpy(feature_np).to(self.fusion.device)

        spatial_matrix = np.array(msg.spatial_correction, dtype=np.float32)
        if spatial_matrix.size != 16:
            self.get_logger().error("Spatial correction matrix must contain 16 floats.")
            return
        spatial_matrix = spatial_matrix.reshape(4, 4)

        buffer.entries[msg.cav_index] = BufferedFeature(
            feature=feature_tensor,
            cav_index=msg.cav_index,
            velocity=msg.velocity,
            time_delay=msg.time_delay,
            infra=msg.infra,
            spatial_correction=spatial_matrix,
        )

        if len(buffer.entries) >= buffer.expected_agents:
            self._process_sample(msg.sample_idx, buffer)
            del self.buffers[msg.sample_idx]

    def _process_sample(self, sample_idx: int, buffer: SampleBuffer) -> None:
        sample = self.dataset[sample_idx % self.dataset_length]
        batch_cpu = self.dataset.collate_batch_test([sample])
        ego_dict = sample["ego"]

        sorted_indices = sorted(buffer.entries.keys())
        # Each feature is (C, H, W), add batch dimension to make it (1, C, H, W)
        feature_list = [buffer.entries[idx].feature.unsqueeze(0) for idx in sorted_indices]

        velocity = np.zeros(buffer.max_cav, dtype=np.float32)
        time_delay = np.zeros(buffer.max_cav, dtype=np.float32)
        infra = np.zeros(buffer.max_cav, dtype=np.float32)
        spatial_matrix = np.tile(np.eye(4, dtype=np.float32), (buffer.max_cav, 1, 1))

        for idx in sorted_indices:
            entry = buffer.entries[idx]
            velocity[idx] = entry.velocity
            time_delay[idx] = entry.time_delay
            infra[idx] = entry.infra
            spatial_matrix[idx] = entry.spatial_correction

        metadata = {
            "record_len": [buffer.expected_agents],
            "velocity": velocity.tolist(),
            "time_delay": time_delay.tolist(),
            "infra": infra.tolist(),
            "spatial_correction_matrix": [spatial_matrix],
        }

        self.get_logger().info(
            f"Processing sample {sample_idx}: fusing {len(feature_list)} features from CAVs {sorted_indices}"
        )

        with torch.no_grad():
            outputs = self.fusion.fuse(feature_list, metadata)

        output_dict = {
            "ego": {
                "psm": outputs["psm"].detach().cpu(),
                "rm": outputs["rm"].detach().cpu(),
            }
        }

        pred_box_tensor, pred_score, gt_box_tensor = self.dataset.post_process(batch_cpu, output_dict)
        
        num_pred = len(pred_box_tensor) if pred_box_tensor is not None else 0
        num_gt = len(gt_box_tensor) if gt_box_tensor is not None else 0
        self.get_logger().info(
            f"Sample {sample_idx}: detected {num_pred} objects (GT: {num_gt})"
        )
        if not self.publish_visualization:
            self._update_metrics(pred_box_tensor, pred_score, gt_box_tensor)
        else:
            self._update_metrics(pred_box_tensor, pred_score, gt_box_tensor)
            detection_msg = self._build_detection_message(pred_box_tensor, pred_score, sample_idx)
            self.detections_publisher.publish(detection_msg)

        self.processed_samples += 1
        if self.sample_limit > 0 and self.processed_samples >= self.sample_limit:
            self._log_final_metrics()

    def _build_detection_message(
        self, pred_box_tensor: Optional[torch.Tensor], pred_score: Optional[torch.Tensor], sample_idx: int
    ) -> Object3dArray:
        detection_array = Object3dArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "map"

        if pred_box_tensor is None or pred_score is None:
            return detection_array

        boxes = pred_box_tensor.cpu().numpy()
        scores = pred_score.cpu().numpy()
        for box_corners, score in zip(boxes, scores):
            detection = Object3d()
            detection.label = Object3d.CAR
            detection.confidence_score = float(score)
            for i, corner in enumerate(box_corners):
                point = Point()
                point.x = float(corner[0])
                point.y = float(corner[1])
                point.z = float(corner[2])
                detection.bounding_box.corners[i] = point
            detection_array.objects.append(detection)

        return detection_array

    def _update_metrics(
        self,
        pred_box_tensor: Optional[torch.Tensor],
        pred_score: Optional[torch.Tensor],
        gt_box_tensor: Optional[torch.Tensor],
    ) -> None:
        for thr in self.result_stat.keys():
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, self.result_stat, thr)

    def _log_final_metrics(self) -> None:
        ap30, ap50, ap70 = eval_utils.eval_final_results(
            self.result_stat, str(Path.cwd()), global_sort_detections=False
        )
        self.get_logger().info(
            f"Evaluation complete after {self.processed_samples} samples. AP@0.3={ap30:.2f}, "
            f"AP@0.5={ap50:.2f}, AP@0.7={ap70:.2f}"
        )

    def destroy_node(self) -> bool:
        if self.processed_samples > 0:
            self._log_final_metrics()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CooperativeFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

