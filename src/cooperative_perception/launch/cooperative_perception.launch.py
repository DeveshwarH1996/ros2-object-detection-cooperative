from os import path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration("config_path").perform(context)
    checkpoint = LaunchConfiguration("checkpoint_path").perform(context)
    data_root = LaunchConfiguration("data_root").perform(context)
    cav_indices_str = LaunchConfiguration("cav_indices").perform(context)
    session_id = LaunchConfiguration("session_id").perform(context)

    indices = [idx.strip() for idx in cav_indices_str.split(",") if idx.strip()]
    if not indices:
        indices = ["0"]

    nodes = []
    for idx_str in indices:
        try:
            idx_int = int(idx_str)
        except ValueError:
            raise ValueError(f"Invalid CAV index '{idx_str}' in cav_indices launch argument.")

        nodes.append(
            Node(
                package="cooperative_perception",
                executable="encoder_node",
                name=f"encoder_node_{idx_int}",
                parameters=[
                    {"config_path": config},
                    {"checkpoint_path": checkpoint},
                    {"data_root": data_root},
                    {"cav_index": idx_int},
                    {"session_id": session_id},
                ],
                output="screen",
            )
        )

    nodes.append(
        Node(
            package="cooperative_perception",
            executable="fusion_node",
            name="fusion_node",
            parameters=[
                {"config_path": config},
                {"checkpoint_path": checkpoint},
                {"data_root": data_root},
                {"session_id": session_id},
            ],
            output="screen",
        )
    )
    return nodes


def generate_launch_description():
    default_config = path.abspath(
        path.join(
            path.dirname(path.realpath(__file__)),
            "..",
            "..",
            "..",
            "opencood",
            "hypes_yaml",
            "point_pillar_v2xvit_PointTransformer.yaml",
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_config,
                description="Path to the model configuration YAML.",
            ),
            DeclareLaunchArgument(
                "checkpoint_path",
                description="Path to the pretrained checkpoint.",
            ),
            DeclareLaunchArgument(
                "data_root",
                description="Root directory of the V2XSet split.",
            ),
            DeclareLaunchArgument(
                "cav_indices",
                default_value="0,1,2",
                description="Comma separated list of CAV indices to launch encoder nodes for.",
            ),
            DeclareLaunchArgument(
                "session_id",
                default_value="ros2_cooperative_demo",
                description="Session identifier used to synchronize encoder and fusion nodes.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
