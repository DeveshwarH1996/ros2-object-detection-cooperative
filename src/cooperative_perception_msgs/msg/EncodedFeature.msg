# Encoded BEV feature and metadata from a single cooperative agent.

std_msgs/Header header

# Identifier for synchronization across agents (e.g. scenario name or run id).
string session_id

# Global sample/frame index within the dataset split.
int32 sample_idx

# Agent identifiers.
int32 cav_id
int32 cav_index

# Number of participating agents in this frame and configured maximum.
int32 cav_count
int32 max_cav

# BEV feature tensor shape (C, H, W) and flattened data (row-major).
int32 feature_channels
int32 feature_height
int32 feature_width
float32[] feature

# Per-agent metadata required for fusion.
float32 velocity
float32 time_delay
float32 infra

# Flattened 4x4 spatial correction matrix (row-major).
float32[] spatial_correction

