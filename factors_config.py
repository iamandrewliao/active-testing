"""
Configuration file for robot policy evaluation factors.

This file centralizes all factor definitions, bounds, valid values, and task configurations.
Import this module in other files to access factor configurations.
"""
import torch

# Set up torch dtype (use CPU by default to avoid CUDA OOM at import time)
# Tensors can be moved to GPU explicitly where needed.
tkwargs = {"dtype": torch.double}

# ============================================================================
# Factor Definitions
# ============================================================================

# Object position x: 0 to 1 in 0.1 increments (11 values: 0.0, 0.1, ..., 1.0)
OBJECT_POS_X_VALUES = torch.tensor([i * 0.1 for i in range(11)], **tkwargs)

# Object position y: 0 to 1 in 0.1 increments (11 values: 0.0, 0.1, ..., 1.0)
OBJECT_POS_Y_VALUES = torch.tensor([i * 0.1 for i in range(11)], **tkwargs)

# Table height: 1, 2, 3 inches
TABLE_HEIGHT_VALUES = torch.tensor([1.0, 2.0, 3.0], **tkwargs)

# Camera viewpoint indices: 0=back, 1=right, 2=left
VIEWPOINT_VALUES = torch.tensor([0.0, 1.0, 2.0], **tkwargs)

# ============================================================================
# Camera Viewpoint Definitions
# ============================================================================

# Camera viewpoint parameters (azimuth, elevation, distance)
# These can be updated with actual values for your setup
CAMERA_VIEWPOINTS = {
    0: {'name': 'back',  'azimuth': 90.0,  'elevation': 70.0, 'distance': 84.0},
    1: {'name': 'right', 'azimuth': -30.0, 'elevation': 30.0, 'distance': 102.0},
    2: {'name': 'left',  'azimuth': 210.0, 'elevation': 50.0, 'distance': 76.0},
}

# How to represent camera viewpoint in the factor / model space:
# - "index": factors are [x, y, table_height, viewpoint] with viewpoint in {0,1,2}
# - "params": factors are [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
#             where the last 3 entries are one of the 3 CAMERA_VIEWPOINTS parameter triples.
VIEWPOINT_REPRESENTATION = "params"  # default; set to "index" for 0/1/2 representation

def get_viewpoint_name(viewpoint_idx):
    """Returns the name of a viewpoint given its index."""
    if viewpoint_idx in CAMERA_VIEWPOINTS:
        return CAMERA_VIEWPOINTS[viewpoint_idx]['name']
    return f"unknown_{viewpoint_idx}"

def get_viewpoint_params(viewpoint_idx):
    """Returns the camera parameters (azimuth, elevation, distance) for a viewpoint."""
    if viewpoint_idx in CAMERA_VIEWPOINTS:
        return CAMERA_VIEWPOINTS[viewpoint_idx]
    return None

def get_viewpoint_index_from_params(azimuth, elevation, distance, tol=1e-6):
    """
    Given a camera parameter triple (azimuth, elevation, distance),
    returns the corresponding viewpoint index (0, 1, or 2) if it matches
    one of the predefined CAMERA_VIEWPOINTS within a small tolerance.
    Returns None if no match is found.
    """
    candidate = torch.tensor([azimuth, elevation, distance], **tkwargs)
    diffs = torch.abs(_VIEWPOINT_PARAM_TENSOR - candidate)
    matches = torch.all(diffs < tol, dim=1)
    if not torch.any(matches).item():
        return None
    # Match index in the sorted key order used to build _VIEWPOINT_PARAM_TENSOR
    sorted_keys = sorted(CAMERA_VIEWPOINTS.keys())
    match_pos = torch.nonzero(matches, as_tuple=False)[0, 0].item()
    return sorted_keys[match_pos]

def set_viewpoint_params(viewpoint_idx, azimuth, elevation, distance):
    """Sets the camera parameters for a viewpoint."""
    if viewpoint_idx in CAMERA_VIEWPOINTS:
        CAMERA_VIEWPOINTS[viewpoint_idx]['azimuth'] = azimuth
        CAMERA_VIEWPOINTS[viewpoint_idx]['elevation'] = elevation
        CAMERA_VIEWPOINTS[viewpoint_idx]['distance'] = distance
    else:
        raise ValueError(f"Invalid viewpoint index: {viewpoint_idx}")

# Precompute viewpoint parameter tensor for convenience (3 x 3: [az, el, dist])
_VIEWPOINT_PARAM_TENSOR = torch.tensor(
    [
        [
            CAMERA_VIEWPOINTS[i]["azimuth"],
            CAMERA_VIEWPOINTS[i]["elevation"],
            CAMERA_VIEWPOINTS[i]["distance"],
        ]
        for i in sorted(CAMERA_VIEWPOINTS.keys())
    ],
    **tkwargs,
)

# ============================================================================
# Factor Column Names and Bounds (for CSV files and normalization)
# ============================================================================

if VIEWPOINT_REPRESENTATION == "index":
    # Factors: [x, y, table_height, viewpoint_index]
    FACTOR_COLUMNS = ["x", "y", "table_height", "viewpoint"]
    BOUNDS = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],  # Minimum values
            [1.0, 1.0, 3.0, 2.0],  # Maximum values
        ],
        **tkwargs,
    )
elif VIEWPOINT_REPRESENTATION == "params":
    # Factors: [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
    FACTOR_COLUMNS = [
        "x",
        "y",
        "table_height",
        "camera_azimuth",
        "camera_elevation",
        "camera_distance",
    ]
    # Bounds: x,y in [0,1]; table_height in [1,3]; camera params in [min,max] over the 3 viewpoints
    az_vals = _VIEWPOINT_PARAM_TENSOR[:, 0]
    el_vals = _VIEWPOINT_PARAM_TENSOR[:, 1]
    dist_vals = _VIEWPOINT_PARAM_TENSOR[:, 2]
    BOUNDS = torch.tensor(
        [
            [0.0, 0.0, 1.0, az_vals.min().item(), el_vals.min().item(), dist_vals.min().item()],
            [1.0, 1.0, 3.0, az_vals.max().item(), el_vals.max().item(), dist_vals.max().item()],
        ],
        **tkwargs,
    )
else:
    raise ValueError(f"Unknown VIEWPOINT_REPRESENTATION: {VIEWPOINT_REPRESENTATION}")

# Number of dimensions in the factor space
DIMS = BOUNDS.shape[1]

# ============================================================================
# Design Space Generation
# ============================================================================

def get_design_points_robot():
    """
    Creates a tensor of all valid design points for robot policy evaluation factors.

    Representation depends on VIEWPOINT_REPRESENTATION:
    - \"index\":  [x, y, table_height, viewpoint_index]
    - \"params\": [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]

    In both cases, only the 3 discrete camera viewpoints in CAMERA_VIEWPOINTS are used,
    so the design space contains exactly 11 x 11 x 3 x 3 = 1089 combinations.

    Iteration order (e.g. for brute_force): slowest to fastest =
    camera viewpoint (back, right, left) -> table height (1,2,3) -> object x -> object y.
    So for a fixed viewpoint and table height we sweep all (x,y) positions first.
    """
    # Meshgrid order (slowest to fastest) = viewpoint, table_height, x, y so that
    # flatten() gives: for viewpoint in [back, right, left]: for h in [1,2,3]: for x: for y
    v_grid, h_grid, x_grid, y_grid = torch.meshgrid(
        VIEWPOINT_VALUES,
        TABLE_HEIGHT_VALUES,
        OBJECT_POS_X_VALUES,
        OBJECT_POS_Y_VALUES,
        indexing="ij",
    )

    v_flat = v_grid.flatten().long()  # indices 0,1,2
    h_flat = h_grid.flatten()
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    if VIEWPOINT_REPRESENTATION == "index":
        # Factors are [x, y, table_height, viewpoint_index]
        all_points = torch.stack([x_flat, y_flat, h_flat, v_flat.to(**tkwargs)], dim=1)
    elif VIEWPOINT_REPRESENTATION == "params":
        # Factors are [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
        # Map each viewpoint index to its parameter triple
        view_params = _VIEWPOINT_PARAM_TENSOR[v_flat]  # shape [N, 3]
        all_points = torch.cat(
            [
                x_flat.unsqueeze(1),
                y_flat.unsqueeze(1),
                h_flat.unsqueeze(1),
                view_params,
            ],
            dim=1,
        )

    total_points = (
        len(OBJECT_POS_X_VALUES)
        * len(OBJECT_POS_Y_VALUES)
        * len(TABLE_HEIGHT_VALUES)
        * len(VIEWPOINT_VALUES)
    )
    print(
        f"Generated {all_points.shape[0]} total design points "
        f"({len(OBJECT_POS_X_VALUES)}x{len(OBJECT_POS_Y_VALUES)}x"
        f"{len(TABLE_HEIGHT_VALUES)}x{len(VIEWPOINT_VALUES)} = {total_points})."
    )

    return all_points

# ============================================================================
# Task Configuration (Continuous Outcome Ranges)
# ============================================================================

# Define continuous outcome ranges and descriptions for different tasks
# increment of 0.5 is used in case the outcome seems like it's in between two descriptions
TASK_CONFIGS = {
    'pickblueblock': {
        'increment': 0.5,
        'descriptions': {
            0.0: 'failed completely',
            1.0: 'moved to block',
            2.0: 'grasped block (success)',
        }
    },
    'uprightcup': {
        'increment': 0.5,
        'descriptions': {
            0.0: 'failed completely',
            1.0: 'moved toward the cup',
            2.0: 'grabbed the cup',
            3.0: 'dropped the cup on its bottom rim',
            4.0: 'set the cup upright without falling (success)'
        }
    },
    'putgreeninpot': {
        'increment': 0.5,
        'descriptions': {
            0.0: 'failed completely',
            1.0: 'moved toward the correct object(s) (lid or block) in the right order',
            2.0: 'grabbed the object(s) in the right order',
            3.0: 'moved object(s) toward their appropriate locations in the right order (e.g. moved block towards open pot, moved lid on pot to the table)',
            4.0: 'placed the object(s) at their appropriate locations in the right order (success)'
        }
    }
    # Add more task configurations here as needed
}

def get_task_config(task_name=None):
    """Returns the configuration for a given task."""
    if task_name is None:
        raise ValueError("Task name is required")
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task_name]

def _derive_outcome_values(task_name=None):
    """Derives min_outcome and max_outcome from descriptions."""
    config = get_task_config(task_name)
    descriptions = config['descriptions']
    outcome_keys = sorted(descriptions.keys())
    min_outcome = outcome_keys[0]
    max_outcome = outcome_keys[-1]
    return min_outcome, max_outcome

def get_outcome_range(task_name=None):
    """Returns the valid outcome range for a task."""
    config = get_task_config(task_name)
    min_outcome, max_outcome = _derive_outcome_values(task_name)
    return min_outcome, max_outcome, config['increment']

def get_success_outcome(task_name=None):
    """Returns the success outcome value for a task (assumed to be max_outcome)."""
    _, success_outcome = _derive_outcome_values(task_name)
    return success_outcome

def get_outcome_descriptions(task_name=None):
    """Returns the outcome descriptions for a task."""
    config = get_task_config(task_name)
    return config['descriptions']

# ============================================================================
# Reachability (must match data_collection/factors_utils.py)
# ============================================================================
# Dictionary defining the reachability line for each table height (how far the
# robot arm can reach). Invalid condition: (y <= mx + c) and (x >= x_thresh).
REACHABILITY_BOUNDARIES = {
    1: {"m": 0.857, "c": -0.257, "x_thresh": 0.0},
    2: {"m": 0.867, "c": -0.2167, "x_thresh": 0.15},
    3: {"m": 0.85, "c": -0.17, "x_thresh": 0.2},
}


def _is_reachable(x, y, table_height):
    """
    Returns True if (x, y) is reachable at the given table height.
    Logic must match data_collection/factors_utils.is_valid_point().
    """
    h = int(round(table_height))
    if h not in REACHABILITY_BOUNDARIES:
        return True  # unknown height: allow (safety fallback)
    params = REACHABILITY_BOUNDARIES[h]
    m, c, x_thresh = params["m"], params["c"], params["x_thresh"]
    is_invalid = (y <= m * x + c) and (x >= x_thresh)
    return not is_invalid


# ============================================================================
# Validation
# ============================================================================

def is_valid_point(point):
    """
    Validates a point in the robot policy evaluation factor space, including
    reachability.

    Representation depends on VIEWPOINT_REPRESENTATION:
    - \"index\":  [x, y, table_height, viewpoint] with viewpoint in {0,1,2}
    - \"params\": [x, y, table_height, camera_azimuth, camera_elevation, camera_distance]
                 where the last 3 entries equal one of the 3 CAMERA_VIEWPOINTS parameter triples.
    """
    if point.shape[0] != DIMS:
        return False

    # Basic factor validation: x, y, table_height
    x = point[0].item()
    y = point[1].item()
    table_height = point[2].item()

    # Check x, y are in [0, 1] and multiples of 0.1 (with tolerance)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False
    if abs(round(x * 10) / 10 - x) > 1e-6 or abs(round(y * 10) / 10 - y) > 1e-6:
        return False

    # Check table_height is 1, 2, or 3
    if table_height not in [1.0, 2.0, 3.0]:
        return False

    # Reachability: same as data collection (factors_utils.is_valid_point)
    if not _is_reachable(x, y, table_height):
        return False

    if VIEWPOINT_REPRESENTATION == "index":
        # Last dimension is viewpoint index (0,1,2)
        viewpoint = point[3].item()
        if viewpoint not in [0.0, 1.0, 2.0]:
            return False
        return True

    # VIEWPOINT_REPRESENTATION == "params"
    if point.shape[0] < 6:
        return False
    cam_az = point[3].item()
    cam_el = point[4].item()
    cam_dist = point[5].item()

    # Construct tensor for comparison
    candidate = torch.tensor([cam_az, cam_el, cam_dist], **tkwargs)
    # Check if candidate matches (within tolerance) one of the allowed parameter triples
    diffs = torch.abs(_VIEWPOINT_PARAM_TENSOR - candidate)
    # All three components must be close for at least one viewpoint
    matches = torch.all(diffs < 1e-6, dim=1)
    if not torch.any(matches).item():
        return False

    return True
