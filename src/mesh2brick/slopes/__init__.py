from .prepare_slopes import SlopeConfig, SlopeResult, prepare_slopes
from .detection import detect_features, compute_optimal_scale, match_slope_to_bricks, SlopeRegion, Features
from .deformation import deform_mesh, apply_scale, DeformationResult
from .tiling import place_slope_bricks
