import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.cm

MAGMA_LUT = None
VIRIDIS_LUT = None

def __get_magma_lut(num_colors: int = 256, device='cuda'):
    """Calculates the Magma color lookup table."""
    global MAGMA_LUT
    if MAGMA_LUT != None:
        return MAGMA_LUT
    magma_map = matplotlib.cm.get_cmap('magma', num_colors)
    lut_np = magma_map(np.linspace(0.0, 1.0, num_colors))[:, :3]
    lut = torch.from_numpy(lut_np).float().to(device)
    # The shape will be (num_colors, 3)
    MAGMA_LUT = lut
    return lut

def __get_viridis_lut(num_colors: int = 256, device='cuda'):
    """Calculates the Viridis color lookup table."""
    global VIRIDIS_LUT
    if VIRIDIS_LUT != None:
        return VIRIDIS_LUT
    viridis_map = matplotlib.cm.get_cmap('viridis', num_colors) 
    
    lut_np = viridis_map(np.linspace(0.0, 1.0, num_colors))[:, :3]
    lut = torch.from_numpy(lut_np).float().to(device)
    # The shape will be (num_colors, 3)
    VIRIDIS_LUT = lut
    return lut

LUT_MAP = {
    "viridis": __get_viridis_lut,
    "magma": __get_magma_lut
}

def map_single_val_to_image(val: torch.Tensor, cm: str="magma", low_percentile: float=0.01, high_percentile: float=0.99):
    """
    Converts a single value tensor into a robust Magma heatmap for visualization.
    
    Args:
        val (torch.Tensor): The single value map (H, W, 1).
        low_percentile (float): The lower percentile to use for robust min clipping.
        high_percentile (float): The upper percentile to use for robust max clipping.
        
    Returns:
        torch.Tensor: The colorized depth map (H, W, 3) ready for logging.
    """
    # 0. Get color LUT
    color_lut = LUT_MAP[cm]()

    # 1. Reshape val
    if val.ndim == 4:
        assert val.shape[0] == 1
        val = val.squeeze(0)

    # 2. Identify and isolate the object pixels (non-zero)
    object_mask = val > 0
    object_depth = val[object_mask]
    
    # Handle the edge case where the object mask is empty
    if object_depth.numel() == 0:
        # Return a black image if there is no object
        h, w, _ = val.shape
        return torch.zeros((h, w, 3), dtype=val.dtype)

    # Ensure the tensor is float for quantile calculation
    object_depth = object_depth.float()
    
    # 3. Calculate robust bounds (v_min and v_max) using quantiles
    # This ignores outlier noise/sparkles at the extremes
    vmin = torch.quantile(object_depth, low_percentile)
    vmax = torch.quantile(object_depth, high_percentile)
    
    # 4. Create a normalized tensor (0 to 1) for colormapping
    normalized_depth = (val - vmin) / (vmax - vmin + 1e-16)
    
    # 5. Clamp the normalized values to [0, 1]
    # Background (0) will map to a negative value, which clamps to 0 (dark/black).
    # Overly far objects will map to values > 1, which clamp to 1 (bright/white).
    clamped_x = torch.clamp(normalized_depth, 0.0, 1.0)
    
    # 6. Map to integer indices for LUT lookup
    num_colors = color_lut.shape[0] - 1
    # Scale from [0, 1] to [0, num_colors] and cast to long for indexing
    indices = (clamped_x * num_colors).long()
    
    # 7. Apply LUT lookup (GPU operation)
    # Use torch.index_select on the flattened indices for fast color mapping
    # This takes the (H*W) indices and returns (H*W, 3) color values
    colored_flat = color_lut.index_select(0, indices.flatten())
    
    # 8. Reshape for visualization
    h, w, _ = val.shape
    colored_image = colored_flat.view(h, w, 3)
    
    return colored_image
