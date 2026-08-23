"""Unified preprocessor interface for ZIT ControlNet."""

import numpy as np


def preprocess(mode: str, image: np.ndarray, model_dir: str, **kwargs) -> np.ndarray:
    """Run preprocessor on image, return control map as RGB numpy array.

    Args:
        mode: One of "canny", "pose", "depth", "hed", "scribble", "gray"
        image: RGB numpy array (H, W, 3), uint8
        model_dir: Path to model directory (for loading preprocessor weights)
    """
    if mode == "canny":
        from .canny import apply_canny
        return apply_canny(image, **kwargs)
    elif mode == "pose":
        from .dwpose import apply_dwpose
        return apply_dwpose(image, model_dir, **kwargs)
    elif mode == "depth":
        from .depth import apply_zoedepth
        return apply_zoedepth(image, model_dir, **kwargs)
    elif mode == "hed":
        from .hed import apply_hed
        return apply_hed(image, model_dir, **kwargs)
    elif mode == "scribble":
        from .hed import apply_scribble
        return apply_scribble(image, model_dir, **kwargs)
    elif mode == "gray":
        from .gray import apply_gray
        return apply_gray(image)
    else:
        raise ValueError(f"Unknown preprocessor mode: {mode}")
