"""5-point similarity alignment, identical to InsightFace `face_align.norm_crop`.

kps order (InsightFace/SCRFD): image-left eye, image-right eye, nose, image-left mouth, image-right mouth.
"""
import cv2
import numpy as np
from skimage.transform import SimilarityTransform

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def estimate_norm(kps: np.ndarray, size: int = 112) -> np.ndarray:
    """2x3 affine mapping image coords -> aligned crop of `size` (ArcFace template scaled)."""
    dst = ARCFACE_DST * (size / 112.0)
    t = SimilarityTransform.from_estimate(np.asarray(kps, dtype=np.float32), dst)
    if not t:
        raise ValueError("degenerate keypoints")
    return t.params[:2].astype(np.float32)


def norm_crop(img: np.ndarray, kps: np.ndarray, size: int = 112, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Warp any HxW[xC] image (RGB, depth, IR) into the aligned crop. Use INTER_NEAREST for depth."""
    return cv2.warpAffine(img, estimate_norm(kps, size), (size, size), flags=interp, borderValue=0)
