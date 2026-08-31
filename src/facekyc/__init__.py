"""FaceKYC core package."""

from facekyc.config import Settings, load_settings
from facekyc.pipeline import FaceKYCPipeline

__all__ = ["FaceKYCPipeline", "Settings", "load_settings"]
__version__ = "1.0.0"
