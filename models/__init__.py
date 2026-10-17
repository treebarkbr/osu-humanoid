"""The three humanoid osu! heads: reading, aim, tapping."""
from .aim import AimNet
from .reading import ReadingNet
from .tapping import TappingNet

__all__ = ["AimNet", "TappingNet", "ReadingNet"]
