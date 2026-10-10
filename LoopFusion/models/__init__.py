"""LoopDeblur package.

Importing ``models`` registers architectures, datasets, losses and the model
into the official BasicSR registries. ``basicsr`` is used as an external
dependency; nothing inside ``site-packages`` is patched.
"""

from . import arch       # noqa: F401  registers architectures
from . import data       # noqa: F401  registers datasets
from . import losses     # noqa: F401  registers losses
from .loopdeblur_model import LoopDeblurModel  # noqa: F401  registers the model
from .arch import LoopDeblur

__all__ = ['LoopDeblur', 'LoopDeblurModel']
