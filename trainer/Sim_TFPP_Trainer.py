"""Deprecated trainer alias; simulations now use the production TFPPTrainer."""

import warnings

from .TFPP_Trainer import TFPPTrainer


class SimTFPPTrainer(TFPPTrainer):
    """Compatibility alias without a parallel training implementation."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SimTFPPTrainer is deprecated; use TFPPTrainer",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
