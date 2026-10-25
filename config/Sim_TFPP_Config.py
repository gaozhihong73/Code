"""Deprecated compatibility configuration for unified simulation training."""

import warnings

from .TFPP_Config import TFPPConfig


class SimTFPPConfig(TFPPConfig):
    """TFPPConfig compatibility alias; six pre-split CSV fields were removed."""

    def __init__(self, **data):
        warnings.warn(
            "SimTFPPConfig is deprecated; use TFPPConfig with prepared simulation data",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
