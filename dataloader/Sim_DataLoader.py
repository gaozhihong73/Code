"""Deprecated compatibility names for the unified TFPP data pipeline."""

import warnings

from .TFPP_DataLoader import TFPPDataset, get_dataloader


class SimTFPPDataset(TFPPDataset):
    """Deprecated alias; use TFPPDataset with prepared complete simulation data."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SimTFPPDataset is deprecated; prepare complete data and use TFPPDataset",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def get_sim_dataloader(*args, **kwargs):
    """Deprecated alias for the shared TFPP get_dataloader implementation."""
    warnings.warn(
        "get_sim_dataloader is deprecated; use get_dataloader",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_dataloader(*args, **kwargs)
