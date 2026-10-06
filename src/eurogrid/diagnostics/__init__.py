"""Scientific diagnostics for blocking, renewable drought, and ramps."""

from eurogrid.diagnostics.blocking import (
    detect_tm1990,
    meridional_gradients,
    persistence_summary,
)

__all__ = ["detect_tm1990", "meridional_gradients", "persistence_summary"]
