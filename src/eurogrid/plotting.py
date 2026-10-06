"""Shared plotting style and helpers for the notebooks.

One matplotlib style, colour-blind-safe defaults (Okabe–Ito), consistent
figure sizes and font sizes, and small helpers for panel labels, cartopy
maps and 3-line captions. Keeping these here keeps notebook cells short.
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt

#: Okabe–Ito colour-blind-safe palette.
PALETTE: Final[list[str]] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",
]

SEQUENTIAL_CMAP: Final[str] = "viridis"
DIVERGING_CMAP: Final[str] = "RdBu_r"


def apply_style() -> None:
    """Project-wide matplotlib defaults; call once per notebook."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 11,
        }
    )


def label_panels(axes, labels: str = "abcdefgh") -> None:
    """Write (a), (b), (c) ... in the top-left corner of each axes."""
    import numpy as np

    for ax, tag in zip(np.asarray(axes).ravel(), labels, strict=False):
        ax.text(
            0.02,
            0.97,
            f"({tag})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )


def map_axes(figsize=(8, 6), extent=(-40.0, 40.0, 30.0, 75.0)):
    """Figure + cartopy axes for Europe with coastlines, borders, lat/lon grid.

    ``extent`` is (lon_min, lon_max, lat_min, lat_max). Returns (fig, ax).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.LambertConformal(central_longitude=10.0, central_latitude=52.0)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj}, layout="constrained")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    return fig, ax


def caption(fig, text: str) -> None:
    """Figure caption of at most ~3 lines below the axes."""
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=8.5, wrap=True)
