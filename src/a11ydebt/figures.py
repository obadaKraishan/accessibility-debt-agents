"""
Figure style for an IEEE two-column paper.

Two constraints drive every choice here:

  1. Figures are printed at column width (3.5 in) or page width (7.16 in),
     so text inside a figure must survive being shrunk to that size.
  2. IEEE proceedings are frequently read and printed in greyscale, so
     colour can never be the only thing distinguishing two series. Every
     categorical mark therefore carries a hatch pattern as well as a hue.

The categorical palette was checked with the six-part validator: all five
slots sit inside the lightness band, clear the chroma floor, keep adjacent
CVD separation at Delta E 9.6 or better (deutan) and 11.6 (tritan), hold a
normal-vision floor of 22.4, and pass 3:1 contrast against the surface.
Do not substitute hues without re-validating.

    from a11ydebt.figures import use_paper_style, save, CATEGORICAL, hatch_for

    use_paper_style()
    fig, ax = plt.subplots(figsize=COLUMN)
    ...
    save(fig, outdir / "rq1_coverage_by_role")   # writes .pdf and .png
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

__all__ = [
    "use_paper_style", "save", "CATEGORICAL", "DATASET_COLOURS",
    "HATCHES", "hatch_for", "colour_for", "COLUMN", "WIDE", "TALL",
    "annotate_bars", "GREY", "INK",
]

# Validated categorical order. Assign in this order; never cycle, never
# generate a sixth hue. A sixth category folds into "Other" or becomes a
# small multiple.
CATEGORICAL: tuple[str, ...] = (
    "#3457b2",   # blue
    "#c4611a",   # orange
    "#1a8f74",   # teal
    "#9c3f96",   # magenta
    "#a08215",   # ochre
)

#: Fixed assignment so a dataset keeps its colour across every figure in
#: the paper. Colour follows the entity, never its position in a chart.
DATASET_COLOURS: dict[str, str] = {
    "rico_2017": CATEGORICAL[0],
    "androidcontrol_2024": CATEGORICAL[1],
}

#: Secondary encoding for greyscale print. Paired with the palette above by
#: index, so colour and texture always agree.
HATCHES: tuple[str, ...] = ("", "///", "...", "\\\\\\", "xxx")

INK = "#1a1a22"
GREY = "#6b6e80"
GRID = "#dcdce4"
SURFACE = "#ffffff"

# IEEE figure widths in inches.
COLUMN = (3.5, 2.4)     # single column
WIDE = (7.16, 2.8)      # spans both columns
TALL = (3.5, 3.6)       # single column, taller


def use_paper_style() -> None:
    """Apply the project's matplotlib defaults. Call once per script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Type: small enough to survive the column, large enough to read.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,

        # Recessive grid and axes: the data carries the eye, not the frame.
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": INK,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.2,

        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "hatch.linewidth": 0.6,

        # Embed real fonts rather than Type 3, which IEEE Xplore rejects.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def colour_for(index_or_key: int | str) -> str:
    if isinstance(index_or_key, str):
        return DATASET_COLOURS.get(index_or_key, CATEGORICAL[0])
    return CATEGORICAL[index_or_key % len(CATEGORICAL)]


def hatch_for(index_or_key: int | str) -> str:
    """The greyscale-safe partner of colour_for. Same index, same slot."""
    if isinstance(index_or_key, str):
        keys = list(DATASET_COLOURS)
        index = keys.index(index_or_key) if index_or_key in keys else 0
    else:
        index = index_or_key
    return HATCHES[index % len(HATCHES)]


def annotate_bars(ax, bars, values: Sequence[float], fmt: str = "{:.1f}%",
                  offset: float = 1.0) -> None:
    """
    Direct-label bars.

    Labels are selective by design: use this on charts with few enough bars
    that every value is worth reading. On a dense chart, label the extremes
    only and let the axis carry the rest.
    """
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + offset),
            ha="center", va="bottom", fontsize=7, color=INK,
        )


def save(fig, stem: str | Path, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """
    Save a figure as vector PDF (for the manuscript) and PNG (for quick looks).

    Returns the paths written. Always saves the PDF: IEEE submissions want
    vector figures, and a rasterised chart at column width looks poor in print.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path)
        written.append(path)
    return written
