"""Shared matplotlib style for publication-quality figures.

Apply once at the top of any plotting entry point with :func:`apply_style`,
size the canvas with :func:`figure_size`, and write it with
:func:`save_figure`.

Three rules hold for every paper figure:

* The canvas is authored at the width it prints at, so the point sizes set in
  :func:`print_sizes` are the sizes the reader sees (9 pt body, 8 pt ticks and
  legends, never below 7 pt).
* Text and mathematics share one face. STIX is bundled with matplotlib, is
  metrically close to the Times face Elsevier typesets in, and ships a
  matching mathtext set; the previous serif body with DejaVu Sans mathematics
  mixed two families in every label.
* The saved file is exactly the authored width at 300 dpi, RGB, with the
  resolution recorded in its metadata. No ``bbox_inches="tight"``: trimming
  after layout changes the width and therefore the printed type size.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#: Raster resolution of every saved figure, at its printed size.
DPI = 300


def apply_style(*, printed: bool = True) -> None:
    """Serif faces, at the size they will print.

    ``printed=True`` assumes the figure is authored at its final width via
    :func:`figure_size`, so the point sizes below are what the reader sees.
    Pass ``False`` only for a figure meant to be scaled after the fact.
    """
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
            # Weight is reserved for data. Bold labels and titles compete with
            # the marks and read as headings, which captions already provide.
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.edgecolor": "#52514e",
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            # Embed TrueType in vector output so the text stays editable and
            # searchable in the production PDF.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            # Lay out at the resolution the file is written at: text extents
            # measured at 100 dpi are hinted differently and run ~10% wide,
            # which throws off any layout computed from them.
            "figure.dpi": DPI,
        }
    )
    if printed:
        print_sizes()


# The manuscript's text width: elsarticle's 3p layout, 468 pt = 16.45 cm.
# Every figure is authored at a fraction of this and included at the same
# fraction of \linewidth, so a point of type in one figure prints the same
# size as a point in every other, and at the size it was set in. (Authored at
# 16.0 cm, every figure printed 2.8% larger than set.)
#
# A change of document class changes this number and nothing else: set
# ``AWWL_TEXT_WIDTH_CM`` (for example 13.55 for elsarticle's preprint layout)
# and regenerate. The include fractions in the .tex files stay valid.
TEXT_WIDTH_CM = float(os.environ.get("AWWL_TEXT_WIDTH_CM", "16.45"))
TEXT_WIDTH_IN = TEXT_WIDTH_CM / 2.54


def figure_size(fraction: float, aspect: float = 1.6) -> tuple[float, float]:
    """Figure size in inches for a figure printed at ``fraction`` of the line.

    Args:
        fraction: The ``width=<fraction>\\linewidth`` the figure is included
            at. Author and include must agree; that agreement is the point.
        aspect: Width divided by height. 1.6 suits a single plot; use a
            larger value for wide small-multiples and a smaller one for a
            square scatter.
    """
    width = TEXT_WIDTH_IN * fraction
    return width, width / aspect


def print_sizes() -> None:
    """Type sizes chosen for the printed size, not for the canvas.

    Authoring at the printed width means the rcParams below are the sizes a
    reader actually sees. They sit a step below the manuscript's 11 pt body:
    9 pt labels, 8 pt ticks and legends, and nothing under 7 pt anywhere.
    Line weights follow the same logic: 1.2 pt data lines against 0.8 pt axes.
    """
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 9,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "patch.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.handlelength": 2.2,
            "legend.borderaxespad": 0.3,
        }
    )


#: Set by ``awwl figures --pdf``: also write a vector PDF beside each PNG, for
#: the production stage, where Elsevier prefers vector line art.
SAVE_PDF = os.environ.get("AWWL_FIGURE_PDF", "") == "1"


def save_figure(fig, out_path: str | Path, *, layout: bool = True) -> Path:
    """Write ``fig`` at exactly its authored size: 300 dpi RGB PNG (+ PDF).

    ``layout=False`` skips ``tight_layout`` for figures that place their axes
    by hand (the sample grid, the protocol diagram).
    """
    from PIL import Image

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if layout:
        fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=DPI, facecolor="white")
    if SAVE_PDF:
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    # Matplotlib writes RGBA. Production systems expect RGB, and without the
    # physical resolution in the metadata a viewer cannot tell the print size.
    with Image.open(out) as im:
        rgb = im.convert("RGB")
    rgb.save(out, dpi=(DPI, DPI))
    logger.info("wrote %s", out)
    return out
