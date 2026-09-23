"""Chart theme for the exploration notebook.

Implements the reference data-viz palette for matplotlib so every figure in the
notebook reads as one system. The rules that matter downstream:

* Categorical hues are assigned in fixed slot order and never cycled. Past the
  8th series, fold the tail into "Other" or facet -- never generate a 9th hue.
* Scatter / small-multiple forms (all-pairs) are capped at the first 3 slots.
* Sequential encoding is one hue, light to dark. Diverging is blue/red with a
  neutral gray midpoint -- never a rainbow, never a hue at the midpoint.
* Nominal categories get ONE color, not a value ramp: bar length already encodes
  magnitude, so coloring by magnitude burns the only free channel.
* Chrome (grid, axis, labels) stays recessive; marks stay thin.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in validated order. Do not reorder: the ordering is the
# colorblind-safety mechanism, not cosmetic.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_ALLPAIRS = CAT[:3]   # cap for scatter / small multiples

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Sequential: blue, light -> dark (steps 100-700 of the documented ramp).
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

# Diverging: blue <-> red, neutral gray midpoint (reads as "nothing").
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#cde2fb", "#f0efec",
                     "#f6c6c6", "#e34948", "#a32424"])

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}


def apply_theme() -> None:
    """Install the theme globally. Call once at the top of the notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,

        # Recessive chrome: hairline grid one shade off the surface, solid
        # (never dashed -- dashing reads as "threshold" when it is just a grid).
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "patch.linewidth": 0,

        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "image.cmap": "seq_blue",
    })
    for name, cm in (("seq_blue", SEQ), ("div_blue_red", DIV)):
        if name not in mpl.colormaps:
            mpl.colormaps.register(cm, name=name)


def finish(ax, title=None, subtitle=None, xlabel=None, ylabel=None,
           source=None):
    """Apply the standard title block and axis labels to an axes."""
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight="semibold",
                     loc="left", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=MUTED,
                fontsize=9, va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    if source:
        ax.text(0, -0.16, source, transform=ax.transAxes, color=MUTED,
                fontsize=8, va="top", ha="left")
    return ax


def label_bars(ax, bars, values, fmt="{:,.0f}", pad=0.01, horizontal=True):
    """Direct-label bar ends, outside the mark so nothing is ever clipped.

    Labels wear text ink, never the series color -- the mark carries identity.
    """
    span = (ax.get_xlim()[1] if horizontal else ax.get_ylim()[1])
    for bar, v in zip(bars, values):
        if horizontal:
            ax.text(bar.get_width() + span * pad,
                    bar.get_y() + bar.get_height() / 2, fmt.format(v),
                    va="center", ha="left", fontsize=8.5, color=INK_2)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + span * pad, fmt.format(v),
                    va="bottom", ha="center", fontsize=8.5, color=INK_2)


def fold_other(series, k=7, other_label="Other"):
    """Keep the top k categories and fold the tail into one bucket.

    The alternative -- a 9th generated hue -- is indistinguishable from an
    existing slot under colorblind simulation.
    """
    top = series.nlargest(k)
    rest = series.drop(top.index).sum()
    if rest > 0:
        top[other_label] = rest
    return top


def stat_tile(ax, value, label, note=None, fmt="{:,.0f}"):
    """A hero number. When the story is one number, it is not a bar chart."""
    ax.axis("off")
    txt = fmt.format(value) if not isinstance(value, str) else value
    ax.text(0, 0.62, txt, fontsize=30, color=INK, ha="left", va="center",
            fontweight="semibold")
    ax.text(0, 0.24, label, fontsize=10, color=INK_2, ha="left", va="center")
    if note:
        ax.text(0, 0.05, note, fontsize=8.5, color=MUTED, ha="left",
                va="center")
    return ax
