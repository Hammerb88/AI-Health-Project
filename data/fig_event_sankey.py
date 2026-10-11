"""Sankey of the order of clinical events.

Same cohort and event definitions as fig_event_order.py (which it reuses).
Columns are event positions 1-4; each node is "the k-th event was X". Unlike
the icicle, a Sankey merges patients who reach the same node by different
routes, so it shows step-to-step flows rather than each full history (those
are in the icicle and reports/event_order_summary.csv). Links take the colour
of the event they lead to; patients with no later recorded event flow into a
gray "no further events" node.

Run: ``python fig_event_sankey.py`` -> reports/fig_event_sankey.png
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Patch, Rectangle
from matplotlib.path import Path

import viz
from fig_event_order import COLOR, EVENT_NAMES, EVENTS, OUT, build_event_table

END = "End"
NODE_ORDER = EVENTS + [END]
NODE_COLOR = {**COLOR, END: viz.AXIS}


def band(ax, x0, x1, y0a, y0b, y1a, y1b, color):
    """Filled cubic-Bezier ribbon from [y0a, y0b] at x0 to [y1a, y1b] at x1."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = ([Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO]
             + [Path.CURVE4] * 3 + [Path.CLOSEPOLY])
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, alpha=0.42,
                           edgecolor="none", zorder=1))


def draw(o) -> plt.Figure:
    paths = o.path.tolist()
    n_total = len(paths)
    depth = max(len(p) for p in paths)
    ncol = depth + 1                       # last column holds only End

    def at(p, j):
        return p[j] if j < len(p) else END

    # Node sizes per column. A patient appears in column j if their path has
    # at least j entries (column j == len(path) is their End node).
    nodes = [defaultdict(int) for _ in range(ncol)]
    for p in paths:
        for j in range(len(p) + 1):
            if j < ncol:
                nodes[j][at(p, j)] += 1
    flows = defaultdict(int)
    for p in paths:
        for j in range(len(p)):
            flows[(j, p[j], at(p, j + 1))] += 1

    gap = 0.018 * n_total
    node_w = 0.05
    xs = np.arange(ncol) * 1.0
    ypos = []                              # (col) -> {node: (top, bottom)}
    for j in range(ncol):
        y, pos = 0.0, {}
        for e in NODE_ORDER:
            if nodes[j].get(e):
                pos[e] = (y, y + nodes[j][e])
                y += nodes[j][e] + gap
        ypos.append(pos)

    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_axis_off()
    ax.grid(False)

    # Links: leave each source in target order, enter each target in source
    # order, so ribbons do not cross inside a node.
    out_off = [defaultdict(float) for _ in range(ncol)]
    in_off = [defaultdict(float) for _ in range(ncol)]
    for j in range(ncol - 1):
        for s in NODE_ORDER:
            for t in NODE_ORDER:
                n = flows.get((j, s, t))
                if not n:
                    continue
                s0 = ypos[j][s][0] + out_off[j][s]
                t0 = ypos[j + 1][t][0] + in_off[j + 1][t]
                band(ax, xs[j] + node_w, xs[j + 1], s0, s0 + n, t0, t0 + n,
                     NODE_COLOR[t])
                out_off[j][s] += n
                in_off[j + 1][t] += n

    min_label = 0.015 * n_total
    for j in range(ncol):
        for e, (top, bot) in ypos[j].items():
            n = int(round(bot - top))
            ax.add_patch(Rectangle((xs[j], top), node_w, n, facecolor=NODE_COLOR[e],
                                   edgecolor="none", zorder=2))
            if n < min_label:
                continue
            name = "No further events" if e == END else e
            sub = o[o.path.apply(lambda p: at(p, j) == e and j <= len(p))]
            multi = sub.multi.mean()
            ax.text(xs[j] + node_w + 0.025, (top + bot) / 2,
                    f"{name}  {n:,} ({n / n_total:.0%})\n≥2 samples {multi:.0%}",
                    va="center", ha="left", fontsize=8.3, color=viz.INK,
                    linespacing=1.25, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=viz.SURFACE,
                              edgecolor="none", alpha=0.85))

    col_bottom = max(max(b for _, b in pos.values()) for pos in ypos)
    for j in range(ncol):
        ax.text(xs[j] + node_w / 2, -0.03 * n_total,
                f"event {j + 1}" if j < depth else "end", ha="left",
                va="bottom", fontsize=9, color=viz.MUTED)
    ax.set_xlim(-0.05, xs[-1] + 0.75)
    ax.set_ylim(col_bottom + 0.01 * n_total, -0.06 * n_total)

    handles = [Patch(facecolor=COLOR[e], label=EVENT_NAMES[e]) for e in EVENTS]
    handles.append(Patch(facecolor=viz.AXIS, label="No further recorded events"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0.0),
              ncol=5, fontsize=8.6, handlelength=1.6)

    pairs = int(o.dated_pair.sum())
    ax.set_title("Order of clinical events per patient", loc="left",
                 fontsize=12, fontweight="semibold", color=viz.INK, pad=34)
    ax.text(0, 1.035,
            f"{n_total:,} MSK-MET ∩ MSK-CHORD patients (excluding histories that start "
            f"with metastasis or have therapy before diagnosis) · ribbon width = patients, "
            f"coloured by the next event · {int(o.multi.sum()):,} sequenced ≥2 times · "
            f"{pairs} dated pre/post-therapy pairs",
            transform=ax.transAxes, fontsize=9, color=viz.MUTED, ha="left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    viz.apply_theme()
    OUT.mkdir(exist_ok=True)
    o = build_event_table()
    draw(o).savefig(OUT / "fig_event_sankey.png", dpi=150, bbox_inches="tight")
    print(f"patients: {len(o):,}")
