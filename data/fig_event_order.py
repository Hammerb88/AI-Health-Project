"""Icicle tree of the order of clinical events, with repeat-sample overlay.

Each column is one step in a patient's history; each bar is a node whose
height is its patient count, so a left-to-right path is one complete event
order (Dx -> Met -> Sample -> Tx, ...). Unlike a Sankey, nodes are never merged
across histories. Hatching marks the share of the node with >= 2 sequenced
samples; leaf labels add the dated pre/post-treatment sample pairs.

Cohort: MSK-MET patients also in MSK-CHORD with a primary diagnosis date.
"Sample" is the collection date of the MSK-MET sample (not its sequencing
date); "Met" is the first distant metastasis (regional lymph nodes excluded);
"Tx" is the first systemic therapy. Same-day ties are ordered Dx, Met, Tx,
Sample. Patients whose history starts with Met, or whose first systemic
therapy precedes the primary diagnosis, are dropped. Patients with no event
after a node appear as a gray "no further events" node in the next column.

Run: ``python fig_event_order.py`` -> reports/fig_event_order.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

import loaders as L
import viz

EVENTS = ["Dx", "Met", "Tx", "Sample"]
EVENT_NAMES = {"Dx": "Primary diagnosis", "Met": "First distant metastasis",
               "Tx": "First systemic therapy", "Sample": "Sample collected"}
COLOR = dict(zip(EVENTS, viz.CAT[:4]))
OUT = L.DATA_DIR / "reports"


def build_event_table() -> pd.DataFrame:
    """One row per patient: event days on the MSK-CHORD axis + sample counts."""
    ms = L.load_met_sample()
    imp = L.load_impact_sample()
    ss = L.load_chord_timeline("specimen_surgery")

    # The MSK-MET sample's row in CHORD gives both its collection day and its
    # sequencing day on the CHORD axis; the sequencing day ties the axes.
    metss = ss[ss.SAMPLE_ID.isin(ms.SAMPLE_ID)].set_index("PATIENT_ID")
    seq_birth = L.load_met_timeline("sequencing").set_index("PATIENT_ID").START_DATE
    offset = (seq_birth.reindex(metss.index) - metss.SEQ_DATE).dropna()
    pts = offset.index

    dx = L.load_met_timeline("diagnosis")
    dx = dx[~dx.SUBTYPE.isin(["Regional Lymph", "Lymph"]) & dx.PATIENT_ID.isin(pts)]
    dx = dx.assign(d=dx.START_DATE - dx.PATIENT_ID.map(offset))

    chdx = L.load_chord_timeline("diagnosis")
    tx = L.load_chord_timeline("treatment")
    o = pd.DataFrame({
        "Dx": chdx.groupby("PATIENT_ID").START_DATE.min().reindex(pts),
        "Met": dx.groupby("PATIENT_ID").d.min().reindex(pts),
        "Tx": tx.groupby("PATIENT_ID").START_DATE.min().reindex(pts),
        "Sample": metss.START_DATE.reindex(pts),
    }).dropna(subset=["Dx"])

    rank = {e: i for i, e in enumerate(EVENTS)}
    o["path"] = o.apply(lambda r: tuple(
        e for e, _ in sorted(((e, r[e]) for e in EVENTS if pd.notna(r[e])),
                             key=lambda z: (z[1], rank[z[0]]))), axis=1)
    # Drop histories that start with a metastasis, and those where systemic
    # therapy predates the primary diagnosis (likely treated before MSK).
    o = o[o.path.str[0] != "Met"]
    o = o[o.path.apply(lambda p: "Tx" not in p or p.index("Tx") > p.index("Dx"))]

    o["multi"] = imp.PATIENT_ID.value_counts().reindex(o.index).fillna(0) >= 2

    # Dated pre/post pair: another dated sample on the opposite side of first Tx.
    oth = ss[~ss.SAMPLE_ID.isin(ms.SAMPLE_ID) & ss.PATIENT_ID.isin(o.index)]
    t0 = oth.PATIENT_ID.map(o.Tx)
    s0 = oth.PATIENT_ID.map(o.Sample)
    straddle = ((oth.START_DATE > t0) & (s0 <= t0)) | ((oth.START_DATE <= t0) & (s0 > t0))
    o["dated_pair"] = o.index.isin(oth.loc[straddle, "PATIENT_ID"])
    return o


def draw(o: pd.DataFrame) -> plt.Figure:
    n_total = len(o)
    depth = int(o.path.str.len().max())
    col_w, gap = 1.0, 0.18
    min_label = 0.022 * n_total   # nodes smaller than this stay unlabeled

    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.grid(False)
    ax.set_axis_off()

    def node(prefix: tuple, y0: float, level: int) -> None:
        sub = o[o.path.apply(lambda p: p[:level + 1] == prefix)]
        n = len(sub)
        x = level * (col_w + gap)
        ev = prefix[-1]
        # 2px-style surface gap between stacked nodes.
        pad = min(n * 0.04, n_total * 0.0015)
        ax.add_patch(Rectangle((x, y0 + pad / 2), col_w, n - pad,
                               facecolor=COLOR[ev], edgecolor="none"))
        k = int(sub.multi.sum())
        if k:
            ax.add_patch(Rectangle((x, y0 + pad / 2), col_w, k, facecolor="none",
                                   edgecolor=viz.SURFACE, hatch="////",
                                   linewidth=0))
        if n >= min_label:
            ax.text(x + 0.06, y0 + n / 2,
                    f"{ev}\n{n:,}  ({n / n_total:.0%})",
                    va="center", ha="left", fontsize=8.6, color="#ffffff",
                    fontweight="semibold", linespacing=1.25)

        ends = sub[sub.path.str.len() == level + 1]
        children = (sub[sub.path.str.len() > level + 1].path
                    .str[level + 1].value_counts())
        # Children largest-first; patients with no further recorded event get
        # a neutral "no further events" node in the next column, below them.
        y = y0
        for ch, cn in children.items():
            node(prefix + (ch,), y, level + 1)
            y += cn
        if len(ends):
            ne = len(ends)
            xe = x + col_w + gap
            pe = min(ne * 0.04, n_total * 0.0015)
            ax.add_patch(Rectangle((xe, y + pe / 2), col_w, ne - pe,
                                   facecolor=viz.GRID, edgecolor="none"))
            # Two lines need ~3.5% of the height; smaller ones are in the CSV.
            if ne >= 0.035 * n_total:
                ax.text(xe + 0.06, y + ne / 2,
                        f"No further events  {ne:,}  ({ne / n_total:.0%})\n"
                        f"≥2 samples {ends.multi.mean():.0%} · "
                        f"{int(ends.dated_pair.sum())} dated pairs",
                        va="center", ha="left", fontsize=8, color=viz.INK_2,
                        linespacing=1.3)

    roots = o.path.str[0].value_counts()
    y = 0
    for r, rn in roots.items():
        node((r,), y, 0)
        y += rn

    for lv in range(depth + 1):
        ax.text(lv * (col_w + gap) + col_w / 2, -n_total * 0.012,
                f"event {lv + 1}" if lv < depth else "end", ha="center",
                va="bottom", fontsize=9, color=viz.MUTED)

    ax.set_xlim(-0.05, (depth + 1) * (col_w + gap))
    ax.set_ylim(n_total * 1.005, -n_total * 0.04)

    handles = [Patch(facecolor=COLOR[e], label=EVENT_NAMES[e]) for e in EVENTS]
    handles.append(Patch(facecolor=viz.GRID, label="No further recorded events"))
    handles.append(Patch(facecolor=viz.MUTED, edgecolor=viz.SURFACE,
                         hatch="////", label="≥2 samples sequenced (timing unknown)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.005),
              ncol=6, fontsize=8.6, handlelength=1.6)

    multi = o.multi.sum()
    pairs = o.dated_pair.sum()
    ax.set_title("Order of clinical events per patient", loc="left",
                 fontsize=12, fontweight="semibold", color=viz.INK, pad=34)
    ax.text(0, 1.035,
            f"{n_total:,} MSK-MET ∩ MSK-CHORD patients (excluding histories that "
            f"start with metastasis or have therapy before diagnosis) · bar height = patients · "
            f"{multi:,} ({multi / n_total:.1%}) sequenced ≥2 times · "
            f"only {pairs} have a dated sample on each side of first therapy",
            transform=ax.transAxes, fontsize=9, color=viz.MUTED, ha="left")
    fig.tight_layout()
    return fig


def summary_table(o: pd.DataFrame) -> pd.DataFrame:
    g = o.assign(order=o.path.apply(" → ".join)).groupby("order").agg(
        patients=("order", "size"), two_plus_samples=("multi", "sum"),
        dated_pre_post_pairs=("dated_pair", "sum"))
    g["pct"] = (g.patients / g.patients.sum() * 100).round(1)
    return g.sort_values("patients", ascending=False)


if __name__ == "__main__":
    viz.apply_theme()
    OUT.mkdir(exist_ok=True)
    o = build_event_table()
    fig = draw(o)
    fig.savefig(OUT / "fig_event_order.png", dpi=150, bbox_inches="tight")
    tab = summary_table(o)
    tab.to_csv(OUT / "event_order_summary.csv")
    print(tab.to_string())
    print(f"\npatients: {len(o):,}")
