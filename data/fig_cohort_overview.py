"""Cohort overview figures.

fig_cohort_overview.png
  Left   three-study Venn (patients). Areas are schematic, not to scale; the
         regions the project uses are shaded: MET ∩ IMPACT (pan-cancer) and,
         nested inside it, MET ∩ IMPACT ∩ CHORD (treatment-aware).
  Right  one row per cancer type in the core cohort (MET ∩ IMPACT, FACETS QC
         pass): a stacked bar of patients with / without a distant metastasis
         (MSK-MET curated flags, ever), and an aligned heatmap of the share of
         patients receiving each treatment class (MSK-CHORD, any time).
         Treatment classes overlap within a patient, so they are a heatmap,
         not further stacked bars.

fig_metastatic_spread.png
  Cancer type × metastatic organ heatmap: share of each cancer type's
  patients with a distant metastasis to that organ. Same row order as above.

Run: ``python fig_cohort_overview.py`` -> reports/
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch

import loaders as L
import viz

OUT = L.DATA_DIR / "reports"
MIN_TX_PATIENTS = 50   # below this, a cancer type shows "no treatment data"

TX_COLS = ["Chemo", "Hormone", "Targeted", "Immuno", "Biologic",
           "Bone Treatment", "Investigational", "Radiation"]

ORGAN_LABEL = {
    "DIST_LN": "Distant LN", "LUNG": "Lung", "PLEURA": "Pleura",
    "MEDIASTINUM": "Mediastinum", "LIVER": "Liver",
    "BILIARY_TRACT": "Biliary tract", "INTRA_ABDOMINAL": "Intra-abdominal",
    "BOWEL": "Bowel", "CNS_BRAIN": "CNS / brain", "BONE": "Bone",
    "PNS": "Peripheral nerve", "ADRENAL_GLAND": "Adrenal",
    "KIDNEY": "Kidney", "BLADDER_UT": "Bladder / UT",
    "FEMALE_GENITAL": "Female genital", "OVARY": "Ovary",
    "MALE_GENITAL": "Male genital", "SKIN": "Skin",
    "HEAD_NECK": "Head & neck", "BREAST": "Breast",
}


# --------------------------------------------------------------------- data

def venn_counts() -> dict[str, int]:
    s = L.cohorts()
    m, i, c = s["met"], s["impact"], s["chord"]
    return {
        "M": len(m - i - c), "I": len(i - m - c), "C": len(c - m - i),
        "MI": len((m & i) - c), "MC": len((m & c) - i), "IC": len((i & c) - m),
        "MIC": len(m & i & c),
    }


def cancer_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """Per cancer type: patients, metastatic share, treatment-class shares."""
    met = cohort[L.DMETS_COLS].eq("Yes").drop(columns="DMETS_DX_UNSPECIFIED")
    t = pd.DataFrame({
        "patients": cohort.groupby("CANCER_TYPE").size(),
        "metastatic": met.any(axis=1).groupby(cohort.CANCER_TYPE).sum(),
    })
    t["no_met"] = t.patients - t.metastatic

    chord_ids = set(L.load_chord_sample().PATIENT_ID)
    in_chord = cohort[cohort.PATIENT_ID.isin(chord_ids)]
    t["tx_n"] = in_chord.groupby("CANCER_TYPE").size().reindex(t.index).fillna(0)

    tx = L.load_chord_timeline("treatment")
    rad = L.load_chord_timeline("radiation")
    got = {cl: set(tx.loc[tx.SUBTYPE == cl, "PATIENT_ID"])
           for cl in TX_COLS if cl != "Radiation"}
    got["Radiation"] = set(rad.PATIENT_ID)
    for cl in TX_COLS:
        share = in_chord.PATIENT_ID.isin(got[cl]).groupby(in_chord.CANCER_TYPE).mean()
        t[cl] = share.reindex(t.index)
    t.loc[t.tx_n < MIN_TX_PATIENTS, TX_COLS] = np.nan
    return t.sort_values("patients", ascending=False)


def tropism_table(cohort: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    cols = [c for c in L.DMETS_COLS if c != "DMETS_DX_UNSPECIFIED"]
    trop = cohort[cols].eq("Yes").groupby(cohort.CANCER_TYPE).mean()
    trop.columns = [c.replace("DMETS_DX_", "") for c in cols]
    organ_order = trop.mul(cohort.CANCER_TYPE.value_counts(), axis=0).sum()
    trop = trop[organ_order.sort_values(ascending=False).index]
    trop.columns = [ORGAN_LABEL[c] for c in trop.columns]
    return trop.loc[order]


# -------------------------------------------------------------------- draw

def draw_venn(ax, counts: dict[str, int], core_n: int, tx_core_n: int) -> None:
    centers = {"M": (-0.55, 0.32), "I": (0.55, 0.32), "C": (0.0, -0.58)}
    r = 1.0
    names = {"M": "MSK-MET", "I": "MSK-IMPACT 50K", "C": "MSK-CHORD"}

    # Region fills are rasterised from circle membership so each of the seven
    # regions can be coloured independently.
    xs = np.linspace(-2.0, 2.0, 900)
    ys = np.linspace(-2.0, 1.8, 855)
    X, Y = np.meshgrid(xs, ys)
    inside = {k: (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
              for k, (cx, cy) in centers.items()}
    M, I, C = inside["M"], inside["I"], inside["C"]
    regions = {"M": M & ~I & ~C, "I": I & ~M & ~C, "C": C & ~M & ~I,
               "MI": M & I & ~C, "MC": M & C & ~I, "IC": I & C & ~M,
               "MIC": M & I & C}

    light, dark = viz.SEQ_STEPS[1], viz.SEQ_STEPS[4]
    fill = {"MI": light, "MIC": dark}
    img = np.zeros(X.shape + (4,))
    for k, mask in regions.items():
        img[mask] = to_rgba(fill.get(k, "#efeee9"))
    ax.imshow(img, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
              interpolation="antialiased", zorder=1)

    for k, (cx, cy) in centers.items():
        ax.add_patch(Circle((cx, cy), r, fill=False, lw=1.4,
                            edgecolor=viz.INK_2, zorder=2))

    label_pos = {"M": (-1.05, 0.62), "I": (1.05, 0.62), "C": (0.0, -1.15),
                 "MI": (0.0, 0.85), "MC": (-0.62, -0.38), "IC": (0.62, -0.38),
                 "MIC": (0.0, -0.02)}
    for k, (x, y) in label_pos.items():
        on_dark = k == "MIC"
        ax.text(x, y, f"{counts[k]:,}", ha="center", va="center", zorder=3,
                fontsize=11 if k in fill else 9.5,
                fontweight="semibold" if k in fill else "normal",
                color="#ffffff" if on_dark else (viz.INK if k in fill else viz.INK_2))

    title_pos = {"M": (-1.35, 1.42, "right"), "I": (1.35, 1.42, "left"),
                 "C": (0.0, -1.72, "center")}
    for k, (x, y, ha) in title_pos.items():
        n = len(L.cohorts()[{"M": "met", "I": "impact", "C": "chord"}[k]])
        ax.text(x, y, f"{names[k]}\n{n:,} patients", ha=ha, va="center",
                fontsize=9.5, color=viz.INK, fontweight="semibold", zorder=3)

    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.05, 1.85)
    ax.set_aspect("equal")
    ax.set_axis_off()

    pan = counts["MI"] + counts["MIC"]
    handles = [
        Patch(facecolor=light, label=f"Pan-cancer: MET ∩ IMPACT, {pan:,} patients"
              f" → {core_n:,} after FACETS QC"),
        Patch(facecolor=dark, label=f"Treatment-aware: all three, {counts['MIC']:,}"
              f" → {tx_core_n:,} in the 5 CHORD cancer types"),
        Patch(facecolor="#efeee9", edgecolor=viz.AXIS, label="Not used"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.0),
              fontsize=8.6, handlelength=1.4)
    viz.finish(ax, "Patients in each study",
               subtitle="patient counts per region · circle areas are schematic")


def draw_bars(ax, t: pd.DataFrame) -> None:
    y = np.arange(len(t))
    ax.barh(y, t.metastatic, height=0.7, color=viz.CAT[1],
            label="Distant metastasis (ever)")
    ax.barh(y, t.no_met, left=t.metastatic, height=0.7, color=viz.SEQ_STEPS[0],
            label="No distant metastasis recorded", edgecolor=viz.SURFACE,
            linewidth=1.5)
    for yi, (n, m) in enumerate(zip(t.patients, t.metastatic)):
        ax.text(n + t.patients.max() * 0.012, yi, f"{n:,}  ({m / n:.0%} met)",
                va="center", ha="left", fontsize=7.8, color=viz.INK_2)
    ax.set_yticks(y, t.index, fontsize=8.6)
    ax.set_ylim(len(t) - 0.5, -0.5)
    ax.set_xlim(0, t.patients.max() * 1.42)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=8.4)
    viz.finish(ax, "Patients by cancer type",
               subtitle="core cohort (MET ∩ IMPACT, FACETS QC pass)",
               xlabel="patients")


def draw_tx(ax, t: pd.DataFrame) -> None:
    vals = t[TX_COLS].to_numpy(dtype=float)
    shown = np.ma.masked_invalid(vals)
    ax.imshow(np.isnan(vals), cmap=plt.matplotlib.colors.ListedColormap(
        [viz.SURFACE, "#efeee9"]), aspect="auto", vmin=0, vmax=1)
    im = ax.imshow(shown, cmap="seq_blue", vmin=0, vmax=1, aspect="auto")
    empty = np.isnan(vals).all(axis=1)
    if empty.any():
        rows = np.flatnonzero(empty)
        ax.text((len(TX_COLS) - 1) / 2, rows.mean(),
                f"No treatment data\n\nMSK-CHORD covers only\nNSCLC, colorectal, breast,\n"
                f"prostate and pancreatic cancer\n({len(rows)} cancer types not covered)",
                ha="center", va="center", fontsize=9, color=viz.INK_2,
                linespacing=1.35)
    for i in np.flatnonzero(~empty):
        for j, v in enumerate(vals[i]):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if v > 0.55 else viz.INK)
    ax.set_xticks(range(len(TX_COLS)), TX_COLS, rotation=40, ha="right",
                  fontsize=8.4)
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", length=0)
    n_tx = int(t.tx_n[t[TX_COLS].notna().any(axis=1)].sum())
    viz.finish(ax, "Treatment received",
               subtitle=f"% of patients ever given each class · MSK-CHORD, n={n_tx:,}")
    return im


def draw_tropism(trop: pd.DataFrame, counts: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 9.2))
    vals = trop.to_numpy()
    im = ax.imshow(vals, cmap="seq_blue", vmin=0, vmax=0.6, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center",
                        fontsize=7.4, color="#ffffff" if v > 0.33 else viz.INK)
    ax.set_xticks(range(trop.shape[1]), trop.columns, rotation=45, ha="right",
                  fontsize=8.6)
    ax.set_yticks(range(trop.shape[0]),
                  [f"{ct}  (n={counts[ct]:,})" for ct in trop.index], fontsize=8.6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.012)
    cb.set_label("share of that cancer type's patients", color=viz.INK_2, fontsize=9)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    cb.outline.set_visible(False)
    viz.finish(ax, "Where each cancer type spreads",
               subtitle="% of patients with a distant metastasis in each organ "
                        "(MSK-MET curated flags, ever; a patient can count in several "
                        "organs) · cells under 5% unlabelled · organs ordered by total "
                        "patients")
    fig.tight_layout()
    return fig


def main() -> None:
    viz.apply_theme()
    OUT.mkdir(exist_ok=True)
    cohort = pd.read_parquet(L.DERIVED / "cohort.parquet")
    counts = venn_counts()
    t = cancer_table(cohort)
    tx_core_n = int(t.loc[t[TX_COLS].notna().any(axis=1), "tx_n"].sum())

    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 0.95], wspace=0.04)
    draw_venn(fig.add_subplot(gs[0]), counts, len(cohort), tx_core_n)
    ax_bar = fig.add_subplot(gs[1])
    draw_bars(ax_bar, t)
    draw_tx(fig.add_subplot(gs[2]), t)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12)
    # Bars need room for cancer-type labels: nudge that axis right.
    pos = ax_bar.get_position()
    ax_bar.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])
    fig.savefig(OUT / "fig_cohort_overview.png", dpi=150, bbox_inches="tight")

    trop = tropism_table(cohort, list(t.index))
    draw_tropism(trop, t.patients).savefig(
        OUT / "fig_metastatic_spread.png", dpi=150, bbox_inches="tight")

    t.to_csv(OUT / "cancer_type_summary.csv")
    trop.to_csv(OUT / "metastatic_spread.csv")
    print(counts)
    print(t[["patients", "metastatic", "tx_n"]].head(8).to_string())


if __name__ == "__main__":
    main()
