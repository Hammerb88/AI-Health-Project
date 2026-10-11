"""Timing between clinical events, companion to fig_event_order.py.

Panel A  one row per main event order (the icicle's paths, same order and
         colours). Each event sits at its median time relative to sample
         collection (day 0), with a bar spanning the interquartile range.
Panel B  distribution of each consecutive interval (Dx -> Met, Met -> Sample,
         ...), pooled over all patients whose history contains that step, on a
         log time axis. Labels give the median and the share within 30 days.

Run: ``python fig_event_timing.py`` -> reports/fig_event_timing.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import viz
from fig_event_order import COLOR, EVENT_NAMES, EVENTS, OUT, build_event_table

MIN_GROUP_SHARE = 0.015        # order groups shown in panel A
YEAR = 365.25


def fmt_days(d: float) -> str:
    if d < 1:
        return "same day"
    if d < 60:
        return f"{d:.0f} d"
    if d < 2 * YEAR:
        return f"{d / 30.44:.0f} mo"
    return f"{d / YEAR:.1f} y"


def panel_timeline(ax, o: pd.DataFrame) -> None:
    o = o.assign(order=o.path.apply(" → ".join))
    sizes = o.order.value_counts()
    groups = sizes[sizes >= MIN_GROUP_SHARE * len(o)].index
    rel = o[EVENTS].sub(o.Sample, axis=0) / YEAR      # years from collection

    for i, g in enumerate(groups):
        sub = rel[o.order == g]
        evs = g.split(" → ")
        meds = [sub[e].median() for e in evs]
        ax.plot(meds, [i] * len(meds), color=viz.AXIS, lw=1.2, zorder=1)
        for k, e in enumerate(evs):
            q1, q3 = sub[e].quantile([0.25, 0.75])
            off = (k - (len(evs) - 1) / 2) * 0.13   # separate overlapping IQRs
            ax.plot([q1, q3], [i + off, i + off], color=COLOR[e], lw=2.2,
                    alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot(meds[k], i + off, "o", ms=8.5, color=COLOR[e],
                    markeredgecolor=viz.SURFACE, markeredgewidth=1.6, zorder=3)

    ax.set_yticks(range(len(groups)),
                  [f"{g}   (n={sizes[g]:,})" for g in groups], fontsize=8.6)
    ax.set_ylim(len(groups) - 0.5, -0.6)
    ax.axvline(0, color=COLOR["Sample"], lw=1.2, zorder=0)
    ax.text(0.004, -0.55, "sample collected", color=viz.INK_2, fontsize=8.4,
            va="center")
    # Most events cluster within weeks of collection while some precede it by
    # years: symlog keeps ±1 month linear and compresses the long tails.
    ax.set_xscale("symlog", linthresh=30.44 / YEAR, linscale=1.2)
    ticks = {-5: "−5 y", -1: "−1 y", -3 * 30.44 / YEAR: "−3 mo",
             -30.44 / YEAR: "−1 mo", 0: "0", 30.44 / YEAR: "1 mo",
             3 * 30.44 / YEAR: "3 mo", 1: "1 y"}
    ax.set_xticks(list(ticks), list(ticks.values()), fontsize=8.4)
    ax.minorticks_off()
    ax.set_xlim(-7, 3)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    viz.finish(ax, "A · When each event happens, by event order",
               subtitle="dot = median, bar = middle 50% of patients · rows match "
                        "the event-order icicle",
               xlabel="time relative to sample collection (linear within ±1 month, "
                      "log beyond)")


def panel_intervals(ax, o: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in o.iterrows():
        p = r.path
        for a, b in zip(p[:-1], p[1:]):
            rows.append((f"{a} → {b}", a, b, r[b] - r[a]))
    iv = pd.DataFrame(rows, columns=["step", "a", "b", "days"])
    stats = (iv.groupby("step").days
             .agg(n="size", median="median",
                  within30=lambda d: (d <= 30).mean()))
    stats = stats[stats.n >= 100].sort_values("median")
    iv = iv[iv.step.isin(stats.index)]

    data = [np.log10(iv.loc[iv.step == s, "days"].clip(lower=1)) for s in stats.index]
    first = {s: s.split(" → ")[1] for s in stats.index}
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False,
                    patch_artist=True, whis=(5, 95),
                    medianprops=dict(color=viz.SURFACE, lw=2),
                    whiskerprops=dict(color=viz.AXIS, lw=1),
                    capprops=dict(color=viz.AXIS, lw=1))
    for patch, s in zip(bp["boxes"], stats.index):
        patch.set_facecolor(COLOR[first[s]])
        patch.set_linewidth(0)
    ax.set_yticks(range(1, len(stats) + 1), stats.index, fontsize=8.8)
    ax.set_ylim(len(stats) + 0.6, 0.4)
    ticks = {1: "≤1 d", 7: "1 wk", 30.44: "1 mo", 182.6: "6 mo", YEAR: "1 y",
             2 * YEAR: "2 y", 5 * YEAR: "5 y", 10 * YEAR: "10 y"}
    ax.set_xticks(np.log10(list(ticks)), list(ticks.values()), fontsize=8.6)
    ax.set_xlim(-0.05, np.log10(16 * YEAR))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    # Stats sit in a column outside the plot so they never cover a box.
    lab = ax.get_yaxis_transform()
    ax.text(1.02, 0.3, "median   ≤30 d      n", transform=lab, fontsize=8,
            color=viz.MUTED, va="center", ha="left")
    for i, (s, st) in enumerate(stats.iterrows(), start=1):
        ax.text(1.02, i, f"{fmt_days(st['median']):<9}{st.within30:>5.0%}"
                f"   {int(st.n):>6,}", transform=lab, va="center", ha="left",
                fontsize=8.2, color=viz.INK_2, family="monospace")
    viz.finish(ax, "B · Time between consecutive events",
               subtitle="box = middle 50%, whiskers = 5th–95th percentile, coloured "
                        "by the later event · log scale",
               xlabel="time from first to second event")
    return stats


def main() -> None:
    viz.apply_theme()
    OUT.mkdir(exist_ok=True)
    o = build_event_table()

    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.8),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    panel_timeline(a, o)
    stats = panel_intervals(b, o)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=COLOR[e],
                      label=EVENT_NAMES[e]) for e in EVENTS]
    fig.legend(handles=handles, loc="lower left", ncol=4, fontsize=8.8,
               bbox_to_anchor=(0.01, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=3)
    fig.savefig(OUT / "fig_event_timing.png", dpi=150, bbox_inches="tight")

    stats.assign(median_days=stats["median"].round(0)).drop(columns="median") \
        .to_csv(OUT / "event_interval_summary.csv")
    print(stats.to_string())


if __name__ == "__main__":
    main()
