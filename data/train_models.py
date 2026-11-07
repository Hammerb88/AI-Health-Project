"""M3 model training (modeling_plan.md v2, Sections 1-5).

Stages (run in order; each writes derived/models/<stage>_*.parquet):
  phase1   all models x F0-F3, repeats r00-r04                       (5.2 step 1)
  select   paired fold-wise selection per cohort -> selection.csv     (5.2 step 2)
  phase2   selected model + LR on F2 (with OOF SHAP), F0, baselines,
           XGBoost interaction values on r00; all 20 repeats          (5.2 step 3, 6.1)
  lc       learning curve, 25/50/75/100% of each training fold        (5.4)
  sens     sensitivity analyses incl. metastasis-/primary-only        (5.5)
  targetB  dose regression among post samples                         (4.3)
  finalize combined files + per-repeat (pooled OOF) metrics           (5.1, 5.3)

Run: ``python train_models.py all`` or ``python train_models.py phase1 select ...``
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C
import m3_common as M

OUT = M.MODELS_DIR
SHAP_DIR = OUT / "shap"
N_JOBS = 8


# --------------------------------------------------------------------------
# One job = one analysis spec x one repeat (5 outer folds)
# --------------------------------------------------------------------------

def run_job(spec: dict, repeat: str) -> dict:
    cohort, target, kind, fs, variant = spec["cohort"], spec["target"], spec["model"], spec["fs"], spec["variant"]
    frac, shap_mode, inter = spec.get("frac", 1.0), spec.get("shap"), spec.get("inter", False)
    task = "clf" if target == "A" else "reg"
    d, fw, y = M.select_rows(cohort, "main" if variant == "main_inter" else variant, target)
    folds = fw[repeat].to_numpy()
    r = int(repeat[-2:])
    key = {k: spec[k] for k in ("stage", "cohort", "target", "model", "fs", "variant")} | {"frac": frac, "repeat": repeat}
    oof, fm, params, shap_full, shap_sum, inter_rows = [], [], [], [], [], []
    for k in sorted(np.unique(folds)):
        te = folds == k
        tr_idx = np.where(~te)[0]
        # assert 3: a patient never spans training and held-out rows
        assert not set(d.PATIENT_ID.iloc[tr_idx]) & set(d.PATIENT_ID[te]), "3. patient in train and test"
        if frac < 1:   # 5.4: stratified subsample of the training fold
            rng = np.random.default_rng(M.SEED + 100 * r + k)
            tr_idx = np.concatenate([rng.choice(tr_idx[y[tr_idx] == v], int(round(frac * (y[tr_idx] == v).sum())),
                                                replace=False) for v in np.unique(y[tr_idx])]) \
                if task == "clf" else rng.choice(tr_idx, int(round(frac * len(tr_idx))), replace=False)
        dtr, dte, ytr = d.iloc[tr_idx], d[te], y[tr_idx]
        p = {}
        if kind == "prevalence":
            pred = np.full(te.sum(), ytr.mean())
        elif kind == "median":
            pred = np.full(te.sum(), np.median(ytr))
        else:
            scale = kind == "enet" and fs == "B_full"
            unpen = kind == "lr" and fs == "F2"
            fixed = task == "reg" and cohort == "nsclc"
            m, des, p = M.tune_fit(kind, task, dtr, ytr, lambda: M.make_design(cohort, fs, scale, unpen),
                                   seed=M.SEED + 10 * r + k, fixed=fixed)
            assert des.fit_ids_.isdisjoint(set(dte.SAMPLE_ID)), "2. transforms fitted on training rows only"
            if target == "A":
                assert not (des.block_ == "B").any(), "4. block B absent from every classifier"
            Xte = des.transform(dte)
            pred = M.predict(m, task, Xte)
            if shap_mode:
                Xtr = des.transform(dtr)
                sv = M.shap_values(kind, m, Xtr, Xte, task)
                agg = M.aggregate_shap(sv, des.origin_)
                blk = pd.Series(des.block_, index=des.origin_).groupby(level=0).first()
                if shap_mode == "full":
                    agg.insert(0, "SAMPLE_ID", dte.SAMPLE_ID.to_numpy())
                    agg.insert(1, "fold", k)
                    shap_full.append(agg)
                s = pd.DataFrame({"mean_abs": agg.drop(columns=["SAMPLE_ID", "fold"], errors="ignore").abs().mean(),
                                  "mean_signed": agg.drop(columns=["SAMPLE_ID", "fold"], errors="ignore").mean()})
                s["block"] = blk.reindex(s.index).to_numpy()
                # Direction for binary alterations: mean SHAP among altered held-out rows (6.5 gate).
                vals = agg.drop(columns=["SAMPLE_ID", "fold"], errors="ignore")
                alt_v = [v for v in s.index if s.at[v, "block"] == "X"]
                s["n_altered"] = pd.Series({v: int((dte[v].to_numpy() == 1).sum()) for v in alt_v})
                s["mean_altered"] = pd.Series({v: float(vals.loc[dte[v].to_numpy() == 1, v].mean())
                                               if (dte[v].to_numpy() == 1).any() else np.nan for v in alt_v})
                shap_sum.append(s.rename_axis("feature").reset_index().assign(fold=k, n_test=int(te.sum())))
            if inter:
                import xgboost as xgb
                I = m.get_booster().predict(xgb.DMatrix(Xte), pred_interactions=True)[:, :-1, :-1]
                phi = I.sum(2)
                org = des.origin_
                for site in ("biopsy_organ", "panel_version"):
                    cols = org == site
                    part = I[:, :, cols].sum(2)
                    for j in np.where(des.block_ == "X")[0]:
                        inter_rows.append({"feature": org[j], "with": site, "fold": k,
                                           "sum_abs_inter": float(np.abs(part[:, j]).sum()),
                                           "sum_abs_shap": float(np.abs(phi[:, j]).sum())})
        oof.append(pd.DataFrame({"SAMPLE_ID": dte.SAMPLE_ID.to_numpy(), "fold": k, "y": y[te], "pred": pred}))
        met = M.clf_metrics(y[te], pred) if task == "clf" else M.reg_metrics(y[te], pred)
        fm.append({**key, "fold": k, "n_train": len(tr_idx), "n_test": int(te.sum()), **met})
        params.append({**key, "fold": k, "params": json.dumps(p, default=str)})
    res = {"oof": pd.concat(oof).assign(**key), "folds": pd.DataFrame(fm), "params": pd.DataFrame(params)}
    if shap_full:
        res["shap_full"] = pd.concat(shap_full, ignore_index=True).assign(repeat=repeat)
    if shap_sum:
        res["shap_sum"] = pd.concat(shap_sum, ignore_index=True).assign(**key)
    if inter_rows:
        res["inter"] = pd.DataFrame(inter_rows).assign(**key)
    return res


def run_stage(stage: str, specs: list[dict], repeats: list[str]) -> None:
    t0 = time.time()
    specs = [s | {"stage": stage} for s in specs]
    jobs = [(s, r) for s in specs for r in (s.get("repeats") or repeats)]
    # Heaviest first for better load balance.
    weight = {"pooled": 4, "breast": 2, "crc": 2, "prostate": 1, "nsclc": 0.5}
    jobs.sort(key=lambda j: -(weight[j[0]["cohort"]] * {"rf": 3, "xgb": 2}.get(j[0]["model"], 1)))
    print(f"[{stage}] {len(specs)} specs, {len(jobs)} jobs")
    res = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(run_job)(s, r) for s, r in jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    for part in ("oof", "folds", "params", "shap_sum", "inter"):
        frames = [x[part] for x in res if part in x]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(OUT / f"{stage}_{part}.parquet", index=False)
    full = {}
    for (s, r), x in zip(jobs, res):
        if "shap_full" in x:
            full.setdefault((s["cohort"], s["model"], s["fs"], s["variant"], s["target"]), []).append(x["shap_full"])
    for (co, mo, fs, va, ta), frames in full.items():
        pd.concat(frames, ignore_index=True).to_parquet(
            SHAP_DIR / f"{stage}_{co}_{ta}_{mo}_{fs}_{va}.parquet", index=False)
    print(f"[{stage}] done in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def spec(cohort, model, fs, variant="main", target="A", **kw):
    return {"cohort": cohort, "model": model, "fs": fs, "variant": variant, "target": target, **kw}


def stage_phase1():
    specs = []
    for c in M.COHORTS:
        specs += [spec(c, "prevalence", "F0"), spec(c, "covonly", "F0")]
        specs += [spec(c, m, f) for m in ("lr", "rf", "xgb") for f in ("F0", "F1", "F2")]
        specs += [spec(c, m, "F3", "facets") for m in ("rf", "xgb")]
    run_stage("phase1", specs, M.PHASE1_REPEATS)


def select_rule(fm: pd.DataFrame, metric: str, order: list[str], margin=0.01, share=0.8):
    """5.2: start from the simplest model; move up only on a paired gain > margin in >= share of folds."""
    w = fm.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    cur, steps = order[0], []
    for cand in order[1:]:
        if cand not in w:
            continue
        diff = (w[cand] - w[cur]).dropna()
        ok = diff.mean() > margin and (diff > 0).mean() >= share
        steps.append({"from": cur, "candidate": cand, "mean_paired_gain": diff.mean(),
                      "share_folds_positive": (diff > 0).mean(), "moved": ok})
        if ok:
            cur = cand
    return cur, steps


def stage_select():
    fm = pd.read_parquet(OUT / "phase1_folds.parquet")
    rows, steps_all = [], []
    for c in M.COHORTS:
        f = fm[(fm.cohort == c) & (fm.fs == "F2") & (fm.variant == "main") & fm.model.isin(M.SIMPLICITY)]
        sel, steps = select_rule(f, "auc", M.SIMPLICITY)
        mean = f.groupby("model").auc.mean()
        rows.append({"cohort": c, "selected": sel, **{f"auc_{m}": mean.get(m) for m in M.SIMPLICITY},
                     "rationale": "; ".join(
                         f"{s['candidate']} vs {s['from']}: paired gain {s['mean_paired_gain']:+.3f}, "
                         f"positive in {100 * s['share_folds_positive']:.0f}% of folds -> "
                         f"{'moved' if s['moved'] else 'kept ' + s['from']}" for s in steps)})
        steps_all += [{"cohort": c, **s} for s in steps]
    sel = pd.DataFrame(rows)
    sel.to_csv(OUT / "selection.csv", index=False)
    pd.DataFrame(steps_all).to_csv(OUT / "selection_steps.csv", index=False)
    print(sel[["cohort", "selected", "auc_lr", "auc_rf", "auc_xgb"]].to_string(index=False))


def selected() -> dict[str, str]:
    return pd.read_csv(OUT / "selection.csv").set_index("cohort").selected.to_dict()


def stage_phase2():
    sel = selected()
    specs = []
    for c in M.COHORTS:
        models = list(dict.fromkeys([sel[c], "lr"]))
        specs += [spec(c, m, "F2", shap="full") for m in models]
        specs += [spec(c, sel[c], "F0"), spec(c, "covonly", "F0"), spec(c, "prevalence", "F0")]
        specs += [spec(c, "xgb", "F2", inter=True, repeats=["fold_r00"], model_tag="inter")]
    # The interaction run reuses model name 'xgb'; give it its own variant tag so files do not collide.
    for s in specs:
        if s.pop("model_tag", None):
            s["variant"] = "main_inter"
    run_stage("phase2", specs, M.ALL_REPEATS)


def stage_lc():
    sel = selected()
    specs = [spec(c, m, "F2", frac=f, shap="sum") for c in M.COHORTS
             for m in dict.fromkeys([sel[c], "lr"]) for f in (0.25, 0.5, 0.75, 1.0)]
    run_stage("lc", specs, M.PHASE1_REPEATS)


def stage_sens():
    sel = selected()
    specs = []
    for c in M.COHORTS:
        s = sel[c]
        tree = s if s in ("rf", "xgb") else "xgb"
        specs += [spec(c, s, "F2", v) for v in ("s3", "a_on", "strict_trained")]
        specs += [spec(c, tree, "F2", "hyper_incl")]
        for v in ("met_only", "prim_only"):
            specs += [spec(c, s, "F2", v, shap="sum"), spec(c, s, "F0", v)]
            if s != "lr":
                specs += [spec(c, "lr", "F2", v, shap="sum")]
    run_stage("sens", specs, M.PHASE1_REPEATS)


def stage_targetB():
    specs = []
    for c in C.COHORTS:
        outcomes = ["B", "B_t"] if c != "crc" else ["B"]
        for t in outcomes:
            for v in ("B_main", "B_all"):
                specs += [spec(c, "median", "B_cov", v, t), spec(c, "ridge_cov", "B_cov", v, t)]
                specs += [spec(c, m, "B_full", v, t, shap="full" if (t == "B" and v == "B_main") else None)
                          for m in ("enet", "rf", "xgb")]
    run_stage("targetB", specs, M.PHASE1_REPEATS)
    fm = pd.read_parquet(OUT / "targetB_folds.parquet")
    rows = []
    for (c, t), f in fm[fm.variant == "B_main"].groupby(["cohort", "target"]):
        sel, steps = select_rule(f[f.model.isin(["enet", "rf", "xgb"])], "spearman", ["enet", "rf", "xgb"])
        rows.append({"cohort": c, "target": t, "selected": sel,
                     "rationale": "; ".join(f"{s['candidate']} vs {s['from']}: {s['mean_paired_gain']:+.3f}, "
                                            f"{100 * s['share_folds_positive']:.0f}% folds" for s in steps)})
    pd.DataFrame(rows).to_csv(OUT / "selection_targetB.csv", index=False)


def stage_finalize():
    """Combined files named in the plan (9.1) + per-repeat metrics on pooled OOF predictions (5.1, 5.3)."""
    stages = [s for s in ("phase1", "phase2", "lc", "sens", "targetB") if (OUT / f"{s}_oof.parquet").exists()]
    oof = pd.concat([pd.read_parquet(OUT / f"{s}_oof.parquet") for s in stages], ignore_index=True)
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    pd.concat([pd.read_parquet(OUT / f"{s}_folds.parquet") for s in stages], ignore_index=True) \
        .to_parquet(OUT / "metrics_folds.parquet", index=False)
    pd.concat([pd.read_parquet(OUT / f"{s}_params.parquet") for s in stages], ignore_index=True) \
        .to_parquet(OUT / "best_params.parquet", index=False)
    strict_ids = set(pd.concat([M.load_table(c).loc[lambda t: t.main_analysis == 1, "SAMPLE_ID"]
                                for c in C.COHORTS]))
    keys = ["stage", "cohort", "target", "model", "fs", "variant", "frac", "repeat"]
    rows = []
    for k, g in oof.groupby(keys, sort=False):
        base = dict(zip(keys, k))
        if base["target"] == "A":
            rows.append({**base, "eval_rows": "all", "n": len(g), **M.clf_metrics(g.y.to_numpy(), g.pred.to_numpy(), calib=True)})
            if base["stage"] == "phase2" and base["variant"] == "main":
                s = g[g.SAMPLE_ID.isin(strict_ids)]
                rows.append({**base, "eval_rows": "strict", "n": len(s),
                             **M.clf_metrics(s.y.to_numpy(), s.pred.to_numpy(), calib=True)})
        else:
            rows.append({**base, "eval_rows": "all", "n": len(g), **M.reg_metrics(g.y.to_numpy(), g.pred.to_numpy())})
    pd.DataFrame(rows).to_parquet(OUT / "metrics_repeats.parquet", index=False)
    print("finalize: wrote oof_predictions, metrics_folds, best_params, metrics_repeats")


STAGES = {"phase1": stage_phase1, "select": stage_select, "phase2": stage_phase2, "lc": stage_lc,
          "sens": stage_sens, "targetB": stage_targetB, "finalize": stage_finalize}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    a = ap.parse_args()
    N_JOBS = a.jobs
    for st in (list(STAGES) if "all" in a.stages else a.stages):
        STAGES[st]()
