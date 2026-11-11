# Running the modeling analysis

This runs the pre- vs post-treatment classifiers described in `../modeling_plan.md`
and produces the figures and tables for the report.

## 1. Setup (once)

You need Python 3.11 or newer.

```bash
python -m pip install "pandas>=2.2" numpy pyarrow scipy scikit-learn xgboost shap statsmodels joblib matplotlib pyyaml
```

You do **not** need the raw cBioPortal data. The modeling scripts read only the
prepared tables in `data/derived/`, which are in the repository.

Run every command below from the `data/` folder:

```bash
cd data
```

## 2. Run the analysis (in this order)

| Step | Command | What it does |
|---|---|---|
| 1 | `python train_models.py all` | Trains and cross-validates all models. This is the slow step. |
| 2 | `python explain.py` | Computes SHAP feature importance and the control check. |
| 3 | `python confirm_strict.py` | Runs the Firth regression confirmation on the strict-tier samples. |
| 4 | `python report_models.py` | Makes the figures and tables. |

Each step needs the one before it to have finished.

## 3. How long it takes

Step 1 takes many hours. On an 8-core laptop, the first stage alone took about
7 hours, so plan to run it overnight. Steps 2 to 4 take minutes.

Step 1 uses 8 CPU cores by default and will make the computer slow. To leave
room for other work, use fewer cores:

```bash
python train_models.py all --jobs 4
```

Step 1 is made of stages. Every finished job is saved as it completes. If a run
is interrupted, rerun the same command and it picks up where it stopped. You can
run stages one at a time, in this order:

```bash
python train_models.py phase1 select
python train_models.py phase2
python train_models.py lc
python train_models.py sens
python train_models.py targetB
python train_models.py finalize
```

Target B is optional for the report. Without it, `report_models.py` skips the
Target B figure and table. To produce Target A results first and add Target B later:

```bash
python train_models.py phase1 select phase2 lc sens finalize
python explain.py && python confirm_strict.py && python report_models.py
# later:
python train_models.py targetB finalize
python report_models.py
```

To keep a log of a long run:

```bash
python train_models.py all > derived/logs/train_models.txt 2>&1
```

## 4. Where the results go

| Folder | Contents |
|---|---|
| `data/reports/models/` | Figures M1 to M12, tables MT1 to MT7, and a README describing each one. |
| `data/derived/models/` | Intermediate results: predictions, metrics, SHAP values, Firth estimates. |

## 5. If something fails

- **A line starting with `AssertionError`.** A built-in data check failed. The
  message names the check. Do not skip it, because the results would not be valid.
- **`report_models.py` refuses to run.** The control check in `explain.py` failed,
  so the figures would be misleading. Look at `derived/models/control_gate.csv`.
- **`ModuleNotFoundError`.** A package is missing. Rerun the install command in step 1.
