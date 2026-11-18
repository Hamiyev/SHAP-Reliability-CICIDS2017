import datetime
import gc
import json
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import shap
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

PARQUET = Path("data/processed/cicids2017_clean.parquet")
AUDIT = Path("results/data_audit")
SPLITS = Path("results/splits")
OUT = Path("results/run_001_pilot")
OUT.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)

PILOT_SEED, SPLIT_SEED = 1, 42
FRACS, VAL_FRAC, MIN_PER_CLASS, N_JOBS = [0.05, 0.10], 0.10, 20, 4
T_START = time.time()


class PeakRAM:
    def __enter__(self):
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.2)

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


with open(AUDIT / "clean_data_summary.json", encoding="utf-8") as f:
    feat = json.load(f)["feature_names"]
assert len(feat) == 69
n_classes = 15

print("Pilot: ma'lumot o'qilmoqda", flush=True)
df = pd.read_parquet(PARQUET, columns=["row_id", "label_id"] + feat).set_index("row_id")
train_ids = pd.read_parquet(SPLITS / f"split_{SPLIT_SEED}_train.parquet")["row_id"].to_numpy()
val_ids = pd.read_parquet(SPLITS / f"split_{SPLIT_SEED}_validation.parquet")["row_id"].to_numpy()


def stratified_sample(part_ids, frac, seed):
    lab = df["label_id"].loc[part_ids].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(n_classes):
        cls = part_ids[lab == c]
        k = min(len(cls), max(int(round(frac * len(cls))), MIN_PER_CLASS))
        chosen.append(rng.choice(cls, size=k, replace=False))
    return np.sort(np.concatenate(chosen))


def take(ids):
    sub = df.loc[ids]
    return sub[feat].to_numpy(dtype="float32"), sub["label_id"].to_numpy()


train_data = {fr: take(stratified_sample(train_ids, fr, PILOT_SEED)) for fr in FRACS}
Xva, yva = take(stratified_sample(val_ids, VAL_FRAC, PILOT_SEED))
full_train_rows = len(train_ids)
del df
gc.collect()
for fr in FRACS:
    print(f"  train namuna {fr:.0%}: {len(train_data[fr][1]):,} qator; sinflar: "
          f"{len(np.unique(train_data[fr][1]))}", flush=True)
print(f"  validation namuna: {len(yva):,} qator", flush=True)

MODELS = {
    "rf": lambda: RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=PILOT_SEED),
    "xgboost": lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     tree_method="hist", n_jobs=N_JOBS, random_state=PILOT_SEED),
    "lightgbm": lambda: LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                                       n_jobs=N_JOBS, random_state=PILOT_SEED, verbose=-1),
}

results, rf_big = [], None
for fr in FRACS:
    Xtr, ytr = train_data[fr]
    for name, make in MODELS.items():
        model = make()
        gc.collect()
        with PeakRAM() as mon:
            t = time.perf_counter()
            model.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            pred = model.predict(Xva)
            pred_s = time.perf_counter() - t
        tmp = OUT / "_tmp_model.joblib"
        joblib.dump(model, tmp)
        size_mb = tmp.stat().st_size / 1e6
        tmp.unlink()
        row = {"model": name, "train_frac": fr, "train_rows": len(ytr), "fit_seconds": round(fit_s, 2),
               "predict_seconds": round(pred_s, 2), "val_rows": len(yva),
               "peak_rss_gb": round(mon.peak / 1024 ** 3, 2), "model_size_mb": round(size_mb, 1),
               "sanity_macro_f1_val_sample": round(float(f1_score(yva, pred, average="macro")), 4)}
        results.append(row)
        print(f"  {name:9s} {fr:.0%}: fit={fit_s:7.1f}s predict={pred_s:5.1f}s "
              f"RAM={row['peak_rss_gb']}GB size={row['model_size_mb']}MB", flush=True)
        if name == "rf" and fr == FRACS[-1]:
            rf_big = model
        else:
            del model
        gc.collect()

# SHAP vaqti (RF, kichik namuna)
print("SHAP vaqti o'lchanmoqda", flush=True)
Xbig = train_data[FRACS[-1]][0]
rng = np.random.default_rng(PILOT_SEED)
Xs = Xbig[rng.choice(len(Xbig), 100, replace=False)]
explainer = shap.TreeExplainer(rf_big)
shap_rows = []
for n in (50, 100):
    with PeakRAM() as mon:
        t = time.perf_counter()
        sv = explainer.shap_values(Xs[:n], check_additivity=False)
        dt = time.perf_counter() - t
    shap_rows.append({"rows_explained": n, "seconds": round(dt, 1),
                      "seconds_per_row": round(dt / n, 3), "output_shape": str(np.shape(sv)),
                      "peak_rss_gb": round(mon.peak / 1024 ** 3, 2)})
    print(f"  SHAP {n} qator: {dt:.1f}s, shakl {np.shape(sv)}, RAM {shap_rows[-1]['peak_rss_gb']}GB",
          flush=True)

# Proyeksiya: bitta o'qitish to'liq train qismida
res = pd.DataFrame(results)
proj = {}
for name in MODELS:
    a = res[(res.model == name) & (res.train_frac == FRACS[0])].iloc[0]
    b = res[(res.model == name) & (res.train_frac == FRACS[1])].iloc[0]
    expo = float(np.log(b.fit_seconds / a.fit_seconds) / np.log(b.train_rows / a.train_rows))
    p = float(b.fit_seconds * (full_train_rows / b.train_rows) ** expo)
    proj[name] = {"scaling_exponent": round(expo, 2), "projected_full_train_fit_minutes": round(p / 60, 1)}

res.to_csv(OUT / "pilot_resources.csv", index=False)
pd.DataFrame(shap_rows).to_csv(OUT / "pilot_shap_timing.csv", index=False)
summary = {
    "purpose": "technical pilot only; not a scientific result",
    "split_seed": SPLIT_SEED, "pilot_seed": PILOT_SEED, "train_fractions": FRACS,
    "val_fraction": VAL_FRAC, "min_per_class": MIN_PER_CLASS, "n_jobs": N_JOBS,
    "full_train_rows": int(full_train_rows), "projections": proj,
    "shap_seconds_per_row": shap_rows[-1]["seconds_per_row"],
    "projected_shap_minutes_for_10000_rows": round(shap_rows[-1]["seconds_per_row"] * 10000 / 60, 1),
    "versions": {"scikit_learn": sklearn.__version__, "shap": shap.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__},
    "elapsed_seconds": round(time.time() - T_START, 1),
}
with open(OUT / "pilot_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

log_path = Path("reports/experiment_log.csv")
row = {"date_time": datetime.datetime.now().isoformat(timespec="seconds"), "stage": "pilot",
       "experiment_id": "run_001_pilot", "script": "src/04_pilot.py", "config_version": "0.9-draft",
       "status": "success", "duration_seconds": round(time.time() - T_START, 1),
       "peak_ram_gb": float(res["peak_rss_gb"].max()),
       "output_files": "results/run_001_pilot/*", "notes": "technical pilot, not a result"}
pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

print("\n===== PILOT XULOSASI =====")
print(res.to_string(index=False))
print("\nSHAP:")
print(pd.DataFrame(shap_rows).to_string(index=False))
print("\nTo'liq train qismida (%s qator) bitta o'qitish, taxminiy daqiqa:" % f"{full_train_rows:,}")
for k, v in proj.items():
    print(f"  {k}: {v['projected_full_train_fit_minutes']} min (daraja {v['scaling_exponent']})")
print(f"SHAP: 10 000 qator uchun taxminan {summary['projected_shap_minutes_for_10000_rows']} min")
print(f"Vaqt: {time.time() - T_START:.1f} s")