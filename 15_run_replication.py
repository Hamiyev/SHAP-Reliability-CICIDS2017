import argparse
import datetime
import gc
import hashlib
import itertools
import json
import threading
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import shap
import sklearn
import yaml
from lightgbm import LGBMClassifier, record_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*eval_set.*")

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=["base", "shap", "topk", "collect", "all"])
ap.add_argument("--splits", default="all", help="masalan: 42 yoki 42,123")
ap.add_argument("--smoke", action="store_true", help="kichik namunada kodni sinash")
args = ap.parse_args()
SMOKE = args.smoke


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


PROTO = Path("configs/experiment_protocol_v1.1.yaml")
_raw = Path("configs/protocol_v1.1.sha256").read_bytes().decode("latin-1")
EXPECTED_SHA = "".join(ch for ch in _raw if ch in "0123456789abcdefABCDEF").upper()
PROTO_SHA = sha256_of(PROTO)
if PROTO_SHA != EXPECTED_SHA:
    raise SystemExit("PROTOKOL O'ZGARGAN: experiment_protocol_v1.1.yaml hashi mos emas")
with open(PROTO, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PARQUET = Path(cfg["data"]["parquet"])
AUDIT = Path("results/data_audit_corrected")
SPLITS = Path("results/splits_corrected")
OUT = Path("results/run_000_smoke_v11") if SMOKE else Path(cfg["runs"]["main"])
DIRS = {k: OUT / k for k in ["models", "predictions", "metrics", "tuning", "shap", "feature_sets", "status"]}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)
LOG_PATH = Path("reports/experiment_log_v11_smoke.csv" if SMOKE else "reports/experiment_log_v11.csv")

N_CLASSES = cfg["data"]["n_classes"]
N_JOBS = cfg["resources"]["cpu_jobs"]
SEEDS = cfg["splits"]["seeds"] if args.splits == "all" else [int(x) for x in args.splits.split(",")]
MON_ROWS, SHAP_BATCH = 200000, 400
SHAP_CAP = 20 if SMOKE else 300
BASE_TASKS = [("rf_all77", "rf"), ("xgb_all77", "xgb"), ("lgbm_all77", "lgbm")]
GRID_KEY = {"rf": "random_forest", "xgb": "xgboost", "lgbm": "lightgbm"}
PARENT = "rf_all77"
TOPK_TASKS = [("rf_top5", 5), ("rf_top10", 10), ("rf_top20", 20), ("rf_top40", 40)]

with open(cfg["data"]["feature_names_source"], encoding="utf-8") as f:
    FEAT = json.load(f)["feature_names"]
assert len(FEAT) == cfg["data"]["feature_count_expected"], len(FEAT)
with open(cfg["data"]["label_map"], encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ID2NAME = {v: k for k, v in LABEL_MAP.items()}
assert len(LABEL_MAP) == N_CLASSES
ULTRA = [LABEL_MAP[n] for n in cfg["evaluation"]["ultra_rare_classes"]]
NONRARE = [c for c in range(N_CLASSES) if c not in ULTRA]
print(f"Protokol {cfg['project']['protocol_version']} (hash mos), chiqish: {OUT}, smoke={SMOKE}, "
      f"belgilar={len(FEAT)}, sinflar={N_CLASSES}", flush=True)


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


ROW_ID = Y_ALL = X_ALL = None
_cache = {}


def load_all():
    global ROW_ID, Y_ALL, X_ALL
    if sha256_of(PARQUET) != cfg["data"]["parquet_sha256"]:
        raise SystemExit("MA'LUMOT O'ZGARGAN: Parquet hashi protokoldagi bilan mos emas")
    df = pd.read_parquet(PARQUET, columns=["row_id", "label_id"] + FEAT)
    ROW_ID = df["row_id"].to_numpy(dtype="int64")
    Y_ALL = df["label_id"].to_numpy().astype("int64")
    X_ALL = df[FEAT].to_numpy(dtype="float32")
    assert len(df) == cfg["data"]["n_rows"] and bool(np.all(np.diff(ROW_ID) > 0))
    del df
    gc.collect()
    print(f"Ma'lumot yuklandi: {X_ALL.shape}", flush=True)


def smoke_subsample(pos, seed, frac=0.04, minimum=20):
    rng = np.random.default_rng(seed)
    lab = Y_ALL[pos]
    chosen = []
    for c in range(N_CLASSES):
        cls = pos[lab == c]
        k = min(len(cls), max(int(round(frac * len(cls))), minimum))
        chosen.append(rng.choice(cls, size=k, replace=False))
    return np.sort(np.concatenate(chosen))


def get_data(seed):
    if seed in _cache:
        return _cache[seed]
    _cache.clear()
    gc.collect()
    d = {}
    for part, key in (("train", "tr"), ("validation", "va"), ("test", "te")):
        ids = pd.read_parquet(SPLITS / f"split_{seed}_{part}.parquet")["row_id"].to_numpy()
        pos = np.searchsorted(ROW_ID, ids)
        assert np.array_equal(ROW_ID[pos], ids)
        if SMOKE:
            pos = smoke_subsample(pos, seed)
        d["X" + key], d["y" + key], d["ids_" + key] = X_ALL[pos], Y_ALL[pos], ROW_ID[pos]
    _cache[seed] = d
    return d


def expand(grid):
    keys = [k for k, v in grid.items() if isinstance(v, list)]
    fixed = {k: v for k, v in grid.items() if not isinstance(v, list)}
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        p.update(fixed)
        out.append(p)
    return out


def build(kind, params, seed):
    n_est = cfg["tuning"]["n_estimators_fixed"]
    if kind == "rf":
        return RandomForestClassifier(n_estimators=n_est, max_features=params.get("max_features", "sqrt"),
                                      max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                      n_jobs=N_JOBS, random_state=seed)
    if kind == "xgb":
        return XGBClassifier(n_estimators=n_est, max_depth=params["max_depth"],
                             learning_rate=params["learning_rate"], tree_method="hist",
                             eval_metric="mlogloss", n_jobs=N_JOBS, random_state=seed)
    fx = cfg["model_fixed_params"]["lightgbm"]
    return LGBMClassifier(n_estimators=n_est, num_leaves=params["num_leaves"],
                          learning_rate=params["learning_rate"], reg_lambda=fx["reg_lambda"],
                          min_child_weight=fx["min_child_weight"], n_jobs=N_JOBS,
                          random_state=seed, verbose=-1)


def fit_monitored(kind, m, X, y, Xmon, ymon):
    if kind == "rf":
        m.fit(X, y)
        return False
    if kind == "xgb":
        m.fit(X, y, eval_set=[(Xmon, ymon)], verbose=False)
        curve = m.evals_result()["validation_0"]["mlogloss"]
    else:
        ev = {}
        m.fit(X, y, eval_set=[(Xmon, ymon)], eval_names=["mon"], callbacks=[record_evaluation(ev)])
        curve = ev["mon"]["multi_logloss"]
    return bool((not np.isfinite(curve[-1])) or curve[-1] > curve[9])


def select(rows):
    valid = sorted([r for r in rows if not r["invalid_flag"]], key=lambda r: -r["val_macro_f1"])
    if not valid:
        raise RuntimeError("barcha konfiguratsiyalar yaroqsiz deb belgilandi")
    best = valid[0]
    if (len(valid) > 1 and best["val_macro_f1"] - valid[1]["val_macro_f1"] < 0.001
            and valid[1]["fit_seconds"] < best["fit_seconds"]):
        best = valid[1]
    return best["config_index"]


def evaluate(y_true, y_pred):
    labels = list(range(N_CLASSES))
    assert len(y_true) == len(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    assert int(cm.sum()) == len(y_true)
    macro = float(f.mean())
    assert abs(macro - f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) < 1e-9
    m = {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": macro,
         "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
         "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
         "macro_f1_nonrare": float(f[NONRARE].mean())}
    assert all(np.isfinite(v) for v in m.values())
    per_class = [{"label_id": c, "label": ID2NAME[c], "support": int(s[c]), "precision": float(p[c]),
                  "recall": float(r[c]), "f1": float(f[c])} for c in labels]
    return m, per_class, cm.tolist()


def write_results(model_id, seed, d, y_pred, extra):
    exp_id = f"{model_id}_split{seed}"
    pd.DataFrame({"row_id": d["ids_te"], "y_true": d["yte"], "y_pred": y_pred}).to_parquet(
        DIRS["predictions"] / f"{exp_id}_test.parquet", index=False)
    m, per_class, cm = evaluate(d["yte"], y_pred)
    record = {"experiment_id": exp_id, "split_seed": seed, "model_name": model_id,
              "feature_count": extra["feature_count"], "class_weight": None, **m,
              "val_macro_f1": extra["val_macro_f1"], "train_seconds": extra["train_seconds"],
              "inference_seconds": extra["inference_seconds"], "model_size_bytes": extra["model_size_bytes"],
              "per_class": per_class, "confusion_matrix": cm,
              "protocol_version": cfg["project"]["protocol_version"], "protocol_sha256": PROTO_SHA,
              "test_used_for_selection": False, "status": "success"}
    with open(DIRS["metrics"] / f"{exp_id}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    meta = {"model_name": model_id, "experiment_id": exp_id, "split_seed": seed, "params": extra["params"],
            "train_rows": int(len(d["ytr"])), "val_rows": int(len(d["yva"])), "test_rows": int(len(d["yte"])),
            "features_used": extra["features"], "feature_count": extra["feature_count"],
            "train_seconds": extra["train_seconds"],
            "library_versions": {"scikit_learn": sklearn.__version__, "shap": shap.__version__,
                                 "numpy": np.__version__, "pandas": pd.__version__},
            "model_file_path": extra["model_path"], "protocol_version": cfg["project"]["protocol_version"],
            "protocol_sha256": PROTO_SHA, "status": "SUCCESS"}
    with open(DIRS["models"] / f"{exp_id}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"    TEST Macro-F1 ({N_CLASSES} sinf) = {m['macro_f1']:.4f} | "
          f"{len(NONRARE)} sinf = {m['macro_f1_nonrare']:.4f}", flush=True)


def base_task(model_id, kind, seed):
    d = get_data(seed)
    Xtr, ytr, Xva, yva, Xte = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xte"]
    rng = np.random.default_rng(seed)
    mon = rng.choice(len(ytr), size=min(MON_ROWS, len(ytr)), replace=False)
    Xmon, ymon = Xtr[mon], ytr[mon]
    grid = expand(cfg["tuning"]["grids"][GRID_KEY[kind]])
    rows, models = [], []
    for i, params in enumerate(grid):
        m = build(kind, params, seed)
        gc.collect()
        t = time.perf_counter()
        invalid = fit_monitored(kind, m, Xtr, ytr, Xmon, ymon)
        fit_s = time.perf_counter() - t
        vf1 = float(f1_score(yva, m.predict(Xva), labels=list(range(N_CLASSES)), average="macro",
                             zero_division=0))
        rows.append({"model": model_id, "split_seed": seed, "config_index": i, "params": json.dumps(params),
                     "val_macro_f1": round(vf1, 6), "fit_seconds": round(fit_s, 1), "invalid_flag": invalid})
        models.append(m)
        flag = " [YAROQSIZ]" if invalid else ""
        print(f"    konfiguratsiya {i + 1}/{len(grid)} {params}: val Macro-F1={vf1:.4f}, {fit_s:.0f}s{flag}",
              flush=True)
    sel = select(rows)
    for r in rows:
        r["selected"] = r["config_index"] == sel
    pd.DataFrame(rows).to_csv(DIRS["tuning"] / f"{model_id}_split{seed}_tuning.csv", index=False)
    best, best_row = models[sel], rows[sel]
    del models
    gc.collect()
    with open(DIRS["tuning"] / f"{model_id}_split{seed}_selected.json", "w", encoding="utf-8") as f:
        json.dump({"params": grid[sel], "val_macro_f1": best_row["val_macro_f1"],
                   "fit_seconds": best_row["fit_seconds"]}, f, indent=2)
    t = time.perf_counter()
    prob = best.predict_proba(Xte)
    inf_s = time.perf_counter() - t
    assert list(best.classes_) == list(range(N_CLASSES))
    mp = DIRS["models"] / f"{model_id}_split{seed}.joblib"
    joblib.dump(best, mp)
    write_results(model_id, seed, d, prob.argmax(axis=1), {
        "feature_count": len(FEAT), "val_macro_f1": best_row["val_macro_f1"],
        "train_seconds": best_row["fit_seconds"], "inference_seconds": round(inf_s, 2),
        "model_size_bytes": mp.stat().st_size, "params": grid[sel], "features": FEAT, "model_path": str(mp)})
    return f"tanlangan konfiguratsiya {sel}: {grid[sel]}"


def shap_task(seed):
    d = get_data(seed)
    model = joblib.load(DIRS["models"] / f"{PARENT}_split{seed}.joblib")
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(N_CLASSES):
        cls = np.where(d["ytr"] == c)[0]
        idx.append(rng.choice(cls, size=min(len(cls), SHAP_CAP), replace=False))
    idx = np.sort(np.concatenate(idx))
    Xs = d["Xtr"][idx]
    print(f"    SHAP namuna: {len(idx):,} qator", flush=True)
    explainer = shap.TreeExplainer(model)
    chk = np.random.default_rng(seed).choice(len(Xs), size=min(20, len(Xs)), replace=False)
    sv_chk = np.asarray(explainer.shap_values(Xs[chk], check_additivity=False))
    recon = np.asarray(explainer.expected_value)[None, :] + sv_chk.sum(axis=1)
    max_diff = float(np.abs(recon - model.predict_proba(Xs[chk])).max())
    if max_diff > 0.01:
        raise RuntimeError(f"SHAP additivlik tekshiruvi o'tmadi: eng katta farq {max_diff:.4f}")
    sv_all = np.zeros((len(Xs), len(FEAT), N_CLASSES), dtype="float32")
    t0 = time.time()
    for s in range(0, len(Xs), SHAP_BATCH):
        sv_all[s:s + SHAP_BATCH] = np.asarray(
            explainer.shap_values(Xs[s:s + SHAP_BATCH], check_additivity=False), dtype="float32")
        print(f"    SHAP {min(s + SHAP_BATCH, len(Xs))}/{len(Xs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    abs_mean = np.abs(sv_all).mean(axis=0)
    imp = pd.DataFrame({"feature": FEAT, "mean_abs_shap": abs_mean.mean(axis=1)})
    for c in range(N_CLASSES):
        imp[f"abs_{ID2NAME[c]}"] = abs_mean[:, c]
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    imp = imp.sort_values("rank")
    imp.insert(0, "model_id", PARENT)
    imp.to_csv(DIRS["shap"] / f"shap_importance_rf_split{seed}.csv", index=False)
    np.save(DIRS["shap"] / f"shap_values_rf_split{seed}.npy", sv_all)
    np.save(DIRS["shap"] / f"shap_sample_row_ids_split{seed}.npy", d["ids_tr"][idx])
    for k in cfg["feature_selection"]["feature_counts"]:
        top = imp.head(k)
        with open(DIRS["feature_sets"] / f"rf_split{seed}_top{k}.json", "w", encoding="utf-8") as f:
            json.dump({"model_id": PARENT, "split_seed": seed, "k": k, "features": top["feature"].tolist(),
                       "importance": [float(v) for v in top["mean_abs_shap"]], "rank": top["rank"].tolist()},
                      f, indent=2)
    return f"SHAP namuna {len(idx)} qator, additivlik farqi {max_diff:.2e}"


def topk_task(model_id, k, seed):
    d = get_data(seed)
    ranking = pd.read_csv(DIRS["shap"] / f"shap_importance_rf_split{seed}.csv").sort_values("rank")
    feats_k = ranking["feature"].tolist()[:k]
    col_idx = [FEAT.index(f) for f in feats_k]
    with open(DIRS["tuning"] / f"{PARENT}_split{seed}_selected.json", encoding="utf-8") as f:
        params = json.load(f)["params"]
    model = build("rf", params, seed)
    Xtr, Xva, Xte = d["Xtr"][:, col_idx], d["Xva"][:, col_idx], d["Xte"][:, col_idx]
    t = time.perf_counter()
    model.fit(Xtr, d["ytr"])
    fit_s = time.perf_counter() - t
    vf1 = float(f1_score(d["yva"], model.predict(Xva), labels=list(range(N_CLASSES)), average="macro",
                         zero_division=0))
    t = time.perf_counter()
    prob = model.predict_proba(Xte)
    inf_s = time.perf_counter() - t
    mp = DIRS["models"] / f"{model_id}_split{seed}.joblib"
    joblib.dump(model, mp)
    write_results(model_id, seed, d, prob.argmax(axis=1), {
        "feature_count": k, "val_macro_f1": round(vf1, 6), "train_seconds": round(fit_s, 1),
        "inference_seconds": round(inf_s, 2), "model_size_bytes": mp.stat().st_size, "params": params,
        "features": feats_k, "model_path": str(mp)})
    return f"{model_id}: {k} belgi"


def collect():
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(DIRS["metrics"].glob("*_metrics.json"))]
    cols = ["experiment_id", "split_seed", "model_name", "feature_count", "accuracy", "macro_f1", "weighted_f1",
            "balanced_accuracy", "macro_f1_nonrare", "val_macro_f1", "train_seconds", "inference_seconds",
            "model_size_bytes", "status"]
    pd.DataFrame([{k: r[k] for k in cols} for r in recs]).to_csv(OUT / "all_experiments.csv", index=False)
    pc = [{"experiment_id": r["experiment_id"], "split_seed": r["split_seed"], "model_name": r["model_name"], **c}
          for r in recs for c in r["per_class"]]
    pd.DataFrame(pc).to_csv(OUT / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    done = {(r["model_name"], r["split_seed"]) for r in recs}
    want = [(m["id"], s) for m in cfg["models"] for s in cfg["splits"]["seeds"]]
    missing = [w for w in want if w not in done]
    print(f"\nYig'ildi: {len(recs)} natija; protokol bo'yicha kutilgan {len(want)}; yetishmayotgani {len(missing)}")
    if missing and len(missing) <= 20:
        print("  yetishmayotganlar:", missing)


def log_row(task, seed, status, dur, peak, notes):
    row = {"date_time": datetime.datetime.now().isoformat(timespec="seconds"), "stage": task.split("_")[0],
           "experiment_id": f"{task}_split{seed}", "script": "src/15_run_replication.py",
           "config_version": cfg["project"]["protocol_version"], "status": status,
           "duration_seconds": round(dur, 1), "peak_ram_gb": round(peak / 1024 ** 3, 2),
           "output_files": str(OUT), "notes": str(notes)[:300].replace("\n", " ")}
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def run_task(task, seed, fn):
    mk = DIRS["status"] / f"{task}_{seed}.json"
    if mk.exists() and json.load(open(mk, encoding="utf-8"))["status"] == "success":
        print(f"  [o'tkazib yuborildi] {task} seed {seed}", flush=True)
        return
    print(f"\n>>> {task} | seed {seed}", flush=True)
    t0 = time.time()
    with PeakRAM() as mon:
        try:
            notes, status = fn() or "", "success"
        except Exception:
            notes, status = traceback.format_exc(), "failed"
    dur = time.time() - t0
    with open(mk, "w", encoding="utf-8") as f:
        json.dump({"task": task, "seed": seed, "status": status, "duration_seconds": round(dur, 1),
                   "peak_ram_gb": round(mon.peak / 1024 ** 3, 2), "notes": str(notes)[-2000:]}, f, indent=2)
    log_row(task, seed, status, dur, mon.peak, notes)
    print(f"<<< {task} | seed {seed}: {status} ({dur / 60:.1f} min)", flush=True)
    if status == "failed":
        print(notes)
        raise SystemExit(1)


if args.stage == "collect":
    collect()
    raise SystemExit(0)

load_all()
stages = ["base", "shap", "topk"] if args.stage == "all" else [args.stage]
for stage in stages:
    for seed in SEEDS:
        if stage == "base":
            for mid, kind in BASE_TASKS:
                run_task(f"base_{mid}", seed, lambda mid=mid, kind=kind, seed=seed: base_task(mid, kind, seed))
        elif stage == "shap":
            run_task("shap_rf", seed, lambda seed=seed: shap_task(seed))
        elif stage == "topk":
            for mid, k in TOPK_TASKS:
                run_task(f"topk_{mid}", seed, lambda mid=mid, k=k, seed=seed: topk_task(mid, k, seed))
collect()
print("\n[Bajarildi] tanlangan bosqich(lar) yakunlandi")