import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUT = Path("results/run_002_main")
SEEDS = [42, 123, 2024, 3407, 7777]
FIX = "--fix" in sys.argv
D = {k: OUT / k for k in ["models", "predictions", "metrics", "tuning", "shap",
                          "feature_sets", "status", "probs"]}
BASE_IDS = ["rf_all69", "xgb_all69", "lgbm_all69"]
TASKS = ([f"base_{m}" for m in ("rf_all69", "rf_cw_all69", "xgb_all69", "lgbm_all69")]
         + ["shap_rf", "shap_rf_cw"]
         + [f"topk_{m}" for m in ("rf_top5", "rf_top10", "rf_top20", "rf_top40", "rf_cw_top20")]
         + ["ens_soft_voting", "ens_stacking"])


def files_for(task, s):
    def std(m):
        return [D["models"] / f"{m}_split{s}.joblib", D["models"] / f"{m}_split{s}_metadata.json",
                D["metrics"] / f"{m}_split{s}_metrics.json", D["predictions"] / f"{m}_split{s}_test.parquet"]
    if task.startswith("base_"):
        m = task[len("base_"):]
        f = std(m) + [D["tuning"] / f"{m}_split{s}_tuning.csv", D["tuning"] / f"{m}_split{s}_selected.json"]
        if m in BASE_IDS:
            f.append(D["probs"] / f"{m}_split{s}_test.npy")
        return f
    if task.startswith("shap_"):
        tag = task[len("shap_"):]
        return ([D["shap"] / f"shap_importance_{tag}_split{s}.csv",
                 D["shap"] / f"shap_values_{tag}_split{s}.npy",
                 D["shap"] / f"shap_sample_row_ids_split{s}.npy"]
                + [D["feature_sets"] / f"{tag}_split{s}_top{k}.json" for k in (5, 10, 20, 40)])
    if task.startswith("topk_"):
        return std(task[len("topk_"):])
    if task == "ens_soft_voting":
        m = "soft_voting"
        return [D["models"] / f"{m}_split{s}_metadata.json", D["metrics"] / f"{m}_split{s}_metrics.json",
                D["predictions"] / f"{m}_split{s}_test.parquet"]
    m = "stacking_oof"
    return [D["models"] / f"{m}_split{s}_meta.joblib", D["models"] / f"{m}_split{s}_metadata.json",
            D["metrics"] / f"{m}_split{s}_metrics.json", D["predictions"] / f"{m}_split{s}_test.parquet"]


def readable(p):
    if not p.exists():
        return "yo'q"
    try:
        if p.suffix == ".json":
            json.load(open(p, encoding="utf-8"))
        elif p.suffix == ".csv":
            pd.read_csv(p)
        elif p.suffix == ".parquet":
            pd.read_parquet(p)
        elif p.suffix == ".npy":
            np.load(p)
        elif p.suffix == ".joblib":
            joblib.load(p)
        return "ok"
    except Exception as e:
        return f"BUZILGAN ({type(e).__name__})"


rows, bad_markers = [], []
for seed in SEEDS:
    for task in TASKS:
        mk = D["status"] / f"{task}_{seed}.json"
        if not mk.exists():
            rows.append((task, seed, "bajarilmagan", ""))
            continue
        try:
            st = json.load(open(mk, encoding="utf-8"))["status"]
        except Exception:
            st = "buzilgan marker"
        if st != "success":
            rows.append((task, seed, st, ""))
            bad_markers.append(mk)
            continue
        problems = [f"{p.name}: {r}" for p in files_for(task, seed) for r in [readable(p)] if r != "ok"]
        if problems:
            rows.append((task, seed, "success, lekin fayl muammosi", "; ".join(problems)))
            bad_markers.append(mk)
        else:
            rows.append((task, seed, "success", ""))

res = pd.DataFrame(rows, columns=["task", "seed", "holat", "muammo"])
print(f"Jami vazifalar: {len(res)}")
print(res["holat"].value_counts().to_string())
print("\nHar bir bo'linish bo'yicha tugagan vazifalar:")
print(res[res["holat"] == "success"].groupby("seed").size().reindex(SEEDS, fill_value=0).to_string())
problem = res[(res["holat"] != "success") & (res["holat"] != "bajarilmagan")]
print("\nMuammoli vazifalar:")
print(problem.to_string(index=False) if len(problem) else "yo'q")
if FIX and bad_markers:
    for mk in bad_markers:
        mk.unlink()
    print(f"\n{len(bad_markers)} ta muammoli marker o'chirildi, ular qayta hisoblanadi")