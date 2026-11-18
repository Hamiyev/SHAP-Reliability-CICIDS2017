import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PARQUET = Path("data/processed/cicids2017_clean.parquet")
AUDIT = Path("results/data_audit")
SPLITS = Path("results/splits")
OUT = Path("results/run_001_pilot")
OUT.mkdir(parents=True, exist_ok=True)
PILOT_SEED, SPLIT_SEED, FRAC, VAL_FRAC, MIN_PER_CLASS = 1, 42, 0.05, 0.10, 20
n_classes = 15

with open(AUDIT / "clean_data_summary.json", encoding="utf-8") as f:
    feat = json.load(f)["feature_names"]
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


Xtr, ytr = take(stratified_sample(train_ids, FRAC, PILOT_SEED))
Xva, yva = take(stratified_sample(val_ids, VAL_FRAC, PILOT_SEED))
del df
print(f"train {Xtr.shape}, val {Xva.shape}, barcha qiymatlar chekli: "
      f"{bool(np.isfinite(Xtr).all())}, eng katta |qiymat|: {float(np.abs(Xtr).max()):.3g}")

VARIANTS = {
    "default": {},
    "lr0.05": {"learning_rate": 0.05},
    "l2=1": {"reg_lambda": 1.0},
    "hess=1": {"min_child_weight": 1.0},
    "l2=1+hess=1": {"reg_lambda": 1.0, "min_child_weight": 1.0},
}
rows = []
for name, extra in VARIANTS.items():
    params = dict(n_estimators=100, num_leaves=31, learning_rate=0.1, n_jobs=4,
                  random_state=PILOT_SEED, verbose=-1)
    params.update(extra)
    m = lgb.LGBMClassifier(**params)
    ev = {}
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "val"],
          callbacks=[lgb.record_evaluation(ev)])
    tl, vl = ev["train"]["multi_logloss"], ev["val"]["multi_logloss"]
    ptr, pva = m.predict(Xtr), m.predict(Xva)
    rows.append({
        "variant": name,
        "train_ll_1": round(tl[0], 3), "train_ll_10": round(tl[9], 3),
        "train_ll_50": round(tl[49], 3), "train_ll_100": round(tl[99], 3),
        "val_ll_100": round(vl[99], 3),
        "train_macro_f1": round(float(f1_score(ytr, ptr, average="macro")), 4),
        "val_macro_f1": round(float(f1_score(yva, pva, average="macro")), 4),
        "val_classes_predicted": int(len(np.unique(pva))),
    })
    if name == "default":
        print("\nDefault: haqiqiy sinflar soni (val):  ", np.bincount(yva, minlength=n_classes).tolist())
        print("Default: bashorat qilingan soni (val):", np.bincount(pva, minlength=n_classes).tolist())

res = pd.DataFrame(rows)
res.to_csv(OUT / "lgbm_diagnostic.csv", index=False)
print("\n===== LIGHTGBM DIAGNOSTIKA =====")
print(res.to_string(index=False))