import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

VER = Path("results/run_003_verified")
OUT = Path("results/run_006_baselines")
AUDIT = Path("results/data_audit")
SPLITS = Path("results/splits")
for d in ("status", "metrics"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
T0 = time.time()
KS = [10, 20, 40]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


PROTO = Path("configs/experiment_protocol_v1.0.yaml")
raw = Path("configs/protocol_v1.0.sha256").read_bytes().decode("latin-1")
expected = "".join(ch for ch in raw if ch in "0123456789abcdefABCDEF").upper()
if sha256_of(PROTO) != expected:
    raise SystemExit("PROTOKOL O'ZGARGAN: hash mos emas")
with open(PROTO, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
SEEDS = cfg["splits"]["seeds"]
J = len(SEEDS)
N_CLASSES = cfg["data"]["n_classes"]
N_JOBS = cfg["resources"]["cpu_jobs"]
manifest = pd.read_csv(VER / "MANIFEST_SHA256.csv").set_index("relative_path")["sha256"].to_dict()


def ver_file(rel):
    p = VER / rel
    if sha256_of(p) != manifest[rel]:
        raise SystemExit(f"TEKSHIRILGAN FAYL O'ZGARGAN: {rel}")
    return p


with open(AUDIT / "clean_data_summary.json", encoding="utf-8") as f:
    FEAT = json.load(f)["feature_names"]
with open(AUDIT / "label_map.json", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ULTRA = [LABEL_MAP[n] for n in cfg["evaluation"]["ultra_rare_classes"]]
NONRARE = [c for c in range(N_CLASSES) if c not in ULTRA]

print("DIZAYN (oldindan belgilangan): impurity Top-k va tasodifiy k ta belgi, k = 10, 20, 40; "
      "o'sha bo'linishlar va rf_all69 parametrlari; SHAP natijalari asosiy tajribadan", flush=True)
if sha256_of(Path(cfg["data"]["parquet"])) != cfg["data"]["parquet_sha256"]:
    raise SystemExit("MA'LUMOT O'ZGARGAN: Parquet hashi mos emas")
df = pd.read_parquet(cfg["data"]["parquet"], columns=["row_id", "label_id"] + FEAT)
ROW_ID = df["row_id"].to_numpy(dtype="int64")
Y_ALL = df["label_id"].to_numpy().astype("int64")
X_ALL = df[FEAT].to_numpy(dtype="float32")
del df
gc.collect()
print(f"Ma'lumot yuklandi: {X_ALL.shape}", flush=True)


def positions(seed, part):
    ids = pd.read_parquet(SPLITS / f"split_{seed}_{part}.parquet")["row_id"].to_numpy()
    pos = np.searchsorted(ROW_ID, ids)
    assert np.array_equal(ROW_ID[pos], ids)
    return pos


def run_variant(method, k, seed, idx):
    mk = OUT / "status" / f"{method}_k{k}_{seed}.json"
    if mk.exists():
        print(f"  [o'tkazib yuborildi] {method} k={k} seed {seed}", flush=True)
        return
    idx = sorted(int(i) for i in idx)
    pos_tr, pos_te = positions(seed, "train"), positions(seed, "test")
    Xtr, ytr = X_ALL[np.ix_(pos_tr, idx)], Y_ALL[pos_tr]
    Xte, yte = X_ALL[np.ix_(pos_te, idx)], Y_ALL[pos_te]
    with open(ver_file(f"tuning/rf_all69_split{seed}_selected.json"), encoding="utf-8") as fh:
        params = json.load(fh)["params"]
    model = RandomForestClassifier(
        n_estimators=cfg["tuning"]["n_estimators_fixed"], max_features=params.get("max_features", "sqrt"),
        max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
        n_jobs=N_JOBS, random_state=seed)
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    yh = model.predict(Xte)
    _, _, fv, _ = precision_recall_fscore_support(yte, yh, labels=list(range(N_CLASSES)), zero_division=0)
    rec = {"method": method, "k": k, "split_seed": seed, "features": [FEAT[i] for i in idx],
           "macro_f1": float(fv.mean()), "macro_f1_12": float(fv[NONRARE].mean()),
           "f1": [float(x) for x in fv], "train_seconds": round(fit_s, 1), "params": params,
           "note": "post hoc baseline, not part of the frozen protocol"}
    with open(OUT / "metrics" / f"{method}_k{k}_split{seed}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    with open(mk, "w", encoding="utf-8") as fh:
        json.dump({"status": "success"}, fh)
    print(f"  {method:8s} k={k:2d} seed {seed}: macro-F1={rec['macro_f1']:.4f}, 12 sinf={rec['macro_f1_12']:.4f}, "
          f"{fit_s / 60:.1f} min", flush=True)
    del model, Xtr, Xte
    gc.collect()


for seed in SEEDS:
    model = joblib.load(ver_file(f"models/rf_all69_split{seed}.joblib"))
    imp = np.asarray(model.feature_importances_)
    assert len(imp) == len(FEAT)
    order = np.argsort(-imp, kind="stable")
    del model
    gc.collect()
    for k in KS:
        run_variant("impurity", k, seed, order[:k])
        rng = np.random.default_rng(1000 * seed + k)
        run_variant("random", k, seed, rng.choice(len(FEAT), size=k, replace=False))

# ---- taqqoslash ----
ex = pd.read_csv(ver_file("all_experiments.csv"))
recs, feats = [], {}
for k in KS:
    for seed in SEEDS:
        r = ex[(ex.model_name == f"rf_top{k}") & (ex.split_seed == seed)].iloc[0]
        recs.append({"method": "shap", "k": k, "seed": seed, "macro_f1": r["macro_f1"],
                     "macro_f1_12": r["macro_f1_12_classes"]})
        with open(ver_file(f"feature_sets/rf_split{seed}_top{k}.json"), encoding="utf-8") as fh:
            feats[("shap", k, seed)] = set(json.load(fh)["features"])
        for m in ("impurity", "random"):
            with open(OUT / "metrics" / f"{m}_k{k}_split{seed}_metrics.json", encoding="utf-8") as fh:
                j = json.load(fh)
            recs.append({"method": m, "k": k, "seed": seed, "macro_f1": j["macro_f1"],
                         "macro_f1_12": j["macro_f1_12"]})
            feats[(m, k, seed)] = set(j["features"])
rd = pd.DataFrame(recs)
rd.to_csv(OUT / "baseline_metrics.csv", index=False)

ratio = float(np.mean([len(pd.read_parquet(SPLITS / f"split_{s}_test.parquet")) /
                       len(pd.read_parquet(SPLITS / f"split_{s}_train.parquet")) for s in SEEDS]))
T975 = stats.t.ppf(0.975, J - 1)


def arr(method, k, key):
    sub = rd[(rd.method == method) & (rd.k == k)].set_index("seed")[key]
    return sub.loc[SEEDS].to_numpy()


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


comp_rows = []
for key in ("macro_f1", "macro_f1_12"):
    block = []
    for k in KS:
        for comp in ("impurity", "random"):
            d = arr("shap", k, key) - arr(comp, k, key)
            mean, sd = float(d.mean()), float(d.std(ddof=1))
            res = {"metric": key, "k": k, "comparator": comp, "mean_diff_shap_minus_comp": mean,
                   "n_splits_shap_better": int((d > 0).sum())}
            for tag, se in (("std", sd / np.sqrt(J)), ("nb", np.sqrt((1 / J + ratio) * sd ** 2))):
                if se == 0:
                    p, lo, hi = (1.0 if mean == 0 else 0.0), mean, mean
                else:
                    t_ = mean / se
                    p = float(2 * stats.t.sf(abs(t_), J - 1))
                    lo, hi = mean - T975 * se, mean + T975 * se
                res.update({f"ci95_{tag}_low": lo, f"ci95_{tag}_high": hi, f"p_{tag}": p})
            block.append(res)
    for tag in ("std", "nb"):
        for b, a in zip(block, holm([b[f"p_{tag}"] for b in block])):
            b[f"p_{tag}_holm"] = float(a)
    comp_rows.extend(block)
cmp = pd.DataFrame(comp_rows)
cmp.to_csv(OUT / "baseline_comparisons.csv", index=False)


def jac(a, b):
    return len(a & b) / len(a | b)


jr = []
for k in KS:
    jr.append({"k": k,
               "shap_vs_impurity": np.mean([jac(feats[("shap", k, s)], feats[("impurity", k, s)]) for s in SEEDS]),
               "shap_vs_random": np.mean([jac(feats[("shap", k, s)], feats[("random", k, s)]) for s in SEEDS]),
               "impurity_vs_random": np.mean([jac(feats[("impurity", k, s)], feats[("random", k, s)])
                                              for s in SEEDS])})
jd = pd.DataFrame(jr)
jd.to_csv(OUT / "baseline_jaccard.csv", index=False)

pd.set_option("display.width", 250)
print("\n===== MACRO-F1 (5 bo'linish: o'rtacha +/- sd) =====")
rows = []
for key, nm in (("macro_f1", "15 sinf"), ("macro_f1_12", "12 sinf")):
    for k in KS:
        row = {"metric": nm, "k": k}
        for m in ("shap", "impurity", "random"):
            a = arr(m, k, key)
            row[m] = f"{a.mean():.4f} +/- {a.std(ddof=1):.4f}"
        rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("\n===== TAQQOSLASH (farq = SHAP - taqqoslanuvchi; Holm 6 ta taqqoslash ichida) =====")
print(cmp[["metric", "k", "comparator", "mean_diff_shap_minus_comp", "ci95_nb_low", "ci95_nb_high",
           "p_std_holm", "p_nb_holm", "n_splits_shap_better"]].round(4).to_string(index=False))
print("\n===== BELGILAR TO'PLAMLARI O'XSHASHLIGI (o'rtacha Jaccard) =====")
print(jd.round(3).to_string(index=False))
print(f"\n[Bajarildi] vaqt: {(time.time() - T0) / 60:.1f} min")