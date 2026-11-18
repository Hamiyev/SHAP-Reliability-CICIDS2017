import hashlib
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

VER = Path("results/run_003_verified")
OUT = Path("results/run_004_posthoc")
AUDIT = Path("results/data_audit")
SPLITS = Path("results/splits")
for d in ("status", "predictions", "metrics"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
T0 = time.time()


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
ID2NAME = {v: k for k, v in LABEL_MAP.items()}
ULTRA = [LABEL_MAP[n] for n in cfg["evaluation"]["ultra_rare_classes"]]
NONRARE = [c for c in range(N_CLASSES) if c not in ULTRA]
CONFLICT_IDS = pd.read_csv(AUDIT / "conflict_rows.csv")["row_id"].to_numpy()

PORT = ["Destination Port"]
ENGELEN = ["Fwd Header Length", "Total Fwd Packets", "min_seg_size_forward",
           "Init_Win_bytes_forward", "SYN Flag Count", "Bwd Packet Length Mean"]
VARIANTS = {"no_port": PORT, "no_engelen6": ENGELEN, "no_port_no_engelen6": PORT + ENGELEN}
assert all(f in FEAT for f in PORT + ENGELEN)

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


def evaluate(y, yh, ids):
    labels = list(range(N_CLASSES))
    _, _, f, _ = precision_recall_fscore_support(y, yh, labels=labels, zero_division=0)
    keep = ~np.isin(ids, CONFLICT_IDS)
    _, _, f2, _ = precision_recall_fscore_support(y[keep], yh[keep], labels=labels, zero_division=0)
    return float(f.mean()), float(f[NONRARE].mean()), float(f2.mean()), f


# 1) baseline: saqlangan bashoratlardan qayta hisoblash va asosiy natijaga solishtirish
exp = pd.read_csv(ver_file("all_experiments.csv"))
base = {}
for s in SEEDS:
    p = pd.read_parquet(ver_file(f"predictions/rf_all69_split{s}_test.parquet"))
    m15, m12, mx, f = evaluate(p["y_true"].to_numpy(), p["y_pred"].to_numpy(), p["row_id"].to_numpy())
    ref = exp[(exp.model_name == "rf_all69") & (exp.split_seed == s)]["macro_f1"].iloc[0]
    if abs(m15 - ref) > 1e-9:
        raise SystemExit(f"BASELINE MOS EMAS (split {s})")
    base[s] = {"macro_f1": m15, "macro_f1_12": m12, "macro_f1_excl": mx, "f1": f}
print("OK: baseline (rf_all69) asosiy natijaga aynan mos", flush=True)


# 2) variantlarni o'qitish
def run_variant(vname, drop, seed):
    mk = OUT / "status" / f"{vname}_{seed}.json"
    if mk.exists():
        print(f"  [o'tkazib yuborildi] {vname} seed {seed}", flush=True)
        return
    t0 = time.time()
    keep_idx = [i for i, f in enumerate(FEAT) if f not in drop]
    pos_tr, pos_te = positions(seed, "train"), positions(seed, "test")
    Xtr = X_ALL[np.ix_(pos_tr, keep_idx)]
    ytr = Y_ALL[pos_tr]
    Xte = X_ALL[np.ix_(pos_te, keep_idx)]
    yte, ids_te = Y_ALL[pos_te], ROW_ID[pos_te]
    with open(ver_file(f"tuning/rf_all69_split{seed}_selected.json"), encoding="utf-8") as f:
        params = json.load(f)["params"]
    model = RandomForestClassifier(
        n_estimators=cfg["tuning"]["n_estimators_fixed"], max_features=params.get("max_features", "sqrt"),
        max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
        n_jobs=N_JOBS, random_state=seed)
    t = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    yh = model.predict(Xte)
    m15, m12, mx, f = evaluate(yte, yh, ids_te)
    pd.DataFrame({"row_id": ids_te, "y_true": yte, "y_pred": yh}).to_parquet(
        OUT / "predictions" / f"{vname}_split{seed}_test.parquet", index=False)
    rec = {"variant": vname, "split_seed": seed, "dropped": drop, "n_features": len(keep_idx),
           "params": params, "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
           "macro_f1": m15, "macro_f1_12": m12, "macro_f1_excl": mx, "f1": [float(x) for x in f],
           "train_seconds": round(fit_s, 1), "protocol_version": cfg["project"]["protocol_version"],
           "note": "post hoc ablation, not part of the frozen protocol"}
    with open(OUT / "metrics" / f"{vname}_split{seed}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    with open(mk, "w", encoding="utf-8") as fh:
        json.dump({"status": "success", "seconds": round(time.time() - t0, 1)}, fh)
    print(f"  {vname} seed {seed}: macro-F1={m15:.4f} (baseline {base[seed]['macro_f1']:.4f}), "
          f"12 sinf={m12:.4f}, {fit_s / 60:.1f} min", flush=True)
    del model, Xtr, Xte
    gc.collect()


for vname, drop in VARIANTS.items():
    for seed in SEEDS:
        run_variant(vname, drop, seed)

# 3) juft taqqoslash: oddiy va Nadeau-Bengio tuzatilgan
recs = {v: {s: json.load(open(OUT / "metrics" / f"{v}_split{s}_metrics.json", encoding="utf-8"))
            for s in SEEDS} for v in VARIANTS}
J = len(SEEDS)
ratio = float(np.mean([recs["no_port"][s]["test_rows"] / recs["no_port"][s]["train_rows"] for s in SEEDS]))
tcrit = stats.t.ppf(0.975, J - 1)


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def one(mean, se):
    if se == 0:
        return mean, mean, 1.0 if mean == 0 else 0.0
    t = mean / se
    return mean - tcrit * se, mean + tcrit * se, float(2 * stats.t.sf(abs(t), J - 1))


rows = []
for metric, key in (("macro_f1_15", "macro_f1"), ("macro_f1_12", "macro_f1_12"),
                    ("macro_f1_excl_conflicts", "macro_f1_excl")):
    block = []
    for v in VARIANTS:
        d = np.array([recs[v][s][key] - base[s][key] for s in SEEDS])
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        lo_s, hi_s, p_s = one(mean, sd / np.sqrt(J))
        lo_n, hi_n, p_n = one(mean, np.sqrt((1 / J + ratio) * sd ** 2))
        block.append({"metric": metric, "variant": v, "baseline_mean": float(np.mean([base[s][key] for s in SEEDS])),
                      "variant_mean": float(np.mean([recs[v][s][key] for s in SEEDS])), "mean_diff": mean,
                      "ci95_std_low": lo_s, "ci95_std_high": hi_s, "p_std": p_s,
                      "ci95_nb_low": lo_n, "ci95_nb_high": hi_n, "p_nb": p_n,
                      "n_splits_worse": int((d < 0).sum())})
    for b, a in zip(block, holm([b["p_std"] for b in block])):
        b["p_std_holm"] = float(a)
    for b, a in zip(block, holm([b["p_nb"] for b in block])):
        b["p_nb_holm"] = float(a)
    rows.extend(block)
summ = pd.DataFrame(rows)
summ.to_csv(OUT / "ablation_summary.csv", index=False)

pc = pd.DataFrame({"class": [ID2NAME[c] for c in range(N_CLASSES)],
                   "baseline_f1": np.mean([base[s]["f1"] for s in SEEDS], axis=0)})
for v in VARIANTS:
    pc[v] = np.mean([recs[v][s]["f1"] for s in SEEDS], axis=0)
    pc["delta_" + v] = pc[v] - pc["baseline_f1"]
pc.to_csv(OUT / "ablation_per_class_f1.csv", index=False)

pd.set_option("display.width", 250)
print(f"\n===== ABLATION XULOSASI (post hoc; n_test/n_train = {ratio:.3f}) =====")
cols = ["metric", "variant", "baseline_mean", "variant_mean", "mean_diff", "ci95_std_low", "ci95_std_high",
        "ci95_nb_low", "ci95_nb_high", "p_std_holm", "p_nb_holm", "n_splits_worse"]
print(summ[cols].round(4).to_string(index=False))
print("\n===== SINF BO'YICHA F1 (5 bo'linish o'rtachasi) =====")
print(pc[["class", "baseline_f1"] + ["delta_" + v for v in VARIANTS]].round(3).to_string(index=False))
print(f"\n[Bajarildi] vaqt: {(time.time() - T0) / 60:.1f} min")