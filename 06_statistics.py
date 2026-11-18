import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

OUT = Path("results/run_002_main")
AUDIT = Path("results/data_audit")
STAT = OUT / "statistics"
STAT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)


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
MODELS = [m["id"] for m in cfg["models"]]
MARGIN = cfg["statistics"]["non_inferiority"]["margin"]
N_CLASSES = cfg["data"]["n_classes"]
assert cfg["statistics"]["non_inferiority"]["metric"] == "macro_f1_12_classes_excluding_ultra_rare"
with open(AUDIT / "label_map.json", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ULTRA = [LABEL_MAP[n] for n in cfg["evaluation"]["ultra_rare_classes"]]
NONRARE = [c for c in range(N_CLASSES) if c not in ULTRA]
CONFLICT_IDS = pd.read_csv(AUDIT / "conflict_rows.csv")["row_id"].to_numpy()

# 1) to'liqlik tekshiruvi
exp = pd.read_csv(OUT / "all_experiments.csv")
have = set(zip(exp["model_name"], exp["split_seed"]))
want = {(m, s) for m in MODELS for s in SEEDS}
if have != want or len(exp) != len(want):
    raise SystemExit(f"NATIJALAR TO'LIQ EMAS: yetishmaydi {sorted(want - have)[:10]}")
print(f"OK: {len(exp)} ta natija to'liq ({len(MODELS)} model x {len(SEEDS)} bo'linish)")

# 2) metrikalarni saqlangan bashoratlardan qayta hisoblash
rows = []
for m, s in sorted(want):
    p = pd.read_parquet(OUT / "predictions" / f"{m}_split{s}_test.parquet")
    y, yh, ids = p["y_true"].to_numpy(), p["y_pred"].to_numpy(), p["row_id"].to_numpy()
    labels = list(range(N_CLASSES))
    _, _, f, _ = precision_recall_fscore_support(y, yh, labels=labels, zero_division=0)
    keep = ~np.isin(ids, CONFLICT_IDS)
    _, _, f2, _ = precision_recall_fscore_support(y[keep], yh[keep], labels=labels, zero_division=0)
    rows.append({"model_name": m, "split_seed": s, "macro_f1": f.mean(),
                 "macro_f1_12_classes": f[NONRARE].mean(), "macro_f1_excl_conflicts": f2.mean()})
rc = pd.DataFrame(rows)
METRICS = ["macro_f1", "macro_f1_12_classes", "macro_f1_excl_conflicts"]
mg = rc.merge(exp, on=["model_name", "split_seed"], suffixes=("", "_run"))
for mt in METRICS:
    if not np.allclose(mg[mt], mg[mt + "_run"], atol=1e-9):
        raise SystemExit(f"QAYTA HISOBLASH MOS EMAS: {mt}")
print("OK: metrikalar saqlangan bashoratlardan qayta hisoblandi va runner natijalariga mos")

# 3) juft taqqoslashlar
T975, T95 = stats.t.ppf(0.975, len(SEEDS) - 1), stats.t.ppf(0.95, len(SEEDS) - 1)


def paired(d):
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    if sd == 0:
        return {"mean_diff": mean, "sd_diff": 0.0, "ci95_low": mean, "ci95_high": mean, "t_stat": np.nan,
                "p_raw": 1.0 if mean == 0 else 0.0, "cohens_dz": np.nan,
                "n_splits_positive": int((d > 0).sum()), "one_sided_lower_95": mean}
    t = mean / se
    return {"mean_diff": mean, "sd_diff": sd, "ci95_low": mean - T975 * se, "ci95_high": mean + T975 * se,
            "t_stat": float(t), "p_raw": float(2 * stats.t.sf(abs(t), n - 1)), "cohens_dz": mean / sd,
            "n_splits_positive": int((d > 0).sum()), "one_sided_lower_95": mean - T95 * se}


def holm(p):
    p = np.asarray(p, dtype=float)
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


pivots = {mt: rc.pivot(index="split_seed", columns="model_name", values=mt).loc[SEEDS] for mt in METRICS}
out_rows = []
for mt in METRICS:
    block = []
    for A, B in cfg["statistics"]["comparisons"]:
        d = (pivots[mt][B] - pivots[mt][A]).to_numpy()
        r = paired(d)
        rec = {"metric": mt, "reference": A, "comparison_model": B,
               **{f"diff_split_{s}": float(x) for s, x in zip(SEEDS, d)}, **r}
        if mt == "macro_f1_12_classes":
            rec["noninferior_margin"] = bool(r["one_sided_lower_95"] > -MARGIN)
        block.append(rec)
    for b, a in zip(block, holm([b["p_raw"] for b in block])):
        b["p_holm"] = float(a)
        b["reject_holm_0.05"] = bool(a < 0.05)
    out_rows.extend(block)
pdiff = pd.DataFrame(out_rows)
pdiff.to_csv(STAT / "paired_differences.csv", index=False)
mc = pdiff[["metric", "reference", "comparison_model", "p_raw", "p_holm", "reject_holm_0.05"]].copy()
mc["holm_rank"] = mc.groupby("metric")["p_raw"].rank(method="first").astype(int)
mc.sort_values(["metric", "holm_rank"]).to_csv(STAT / "multiple_comparison_correction.csv", index=False)

# 4) modellar bo'yicha umumiy jadval
summ = exp.groupby("model_name").agg(
    macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
    macro_f1_12_mean=("macro_f1_12_classes", "mean"), macro_f1_12_sd=("macro_f1_12_classes", "std"),
    accuracy_mean=("accuracy", "mean"), weighted_f1_mean=("weighted_f1", "mean"),
    balanced_acc_mean=("balanced_accuracy", "mean"), train_seconds_mean=("train_seconds", "mean"),
    inference_seconds_mean=("inference_seconds", "mean"), model_size_mb_mean=("model_size_bytes", "mean"))
summ["model_size_mb_mean"] = summ["model_size_mb_mean"] / 1e6
summ = summ.loc[MODELS]
summ.to_csv(STAT / "model_summary.csv")

primary = pdiff[pdiff["metric"] == "macro_f1"]
ni = pdiff[pdiff["metric"] == "macro_f1_12_classes"]
report = {
    "protocol_version": cfg["project"]["protocol_version"], "n_splits": len(SEEDS),
    "test": "paired t-test, df = n_splits - 1, two-sided", "correction": "Holm within each metric (10 comparisons)",
    "ci": "95% two-sided t interval of the mean paired difference (B - A)",
    "non_inferiority_rule": cfg["statistics"]["non_inferiority"]["rule"], "margin": MARGIN,
    "significant_after_holm_macro_f1": primary[primary["reject_holm_0.05"]][["reference", "comparison_model"]]
    .values.tolist(),
    "non_inferior_12_class": ni[ni["noninferior_margin"]][["reference", "comparison_model"]].values.tolist(),
    "caveats": ["splits are repeated random partitions of the same data, not independent datasets",
                "n=5 gives 4 degrees of freedom: intervals are wide and p-values are indicative",
                "ultra-rare classes (2-5 test rows) make 15-class Macro-F1 noisy",
                "non-inferiority bounds are not adjusted for multiple comparisons"],
}
with open(STAT / "statistical_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# 5) ekranga qisqa xulosa
pd.set_option("display.width", 250)
print("\n===== MODELLAR (5 bo'linish bo'yicha o'rtacha) =====")
print(summ[["macro_f1_mean", "macro_f1_sd", "macro_f1_12_mean", "macro_f1_12_sd", "accuracy_mean",
            "train_seconds_mean"]].round(4).to_string())
cols = ["reference", "comparison_model", "mean_diff", "ci95_low", "ci95_high", "p_raw", "p_holm", "n_splits_positive"]
print("\n===== TAQQOSLASHLAR: 15 sinfli Macro-F1 (farq = B - A) =====")
print(pdiff[pdiff["metric"] == "macro_f1"][cols].round(4).to_string(index=False))
print("\n===== TAQQOSLASHLAR: 12 sinfli Macro-F1 + ekvivalentlik (chegara %.2f) =====" % MARGIN)
print(ni[cols[:5] + ["p_holm", "one_sided_lower_95", "noninferior_margin"]].round(4).to_string(index=False))
print("\nHolm dan keyin 0.05 da ahamiyatli (15 sinf):", report["significant_after_holm_macro_f1"])
print("[Bajarildi] statistik tahlil yakunlandi")