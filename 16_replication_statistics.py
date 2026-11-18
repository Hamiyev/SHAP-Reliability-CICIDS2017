import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


PROTO = Path("configs/experiment_protocol_v1.1.yaml")
raw = Path("configs/protocol_v1.1.sha256").read_bytes().decode("latin-1")
expected = "".join(ch for ch in raw if ch in "0123456789abcdefABCDEF").upper()
if sha256_of(PROTO) != expected:
    raise SystemExit("PROTOKOL v1.1 O'ZGARGAN: hash mos emas")
with open(PROTO, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

OUT = Path(cfg["runs"]["main"])
STAT = OUT / "statistics"
STAT.mkdir(parents=True, exist_ok=True)
SPLITS = Path("results/splits_corrected")
V10 = Path("results/run_003_verified")
SEEDS = cfg["splits"]["seeds"]
J = len(SEEDS)
MODELS = [m["id"] for m in cfg["models"]]
N = cfg["data"]["n_classes"]
MARGIN = cfg["statistics"]["non_inferiority"]["margin"]
with open(cfg["data"]["label_map"], encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
ID2NAME = {v: k for k, v in LABEL_MAP.items()}
ULTRA = [LABEL_MAP[n] for n in cfg["evaluation"]["ultra_rare_classes"]]
NONRARE = [c for c in range(N) if c not in ULTRA]

exp = pd.read_csv(OUT / "all_experiments.csv")
want = {(m, s) for m in MODELS for s in SEEDS}
have = set(zip(exp["model_name"], exp["split_seed"]))
if have != want or len(exp) != len(want):
    raise SystemExit(f"NATIJALAR TO'LIQ EMAS: yetishmaydi {sorted(want - have)[:10]}")
print(f"OK: {len(exp)} ta natija to'liq ({len(MODELS)} model x {J} bo'linish)")

rows, fmat = [], {}
cm_rf = np.zeros((N, N), dtype="int64")
for m, s in sorted(want):
    p = pd.read_parquet(OUT / "predictions" / f"{m}_split{s}_test.parquet")
    y, yh = p["y_true"].to_numpy(), p["y_pred"].to_numpy()
    _, _, f, _ = precision_recall_fscore_support(y, yh, labels=list(range(N)), zero_division=0)
    fmat[(m, s)] = f
    rows.append({"model_name": m, "split_seed": s, "macro_f1": f.mean(), "macro_f1_nonrare": f[NONRARE].mean()})
    if m == "rf_all77":
        cm_rf += confusion_matrix(y, yh, labels=list(range(N)))
rc = pd.DataFrame(rows)
mg = rc.merge(exp, on=["model_name", "split_seed"], suffixes=("", "_run"))
for k in ("macro_f1", "macro_f1_nonrare"):
    if not np.allclose(mg[k], mg[k + "_run"], atol=1e-9):
        raise SystemExit(f"QAYTA HISOBLASH MOS EMAS: {k}")
print("OK: metrikalar saqlangan bashoratlardan qayta hisoblandi va runner natijalariga mos")

ratio = float(np.mean([len(pd.read_parquet(SPLITS / f"split_{s}_test.parquet")) /
                       len(pd.read_parquet(SPLITS / f"split_{s}_train.parquet")) for s in SEEDS]))
T975, T95 = stats.t.ppf(0.975, J - 1), stats.t.ppf(0.95, J - 1)
METRICS = {"macro_f1_16": "macro_f1", "macro_f1_nonrare": "macro_f1_nonrare"}
piv = {k: rc.pivot(index="split_seed", columns="model_name", values=c).loc[SEEDS] for k, c in METRICS.items()}


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


out_rows = []
for mk in METRICS:
    block = []
    for A, B in cfg["statistics"]["comparisons"]:
        d = (piv[mk][B] - piv[mk][A]).to_numpy()
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        res = {"metric": mk, "reference": A, "comparison_model": B, "mean_diff": mean, "sd_diff": sd,
               "n_splits_worse": int((d < 0).sum())}
        for tag, se in (("std", sd / np.sqrt(J)), ("nb", np.sqrt((1 / J + ratio) * sd ** 2))):
            if se == 0:
                p, lo, hi, low1 = (1.0 if mean == 0 else 0.0), mean, mean, mean
            else:
                t = mean / se
                p = float(2 * stats.t.sf(abs(t), J - 1))
                lo, hi, low1 = mean - T975 * se, mean + T975 * se, mean - T95 * se
            res.update({f"ci95_{tag}_low": lo, f"ci95_{tag}_high": hi, f"p_{tag}": p, f"lower1s_{tag}": low1})
        block.append(res)
    for tag in ("std", "nb"):
        for b, a in zip(block, holm([b[f"p_{tag}"] for b in block])):
            b[f"p_{tag}_holm"] = float(a)
    out_rows.extend(block)
summ = pd.DataFrame(out_rows)
summ.to_csv(STAT / "paired_differences.csv", index=False)

a16 = summ[summ.metric == "macro_f1_16"].set_index("comparison_model")
nr = summ[summ.metric == "macro_f1_nonrare"].set_index("comparison_model")
c1p = {m: bool(a16.loc[m, "mean_diff"] < 0 and a16.loc[m, "p_std_holm"] < 0.05) for m in ("rf_top5", "rf_top10")}
C1 = all(c1p.values())
C2 = bool(nr.loc["rf_top40", "lower1s_std"] > -MARGIN)
c3p = {m: bool(a16.loc[m, "p_std_holm"] >= 0.05) for m in ("xgb_all77", "lgbm_all77")}
C3 = all(c3p.values())
c1n = all(bool(a16.loc[m, "mean_diff"] < 0 and a16.loc[m, "p_nb_holm"] < 0.05) for m in ("rf_top5", "rf_top10"))
c2n = bool(nr.loc["rf_top40", "lower1s_nb"] > -MARGIN)
c3n = all(bool(a16.loc[m, "p_nb_holm"] >= 0.05) for m in ("xgb_all77", "lgbm_all77"))
crit = {"C1_top5_top10_significantly_worse": {"met": C1, "parts": c1p, "met_with_nadeau_bengio": c1n},
        "C2_top40_noninferior_nonrare": {"met": C2, "lower_bound_std": float(nr.loc["rf_top40", "lower1s_std"]),
                                         "lower_bound_nb": float(nr.loc["rf_top40", "lower1s_nb"]),
                                         "met_with_nadeau_bengio": c2n},
        "C3_xgb_lgbm_not_different": {"met": C3, "parts": c3p, "met_with_nadeau_bengio": c3n}}
with open(STAT / "replication_criteria.json", "w", encoding="utf-8") as f:
    json.dump(crit, f, indent=2)

old_man = pd.read_csv(V10 / "MANIFEST_SHA256.csv").set_index("relative_path")["sha256"].to_dict()
old_path = V10 / "statistics" / "paired_differences.csv"
if sha256_of(old_path) != old_man["statistics/paired_differences.csv"]:
    raise SystemExit("v1.0 STATISTIKA FAYLI O'ZGARGAN")
old = pd.read_csv(old_path)
cmp_rows = []
for lab, m10, m11 in [("xgb", "xgb_all69", "xgb_all77"), ("lgbm", "lgbm_all69", "lgbm_all77"),
                      ("top5", "rf_top5", "rf_top5"), ("top10", "rf_top10", "rf_top10"),
                      ("top20", "rf_top20", "rf_top20"), ("top40", "rf_top40", "rf_top40")]:
    for met10, met11, nm in (("macro_f1", "macro_f1_16", "all classes"),
                             ("macro_f1_12_classes", "macro_f1_nonrare", "non-ultra-rare")):
        o = old[(old.metric == met10) & (old.reference == "rf_all69") & (old.comparison_model == m10)].iloc[0]
        n = summ[(summ.metric == met11) & (summ.comparison_model == m11)].iloc[0]
        cmp_rows.append({"comparison": lab, "metric": nm, "v10_mean_diff": o["mean_diff"], "v10_p_holm": o["p_holm"],
                         "v11_mean_diff": n["mean_diff"], "v11_p_holm_std": n["p_std_holm"],
                         "v11_p_holm_nb": n["p_nb_holm"]})
cmpdf = pd.DataFrame(cmp_rows)
cmpdf.to_csv(STAT / "v10_vs_v11.csv", index=False)

ms = exp.groupby("model_name").agg(
    macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
    nonrare_mean=("macro_f1_nonrare", "mean"), nonrare_sd=("macro_f1_nonrare", "std"),
    accuracy_mean=("accuracy", "mean"), train_seconds_mean=("train_seconds", "mean")).loc[MODELS]
ms.to_csv(STAT / "model_summary.csv")
pcf = pd.DataFrame({m: np.mean([fmat[(m, s)] for s in SEEDS], axis=0) for m in MODELS},
                   index=[ID2NAME[c] for c in range(N)])
pcf.to_csv(STAT / "per_class_f1_mean.csv")
pd.DataFrame(cm_rf, index=[ID2NAME[c] for c in range(N)], columns=[ID2NAME[c] for c in range(N)]).to_csv(
    STAT / "confusion_rf_all77_total.csv")

pd.set_option("display.width", 250)
print(f"\nn_test/n_train = {ratio:.4f}")
print("\n===== MODELLAR (5 bo'linish o'rtachasi) =====")
print(ms.round(4).to_string())
cols = ["comparison_model", "mean_diff", "ci95_std_low", "ci95_std_high", "ci95_nb_low", "ci95_nb_high",
        "p_std_holm", "p_nb_holm", "n_splits_worse"]
print("\n===== TAQQOSLASH: 16 sinfli Macro-F1 (farq = B - rf_all77) =====")
print(summ[summ.metric == "macro_f1_16"][cols].round(4).to_string(index=False))
print("\n===== TAQQOSLASH: 11 sinfli Macro-F1 va ekvivalentlik (chegara %.2f) =====" % MARGIN)
print(summ[summ.metric == "macro_f1_nonrare"][["comparison_model", "mean_diff", "lower1s_std", "lower1s_nb",
                                               "p_std_holm", "p_nb_holm"]].round(4).to_string(index=False))
print("\n===== TAKRORLASH MEZONLARI (protokol v1.1) =====")
print(f"C1 (Top-5 va Top-10 ahamiyatli yomon, 16 sinf): {'BAJARILDI' if C1 else 'BAJARILMADI'}  {c1p} | "
      f"Nadeau-Bengio bilan: {c1n}")
print(f"C2 (Top-40 ekvivalent, 11 sinf, quyi chegara {crit['C2_top40_noninferior_nonrare']['lower_bound_std']:.4f}): "
      f"{'BAJARILDI' if C2 else 'BAJARILMADI'} | Nadeau-Bengio bilan: {c2n}")
print(f"C3 (XGBoost va LightGBM RF dan farq qilmaydi, 16 sinf): {'BAJARILDI' if C3 else 'BAJARILMADI'}  {c3p} | "
      f"Nadeau-Bengio bilan: {c3n}")
print("\n===== v1.0 va v1.1 YONMA-YON (farq = B - RF; p = Holm) =====")
print(cmpdf.round(4).to_string(index=False))
print("\n===== SINF BO'YICHA F1 (o'rtacha), RF / XGBoost / LightGBM =====")
print(pcf[["rf_all77", "xgb_all77", "lgbm_all77"]].round(3).to_string())
ps, ip = LABEL_MAP["Portscan"], LABEL_MAP["Infiltration - Portscan"]
sub = cm_rf[np.ix_([ps, ip], [ps, ip])]
tot = cm_rf[[ps, ip]].sum(axis=1)
print("\n===== Portscan va Infiltration - Portscan (RF, 5 bo'linish yig'indisi) =====")
print(pd.DataFrame(sub, index=["true Portscan", "true Infiltration - Portscan"],
                   columns=["pred Portscan", "pred Infiltration - Portscan"]).to_string())
print(f"Qatorlar jami: {tot.tolist()}; boshqa sinflarga ketgani: {(tot - sub.sum(axis=1)).tolist()}")
print("\n[Bajarildi] takroriy tajriba statistikasi yakunlandi")