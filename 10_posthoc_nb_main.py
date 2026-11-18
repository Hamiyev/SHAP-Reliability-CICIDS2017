import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

VER = Path("results/run_003_verified")
OUT = Path("results/run_004_posthoc")
SPLITS = Path("results/splits")
OUT.mkdir(parents=True, exist_ok=True)


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
manifest = pd.read_csv(VER / "MANIFEST_SHA256.csv").set_index("relative_path")["sha256"].to_dict()


def ver_file(rel):
    p = VER / rel
    if sha256_of(p) != manifest[rel]:
        raise SystemExit(f"TEKSHIRILGAN FAYL O'ZGARGAN: {rel}")
    return p


SEEDS = cfg["splits"]["seeds"]
J = len(SEEDS)
MARGIN = cfg["statistics"]["non_inferiority"]["margin"]
ex = pd.read_csv(ver_file("all_experiments.csv"))
ratio = float(np.mean([len(pd.read_parquet(SPLITS / f"split_{s}_test.parquet")) /
                       len(pd.read_parquet(SPLITS / f"split_{s}_train.parquet")) for s in SEEDS]))
factor = (1 / J + ratio) / (1 / J)
T975, T95 = stats.t.ppf(0.975, J - 1), stats.t.ppf(0.95, J - 1)
METRICS = {"macro_f1_15": "macro_f1", "macro_f1_12": "macro_f1_12_classes",
           "macro_f1_excl_conflicts": "macro_f1_excl_conflicts"}
piv = {k: ex.pivot(index="split_seed", columns="model_name", values=c).loc[SEEDS] for k, c in METRICS.items()}


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


rows = []
for mkey in METRICS:
    block = []
    for A, B in cfg["statistics"]["comparisons"]:
        d = (piv[mkey][B] - piv[mkey][A]).to_numpy()
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        res = {"metric": mkey, "reference": A, "comparison_model": B, "mean_diff": mean, "sd_diff": sd}
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
    rows.extend(block)
summ = pd.DataFrame(rows)
summ["sig_std"] = summ["p_std_holm"] < 0.05
summ["sig_nb"] = summ["p_nb_holm"] < 0.05
summ["noninf_std"] = summ["lower1s_std"] > -MARGIN
summ["noninf_nb"] = summ["lower1s_nb"] > -MARGIN
summ.to_csv(OUT / "nb_corrected_main_comparisons.csv", index=False)

pd.set_option("display.width", 250)
print(f"n_test/n_train = {ratio:.4f}; standart xato koeffitsiyenti = {np.sqrt(factor):.3f}")
a = summ[summ.metric == "macro_f1_15"]
print("\n===== 15 SINFLI MACRO-F1 (post hoc, Nadeau-Bengio) =====")
print(a[["reference", "comparison_model", "mean_diff", "ci95_nb_low", "ci95_nb_high",
         "p_std_holm", "p_nb_holm", "sig_std", "sig_nb"]].round(4).to_string(index=False))
b = summ[summ.metric == "macro_f1_12"]
print("\n===== 12 SINFLI MACRO-F1 VA EKVIVALENTLIK (chegara %.2f) =====" % MARGIN)
print(b[["reference", "comparison_model", "mean_diff", "lower1s_std", "lower1s_nb",
         "noninf_std", "noninf_nb"]].round(4).to_string(index=False))
print("\n[Bajarildi]")