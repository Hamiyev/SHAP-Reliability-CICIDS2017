import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_fscore_support

RUN = Path("results/run_002_main")
VER = Path("results/run_003_verified")
AUDIT = Path("results/data_audit")
SPLITS = Path("results/splits")
PROTO = Path("configs/experiment_protocol_v1.0.yaml")
SEEDS = [42, 123, 2024, 3407, 7777]
problems = []


def check(cond, msg):
    print(("  OK: " if cond else "  XATO: ") + msg, flush=True)
    if not cond:
        problems.append(msg)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


with open(PROTO, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
models = [m["id"] for m in cfg["models"]]
check(sha256_of(PROTO) == "".join(ch for ch in Path("configs/protocol_v1.0.sha256").read_text(encoding="latin-1")
                                 if ch in "0123456789abcdefABCDEF").upper(),
      "protokol hashi muzlatilgan qiymatga mos")
check(sha256_of(AUDIT / "clean_data_summary.json") is not None, "tozalash hisoboti mavjud")
with open(AUDIT / "label_map.json", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
N = 15
ULTRA = [LABEL_MAP[n] for n in cfg["evaluation"]["ultra_rare_classes"]]
NONRARE = [c for c in range(N) if c not in ULTRA]
CONF = set(pd.read_csv(AUDIT / "conflict_rows.csv")["row_id"].to_numpy())

ex = pd.read_csv(RUN / "all_experiments.csv")
check(len(ex) == len(models) * len(SEEDS), f"natijalar soni {len(ex)} (kutilgan {len(models) * len(SEEDS)})")
check(ex["status"].eq("success").all(), "barcha natijalar status=success")

# 1) metrikalarni bashoratlardan mustaqil qayta hisoblash
worst = 0.0
for m in models:
    for s in SEEDS:
        p = pd.read_parquet(RUN / "predictions" / f"{m}_split{s}_test.parquet")
        y, yh, ids = p["y_true"].to_numpy(), p["y_pred"].to_numpy(), p["row_id"].to_numpy()
        check(len(p) == len(set(ids)), f"{m} s{s}: test ID lari noyob")
        test_ids = pd.read_parquet(SPLITS / f"split_{s}_test.parquet")["row_id"].to_numpy()
        check(set(ids) == set(test_ids), f"{m} s{s}: bashorat ID lari test qismiga aynan mos")
        _, _, f, _ = precision_recall_fscore_support(y, yh, labels=list(range(N)), zero_division=0)
        row = ex[(ex.model_name == m) & (ex.split_seed == s)].iloc[0]
        worst = max(worst, abs(f.mean() - row["macro_f1"]))
check(worst < 1e-9, f"15 sinfli Macro-F1 qayta hisoblash farqi: {worst:.2e}")

# 2) test ID lari tuning va SHAP tanlovida ishtirok etmagan
for s in SEEDS:
    test_ids = set(pd.read_parquet(SPLITS / f"split_{s}_test.parquet")["row_id"].to_numpy())
    train_ids = set(pd.read_parquet(SPLITS / f"split_{s}_train.parquet")["row_id"].to_numpy())
    shap_ids = set(np.load(RUN / "shap" / f"shap_sample_row_ids_split{s}.npy").tolist())
    check(len(shap_ids & test_ids) == 0, f"s{s}: SHAP namunasi test qismida yo'q")
    check(shap_ids <= train_ids, f"s{s}: SHAP namunasi to'liq train qismidan")
    for tag in ("rf", "rf_cw"):
        imp = pd.read_csv(RUN / "shap" / f"shap_importance_{tag}_split{s}.csv")
        check(len(imp) == 69 and imp["rank"].is_unique, f"s{s} {tag}: 69 belgi, noyob reyting")

# 3) mahsulotlar mavjudligi va bo'linish manifesti
for s in SEEDS:
    for part in ("train", "validation", "test"):
        check((SPLITS / f"split_{s}_{part}.parquet").exists(), f"split_{s}_{part}.parquet mavjud")
check((SPLITS / "split_manifest.json").exists(), "split_manifest.json mavjud")

# 4) natija fayllari heshlari va ko'chirish
if problems:
    raise SystemExit(f"\nAUDIT O'TMADI: {len(problems)} ta muammo. Ko'chirish bajarilmadi.")
if VER.exists():
    raise SystemExit(f"{VER} allaqachon mavjud, o'chirmasdan to'xtatildi")
shutil.copytree(RUN, VER)
rows = []
for p in sorted(VER.rglob("*")):
    if p.is_file():
        rows.append({"relative_path": str(p.relative_to(VER)).replace("\\", "/"), "bytes": p.stat().st_size,
                     "sha256": sha256_of(p)})
pd.DataFrame(rows).to_csv(VER / "MANIFEST_SHA256.csv", index=False, encoding="utf-8-sig")
print(f"\n[Bajarildi] audit o'tdi, {len(rows)} ta fayl run_003_verified ga ko'chirildi")
print(f"Manifest: {VER / 'MANIFEST_SHA256.csv'}")