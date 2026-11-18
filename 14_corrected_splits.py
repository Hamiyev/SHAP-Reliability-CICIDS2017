import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

T0 = time.time()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="int64").tobytes()).hexdigest().upper()


def check(cond, msg):
    if not cond:
        raise SystemExit("TEKSHIRUV O'TMADI: " + msg)
    print("  OK: " + msg, flush=True)


PROTO = Path("configs/experiment_protocol_v1.1.yaml")
raw = Path("configs/protocol_v1.1.sha256").read_bytes().decode("latin-1")
expected = "".join(ch for ch in raw if ch in "0123456789abcdefABCDEF").upper()
if sha256_of(PROTO) != expected:
    raise SystemExit("PROTOKOL v1.1 O'ZGARGAN: hash mos emas")
with open(PROTO, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PARQUET = Path(cfg["data"]["parquet"])
OUT = Path("results/splits_corrected")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = cfg["splits"]["seeds"]
TRAIN_F, VAL_F, TEST_F = (cfg["splits"]["train_fraction"], cfg["splits"]["validation_fraction"],
                          cfg["splits"]["test_fraction"])
with open(cfg["data"]["label_map"], encoding="utf-8") as f:
    label_map = json.load(f)
id_to_name = {v: k for k, v in label_map.items()}
n_classes = len(label_map)
RARE = cfg["evaluation"]["ultra_rare_classes"]


def make_split(y, seed):
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=TEST_F, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=VAL_F / (TRAIN_F + VAL_F), stratify=y[tv], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


print("Bo'linishlar (protokol v1.1) yaratilmoqda", flush=True)
check(sha256_of(PARQUET) == cfg["data"]["parquet_sha256"], "Parquet SHA256 protokolga mos")
df = pd.read_parquet(PARQUET, columns=["row_id", "Label", "label_id"])
check(len(df) == cfg["data"]["n_rows"], f"qatorlar soni {len(df):,}")
check(n_classes == cfg["data"]["n_classes"] and df["Label"].nunique() == n_classes, f"{n_classes} sinf")
check(df["row_id"].is_unique, "row_id noyob")
ids = df["row_id"].to_numpy(dtype="int64")
y = df["label_id"].to_numpy()
N = len(df)
total_by_class = np.bincount(y, minlength=n_classes)

dist_rows, manifest_splits = [], []
for seed in SEEDS:
    print(f"\nSeed {seed}:", flush=True)
    tr, va, te = make_split(y, seed)
    tr2, va2, te2 = make_split(y, seed)
    check(np.array_equal(tr, tr2) and np.array_equal(va, va2) and np.array_equal(te, te2),
          "qayta ishga tushirishda bo'linish aynan bir xil")
    check(len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(tr, te)) == 0
          and len(np.intersect1d(va, te)) == 0, "train, validation va test kesishmaydi")
    check(len(tr) + len(va) + len(te) == N, "har bir qator aynan bitta qismga tushgan")
    check(abs(len(va) - VAL_F * N) <= 2 and abs(len(te) - TEST_F * N) <= 2,
          f"hajmlar: train={len(tr):,} val={len(va):,} test={len(te):,}")
    parts = {"train": tr, "validation": va, "test": te}
    fracs = {"train": TRAIN_F, "validation": VAL_F, "test": TEST_F}
    mins = {}
    for name, part in parts.items():
        cnt = np.bincount(y[part], minlength=n_classes)
        check(int(cnt.min()) >= 1, f"{name}: barcha {n_classes} sinf mavjud (eng kam: {int(cnt.min())})")
        dev = np.abs(cnt - total_by_class * fracs[name]).max()
        check(dev <= 2, f"{name}: sinf ulushlari kutilganga yaqin (eng katta og'ish {dev:.2f})")
        mins[name] = int(cnt.min())
        for c in range(n_classes):
            dist_rows.append({"seed": seed, "part": name, "label_id": c, "label": id_to_name[c],
                              "count": int(cnt[c]), "share_in_part_pct": round(cnt[c] / len(part) * 100, 4)})
        pd.DataFrame({"row_id": ids[part]}).to_parquet(OUT / f"split_{seed}_{name}.parquet", index=False)
    rare_txt = " | ".join(
        f"{r}: " + "/".join(str(int(np.bincount(y[parts[p]], minlength=n_classes)[label_map[r]]))
                            for p in ("train", "validation", "test")) for r in RARE)
    print("  Kam sinflar (train/val/test): " + rare_txt)
    manifest_splits.append({"seed": seed, "train_rows": int(len(tr)), "validation_rows": int(len(va)),
                            "test_rows": int(len(te)), "train_row_id_sha256": arr_hash(ids[tr]),
                            "validation_row_id_sha256": arr_hash(ids[va]),
                            "test_row_id_sha256": arr_hash(ids[te]), "min_class_count": mins})

pd.DataFrame(dist_rows).to_csv(OUT / "split_class_distribution.csv", index=False, encoding="utf-8-sig")
manifest = {"parquet_file": str(PARQUET), "parquet_sha256": cfg["data"]["parquet_sha256"], "n_rows": N,
            "seeds": SEEDS, "fractions": {"train": TRAIN_F, "validation": VAL_F, "test": TEST_F},
            "method": cfg["splits"]["method"], "protocol_version": cfg["project"]["protocol_version"],
            "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "splits": manifest_splits, "elapsed_seconds": round(time.time() - T0, 1)}
with open(OUT / "split_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("\n[Bajarildi] bo'linishlar yaratildi, barcha tekshiruvlar o'tdi")
print(f"Vaqt: {time.time() - T0:.1f} s")