import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

PARQUET = Path("data/processed/cicids2017_clean.parquet")
AUDIT = Path("results/data_audit")
OUT = Path("results/splits")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024, 3407, 7777]
TRAIN_F, VAL_F, TEST_F = 0.70, 0.15, 0.15
EXP_ROWS = 2520798
RARE = ["Heartbleed", "Infiltration", "Web Attack - Sql Injection"]
t0 = time.time()


def check(cond, msg):
    if not cond:
        raise SystemExit("TEKSHIRUV O'TMADI: " + msg)
    print("  OK: " + msg, flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="int64").tobytes()).hexdigest().upper()


def make_split(y, seed):
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=TEST_F, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=VAL_F / (TRAIN_F + VAL_F),
                              stratify=y[tv], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


print("8-bosqich: bo'linishlar yaratilmoqda", flush=True)
with open(AUDIT / "clean_data_summary.json", encoding="utf-8") as f:
    summary = json.load(f)
with open(AUDIT / "label_map.json", encoding="utf-8") as f:
    label_map = json.load(f)
id_to_name = {v: k for k, v in label_map.items()}
n_classes = len(label_map)

check(sha256_of(PARQUET) == summary["parquet_sha256"], "Parquet SHA256 tozalash hisobotiga mos")
df = pd.read_parquet(PARQUET, columns=["row_id", "Label", "label_id"])
check(len(df) == EXP_ROWS, f"qatorlar soni {len(df):,}")
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
          f"seed {seed}: ikkinchi ishga tushirishda bo'linish aynan bir xil")

    check(len(np.intersect1d(tr, va)) == 0, "train va validation kesishmaydi")
    check(len(np.intersect1d(tr, te)) == 0, "train va test kesishmaydi")
    check(len(np.intersect1d(va, te)) == 0, "validation va test kesishmaydi")
    check(len(tr) + len(va) + len(te) == N, "har bir qator aynan bitta qismga tushgan")
    check(abs(len(va) - VAL_F * N) <= 2 and abs(len(te) - TEST_F * N) <= 2,
          f"hajmlar: train={len(tr):,} val={len(va):,} test={len(te):,}")

    parts = {"train": tr, "validation": va, "test": te}
    fracs = {"train": TRAIN_F, "validation": VAL_F, "test": TEST_F}
    mins = {}
    for name, part in parts.items():
        cnt = np.bincount(y[part], minlength=n_classes)
        check(int(cnt.min()) >= 1, f"{name}: barcha 15 sinf mavjud (eng kam sinf: {int(cnt.min())})")
        dev = np.abs(cnt - total_by_class * fracs[name]).max()
        check(dev <= 2, f"{name}: sinf ulushlari kutilganga yaqin (eng katta og'ish {dev:.2f} qator)")
        mins[name] = int(cnt.min())
        for c in range(n_classes):
            dist_rows.append({"seed": seed, "part": name, "label_id": c, "label": id_to_name[c],
                              "count": int(cnt[c]),
                              "share_in_part_pct": round(cnt[c] / len(part) * 100, 4)})
        pd.DataFrame({"row_id": ids[part]}).to_parquet(
            OUT / f"split_{seed}_{name}.parquet", index=False)

    rare_txt = " | ".join(
        f"{r}: " + "/".join(str(int(np.bincount(y[parts[p]], minlength=n_classes)[label_map[r]]))
                            for p in ("train", "validation", "test")) for r in RARE)
    print("  Kam sinflar (train/val/test): " + rare_txt)
    manifest_splits.append({
        "seed": seed, "train_rows": int(len(tr)), "validation_rows": int(len(va)),
        "test_rows": int(len(te)),
        "train_row_id_sha256": arr_hash(ids[tr]), "validation_row_id_sha256": arr_hash(ids[va]),
        "test_row_id_sha256": arr_hash(ids[te]),
        "min_class_count": mins,
    })

pd.DataFrame(dist_rows).to_csv(OUT / "split_class_distribution.csv", index=False, encoding="utf-8-sig")
manifest = {
    "parquet_file": str(PARQUET), "parquet_sha256": summary["parquet_sha256"],
    "n_rows": N, "seeds": SEEDS,
    "fractions": {"train": TRAIN_F, "validation": VAL_F, "test": TEST_F},
    "method": "two-step stratified train_test_split on label_id (test first, then validation from the rest)",
    "files_contain": "row_id values referring to the row_id column of the clean parquet",
    "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    "splits": manifest_splits,
    "elapsed_seconds": round(time.time() - t0, 1),
}
with open(OUT / "split_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("\n[Bajarildi] 8-bosqich yakunlandi, barcha tekshiruvlar o'tdi")
print(f"Bo'linishlar: {len(SEEDS)} ta, har biri train/validation/test")
print(f"Vaqt: {time.time() - t0:.1f} s")