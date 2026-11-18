import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/raw_corrected")
V10 = Path("results/data_audit")
OUT = Path("results/data_audit_corrected")
PROC = Path("data/processed_corrected")
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
ENC = "latin-1"
ULTRA_RULE = 100
T0 = time.time()

RENAME = {
    "Dst Port": "Destination Port", "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets", "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets", "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length", "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size", "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk", "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate", "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk", "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward", "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Seg Size Min": "min_seg_size_forward",
}
META = ["id", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Attempted Category"]
inv = pd.read_csv(V10 / "feature_inventory.csv", encoding="utf-8-sig").sort_values("raw_position")
TARGET = [n for n in inv["clean_name"] if n != "Fwd Header Length.1"]
assert len(TARGET) == 77, len(TARGET)
log = []


def log_step(step, desc, before, after, details=""):
    log.append({"step": step, "description": desc, "rows_before": before, "rows_after": after,
                "rows_removed": before - after, "details": details})
    print(f"[{step}] {desc}: {before:,} -> {after:,} (olib tashlandi {before - after:,}) {details}", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


files = sorted(SRC.glob("*.csv"))
if len(files) != 5:
    raise SystemExit(f"5 ta CSV kutilgan, {len(files)} ta topildi")
frames = []
for p in files:
    d = pd.read_csv(p, encoding=ENC, low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    d["source_file"] = p.name
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
del frames
df["row_id"] = np.arange(len(df), dtype="int64")
n_raw = len(df)
log_step(1, "5 CSV o'qildi va birlashtirildi", n_raw, n_raw, f"ustunlar: {df.shape[1]}")

df = df.rename(columns=RENAME)
missing = [c for c in TARGET if c not in df.columns]
if missing:
    raise SystemExit(f"BELGILAR TOPILMADI: {missing}")
cols = list(df.columns)
bad_names = [c for c in TARGET if cols.count(c) != 1]
if bad_names:
    raise SystemExit(f"TAKRORIY USTUN NOMLARI: {bad_names}")

lab_orig = df["Label"].astype(str).str.strip()
att = lab_orig.str.lower().str.endswith("- attempted")
acat = pd.to_numeric(df["Attempted Category"], errors="coerce")
mismatch = int((att != (acat != -1)).sum())
lab = lab_orig.where(~att, "BENIGN")
raw_counts = lab_orig.value_counts()
log_step(2, "Yorliqlar normalizatsiya qilindi ('Attempted' -> BENIGN)", n_raw, n_raw,
         f"attempted: {int(att.sum()):,}; Attempted Category bilan nomuvofiqlik: {mismatch}")

nonnum = [c for c in TARGET if not pd.api.types.is_numeric_dtype(df[c])]
if nonnum:
    print("OGOHLANTIRISH: raqamli bo'lmagan ustunlar (NaN ga aylantiriladi):", nonnum)
X = df[TARGET].apply(pd.to_numeric, errors="coerce").astype("float64")

key = X.copy()
key["__label"] = lab.to_numpy()
dup = key.duplicated(keep="first").to_numpy()
del key
n1 = n_raw - int(dup.sum())
log_step(3, "To'liq dublikatlar o'chirildi (belgilar + yorliq)", n_raw, n1)

Xv = X.to_numpy()
bad = ~np.isfinite(Xv).all(axis=1)
del Xv
keep = (~dup) & (~bad)
n2 = int(keep.sum())
log_step(4, "NaN/inf qatorlar olib tashlandi", n1, n2, f"dedupdan keyingi NaN/inf qatorlar: {int(((~dup) & bad).sum()):,}")

X = X.loc[keep].reset_index(drop=True)
meta = df.loc[keep, ["row_id", "source_file"] + META].copy()
meta["Label_original"] = lab_orig[keep].to_numpy()
meta = meta.reset_index(drop=True)
lab = lab[keep].reset_index(drop=True)
row_id = meta["row_id"].to_numpy()
src = meta["source_file"].to_numpy()
del df

const = [c for c in TARGET if X[c].nunique(dropna=False) == 1]
X = X.drop(columns=const)
seen, dup_cols = {}, []
for c in list(X.columns):
    k = hashlib.md5(X[c].to_numpy().tobytes()).hexdigest()
    if k in seen and np.array_equal(X[seen[k]].to_numpy(), X[c].to_numpy()):
        dup_cols.append((c, seen[k]))
    else:
        seen.setdefault(k, c)
X = X.drop(columns=[c for c, _ in dup_cols])
feats = list(X.columns)
log_step(5, "Doimiy va bir xil nusxa ustunlar olib tashlandi", n2, n2,
         f"doimiy: {len(const)}; nusxa: {len(dup_cols)}; qolgan belgilar: {len(feats)}")

classes = sorted(lab.unique())
label_map = {c: i for i, c in enumerate(classes)}
final = pd.DataFrame({"row_id": row_id, "source_file": pd.Categorical(src), "Label": lab.to_numpy(),
                      "label_id": lab.map(label_map).astype("int8").to_numpy()})
final = pd.concat([final, X], axis=1)
assert len(final) == n2 and final["row_id"].is_unique
assert bool(np.isfinite(final[feats].to_numpy()).all())
pq = PROC / "cicids2017_improved_clean.parquet"
final.to_parquet(pq, engine="pyarrow", compression="snappy", index=False)
meta.to_parquet(PROC / "cicids2017_improved_meta.parquet", index=False)
sha = sha256_of(pq)

fc = lab.value_counts()
cls = pd.DataFrame({"label": classes, "label_id": [label_map[c] for c in classes]})
cls["raw_rows_original_label"] = cls["label"].map(raw_counts).fillna(0).astype(int)
cls["final_rows"] = cls["label"].map(fc).astype(int)
cls["ultra_rare"] = cls["final_rows"] < ULTRA_RULE
cls.to_csv(OUT / "clean_class_distribution.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(log).to_csv(OUT / "cleaning_log.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"feature": feats}).to_csv(OUT / "feature_list.csv", index=False)
with open(OUT / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2)
summary = {"raw_rows": n_raw, "attempted_rows_to_benign": int(att.sum()), "attempted_category_mismatch": mismatch,
           "duplicates_removed": int(dup.sum()), "nan_inf_rows_removed": int(((~dup) & bad).sum()),
           "final_rows": n2, "n_features": len(feats), "feature_names": feats, "constant_dropped": const,
           "duplicate_columns_dropped": [list(t) for t in dup_cols], "classes": classes,
           "final_class_counts": {c: int(fc[c]) for c in classes},
           "ultra_rare_classes": [c for c in classes if int(fc[c]) < ULTRA_RULE],
           "parquet": str(pq), "parquet_sha256": sha, "elapsed_seconds": round(time.time() - T0, 1)}
with open(OUT / "clean_data_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

pd.set_option("display.width", 220)
print("\n===== SINFLAR =====")
print(cls.to_string(index=False))
print(f"\nDoimiy ustunlar ({len(const)}): {const}")
print(f"Bir xil nusxa ustunlar: {dup_cols}")
print(f"Juda kichik sinflar (<{ULTRA_RULE}): {summary['ultra_rare_classes']}")
print(f"\n[Bajarildi] yakuniy: {n2:,} qator, {len(feats)} belgi, {len(classes)} sinf, vaqt {(time.time() - T0) / 60:.1f} min")
print(f"Parquet SHA256: {sha}")