import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
AUDIT = Path("results/data_audit")
PROCESSED = Path("data/processed")
AUDIT.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

ENC = "latin-1"
RAW_LABEL = " Label"
LABEL_MAP = {
    "Web Attack \u00ef\u00bf\u00bd Brute Force": "Web Attack - Brute Force",
    "Web Attack \u00ef\u00bf\u00bd XSS": "Web Attack - XSS",
    "Web Attack \u00ef\u00bf\u00bd Sql Injection": "Web Attack - Sql Injection",
}
CONSTANT = {
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
}
DROP_DUP_COL = " Fwd Header Length.1"

# Auditda va ta'sir tekshiruvida o'lchangan kutilgan qiymatlar
EXP_RAW = {
    "BENIGN": 2273097, "DoS Hulk": 231073, "PortScan": 158930, "DDoS": 128027,
    "DoS GoldenEye": 10293, "FTP-Patator": 7938, "SSH-Patator": 5897,
    "DoS slowloris": 5796, "DoS Slowhttptest": 5499, "Bot": 1966,
    "Web Attack - Brute Force": 1507, "Web Attack - XSS": 652, "Infiltration": 36,
    "Web Attack - Sql Injection": 21, "Heartbleed": 11,
}
EXP_FINAL = {
    "BENIGN": 2095057, "DoS Hulk": 172846, "PortScan": 90694, "DDoS": 128014,
    "DoS GoldenEye": 10286, "FTP-Patator": 5931, "SSH-Patator": 3219,
    "DoS slowloris": 5385, "DoS Slowhttptest": 5228, "Bot": 1948,
    "Web Attack - Brute Force": 1470, "Web Attack - XSS": 652, "Infiltration": 36,
    "Web Attack - Sql Injection": 21, "Heartbleed": 11,
}
EXP_ROWS_RAW, EXP_DUP, EXP_AFTER_DEDUP, EXP_BAD, EXP_ROWS_FINAL = 2830743, 308381, 2522362, 1564, 2520798

t0 = time.time()
log = []


def check(cond, msg):
    if not cond:
        raise SystemExit("TEKSHIRUV O'TMADI: " + msg)
    print("  OK: " + msg, flush=True)


def log_step(step, desc, rb, ra, cb, ca, details=""):
    log.append({"step": step, "description": desc, "rows_before": rb, "rows_after": ra,
                "rows_removed": rb - ra, "columns_before": cb, "columns_after": ca,
                "details": details})


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


print("7-bosqich: tozalash boshlandi", flush=True)

# 1) O'qish
files = sorted(RAW.glob("*.csv"))
check(len(files) == 8, "8 ta CSV fayl topildi")
frames = [pd.read_csv(p, encoding=ENC, low_memory=False) for p in files]
lens = [len(f) for f in frames]
df = pd.concat(frames, ignore_index=True)
del frames
raw_cols = list(df.columns)
n_raw = len(df)
check(n_raw == EXP_ROWS_RAW and len(raw_cols) == 79, f"xom hajm: {n_raw:,} qator, {len(raw_cols)} ustun")
log_step(1, "8 CSV fayl o'qildi va birlashtirildi", n_raw, n_raw, 79, 79, "fayl nomlari tartibida, latin-1")

# barqaror identifikator: xom birlashtirilgan jadvaldagi o'rni
df["row_id"] = np.arange(n_raw, dtype="int64")
df["source_file"] = pd.Categorical.from_codes(
    np.repeat(np.arange(len(files)), lens), categories=[p.name for p in files])

# 2) Yorliqlar normalizatsiyasi
df["Label"] = df[RAW_LABEL].astype(str).str.strip().replace(LABEL_MAP)
check(df["Label"].value_counts().to_dict() == EXP_RAW, "xom sinflar soni auditga mos (15 sinf)")
log_step(2, "Yorliqlar normalizatsiya qilindi", n_raw, n_raw, 79, 79,
         "3 ta Web Attack yorlig'idagi buzilgan belgi tuzatildi; boshqalari o'zgarmadi")

# 3) To'liq dublikatlar (birinchisi saqlanadi)
dup = df[raw_cols].duplicated(keep="first").to_numpy()
check(int(dup.sum()) == EXP_DUP, f"ortiqcha dublikatlar: {int(dup.sum()):,}")
check(n_raw - int(dup.sum()) == EXP_AFTER_DEDUP, f"dublikatsiz qatorlar: {n_raw - int(dup.sum()):,}")
log_step(3, "To'liq dublikatlar o'chirildi", n_raw, n_raw - int(dup.sum()), 79, 79,
         "belgilar va yorliq bir xil bo'lgan qatorlar; birinchi nusxa saqlandi")

# 4) NaN / cheksiz qiymatli qatorlar
feats_raw = [c for c in raw_cols if c != RAW_LABEL]
check(len(feats_raw) == 78, "xom belgilar soni 78")
raw_dtypes = df[feats_raw].dtypes.astype(str).to_dict()
Xv = df[feats_raw].to_numpy(dtype="float64")
bad = ~np.isfinite(Xv).all(axis=1)
del Xv
bad_after = (~dup) & bad
check(int(bad_after.sum()) == EXP_BAD, f"dedupdan keyin NaN/inf qatorlar: {int(bad_after.sum()):,}")
keep = (~dup) & (~bad)
check(int(keep.sum()) == EXP_ROWS_FINAL, f"yakuniy qatorlar: {int(keep.sum()):,}")
log_step(4, "NaN/inf qatorlar olib tashlandi", n_raw - int(dup.sum()), int(keep.sum()), 79, 79,
         "faqat Flow Bytes/s va Flow Packets/s ustunlarida")

dup_by = df.loc[dup, "Label"].value_counts()
bad_by = df.loc[bad_after, "Label"].value_counts()
out = df.loc[keep].reset_index(drop=True)
del df, dup, bad, bad_after, keep

final_counts = out["Label"].value_counts().to_dict()
check(final_counts == EXP_FINAL, "yakuniy sinflar soni kutilganga mos")

# 5) Ustunlar
check(out[" Fwd Header Length"].equals(out[DROP_DUP_COL]), "Fwd Header Length nusxasi asl ustunga teng")
name_map = {c: c.strip() for c in feats_raw}
check(len(set(name_map.values())) == 78, "tozalangan ustun nomlari takrorlanmaydi")
drop_const = [c for c in feats_raw if name_map[c] in CONSTANT]
check(len(drop_const) == 8, "8 ta doimiy ustun topildi")
check(all(out[c].nunique(dropna=False) == 1 for c in drop_const), "doimiy ustunlar yakuniy ma'lumotda ham doimiy")
kept = [c for c in feats_raw if c not in drop_const and c != DROP_DUP_COL]
check(len(kept) == 69, "qolgan belgilar soni 69")
mn, mx = out[kept].min(), out[kept].max()
check(not bool((mn == mx).any()), "qolgan belgilar orasida doimiylari yo'q")
log_step(5, "Doimiy ustunlar va nusxa ustun olib tashlandi", len(out), len(out), 78, 69,
         "8 doimiy ustun + Fwd Header Length.1 (asl ustunning aynan nusxasi)")

feat_names = [name_map[c] for c in kept]
classes = sorted(EXP_FINAL)
label_map = {c: i for i, c in enumerate(classes)}

final = out[["row_id", "source_file", "Label"]].copy()
final["label_id"] = final["Label"].map(label_map).astype("int8")
feat_df = out[kept].astype("float64")
feat_df.columns = feat_names
final = pd.concat([final, feat_df], axis=1)
del out, feat_df

check(final.shape == (EXP_ROWS_FINAL, 73), f"yakuniy jadval shakli {final.shape}")
check(bool(np.isfinite(final[feat_names].to_numpy()).all()), "belgilarda NaN/inf yo'q")
check(final["row_id"].is_unique, "row_id noyob")
check(final["Label"].nunique() == 15, "15 sinf")
log_step(6, "Belgi nomlari tozalandi, float64 ga o'tkazildi, label_id qo'shildi",
         len(final), len(final), 69, 69, "bo'sh joylar olib tashlandi; label_id sinflar alifbo tartibida")

# 6) Qarama-qarshi yorliqlar (qoldiriladi, faqat qayd etiladi)
h = pd.util.hash_pandas_object(final[feat_names], index=False).to_numpy()
k = pd.DataFrame({"h": h, "y": final["Label"].to_numpy(), "row_id": final["row_id"].to_numpy()})
nun = k.groupby("h")["y"].nunique()
conf_groups = nun[nun > 1].index
kc = k[k["h"].isin(conf_groups)].copy()
kc["group_hash"] = kc["h"].astype(str)
check(len(conf_groups) <= 698, f"qarama-qarshi guruhlar: {len(conf_groups):,}, qatorlar: {len(kc):,}")
log_step(7, "Qarama-qarshi yorliqli qatorlar qayd etildi (o'chirilmadi)", len(final), len(final), 69, 69,
         f"{len(conf_groups)} guruh, {len(kc)} qator; row_id lar conflict_rows.csv da")

# 7) Parquet: avval vaqtinchalik fayl, tekshiruv, keyin asl nom
target = PROCESSED / "cicids2017_clean.parquet"
tmp = PROCESSED / "cicids2017_clean.parquet.tmp"
final.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
chk = pd.read_parquet(tmp)
check(chk.shape == final.shape, "Parquet qayta o'qildi: shakl mos")
check(list(chk.columns) == list(final.columns), "Parquet ustunlari mos")
check(chk["Label"].value_counts().to_dict() == EXP_FINAL, "Parquetdagi sinflar soni mos")
check(chk[feat_names].equals(final[feat_names]), "Parquetdagi belgilar qiymatlari aynan mos")
del chk
os.replace(tmp, target)
parquet_sha = sha256_of(target)

# 8) Hisobot fayllari
pd.DataFrame(log).to_csv(AUDIT / "cleaning_log.csv", index=False, encoding="utf-8-sig")

rows = []
for c in classes:
    raw_n, d, b, f = EXP_RAW[c], int(dup_by.get(c, 0)), int(bad_by.get(c, 0)), final_counts[c]
    assert raw_n - d - b == f, f"{c}: sonlar mos kelmadi"
    rows.append({"label": c, "label_id": label_map[c], "raw_rows": raw_n, "duplicates_removed": d,
                 "nan_inf_removed": b, "final_rows": f,
                 "pct_lost": round((1 - f / raw_n) * 100, 2),
                 "final_share_pct": round(f / EXP_ROWS_FINAL * 100, 4)})
pd.DataFrame(rows).sort_values("final_rows", ascending=False).to_csv(
    AUDIT / "clean_class_distribution.csv", index=False, encoding="utf-8-sig")

inv = []
for pos, c in enumerate(feats_raw):
    if c in drop_const:
        status, reason = "dropped", "constant value in all raw rows"
    elif c == DROP_DUP_COL:
        status, reason = "dropped", "identical copy of 'Fwd Header Length'"
    else:
        status, reason = "kept", ""
    inv.append({"raw_position": pos, "raw_name": c, "clean_name": name_map[c],
                "raw_dtype": raw_dtypes[c], "status": status, "reason": reason})
pd.DataFrame(inv).to_csv(AUDIT / "feature_inventory.csv", index=False, encoding="utf-8-sig")

kc[["row_id", "y", "group_hash"]].rename(columns={"y": "label"}).sort_values(["group_hash", "row_id"]).to_csv(
    AUDIT / "conflict_rows.csv", index=False)
with open(AUDIT / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2, ensure_ascii=False)

summary = {
    "raw_rows": EXP_ROWS_RAW, "duplicates_removed": EXP_DUP, "nan_inf_rows_removed": EXP_BAD,
    "final_rows": int(len(final)), "n_features": len(feat_names), "n_classes": len(classes),
    "feature_names": feat_names, "label_map": label_map,
    "final_class_counts": {k_: int(v) for k_, v in final_counts.items()},
    "conflicting_label_groups": int(len(conf_groups)), "conflicting_label_rows": int(len(kc)),
    "parquet_file": str(target), "parquet_size_bytes": target.stat().st_size,
    "parquet_sha256": parquet_sha, "pandas": pd.__version__, "numpy": np.__version__,
    "elapsed_seconds": round(time.time() - t0, 1),
}
with open(AUDIT / "clean_data_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n[Bajarildi] 7-bosqich yakunlandi, barcha tekshiruvlar o'tdi")
print(f"Qatorlar: {len(final):,} | belgilar: {len(feat_names)} | sinflar: {len(classes)}")
print(f"Qarama-qarshi guruhlar: {len(conf_groups):,} ({len(kc):,} qator)")
print(f"Parquet: {target} ({target.stat().st_size / 1e6:.0f} MB), SHA256: {parquet_sha}")
print(f"Vaqt: {time.time() - t0:.1f} s")