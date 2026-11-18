import csv, hashlib, json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("results/data_audit")
OUT.mkdir(parents=True, exist_ok=True)
ENC = "latin-1"  # dekodlash xatosi bermaydi; g'alati belgilar repr orqali ko'rinadi

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()

files = sorted(RAW.glob("*.csv"))
print(f"Topilgan CSV fayllar: {len(files)}")

inventory, class_rows, missing_rows, dtype_rows = [], [], [], []
full_hashes, feat_label, mins, maxs, headers = [], [], [], [], {}
t0 = time.time()

for path in files:
    print(f"\nTahlil: {path.name}", flush=True)
    with open(path, newline="", encoding=ENC) as f:
        raw_header = next(csv.reader(f))
    label_cols = [c for c in raw_header if c.strip().lower() == "label"]
    if len(label_cols) != 1:
        raise SystemExit(f"{path.name}: yorliq ustuni aniq bitta emas: {label_cols}")
    label_col = label_cols[0]
    dup_names = sorted({c for c in raw_header if raw_header.count(c) > 1})
    headers[path.name] = tuple(raw_header)

    df = pd.read_csv(path, encoding=ENC, low_memory=False)
    feats = [c for c in df.columns if c != label_col]
    non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    num = df[feats].apply(pd.to_numeric, errors="coerce")
    nan_orig = df[feats].isna().sum()
    inf_cnt = np.isinf(num).sum()
    coerced = num.isna().sum() - nan_orig

    for c in feats:
        missing_rows.append({"file": path.name, "column": c, "nan": int(nan_orig[c]),
                             "inf": int(inf_cnt[c]), "non_numeric_coerced": int(coerced[c])})
        dtype_rows.append({"file": path.name, "column": c, "dtype": str(df[c].dtype)})
    dtype_rows.append({"file": path.name, "column": label_col, "dtype": str(df[label_col].dtype)})

    for lab, n in df[label_col].value_counts(dropna=False).items():
        class_rows.append({"file": path.name, "label_repr": repr(lab), "count": int(n)})

    h_full = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h_feat = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
    intra = int(len(h_full) - len(np.unique(h_full)))
    full_hashes.append(h_full)
    feat_label.append(pd.DataFrame({"h": h_feat, "y": df[label_col].astype(str).to_numpy()}))

    clean = num.replace([np.inf, -np.inf], np.nan)
    mins.append(clean.min())
    maxs.append(clean.max())

    inventory.append({
        "file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_of(path),
        "rows": len(df), "columns": len(raw_header),
        "label_column_repr": repr(label_col), "n_classes": int(df[label_col].nunique()),
        "duplicate_header_names": "; ".join(repr(x) for x in dup_names),
        "non_numeric_feature_columns": "; ".join(non_numeric),
        "nan_cells": int(nan_orig.sum()), "inf_cells": int(inf_cnt.sum()),
        "intra_file_duplicate_rows": intra,
    })
    print(f"  qatorlar={len(df):,} ustunlar={len(raw_header)} yorliq={label_col!r} "
          f"sinflar={df[label_col].nunique()} NaN={int(nan_orig.sum()):,} "
          f"inf={int(inf_cnt.sum()):,} dublikat={intra:,}", flush=True)
    del df, num, clean
    gc.collect()

inv = pd.DataFrame(inventory)
cls = pd.DataFrame(class_rows)
inv.to_csv(OUT / "raw_file_inventory.csv", index=False, encoding="utf-8-sig")
cls.to_csv(OUT / "raw_class_distribution.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(missing_rows).to_csv(OUT / "raw_missing_values.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(dtype_rows).to_csv(OUT / "raw_data_types.csv", index=False, encoding="utf-8-sig")

all_full = np.concatenate(full_hashes)
total_rows = int(len(all_full))
excess_all = int(total_rows - len(np.unique(all_full)))
intra_total = int(inv["intra_file_duplicate_rows"].sum())
big = pd.concat(feat_label, ignore_index=True)
nun = big.groupby("h")["y"].nunique()
conflict_groups = nun[nun > 1]
conflict_rows = int(big["h"].isin(conflict_groups.index).sum())

dup_report = {
    "definition": "excess copies of identical full rows (including label)",
    "intra_file_excess_rows": {r["file"]: int(r["intra_file_duplicate_rows"]) for r in inventory},
    "intra_file_total": intra_total,
    "all_files_total_excess_rows": excess_all,
    "cross_file_excess_rows": excess_all - intra_total,
    "feature_groups_with_conflicting_labels": int(len(conflict_groups)),
    "rows_in_conflicting_groups": conflict_rows,
}
with open(OUT / "raw_duplicate_report.json", "w", encoding="utf-8") as f:
    json.dump(dup_report, f, indent=2, ensure_ascii=False)

gmin = pd.concat(mins, axis=1).min(axis=1)
gmax = pd.concat(maxs, axis=1).max(axis=1)
all_nan_cols = [c for c in gmin.index if pd.isna(gmin[c])]
const_cols = [c for c in gmin.index if pd.notna(gmin[c]) and gmin[c] == gmax[c]]
totals = cls.groupby("label_repr")["count"].sum().sort_values(ascending=False)

summary = {
    "n_files": len(files),
    "total_rows": total_rows,
    "columns_per_file": sorted(set(int(x) for x in inv["columns"])),
    "same_header_in_all_files": len(set(headers.values())) == 1,
    "label_column_repr": sorted(set(inv["label_column_repr"])),
    "n_classes_total": int(len(totals)),
    "class_counts_total": {k: int(v) for k, v in totals.items()},
    "nan_cells_total": int(inv["nan_cells"].sum()),
    "inf_cells_total": int(inv["inf_cells"].sum()),
    "duplicate_header_names": sorted(set("; ".join(inv["duplicate_header_names"]).split("; ")) - {""}),
    "non_numeric_feature_columns": sorted(set("; ".join(inv["non_numeric_feature_columns"]).split("; ")) - {""}),
    "constant_columns": const_cols,
    "all_nan_columns": all_nan_cols,
    "duplicates": dup_report,
    "elapsed_seconds": round(time.time() - t0, 1),
}
with open(OUT / "raw_audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n===== AUDIT XULOSASI =====")
print(json.dumps(summary, indent=2, ensure_ascii=True))