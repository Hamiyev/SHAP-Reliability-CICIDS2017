import hashlib
import sys
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw_corrected")
AUDIT = Path("results/data_audit")
OUT = Path("results/data_audit_corrected")
OUT.mkdir(parents=True, exist_ok=True)
ENC = "latin-1"
V10 = set(pd.read_csv(AUDIT / "feature_inventory.csv", encoding="utf-8-sig")["clean_name"])


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def ascii_safe(s):
    return s.astype(str).str.strip().str.encode("ascii", "replace").str.decode("ascii")


files = sorted(p for p in SRC.rglob("*") if p.is_file())
print(f"Papka: {SRC} | fayllar: {len(files)}")
for p in files:
    print(f"  {p.relative_to(SRC)}  {p.stat().st_size / 1e6:.1f} MB")
csvs = [p for p in files if p.suffix.lower() == ".csv"]
if not csvs:
    raise SystemExit("CSV fayl topilmadi")

rows, hashes, labels, cols_by_file = [], [], pd.Series(dtype="float64"), {}
for p in csvs:
    df = pd.read_csv(p, encoding=ENC, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    cols_by_file[p.name] = list(df.columns)
    lab = [c for c in df.columns if "label" in c.lower()]
    lc = lab[0] if lab else None
    vc = ascii_safe(df[lc]).value_counts() if lc else pd.Series(dtype="float64")
    labels = labels.add(vc, fill_value=0)
    rows.append({"file": p.name, "rows": len(df), "columns": df.shape[1], "label_col": lc,
                 "distinct_labels": len(vc), "nan_labels": int(df[lc].isna().sum()) if lc else None})
    hashes.append({"file": str(p.relative_to(SRC)), "bytes": p.stat().st_size, "sha256": sha256_of(p)})
    print(f"O'qildi: {p.name} ({len(df):,} qator)", flush=True)
    del df

pd.set_option("display.width", 250)
info = pd.DataFrame(rows)
print("\n===== 1) FAYLLAR =====")
print(info.to_string(index=False))
print(f"Jami qatorlar: {info['rows'].sum():,}")
print("\n===== 2) YORLIQLAR (barcha fayllar bo'yicha) =====")
print(labels.sort_values(ascending=False).astype("int64").to_string())
same = len({tuple(c) for c in cols_by_file.values()}) == 1
allcols = list(dict.fromkeys(c for cs in cols_by_file.values() for c in cs))
print(f"\n===== 3) USTUNLAR ===== (barcha fayllarda bir xil sarlavha: {same}; ustunlar soni: {len(allcols)})")
print("Barcha ustunlar:", " | ".join(allcols))
missing = sorted(V10 - set(allcols))
extra = [c for c in allcols if c not in V10]
print(f"\nv1.0 dagi {len(V10)} ta belgi nomidan shu datasetda aynan topilmaganlari ({len(missing)}):")
print(" | ".join(missing))
print(f"\nBu datasetda bor, v1.0 ro'yxatida yo'q ustunlar ({len(extra)}):")
print(" | ".join(extra))
info.to_csv(OUT / "files_overview.csv", index=False)
pd.DataFrame(hashes).to_csv(OUT / "files_sha256.csv", index=False)
labels.astype("int64").to_csv(OUT / "label_counts.csv", header=["rows"])
print("\n[Bajarildi]")