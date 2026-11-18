import time
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("results/data_audit")
OUT.mkdir(parents=True, exist_ok=True)
ENC = "latin-1"
LABEL = " Label"
LABEL_MAP = {
    "Web Attack \u00ef\u00bf\u00bd Brute Force": "Web Attack - Brute Force",
    "Web Attack \u00ef\u00bf\u00bd XSS": "Web Attack - XSS",
    "Web Attack \u00ef\u00bf\u00bd Sql Injection": "Web Attack - Sql Injection",
}
t0 = time.time()

df = pd.concat(
    [pd.read_csv(p, encoding=ENC, low_memory=False) for p in sorted(RAW.glob("*.csv"))],
    ignore_index=True,
)
feats = [c for c in df.columns if c != LABEL]
y = df[LABEL].astype(str).str.strip().replace(LABEL_MAP)
print(f"Yuklandi: {df.shape}, belgilar: {len(feats)}, sinflar: {y.nunique()}")
print("Fwd Header Length nusxasi bir xilmi:",
      bool(df[" Fwd Header Length"].equals(df[" Fwd Header Length.1"])))

# 1) to'liq dublikatlar (birinchisi saqlanadi)
h_full = pd.util.hash_pandas_object(df, index=False).to_numpy()
dup = pd.Series(h_full).duplicated(keep="first").to_numpy()
print(f"Ortiqcha dublikatlar: {int(dup.sum()):,} (audit: 308,381)")

# 2) NaN / cheksiz qiymatli qatorlar
Xv = df[feats].to_numpy(dtype="float64")
bad = ~np.isfinite(Xv).all(axis=1)
cols = pd.DataFrame({"column": feats, "nan": np.isnan(Xv).sum(axis=0),
                     "inf": np.isinf(Xv).sum(axis=0)})
cols = cols[(cols.nan > 0) | (cols.inf > 0)]
cols.to_csv(OUT / "cleaning_impact_nan_inf_columns.csv", index=False)
del Xv

# 3) sinflar bo'yicha ta'sir
t = pd.DataFrame({"label": y, "dup": dup, "bad": bad})
t["after_dedup"] = ~t["dup"]
t["bad_after_dedup"] = t["after_dedup"] & t["bad"]
t["final"] = t["after_dedup"] & ~t["bad"]
tab = t.groupby("label").agg(
    raw_rows=("dup", "size"), duplicate_excess=("dup", "sum"),
    after_dedup=("after_dedup", "sum"), nan_inf_rows_after_dedup=("bad_after_dedup", "sum"),
    final_rows=("final", "sum"))
tab["pct_lost"] = ((1 - tab["final_rows"] / tab["raw_rows"]) * 100).round(2)
tab = tab.sort_values("raw_rows", ascending=False)
tab.to_csv(OUT / "cleaning_impact_by_class.csv", encoding="utf-8-sig")

# 4) qarama-qarshi yorliqlar (dublikatlar o'chirilgandan keyin)
h_feat = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
k = pd.DataFrame({"h": h_feat, "y": y.to_numpy()})[~dup]
n = k.groupby("h")["y"].nunique()
kc = k[k["h"].isin(n[n > 1].index)].copy()
combo = kc.groupby("h")["y"].agg(lambda s: " | ".join(sorted(set(s))))
kc["combo"] = kc["h"].map(combo)
conf = pd.DataFrame({"groups": combo.value_counts(),
                     "rows": kc.groupby("combo").size()}).sort_values("rows", ascending=False)
conf.to_csv(OUT / "cleaning_impact_conflicts.csv", encoding="utf-8-sig")

print("\n===== TA'SIR XULOSASI =====")
print(f"Dublikatsiz qatorlar: {int(t['after_dedup'].sum()):,}")
print(f"Shulardan NaN/inf bor qatorlar: {int(t['bad_after_dedup'].sum()):,}")
print(f"Yakuniy qatorlar (dedup + NaN/inf olib tashlangandan keyin): {int(t['final'].sum()):,}")
print("\nSinflar bo'yicha:")
print(tab.to_string())
print("\nNaN/inf bor ustunlar:")
print(cols.to_string(index=False))
print(f"\nDedupdan keyin qarama-qarshi guruhlar: {len(combo):,}, qatorlar: {len(kc):,}")
print(conf.to_string())
print(f"\nVaqt: {time.time() - t0:.1f} s")