import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW_EXTRA = Path("data/raw_extra")
AUDIT = Path("results/data_audit")
OUT = Path("results/run_004_posthoc")
OUT.mkdir(parents=True, exist_ok=True)
ENC = "latin-1"
T0 = time.time()
ATTACKER, VICTIM, INFECTED = "172.16.0.1", "192.168.10.50", "192.168.10.8"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


with open(AUDIT / "clean_data_summary.json", encoding="utf-8") as f:
    FEAT = json.load(f)["feature_names"]
cr = pd.read_csv(AUDIT / "conflict_rows.csv", dtype={"group_hash": str}, encoding="utf-8-sig")
cr["h"] = cr["group_hash"].map(int)
combo = cr.groupby("h")["label"].agg(lambda s: " | ".join(sorted(set(s))))
cmap = {str(k): v for k, v in combo.items()}
HASHES = np.array(sorted(combo.index), dtype=np.uint64)
inv = pd.read_csv(AUDIT / "raw_file_inventory.csv", encoding="utf-8-sig").set_index("file")["rows"].to_dict()
print(f"Qarama-qarshi guruhlar: {len(HASHES)}, qatorlar: {len(cr)}", flush=True)

parts, file_rows, file_hashes = [], [], []
need_meta = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
             "Timestamp", "Label"]
for p in sorted(RAW_EXTRA.glob("*.csv")):
    print(f"O'qilmoqda: {p.name}", flush=True)
    df = pd.read_csv(p, encoding=ENC, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEAT + need_meta if c not in df.columns]
    if missing:
        raise SystemExit(f"{p.name}: ustunlar yetishmaydi: {missing}")
    X = df[FEAT].apply(pd.to_numeric, errors="coerce").astype("float64")
    h = pd.util.hash_pandas_object(X, index=False).to_numpy()
    mask = np.isin(h, HASHES)
    sub = df.loc[mask, need_meta].copy()
    sub["h"] = h[mask]
    sub["file"] = p.name
    sub["row_in_file"] = np.flatnonzero(mask)
    parts.append(sub)
    file_rows.append({"file": p.name, "tl_rows": len(df), "ml_rows": inv.get(p.name),
                      "matched_rows": int(mask.sum())})
    file_hashes.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_of(p)})
    del df, X

pd.DataFrame(file_hashes).to_csv(OUT / "trafficlabelling_hashes.csv", index=False)
m = pd.concat(parts, ignore_index=True)
m["Label"] = (m["Label"].astype(str).str.strip()
              .str.replace(r"^(Web Attack)\s+\S{1,3}\s+", r"\1 - ", regex=True))
m["combo"] = m["h"].astype(str).map(cmap)
src = m["Source IP"].astype(str).str.strip()
dst = m["Destination IP"].astype(str).str.strip()
thu = m["file"].str.contains("Infilteration") & ((src == INFECTED) | (dst == INFECTED))
a2v = (src == ATTACKER) & (dst == VICTIM)
v2a = (src == VICTIM) & (dst == ATTACKER)
m["category"] = np.select([thu, a2v, v2a],
                          ["thu_infil_192.168.10.8", "attacker_to_victim", "reversed_victim_to_attacker"],
                          default="other")
m["is_benign"] = m["Label"] == "BENIGN"
m.to_csv(OUT / "conflict_traffic_detail.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
fr = pd.DataFrame(file_rows)
print("\n===== 1) FAYL MOSLIGI (TrafficLabelling va MachineLearningCSV) =====")
print(fr.to_string(index=False))
print(f"Jami TL qatorlar: {fr['tl_rows'].sum():,} | MachineLearningCSV: {int(fr['ml_rows'].sum()):,}")

tabA = m.groupby("combo").agg(groups_found=("h", "nunique"), rows=("h", "size"),
                              benign_rows=("is_benign", "sum"))
tabA["attack_rows"] = tabA["rows"] - tabA["benign_rows"]
tabA["groups_expected"] = combo.value_counts()
print("\n===== 2) TOPILGAN GURUHLAR =====")
print(tabA.to_string())
print(f"Jami topilgan guruhlar: {m['h'].nunique()} / {len(HASHES)}")

b = m[m["is_benign"]]
cnt = pd.crosstab(b["combo"], b["category"])
pct = (cnt.div(cnt.sum(axis=1), axis=0) * 100).round(1)
print("\n===== 3) BENIGN qatorlar: toifalar bo'yicha (soni) =====")
print(cnt.to_string())
print("\n===== 3b) BENIGN qatorlar: toifalar bo'yicha (%) =====")
print(pct.to_string())
a = m[~m["is_benign"]]
print("\n===== 4) Hujum yorliqli qatorlar: toifalar bo'yicha (taqqoslash uchun) =====")
print(pd.crosstab(a["combo"], a["category"]).to_string())
print("\n===== 5) BENIGN qatorlar: fayl bo'yicha =====")
print(pd.crosstab(b["combo"], b["file"].str.replace(".pcap_ISCX.csv", "", regex=False)).to_string())
t = b[b["category"] == "thu_infil_192.168.10.8"].sort_values("row_in_file")
if len(t):
    print(f"\nThursday/192.168.10.8 BENIGN qatorlar vaqti (matn): birinchi={t['Timestamp'].iloc[0]}, "
          f"oxirgi={t['Timestamp'].iloc[-1]}")
oth = b[b["category"] == "other"]
if len(oth):
    print("\n===== 6) Tushuntirilmagan BENIGN qatorlar: eng ko'p IP juftliklari =====")
    print(oth.groupby(["combo", "Source IP", "Destination IP"]).size()
          .sort_values(ascending=False).head(10).to_string())

h1 = b[b["combo"] == "BENIGN | PortScan"]
h2 = b[b["combo"].isin(["BENIGN | DoS Hulk", "BENIGN | DDoS", "BENIGN | DoS slowloris"])]
r1 = float((h1["category"] == "thu_infil_192.168.10.8").mean() * 100) if len(h1) else float("nan")
r2 = float(h2["category"].isin(["attacker_to_victim", "reversed_victim_to_attacker"]).mean() * 100) \
    if len(h2) else float("nan")
print("\n===== QOIDALAR (50% chegara) =====")
print(f"H1: BENIGN|PortScan guruhlarida Thursday/192.168.10.8 ulushi = {r1:.1f}% -> "
      f"{'TASDIQLANDI' if r1 >= 50 else 'TASDIQLANMADI'}")
print(f"H2: BENIGN|DoS/DDoS guruhlarida hujumchi-qurbon oqimi ulushi = {r2:.1f}% -> "
      f"{'TASDIQLANDI' if r2 >= 50 else 'TASDIQLANMADI'}")
print(f"\n[Bajarildi] vaqt: {(time.time() - T0) / 60:.1f} min")