"""Compare the DOI-based entries of the reference list with Crossref metadata.

Run on a computer with internet access:  python src/19_check_references.py
Writes results/references_check.csv and prints one line per entry. Nothing is changed automatically;
differences must be judged by the author (online-first years and page formats often differ legitimately).
"""
import csv
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MAILTO = "hamiyevakrom@samdu.uz"
# key: (doi, expected title keyword, year, volume, first page or article number)
REFS = {
    "Breiman 2001": ("10.1023/A:1010933404324", "random forests", "2001", "45", "5"),
    "Chen & Guestrin 2016": ("10.1145/2939672.2939785", "xgboost", "2016", "", "785"),
    "Chen et al. 2024": ("10.3390/s24165223", "feature selection", "2024", "24", "5223"),
    "Engelen et al. 2021": ("10.1109/SPW53761.2021.00009", "troubleshooting", "2021", "", "7"),
    "Fryer et al. 2021": ("10.1109/ACCESS.2021.3119110", "shapley values for feature selection", "2021", "9", "144352"),
    "Gaspar et al. 2024": ("10.1109/ACCESS.2024.3368377", "lime and shap", "2024", "12", "30164"),
    "Goldschmidt & Chuda 2025": ("10.1016/j.cose.2025.104510", "network intrusion datasets", "2025", "156", "104510"),
    "Kapoor & Narayanan 2023": ("10.1016/j.patter.2023.100804", "leakage", "2023", "4", "100804"),
    "Lanvin et al. 2023": ("10.1007/978-3-031-31108-6_2", "errors in the cicids2017", "2023", "", "18"),
    "Liu et al. 2022": ("10.1109/CNS56114.2022.9947235", "error prevalence", "2022", "", "254"),
    "Lundberg et al. 2020": ("10.1038/s42256-019-0138-9", "local explanations", "2020", "2", "56"),
    "Maseer et al. 2021": ("10.1109/ACCESS.2021.3056614", "benchmarking of machine learning", "2021", "9", "22351"),
    "Nadeau & Bengio 2003": ("10.1023/A:1024068626366", "generalization error", "2003", "52", "239"),
    "Rafique et al. 2026": ("10.47852/bonviewJDSIS62027552", "correlation-based feature selection", "2026", "4", "193"),
    "Sharafaldin et al. 2018": ("10.5220/0006639801080116", "intrusion detection dataset", "2018", "", "108"),
    "Wolpert 1992": ("10.1016/S0893-6080(05)80023-1", "stacked generalization", "1992", "5", "241"),
}
MANUAL = [
    "Arp et al. 2022 (USENIX Security 2022): check pages on the USENIX presentation page",
    "Holm 1979 (Scand. J. Stat. 6(2), 65-70): check on JSTOR",
    "Ke et al. 2017 (NeurIPS 30, 3146-3154): check in the NeurIPS proceedings",
    "Kumar et al. 2020 (ICML, PMLR 119): check pages on proceedings.mlr.press",
    "Lundberg & Lee 2017 (NeurIPS 30, 4765-4774): check in the NeurIPS proceedings",
    "Pedregosa et al. 2011 (JMLR 12, 2825-2830): check on jmlr.org",
]


def safe(s):
    """ASCII-only text for consoles that cannot print accented letters (CSV keeps the original text)."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": f"reference-check/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def summarize(msg):
    title = (msg.get("title") or [""])[0]
    authors = ", ".join(f"{a.get('family', a.get('name', '?'))}" for a in (msg.get("author") or [])[:3])
    parts = (msg.get("issued") or {}).get("date-parts") or [[None]]
    year = str(parts[0][0]) if parts and parts[0] and parts[0][0] else ""
    return {"title": title, "authors": authors, "year": year,
            "container": (msg.get("container-title") or [""])[0], "volume": str(msg.get("volume", "")),
            "issue": str(msg.get("issue", "")), "page": str(msg.get("page", "")),
            "article_number": str(msg.get("article-number", ""))}


def compare(info, kw, year, vol, first):
    flags = []
    if kw.lower() not in info["title"].lower():
        flags.append("TITLE differs")
    if year and info["year"] and year != info["year"]:
        flags.append(f"YEAR {year} vs {info['year']}")
    if vol and info["volume"] and vol != info["volume"]:
        flags.append(f"VOLUME {vol} vs {info['volume']}")
    where = (info["page"] + " " + info["article_number"]).replace("-", " ").split()
    if first and where and first not in where:
        flags.append(f"PAGE/ARTICLE {first} vs {info['page'] or info['article_number']}")
    return flags


def main(out_path="results/references_check.csv"):
    rows = []
    for key, (doi, kw, year, vol, first) in REFS.items():
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()-._:;")
        try:
            info = summarize(fetch_json(url)["message"])
            flags = compare(info, kw, year, vol, first)
            status = "OK" if not flags else "CHECK: " + "; ".join(flags)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as e:
            info = {k: "" for k in ("title", "authors", "year", "container", "volume", "issue", "page", "article_number")}
            status = f"ERROR: {e}"
        print(safe(f"{key:26s} {status}"))
        print(safe(f"    Crossref: {info['authors']} ({info['year']}) {info['title'][:90]} | {info['container'][:50]} "
                   f"{info['volume']}({info['issue']}) p. {info['page'] or info['article_number']}"), flush=True)
        rows.append({"entry": key, "doi": doi, "status": status, **info})
        time.sleep(0.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r["entry"] for r in rows if r["status"] != "OK"]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} entries match; to review: {bad if bad else 'none'}")
    print("\nNo DOI in Crossref, check by hand:")
    for m in MANUAL:
        print("  - " + m)


if __name__ == "__main__":
    main()