from pathlib import Path

p = Path("src/19_check_references.py") if Path("src/19_check_references.py").exists() else Path("19_check_references.py")
src = p.read_text(encoding="utf-8")


def swap(old, new):
    global src
    if src.count(old) != 1:
        raise SystemExit("PATCH FAILED: text not found exactly once:\n" + old[:80])
    src = src.replace(old, new)


swap("import time\n", "import time\nimport unicodedata\n")
swap("def fetch_json(url):", '''def safe(s):
    """ASCII-only text for consoles that cannot print accented letters (CSV keeps the original text)."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")


def fetch_json(url):''')
swap('''        print(f"{key:26s} {status}")
        print(f"    Crossref: {info['authors']} ({info['year']}) {info['title'][:90]} | {info['container'][:50]} "
              f"{info['volume']}({info['issue']}) p. {info['page'] or info['article_number']}", flush=True)''',
     '''        print(safe(f"{key:26s} {status}"))
        print(safe(f"    Crossref: {info['authors']} ({info['year']}) {info['title'][:90]} | {info['container'][:50]} "
                   f"{info['volume']}({info['issue']}) p. {info['page'] or info['article_number']}"), flush=True)''')
p.write_text(src, encoding="utf-8", newline="\n")
print("patched:", p)