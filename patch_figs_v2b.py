from pathlib import Path

p = Path("src/18_figures_v2.py") if Path("src/18_figures_v2.py").exists() else Path("18_figures_v2.py")
src = p.read_text(encoding="utf-8")


def swap(old, new):
    global src
    if src.count(old) != 1:
        raise SystemExit("PATCH FAILED: text not found exactly once:\n" + old[:80])
    src = src.replace(old, new)


# 1) same font family as the earlier figure set (Arial when installed)
swap('"font.sans-serif": ["DejaVu Sans"]', '"font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]')
swap("import hashlib\nimport sys\n", "import hashlib\nimport itertools\nimport sys\n")
swap("from sklearn.metrics import precision_recall_fscore_support\n",
     "from sklearn.metrics import confusion_matrix, precision_recall_fscore_support\n")

# 2) extra inputs: class-weighted RF predictions of the original dataset
swap("    # content checks\n    if d[\"v10\"] is not None:",
     '''    d["predcw10"] = []
    for s in SEEDS:
        p = RES / f"run_003_verified/predictions/rf_cw_all69_split{s}_test.parquet"
        if not p.exists():
            PROBLEMS.append(f"MISSING FILE: {p}")
            d["predcw10"].append(None)
        else:
            d["predcw10"].append(pd.read_parquet(p)[["y_true", "y_pred"]])
    # content checks
    if d["v10"] is not None:''')

# 3) new figure functions
swap("# ----------------------------------------------------------------------------- main\n",
     '''# ----------------------------------------------------------------------------- figures 9-11 (original dataset)
def draw_confusion(cm, stem, caption, sources):
    classes = list(cm.index)
    row = cm.sum(axis=1).replace(0, np.nan)
    mat = cm.div(row, axis=0).fillna(0.0).to_numpy(dtype=float)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(W2 * 0.86, W2 * 0.78))
    im = ax.imshow(mat, cmap="Blues", norm=PowerNorm(gamma=0.45, vmin=0, vmax=1))
    ax.grid(False)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labs = [short(c) for c in classes]
    ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=6.3)
    ax.set_yticklabels(labs, fontsize=6.3)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if i == j or v >= 0.005:
                txt = "100" if v >= 0.9995 else f"{100 * v:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.2, color="white" if v > 0.55 else "black")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of the true class (row-normalised)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, stem, caption, sources)


def confusion_from_predictions(d, key):
    id2name = {v: k for k, v in d["lm10"].items()}
    labels = sorted(id2name)
    cm = np.zeros((len(labels), len(labels)), dtype="int64")
    for p in d[key]:
        cm += confusion_matrix(p["y_true"].to_numpy(), p["y_pred"].to_numpy(), labels=labels)
    return pd.DataFrame(cm, index=[id2name[i] for i in labels], columns=[id2name[i] for i in labels])


def low_recall_text(cm, limit=0.90):
    tot = cm.sum(axis=1).to_numpy()
    rec = np.divide(np.diag(cm.to_numpy()), tot, out=np.ones(len(tot)), where=tot > 0)
    items = [(short(c), r) for c, r in zip(cm.index, rec) if r < limit]
    return ", ".join(f"{n} ({100 * r:.0f} %)" for n, r in sorted(items, key=lambda t: t[1], reverse=True)) or "none"


def fig_confusion_original(d):
    cm = confusion_from_predictions(d, "pred10")
    draw_confusion(cm, "figv2_09_confusion_rf_original",
                   "Row-normalised confusion matrix of the full Random Forest (69 features) on the original dataset, summed over the test partitions of the five splits. Numbers are percentages of the true class; off-diagonal cells are shown from 0.5 %. Classes with a recall below 90 %: " + low_recall_text(cm) + ". Classes with very few test flows (Heartbleed, Infiltration, Web SQL Injection) have unstable rates.",
                   "run_003_verified/predictions/rf_all69_split*_test.parquet")


def fig_confusion_cw_original(d):
    cm = confusion_from_predictions(d, "predcw10")
    draw_confusion(cm, "figv2_10_confusion_rf_class_weighted_original",
                   "Row-normalised confusion matrix of the class-weighted Random Forest (69 features) on the original dataset, summed over the test partitions of the five splits. Numbers are percentages of the true class; off-diagonal cells are shown from 0.5 %. Classes with a recall below 90 %: " + low_recall_text(cm) + ".",
                   "run_003_verified/predictions/rf_cw_all69_split*_test.parquet")


def fig_stability(d):
    tops = []
    for df in d["shap10"]:
        s = df.set_index("feature")["mean_abs_shap"].sort_values(ascending=False)
        tops.append(list(s.index[:20]))
    counts = pd.Series([f for t in tops for f in t]).value_counts()
    mean_imp = pd.concat([df.set_index("feature")["mean_abs_shap"] for df in d["shap10"]], axis=1).mean(axis=1)
    order = sorted(counts.index, key=lambda f: (-counts[f], -mean_imp[f]))
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(tops, 2)]
    colors = [C["red"] if counts[f] == 5 else (C["blue"] if counts[f] >= 3 else "#999999") for f in order]
    fig, ax = plt.subplots(figsize=(4.9, 0.19 * len(order) + 1.3))
    y = np.arange(len(order))
    ax.barh(y, [counts[f] for f in order], color=colors, height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.3)
    ax.set_xticks(range(6))
    ax.set_xlabel("Splits in which the feature is among the 20 most important (of 5)")
    ax.grid(axis="y", alpha=0.0)
    ax.legend(handles=[Patch(facecolor=C["red"], label="all 5 splits"), Patch(facecolor=C["blue"], label="3-4 splits"),
                       Patch(facecolor="#999999", label="1-2 splits")], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3,
              frameon=False, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    save(fig, "figv2_11_shap_top20_stability_original",
         f"Stability of the SHAP top-20 feature set of the full Random Forest on the original dataset across the five splits (mean pairwise Jaccard index = {np.mean(jac):.2f}). Each bar shows in how many of the five splits the feature was among the 20 most important features.",
         "run_003_verified/shap/shap_importance_rf_split*.csv")


# ----------------------------------------------------------------------------- main
''')
swap("    fig_shap(d)\n    with open(OUT", "    fig_shap(d)\n    fig_confusion_original(d)\n    fig_confusion_cw_original(d)\n    fig_stability(d)\n    with open(OUT")
p.write_text(src, encoding="utf-8", newline="\n")
print("patched:", p)