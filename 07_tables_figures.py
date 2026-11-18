import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

VER = Path("results/run_003_verified")
AUDIT = Path("results/data_audit")
FIG = Path("results/figures")
TAB = Path("results/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
DPI = 500
MIN_PIXELS = 1500
SEEDS = [42, 123, 2024, 3407, 7777]
SCRIPT = "src/07_tables_figures.py"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


manifest = pd.read_csv(VER / "MANIFEST_SHA256.csv").set_index("relative_path")["sha256"].to_dict()


def src(rel):
    p = VER / rel
    if sha256_of(p) != manifest[rel]:
        raise SystemExit(f"SOURCE FILE CHANGED AFTER VERIFICATION: {rel}")
    return p


with open(AUDIT / "label_map.json", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
NAMES = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])]
index_rows = []


def save(fig, name, source):
    png, pdf = FIG / f"{name}.png", FIG / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    img = Image.open(png)
    w, h = img.size
    dpi_meta = round(img.info.get("dpi", (DPI, DPI))[0])
    if min(w, h) < MIN_PIXELS or dpi_meta < 499:
        raise SystemExit(f"FIGURE DOES NOT MEET THE REQUIREMENT: {name} {w}x{h} px, {dpi_meta} dpi")
    index_rows.append({"figure": name, "png_pixels": f"{w}x{h}", "dpi": dpi_meta, "vector": pdf.name,
                       "source": source, "script": SCRIPT,
                       "created": datetime.datetime.now().isoformat(timespec="seconds")})
    print(f"  {name}: {w}x{h} px, {dpi_meta} dpi (+ PDF)", flush=True)


ex = pd.read_csv(src("all_experiments.csv"))
ms = pd.read_csv(src("statistics/model_summary.csv"), index_col=0)
pdiff = pd.read_csv(src("statistics/paired_differences.csv"))

# Figure 1: class distribution of the cleaned data
cd = pd.read_csv(AUDIT / "clean_class_distribution.csv").sort_values("final_rows")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.barh(cd["label"], cd["final_rows"], color=BLUE)
ax.set_xscale("log")
ax.set_xlim(right=cd["final_rows"].max() * 12)
ax.set_xlabel("Number of flows (log scale)")
ax.set_ylabel("Class")
for y, v in enumerate(cd["final_rows"]):
    ax.text(v * 1.08, y, f"{int(v):,}", va="center", fontsize=6.5)
save(fig, "fig01_class_distribution", "results/data_audit/clean_class_distribution.csv")


# Figures 2-3: confusion matrices (sum over the five test partitions, row-normalised)
def confusion_fig(model, name):
    cm = sum(np.array(json.load(open(src(f"metrics/{model}_split{s}_metrics.json"),
                                     encoding="utf-8"))["confusion_matrix"]) for s in SEEDS)
    rown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(15))
    ax.set_xticklabels(NAMES, rotation=50, ha="right")
    ax.set_yticks(range(15))
    ax.set_yticklabels(NAMES)
    for i in range(15):
        for j in range(15):
            if rown[i, j] >= 1.0:
                ax.text(j, i, f"{rown[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if rown[i, j] > 50 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Share of the true class (%)")
    save(fig, name, f"run_003_verified/metrics/{model}_split*_metrics.json (confusion_matrix)")


confusion_fig("rf_all69", "fig02_confusion_matrix_rf")
confusion_fig("rf_cw_all69", "fig03_confusion_matrix_weighted_rf")

# Figure 4: global SHAP importance (random forest)
imp = pd.concat([pd.read_csv(src(f"shap/shap_importance_rf_split{s}.csv")) for s in SEEDS])
g = (imp.groupby("feature")["mean_abs_shap"].agg(["mean", "std"])
     .sort_values("mean", ascending=False).head(20).iloc[::-1])
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.barh(g.index, g["mean"], xerr=g["std"].fillna(0), color=GREEN, ecolor="#333333",
        error_kw={"elinewidth": 0.7, "capsize": 2})
ax.set_xlabel("Mean |SHAP value|")
ax.set_ylabel("Feature")
save(fig, "fig04_shap_global_summary", "run_003_verified/shap/shap_importance_rf_split*.csv")

# Figure 5: number of selected features vs Macro-F1
pts = [("rf_top5", 5), ("rf_top10", 10), ("rf_top20", 20), ("rf_top40", 40), ("rf_all69", 69)]
rows = []
for m, k in pts:
    sub = ex[ex["model_name"] == m]
    rows.append({"k": k, "m15": sub["macro_f1"].mean(), "s15": sub["macro_f1"].std(),
                 "m12": sub["macro_f1_12_classes"].mean(), "s12": sub["macro_f1_12_classes"].std()})
fc = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.errorbar(fc["k"], fc["m15"], yerr=fc["s15"], marker="o", ls="-", color=BLUE, capsize=3,
            label="15 classes")
ax.errorbar(fc["k"], fc["m12"], yerr=fc["s12"], marker="s", ls="--", color=ORANGE, capsize=3,
            label="12 classes (no ultra-rare)")
ax.set_xlabel("Number of features (69 = all)")
ax.set_ylabel("Macro-F1 (mean \u00b1 SD, 5 splits)")
ax.set_xticks(fc["k"])
ax.legend(loc="lower right", frameon=False)
save(fig, "fig05_feature_count_vs_macro_f1", "run_003_verified/all_experiments.csv")

# Figure 6: stability of the SHAP top-20 sets
top = {s: json.load(open(src(f"feature_sets/rf_split{s}_top20.json"), encoding="utf-8"))["features"]
       for s in SEEDS}
cnt = pd.Series([f for s in SEEDS for f in top[s]]).value_counts().sort_values(ascending=True)
jac = pd.DataFrame(index=SEEDS, columns=SEEDS, dtype=float)
for a in SEEDS:
    for b in SEEDS:
        jac.loc[a, b] = len(set(top[a]) & set(top[b])) / len(set(top[a]) | set(top[b]))
jac.round(3).to_csv(TAB / "table_top20_jaccard.csv", encoding="utf-8-sig")
mean_j = float(jac.values[~np.eye(len(SEEDS), dtype=bool)].mean())
colors = [VERM if v == 5 else (BLUE if v >= 3 else GRAY) for v in cnt.values]
fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.2 * len(cnt) + 1)))
ax.barh(cnt.index, cnt.values, color=colors)
ax.set_xticks(range(0, 6))
ax.set_xlabel("Splits with feature in top 20 (of 5)")
ax.set_ylabel("Feature")
ax.legend(handles=[Patch(color=VERM, label="all 5 splits"), Patch(color=BLUE, label="3-4 splits"),
                   Patch(color=GRAY, label="1-2 splits")], loc="lower right", frameon=False)
save(fig, "fig06_feature_selection_stability", "run_003_verified/feature_sets/rf_split*_top20.json")

# Figure 7: computational cost
order = ms.sort_values("train_seconds_mean").index
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
axes[0].barh(order, ms.loc[order, "train_seconds_mean"], color=BLUE)
axes[0].set_xscale("log")
axes[0].set_xlabel("Training time (s, log scale)")
axes[0].set_title("(a)", loc="left", fontsize=8, fontweight="bold")
axes[1].barh(order, ms.loc[order, "inference_seconds_mean"], color=ORANGE)
axes[1].set_xscale("log")
axes[1].set_xlabel("Inference time (s, log scale)")
axes[1].set_title("(b)", loc="left", fontsize=8, fontweight="bold")
axes[0].set_ylabel("Model")
save(fig, "fig07_runtime_comparison", "run_003_verified/statistics/model_summary.csv")

# Tables (English headers)
def pm(m, s, d=4):
    return f"{m:.{d}f} \u00b1 {s:.{d}f}"


main = pd.DataFrame({
    "Macro-F1 (15 classes)": [pm(a, b) for a, b in zip(ms["macro_f1_mean"], ms["macro_f1_sd"])],
    "Macro-F1 (12 classes)": [pm(a, b) for a, b in zip(ms["macro_f1_12_mean"], ms["macro_f1_12_sd"])],
    "Accuracy": ms["accuracy_mean"].round(4),
    "Weighted-F1": ms["weighted_f1_mean"].round(4),
    "Balanced accuracy": ms["balanced_acc_mean"].round(4),
    "Training time (s)": ms["train_seconds_mean"].round(1),
    "Inference time (s)": ms["inference_seconds_mean"].round(2),
    "Model size (MB)": ms["model_size_mb_mean"].round(2),
}, index=ms.index)
main.index.name = "Model"
main.to_csv(TAB / "table_main_results.csv", encoding="utf-8-sig")

pw = pdiff[["metric", "reference", "comparison_model", "mean_diff", "ci95_low", "ci95_high",
            "p_raw", "p_holm", "n_splits_positive"]].round(4)
pw.columns = ["Metric", "Reference model (A)", "Compared model (B)", "Mean difference (B - A)",
              "95% CI lower", "95% CI upper", "p (raw)", "p (Holm)", "Splits with B > A (of 5)"]
pw.to_csv(TAB / "table_pairwise_all.csv", index=False, encoding="utf-8-sig")

pv = ex.pivot(index="model_name", columns="split_seed", values="macro_f1").loc[ms.index, SEEDS].round(4)
pv.index.name = "Model"
pv.columns = [f"Split {s}" for s in SEEDS]
pv.to_csv(TAB / "table_per_split_macro_f1.csv", encoding="utf-8-sig")

# Captions for the manuscript (titles are NOT placed inside the images)
captions = f"""Fig. 1. Class distribution of the cleaned CICIDS2017 data (n = {int(cd['final_rows'].sum()):,} flows; logarithmic axis).
Fig. 2. Confusion matrix of the random forest (69 features), summed over the five test partitions and normalised by true class. Values below 1% are not printed.
Fig. 3. Confusion matrix of the class-weighted random forest (69 features), summed over the five test partitions and normalised by true class. Values below 1% are not printed.
Fig. 4. The 20 most important features according to the global SHAP importance of the random forest (mean absolute SHAP value, averaged over the 15 output classes). Bars show the mean over the five splits; whiskers show the standard deviation.
Fig. 5. Macro-F1 as a function of the number of SHAP-selected features (mean +/- SD over five splits), for all 15 classes and for 12 classes excluding the three ultra-rare classes.
Fig. 6. Stability of the SHAP top-20 feature set across the five splits (mean pairwise Jaccard index = {mean_j:.2f}). Each bar shows in how many of the five splits the feature was among the 20 most important features.
Fig. 7. Computational cost per model, mean of five splits (logarithmic axes): (a) training time; for stacking it includes the base models and the out-of-fold fits; (b) inference time on the test partition; for the ensembles it is the sum of the base-model times.
"""
(FIG / "figure_captions_en.txt").write_text(captions, encoding="utf-8")

pd.DataFrame(index_rows).to_csv(FIG / "figures_index.csv", index=False, encoding="utf-8-sig")
files = [p for d in (FIG, TAB) for p in sorted(d.iterdir()) if p.is_file()]
pd.DataFrame([{"file": str(p).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256_of(p)}
              for p in files]).to_csv("results/figures_tables_manifest.csv", index=False, encoding="utf-8-sig")
print(f"\n[Done] {len(index_rows)} figures ({DPI} dpi PNG + PDF), tables and captions created")
print("Tables:", ", ".join(p.name for p in sorted(TAB.glob("*.csv"))))
print(f"Mean pairwise Jaccard of the top-20 sets: {mean_j:.3f}")