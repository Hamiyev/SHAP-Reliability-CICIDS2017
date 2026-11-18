"""Figures v2 for the SHAP-guided CICIDS2017 article (journal style).

Reads only existing result files, never recomputes models. Output: results/figures_v2
  - PNG (500 dpi) and PDF (vector, TrueType fonts) for every figure, no titles inside the images
  - figure_captions_v2_en.txt, figures_v2_index.csv, MANIFEST_SHA256.csv
Usage: python src/18_figures_v2.py [project_root]
"""
import hashlib
import itertools
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
RES = ROOT / "results"
OUT = RES / "figures_v2"
SEEDS = [42, 123, 2024, 3407, 7777]
DPI = 500
W2 = 7.2  # double-column width in inches (about 183 mm)

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#555555"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2, "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 100,
})

SHORT = {
    "Bot": "Bot", "Botnet": "Bot", "DoS slowloris": "DoS Slowloris", "DoS Slowloris": "DoS Slowloris",
    "PortScan": "PortScan", "Portscan": "PortScan", "Infiltration - Portscan": "Infiltration-PortScan",
    "Web Attack - Brute Force": "Web Brute Force", "Web Attack - Sql Injection": "Web SQL Injection",
    "Web Attack - SQL Injection": "Web SQL Injection", "Web Attack - XSS": "Web XSS",
}
ENGELEN6 = ["Fwd Header Length", "Total Fwd Packets", "min_seg_size_forward", "Init_Win_bytes_forward",
            "SYN Flag Count", "Bwd Packet Length Mean"]


def short(name):
    return SHORT.get(name, name)


# ----------------------------------------------------------------------------- input checks
PROBLEMS = []


def read_csv(rel, cols=None, index_col=None):
    p = RES / rel
    if not p.exists():
        PROBLEMS.append(f"MISSING FILE: {p}")
        return None
    df = pd.read_csv(p, encoding="utf-8-sig", index_col=index_col)
    if cols is not None:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            PROBLEMS.append(f"MISSING COLUMNS in {p.name}: {miss} (has: {list(df.columns)[:12]})")
            return None
    return df


def read_json(rel):
    import json
    p = RES / rel
    if not p.exists():
        PROBLEMS.append(f"MISSING FILE: {p}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_inputs():
    d = {}
    d["base"] = read_csv("run_006_baselines/baseline_metrics.csv", ["method", "k", "seed", "macro_f1", "macro_f1_12"])
    d["v10"] = read_csv("run_003_verified/all_experiments.csv",
                        ["model_name", "split_seed", "macro_f1", "macro_f1_12_classes"])
    d["abl"] = read_csv("run_004_posthoc/ablation_per_class_f1.csv",
                        ["class", "baseline_f1", "delta_no_port", "delta_no_engelen6", "delta_no_port_no_engelen6"])
    d["rep"] = read_csv("run_005_replication/all_experiments.csv",
                        ["model_name", "split_seed", "macro_f1", "macro_f1_nonrare"])
    d["pc11"] = read_csv("run_005_replication/statistics/per_class_f1_mean.csv",
                         ["rf_all77", "rf_top5", "rf_top10", "rf_top20", "rf_top40"], index_col=0)
    d["cm11"] = read_csv("run_005_replication/statistics/confusion_rf_all77_total.csv", None, index_col=0)
    d["lm10"] = read_json("data_audit/label_map.json")
    d["lm11"] = read_json("data_audit_corrected/label_map.json")
    d["shap10"], d["shap11"] = [], []
    for s in SEEDS:
        for key, sub in (("shap10", "run_003_verified/shap"), ("shap11", "run_005_replication/shap")):
            df = read_csv(f"{sub}/shap_importance_rf_split{s}.csv", ["feature", "mean_abs_shap"])
            d[key].append(df)
    d["pred10"] = []
    for s in SEEDS:
        rel = f"run_003_verified/predictions/rf_all69_split{s}_test.parquet"
        p = RES / rel
        if not p.exists():
            PROBLEMS.append(f"MISSING FILE: {p}")
            d["pred10"].append(None)
        else:
            df = pd.read_parquet(p)
            if not {"y_true", "y_pred"} <= set(df.columns):
                PROBLEMS.append(f"MISSING COLUMNS in {p.name}: y_true/y_pred")
                d["pred10"].append(None)
            else:
                d["pred10"].append(df[["y_true", "y_pred"]])
    d["predcw10"] = []
    for s in SEEDS:
        p = RES / f"run_003_verified/predictions/rf_cw_all69_split{s}_test.parquet"
        if not p.exists():
            PROBLEMS.append(f"MISSING FILE: {p}")
            d["predcw10"].append(None)
        else:
            d["predcw10"].append(pd.read_parquet(p)[["y_true", "y_pred"]])
    # content checks
    if d["v10"] is not None:
        need = {"rf_all69", "rf_top5", "rf_top10", "rf_top20", "rf_top40"} - set(d["v10"]["model_name"])
        if need:
            PROBLEMS.append(f"all_experiments.csv (v1.0) lacks models: {sorted(need)}")
    if d["rep"] is not None:
        need = {"rf_all77", "rf_top5", "rf_top10", "rf_top20", "rf_top40"} - set(d["rep"]["model_name"])
        if need:
            PROBLEMS.append(f"all_experiments.csv (v1.1) lacks models: {sorted(need)}")
    if d["base"] is not None:
        if not {"shap", "impurity", "random"} <= set(d["base"]["method"]):
            PROBLEMS.append("baseline_metrics.csv lacks one of: shap, impurity, random")
        if not {10, 20, 40} <= set(d["base"]["k"]):
            PROBLEMS.append("baseline_metrics.csv lacks k = 10, 20 or 40")
    if d["cm11"] is not None and list(d["cm11"].index) != list(d["cm11"].columns):
        PROBLEMS.append("confusion matrix: row and column labels differ")
    return d


# ----------------------------------------------------------------------------- output helpers
INDEX = []
CAPTIONS = {}


def save(fig, stem, caption, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    with Image.open(png) as im:
        w, h = im.size
    if min(w, h) < 800 or w < 1500:
        raise SystemExit(f"{stem}: PNG too small ({w}x{h} px)")
    INDEX.append({"figure": stem, "png_px": f"{w}x{h}", "dpi": DPI, "source_files": sources})
    CAPTIONS[stem] = caption
    print(f"  OK {stem}: {w}x{h} px", flush=True)


def panel_label(ax, text, x=-0.14, y=1.03):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="left")


def mean_sd(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


# ----------------------------------------------------------------------------- figure 1: workflow
def fig_workflow():
    H = 4.3
    fig = plt.figure(figsize=(W2, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W2)
    ax.set_ylim(0, H)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, fs=6.6, wrap=17, twrap=15):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
        t_lines = textwrap.wrap(title, twrap)
        b_lines = []
        for para in body.split("\n"):
            b_lines.extend(textwrap.wrap(para, wrap) or [""])
        ytop = y + h - 0.08
        ax.text(x + w / 2, ytop, "\n".join(t_lines), ha="center", va="top", fontsize=fs, fontweight="bold",
                linespacing=1.15)
        ybody = ytop - 0.118 * len(t_lines) - 0.07
        ax.text(x + w / 2, ybody, "\n".join(b_lines), ha="center", va="top", fontsize=fs, linespacing=1.25)

    def line(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color=C["grey"], lw=0.9, solid_capstyle="butt")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C["grey"], shrinkA=0, shrinkB=0))

    steps = [
        ("1. Raw flows", "Original: 2,830,743 flows, 15 classes\nImproved: 2,099,976 flows"),
        ("2. Cleaning before splitting", "Duplicates, NaN/inf rows, constant columns removed\nOriginal: 2,520,798 flows, 69 features\nImproved: 1,840,242 flows, 77 features"),
        ("3. Five splits", "Stratified 70/15/15; seeds 42, 123, 2024, 3407, 7777"),
        ("4. Tuning", "Small grids per split, chosen on validation Macro-F1\nRF, XGBoost, LightGBM"),
        ("5. SHAP selection", "Ranking from training data only\nRF on the top-k features (k = 5, 10, 20, 40)"),
        ("6. Test and statistics", "Test set used once\nPaired t-test, Holm, Nadeau-Bengio, non-inferiority"),
    ]
    bw, gap, y0, bh = 1.07, 0.13, 2.5, 1.7
    x = (W2 - (6 * bw + 5 * gap)) / 2
    xs = []
    for title, body in steps:
        box(x, y0, bw, bh, title, body, "#EAF3FA", C["blue"])
        xs.append(x)
        x += bw + gap
    for i in range(len(xs) - 1):
        arrow(xs[i] + bw, y0 + bh / 2, xs[i + 1], y0 + bh / 2)

    b2w, b2h, y2, g2 = 2.25, 1.12, 0.72, 0.15
    x2s = [(W2 - (3 * b2w + 2 * g2)) / 2 + i * (b2w + g2) for i in range(3)]
    txt2 = [
        ("Post hoc analyses (original data)", "Ablation: Destination Port and flagged features\nConflicting labels checked against TrafficLabelling\nFeature-selection baselines: impurity and random"),
        ("Replication (improved data)", "Protocol v1.1 frozen before training\nReduced model set, six comparisons\nCriteria C1 to C3 fixed in advance"),
        ("Additional models (original data)", "Class-weighted RF, Soft Voting, Stacking\nTen paired comparisons with the full RF\nEquivalence margin 0.01 on 12-class Macro-F1"),
    ]
    for xx, (t, b) in zip(x2s, txt2):
        box(xx, y2, b2w, b2h, t, b, "#FDF1E3", C["orange"], wrap=42, twrap=40)
    y_b = y2 + b2h + 0.2
    line(W2 / 2, y0, W2 / 2, y_b)
    line(x2s[0] + b2w / 2, y_b, x2s[2] + b2w / 2, y_b)
    for xx in x2s:
        arrow(xx + b2w / 2, y_b, xx + b2w / 2, y2 + b2h)

    ax.add_patch(FancyBboxPatch((x2s[0], 0.07), 3 * b2w + 2 * g2, 0.44, boxstyle="round,pad=0.0,rounding_size=0.05",
                                fc="#F2F2F2", ec=C["grey"], lw=0.8, ls="--"))
    ax.text(W2 / 2, 0.29, "Frozen protocol (YAML with SHA256 hash) verified before every run;\nresults recomputed from saved predictions and stored with SHA256 manifests",
            ha="center", va="center", fontsize=6.6, linespacing=1.25)
    save(fig, "figv2_01_workflow",
         "Study design. Both dataset versions follow the same pipeline: cleaning and deduplication before splitting, five stratified splits, per-split tuning on validation data, SHAP ranking on training data only, and a single test evaluation. Post hoc analyses, the replication on the improved dataset and the additional ensemble models are shown below the main pipeline.",
         "none (schematic; counts from the data audit files)")


# ----------------------------------------------------------------------------- figure 2: baselines
def fig_baselines(d):
    base, v10 = d["base"], d["v10"]
    ks = [10, 20, 40]
    methods = [("shap", "SHAP", C["blue"], "o", "-", -0.17), ("impurity", "Impurity (MDI)", C["orange"], "s", "--", 0.0),
               ("random", "Random subset", C["green"], "^", ":", 0.17)]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0))
    for ax, key, ylab, ref_key, tag in zip(axes, ["macro_f1", "macro_f1_12"],
                                           ["Macro-F1 (15 classes)", "Macro-F1 (12 classes)"],
                                           ["macro_f1", "macro_f1_12_classes"], ["(a)", "(b)"]):
        ref = float(v10[v10.model_name == "rf_all69"][ref_key].mean())
        ax.axhline(ref, color=C["grey"], lw=0.9, ls=(0, (4, 2)), zorder=1)
        allv = [ref]
        for m, lab, col, mk, ls, off in methods:
            mus, xs = [], []
            for i, k in enumerate(ks):
                vals = base[(base.method == m) & (base.k == k)].sort_values("seed")[key].to_numpy()
                allv.extend(vals.tolist())
                mu, sd = mean_sd(vals)
                x = i + off
                jit = np.linspace(-0.035, 0.035, len(vals))
                ax.scatter(x + jit, vals, s=6, color=col, alpha=0.35, linewidths=0, zorder=2)
                ax.errorbar(x, mu, yerr=sd, color=col, marker=mk, ms=4.8, capsize=2, lw=1.0, elinewidth=1.0,
                            zorder=4, ls="none")
                mus.append(mu)
                xs.append(x)
            ax.plot(xs, mus, color=col, ls=ls, lw=1.0, zorder=3)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlim(-0.55, len(ks) - 0.45)
        ax.set_ylim(min(allv) - 0.03, max(allv) + 0.03)
        ax.set_xlabel("Number of selected features (k)")
        ax.set_ylabel(ylab)
        panel_label(ax, tag)
    handles = [Line2D([0], [0], color=c, marker=mk, ls=ls, ms=4.8, lw=1.0, label=lab)
               for _, lab, c, mk, ls, _ in methods]
    handles.append(Line2D([0], [0], color=C["grey"], ls=(0, (4, 2)), lw=0.9, label="Full RF (69 features)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "figv2_02_feature_selection_baselines",
         "Feature-selection criteria compared on the original dataset (post hoc analysis). Macro-F1 of a Random Forest trained on the k features ranked highest by SHAP, by the impurity-based importance of the full model (MDI) or drawn at random (one subset per split and k). Large markers show the mean and error bars the standard deviation over five splits; small points are single splits; the dashed line is the full model. (a) All 15 classes; (b) the 12 classes with at least 100 flows.",
         "run_006_baselines/baseline_metrics.csv; run_003_verified/all_experiments.csv")


# ----------------------------------------------------------------------------- figure 3: top-k, two versions
def fig_topk_versions(d):
    v10, rep = d["v10"], d["rep"]
    names = ["5", "10", "20", "40", "All"]
    m10 = ["rf_top5", "rf_top10", "rf_top20", "rf_top40", "rf_all69"]
    m11 = ["rf_top5", "rf_top10", "rf_top20", "rf_top40", "rf_all77"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0))
    specs = [("macro_f1", "macro_f1", "Macro-F1 (all classes)", "(a)", "Original, 15 classes", "Improved, 16 classes"),
             ("macro_f1_12_classes", "macro_f1_nonrare", "Macro-F1 (without ultra-rare classes)", "(b)",
              "Original, 12 classes", "Improved, 11 classes")]
    for ax, (k10, k11, ylab, tag, lab10, lab11) in zip(axes, specs):
        allv = []
        for models, df, key, col, mk, lab, off in ((m10, v10, k10, C["blue"], "o", lab10, -0.07),
                                                     (m11, rep, k11, C["red"], "D", lab11, 0.07)):
            mus, sds = [], []
            for m in models:
                vals = df[df.model_name == m].sort_values("split_seed")[key].to_numpy()
                allv.extend(vals.tolist())
                mu, sd = mean_sd(vals)
                mus.append(mu)
                sds.append(sd)
            xs = np.arange(len(models)) + off
            ax.errorbar(xs, mus, yerr=sds, color=col, marker=mk, ms=4.5, capsize=2, lw=1.1, elinewidth=1.0,
                        label=lab)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_xlabel("Number of SHAP-ranked features")
        ax.set_ylabel(ylab)
        ax.set_ylim(min(allv) - 0.03, min(1.01, max(allv) + 0.03))
        ax.legend(loc="lower right", frameon=False)
        panel_label(ax, tag)
    fig.tight_layout()
    save(fig, "figv2_03_topk_original_vs_improved",
         "Effect of reducing the Random Forest to the top-k SHAP features on the original and the improved dataset. Mean and standard deviation over five splits; All denotes the full model (69 features in the original, 77 in the improved dataset). (a) Macro-F1 over all classes (15 and 16 classes); (b) Macro-F1 over the classes with at least 100 flows (12 and 11 classes). The ranking is computed on training data only.",
         "run_003_verified/all_experiments.csv; run_005_replication/all_experiments.csv")


# ----------------------------------------------------------------------------- figure 4: ablation heatmap
def fig_ablation(d):
    a = d["abl"].copy().sort_values("delta_no_port_no_engelen6").reset_index(drop=True)
    cols = ["delta_no_port", "delta_no_engelen6", "delta_no_port_no_engelen6"]
    mat = a[cols].to_numpy(dtype=float)
    lim = max(0.05, float(np.ceil(np.abs(mat).max() * 20) / 20))
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(mat, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Without\nDestination\nPort", "Without six\nflagged\nfeatures", "Without\nboth"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels([f"{short(c)} ({b:.2f})" for c, b in zip(a["class"], a["baseline_f1"])])
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                    color="white" if abs(v) > 0.6 * lim else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Change in class F1 relative to the full RF", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "figv2_04_ablation_per_class",
         "Ablation on the original dataset (post hoc). Change in the F1 of each class (mean over five splits) when a Random Forest with the hyperparameters of the full model is trained without Destination Port, without the six features flagged as shortcut candidates by Engelen et al. (Fwd Header Length, Total Fwd Packets, min_seg_size_forward, Init_Win_bytes_forward, SYN Flag Count, Bwd Packet Length Mean), or without both. The number in brackets is the F1 of the full model. Heartbleed, Infiltration and Web SQL Injection have only 2 to 5 test flows per split, so their changes are unstable.",
         "run_004_posthoc/ablation_per_class_f1.csv")


# ----------------------------------------------------------------------------- figure 5: per-class dumbbell
def per_class_f1_v10(d):
    id2name = {v: k for k, v in d["lm10"].items()}
    labels = sorted(id2name)
    acc = []
    for p in d["pred10"]:
        _, _, f, _ = precision_recall_fscore_support(p["y_true"].to_numpy(), p["y_pred"].to_numpy(),
                                                     labels=labels, zero_division=0)
        acc.append(f)
    return pd.Series(np.mean(acc, axis=0), index=[short(id2name[i]) for i in labels])


def fig_dumbbell(d):
    orig = per_class_f1_v10(d)
    imp = d["pc11"]["rf_all77"].copy()
    imp.index = [short(c) for c in imp.index]
    df = pd.DataFrame({"orig": orig, "imp": imp})
    df["key"] = df["orig"].fillna(2.0)
    df = df.sort_values("key", ascending=True)
    fig, ax = plt.subplots(figsize=(4.9, 4.2))
    y = np.arange(len(df))
    for yi, (o, m) in enumerate(zip(df["orig"], df["imp"])):
        if np.isfinite(o) and np.isfinite(m):
            ax.plot([o, m], [yi, yi], color="#BBBBBB", lw=1.4, zorder=1)
    ax.scatter(df["orig"], y, s=28, color=C["blue"], marker="o", zorder=3, label="Original dataset (15 classes)")
    ax.scatter(df["imp"], y, s=28, color=C["red"], marker="D", zorder=3, label="Improved dataset (16 classes)")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel("F1 of the full Random Forest (mean over five splits)")
    ax.grid(axis="y", alpha=0.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "figv2_05_per_class_f1_original_vs_improved",
         "Class-wise F1 of the full Random Forest on the original and on the improved dataset (mean over five splits). Class names are matched between the two versions; Infiltration-PortScan exists only in the improved dataset. The largest differences between the two versions are seen for the classes with the lowest F1 on the original dataset (for example Bot, Web XSS and Web Brute Force); Heartbleed, Infiltration and Web SQL Injection have only 2 to 5 test flows per split.",
         "run_003_verified/predictions/rf_all69_split*_test.parquet; run_005_replication/statistics/per_class_f1_mean.csv")


# ----------------------------------------------------------------------------- figure 6: per-class heatmap
def fig_perclass_heatmap(d):
    pc = d["pc11"][["rf_all77", "rf_top40", "rf_top20", "rf_top10", "rf_top5"]].copy()
    pc.index = [short(c) for c in pc.index]
    pc = pc.loc[pc.min(axis=1).sort_values().index]
    mat = pc.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 4.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Full RF\n(77)", "Top-40", "Top-20", "Top-10", "Top-5"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(pc)))
    ax.set_yticklabels(pc.index)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if v < 0.62 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("F1 (mean over five splits)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "figv2_06_per_class_f1_topk_improved",
         "Class-wise F1 of Random Forests trained on the top-k SHAP features of the improved dataset (mean over five splits). Reducing the model to 5 or 10 features mainly removes the ability to separate the two port-scan classes (Infiltration-PortScan and PortScan); almost all other classes stay near 1. Web SQL Injection has only 2 to 3 test flows per split.",
         "run_005_replication/statistics/per_class_f1_mean.csv")


# ----------------------------------------------------------------------------- figure 7: confusion matrix
def fig_confusion(d):
    cm = d["cm11"].copy()
    classes = list(cm.index)
    row = cm.sum(axis=1).replace(0, np.nan)
    norm = cm.div(row, axis=0).fillna(0.0)
    mat = norm.to_numpy(dtype=float)
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
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.2,
                        color="white" if v > 0.55 else "black")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of the true class (row-normalised)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "figv2_07_confusion_rf_improved",
         "Row-normalised confusion matrix of the full Random Forest on the improved dataset, summed over the test partitions of the five splits. Numbers are percentages of the true class; off-diagonal cells are shown from 0.5 %. Almost all errors occur between PortScan and Infiltration-PortScan. Classes with very few test flows (Heartbleed, Infiltration, Web Brute Force, Web SQL Injection, Web XSS) have unstable rates.",
         "run_005_replication/statistics/confusion_rf_all77_total.csv")


# ----------------------------------------------------------------------------- figure 8: SHAP importance
def shap_mean_sd(frames):
    s = pd.concat([f.set_index("feature")["mean_abs_shap"] for f in frames], axis=1)
    return s.mean(axis=1), s.std(axis=1, ddof=1)


def fig_shap(d):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.9))
    for ax, key, col, tag, lab in zip(axes, ["shap10", "shap11"], [C["blue"], C["red"]], ["(a)", "(b)"],
                                      ["Original dataset", "Improved dataset"]):
        mu, sd = shap_mean_sd(d[key])
        top = mu.sort_values(ascending=False).head(15)
        y = np.arange(len(top))
        bars = ax.barh(y, top.to_numpy(), xerr=sd.loc[top.index].to_numpy(), color=col, alpha=0.85, ecolor="#333333",
                       error_kw=dict(lw=0.8, capsize=1.5), height=0.7)
        for b, name in zip(bars, top.index):
            if name in ENGELEN6:
                b.set_hatch("///")
                b.set_edgecolor("white")
        ax.set_yticks(y)
        ax.set_yticklabels(top.index, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value| (rows and classes, 5 splits)")
        ax.grid(axis="y", alpha=0.0)
        ax.text(0.98, 0.04, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, style="italic")
        panel_label(ax, tag, x=-0.02)
    fig.legend(handles=[Patch(facecolor="#999999", hatch="///", edgecolor="white",
                              label="Feature flagged by Engelen et al. (2021)")],
               loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "figv2_08_shap_importance_top15",
         "Global SHAP importance of the 15 most important features of the full Random Forest (mean over five splits; error bars show the standard deviation). The importance is the mean absolute SHAP value over a class-balanced sample from the training partition and over all output classes. (a) Original dataset; (b) improved dataset. Hatched bars mark the features flagged as shortcut candidates by Engelen et al. (2021).",
         "run_003_verified/shap/shap_importance_rf_split*.csv; run_005_replication/shap/shap_importance_rf_split*.csv")


# ----------------------------------------------------------------------------- figures 9-11 (original dataset)
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
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def main():
    print(f"Project root: {ROOT.resolve()}", flush=True)
    d = load_inputs()
    if PROBLEMS:
        print("\nINPUT PROBLEMS (nothing was drawn):")
        for p in PROBLEMS:
            print("  - " + p)
        raise SystemExit(1)
    print("All inputs found and checked.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_workflow()
    fig_baselines(d)
    fig_topk_versions(d)
    fig_ablation(d)
    fig_dumbbell(d)
    fig_perclass_heatmap(d)
    fig_confusion(d)
    fig_shap(d)
    fig_confusion_original(d)
    fig_confusion_cw_original(d)
    fig_stability(d)
    with open(OUT / "figure_captions_v2_en.txt", "w", encoding="utf-8") as f:
        for i, (stem, cap) in enumerate(CAPTIONS.items(), 1):
            f.write(f"Figure V2-{i} ({stem}). {cap}\n\n")
    pd.DataFrame(INDEX).to_csv(OUT / "figures_v2_index.csv", index=False)
    rows = []
    for p in sorted(OUT.glob("*")):
        if p.name != "MANIFEST_SHA256.csv" and p.is_file():
            rows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_of(p)})
    pd.DataFrame(rows).to_csv(OUT / "MANIFEST_SHA256.csv", index=False)
    print(f"\n[Done] {len(INDEX)} figures (PNG {DPI} dpi + PDF) in {OUT}")


if __name__ == "__main__":
    main()