p = "src/07_tables_figures.py"
s = open(p, encoding="utf-8").read()
pairs = [
 ('ax.set_xlabel("Global importance: mean |SHAP value| (mean over 15 classes)")', 'ax.set_xlabel("Mean |SHAP value|")'),
 ('ax.set_xlabel("Number of features (SHAP top-k; 69 = all features)")', 'ax.set_xlabel("Number of features (69 = all)")'),
 ('label="12 classes (without ultra-rare classes)"', 'label="12 classes (no ultra-rare)"'),
 ('ax.set_xlabel("Number of splits (out of 5) in which the feature was in the top 20")', 'ax.set_xlabel("Splits with feature in top 20 (of 5)")'),
 ('axes[1].set_xlabel("Inference time on the test partition (s, log scale)")', 'axes[1].set_xlabel("Inference time (s, log scale)")'),
 ('(mean pairwise Jaccard index = {mean_j:.2f}).', '(mean pairwise Jaccard index = {mean_j:.2f}). Each bar shows in how many of the five splits the feature was among the 20 most important features.'),
]
for old, new in pairs:
    if old in s:
        s = s.replace(old, new)
        print("replaced:", old[:60])
    else:
        print("NOT FOUND:", old[:60])
open(p, "w", encoding="utf-8").write(s)
