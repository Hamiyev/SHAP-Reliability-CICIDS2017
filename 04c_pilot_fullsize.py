project:
  name: "SHAP-guided CICIDS2017"
  protocol_version: "1.0"
  python: "3.11"

data:
  parquet: "data/processed/cicids2017_clean.parquet"
  parquet_sha256: "ADE5559BB9C9CBA49F9ED2ED5881CF2D8B3707F55D18542B009A1478ABD76DCD"
  n_rows: 2520798
  n_classes: 15
  feature_count_expected: 69
  id_column: "row_id"
  label_column: "Label"
  label_id_column: "label_id"
  feature_names_source: "results/data_audit/clean_data_summary.json"
  label_map: "results/data_audit/label_map.json"
  cleaning_decisions:
    web_attack_labels_normalized: true
    full_row_duplicates_removed: 308381
    nan_inf_rows_removed: 1564
    constant_columns_removed: 8
    identical_duplicate_column_removed: 1
    conflicting_label_rows_kept: 1394   # 697 groups, listed in conflict_rows.csv
    outliers_removed: false

splits:
  seeds: [42, 123, 2024, 3407, 7777]
  train_fraction: 0.70
  validation_fraction: 0.15
  test_fraction: 0.15
  method: "two-step stratified on label_id"
  manifest: "results/splits/split_manifest.json"

evaluation:
  primary_metric: "macro_f1"            # 15 classes
  secondary_metrics: [accuracy, weighted_f1, balanced_accuracy,
                      per_class_precision_recall_f1, confusion_matrix]
  tuning_selection_metric: "validation macro_f1"
  test_used_for_selection: false
  ultra_rare_rule: "classes with fewer than 100 rows after cleaning"
  ultra_rare_classes: ["Heartbleed", "Web Attack - Sql Injection", "Infiltration"]
  sensitivity_analyses:
    - "macro_f1_12_classes_excluding_ultra_rare"
    - "macro_f1_excluding_conflicting_label_rows"

reproducibility:
  model_random_state: "equal to the split seed"

feature_selection:
  method: "global_mean_absolute_shap"
  rankings: [rf, class_weighted_rf]
  feature_counts: [5, 10, 20, 40]
  fit_on_training_data_only: true
  explainer: "shap.TreeExplainer, tree_path_dependent (no background data)"
  sample: "class-balanced draw from the training part only: up to 300 rows per class, all rows for classes with fewer; seed = the split seed"
  sample_size_expected: 3649
  importance: "mean over sampled rows and over the 15 output classes of |SHAP value|"
  additivity_check: "20 rows per ranking"

models:
  - {id: rf_all69,     type: random_forest, features: all69}
  - {id: rf_cw_all69,  type: random_forest, features: all69, class_weight: balanced}
  - {id: xgb_all69,    type: xgboost,       features: all69}
  - {id: lgbm_all69,   type: lightgbm,      features: all69}
  - {id: rf_top5,      type: random_forest, features: shap_top5}
  - {id: rf_top10,     type: random_forest, features: shap_top10}
  - {id: rf_top20,     type: random_forest, features: shap_top20}
  - {id: rf_top40,     type: random_forest, features: shap_top40}
  - {id: rf_cw_top20,  type: random_forest, features: shap_top20_cw, class_weight: balanced}
  - {id: soft_voting,  type: soft_voting,   base: [rf_all69, xgb_all69, lgbm_all69], weights: equal}
  - {id: stacking_oof, type: stacking,      base: [rf_all69, xgb_all69, lgbm_all69],
     oof_folds: 3, meta_model: "multinomial logistic regression (C=1.0, max_iter=1000)"}

model_fixed_params:
  lightgbm:
    reg_lambda: 1.0
    min_child_weight: 1.0
    reason: "defaults diverged on rare classes in the pilot (train logloss 0.32 -> 3.9, macro-F1 0.16); chosen for training stability, not validation score"

tuning:
  per_split: true
  selection_metric: "validation macro_f1 (15 classes)"
  tie_rule: "if the best two configs differ by less than 0.001, take the one with shorter training time"
  n_estimators_fixed: 100
  grids:
    random_forest:     # rf_all69 and rf_cw_all69 tuned separately; top-k models reuse their parent's parameters
      max_features: sqrt
      max_depth: [null, 20]
      min_samples_leaf: [1, 3]
    xgboost:
      max_depth: [6, 10]
      learning_rate: [0.1, 0.3]
    lightgbm:
      num_leaves: [31, 127]
      learning_rate: [0.1, 0.2]
  class_weighting: "RF only (rf_cw_*); boosting models are unweighted"
  refit: "the best config is refitted on the training part, then predicts the test part once"

validity_checks:
  boosting_divergence_rule: "a boosting run is flagged invalid if its logloss on a fixed 200,000-row training subsample is non-finite or higher at the last iteration than at iteration 10; flagged runs stay in the log"

statistics:
  unit: "per-split macro_f1 difference (n = 5, splits are not independent)"
  report: [all_five_differences, mean_difference, ci_95, raw_p_value, holm_adjusted_p_value]
  test: "paired t-test (Wilcoxon cannot reach p<0.05 with n=5)"
  correction: "Holm over the comparisons below"
  non_inferiority:
    metric: "macro_f1_12_classes_excluding_ultra_rare"
    margin: 0.01
    rule: "one-sided 95% lower confidence bound of the paired difference > -0.01"
  macro_f1_15_classes: "differences and CI reported, no non-inferiority claim"
  comparisons:
    - [rf_all69, xgb_all69]
    - [rf_all69, lgbm_all69]
    - [rf_all69, soft_voting]
    - [rf_all69, stacking_oof]
    - [rf_all69, rf_top5]
    - [rf_all69, rf_top10]
    - [rf_all69, rf_top20]
    - [rf_all69, rf_top40]
    - [rf_all69, rf_cw_all69]
    - [rf_cw_all69, rf_cw_top20]

resources:
  cpu_jobs: 4
  save_predictions: true
  save_model_metadata: true

runs:
  pilot: "results/run_001_pilot"
  main: "results/run_002_main"
  verified: "results/run_003_verified"