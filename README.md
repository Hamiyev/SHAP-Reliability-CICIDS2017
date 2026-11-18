# SHAP-guided feature selection for network intrusion detection on CICIDS2017

Reproducibility package for the article *How Reliable Is SHAP-Guided Feature Selection for Network Intrusion Detection on CICIDS2017?* by Akrom Hamiyev (Samarkand State University named after Sharof Rashidov).

The repository holds the code, the frozen protocols, the result tables, the figures and the SHA256 manifests of all experiments. Raw datasets, trained models, predictions and SHAP arrays are not included (see "Not included").

## Data (not redistributed)

| Dataset | Source | Used for |
|---|---|---|
| CICIDS2017, MachineLearningCSV.zip | https://www.unb.ca/cic/datasets/ids-2017.html (Canadian Institute for Cybersecurity) | primary analysis |
| CICIDS2017, GeneratedLabelledFlows / TrafficLabelling | same page | analysis of conflicting labels |
| Improved CIC-IDS2017 (Liu et al., 2022), CICIDS2017_improved.zip | https://intrusion-detection.distrinet-research.be/CNS2022/index.html | replication |

Expected checksums of MachineLearningCSV.zip: SHA256 C3F26274B36C837CCF28FFD2DBF4582941C30B3EE70A635C6E5B2F87C4727928, MD5 4F83860AFBF29CAC8163854095BF6CF7. Per-file SHA256 values of the CSV files are listed in `data/DATASET_INSTRUCTIONS.md`.

Place the files in `data/raw` (original), `data/raw_extra` (TrafficLabelling) and `data/raw_corrected` (improved). Please follow the terms of use of the dataset providers.

Expected SHA256 of the cleaned data files: original `ADE5559BB9C9CBA49F9ED2ED5881CF2D8B3707F55D18542B009A1478ABD76DCD`; improved `865B612F694FACAD0A70D0967051182654D4E189E881F1A378A88B1BD64C1390`.

## Environment

Python 3.11.4 (64 bit), NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, SHAP 0.51.0, XGBoost 3.2.0, LightGBM 4.7.0, PyArrow 26.0.0 (full list in `requirements.txt`). All experiments ran on CPU (Intel Core i5-9400, 6 cores, 15.87 GB RAM). Run time: about 10-14 h for the main experiment and about 5-6 h for the replication.

## Order of the scripts (src/)

| Step | Scripts |
|---|---|
| Original dataset: audit, cleaning, splits | `01_data_audit.py`, `01b_cleaning_impact.py`, `02_data_cleaning.py`, `03_create_splits.py` |
| Pilot (time and memory only) | `04_pilot.py`, `04b_lgbm_diagnostic.py`, `04c_pilot_fullsize.py` |
| Main experiment | `05_run_experiments.py` (resumable; progress: `05b_check_progress.py`) |
| Statistics, tables, figures, audit | `06_statistics.py`, `07_tables_figures.py`, `08_final_audit.py` |
| Post hoc analyses (original data) | `09_posthoc_ablation.py`, `10_posthoc_nb_main.py`, `11_posthoc_conflict_traffic.py`, `17_posthoc_feature_selection_baselines.py` |
| Improved dataset: inspection, cleaning, splits | `12_corrected_inspect.py`, `13_corrected_clean.py`, `14_corrected_splits.py` |
| Replication | `15_run_replication.py` (resumable), `16_replication_statistics.py` |
| Additional figures, reference check | `18_figures_v2.py`, `19_check_references.py` |

Every experiment script verifies the SHA256 of the frozen protocol and of the data at start and stops on a mismatch.

## Frozen protocols

`configs/experiment_protocol_v1.0.yaml` (original dataset) and `configs/experiment_protocol_v1.1.yaml` (improved dataset) were written and hashed before any model was trained; the hashes are in `configs/protocol_v1.0.sha256` and `configs/protocol_v1.1.sha256`.

## Results (results/)

| Folder | Content |
|---|---|
| `data_audit`, `data_audit_corrected` | audit and cleaning logs, class distributions, label maps |
| `splits`, `splits_corrected` | split manifests and class distributions per split |
| `run_003_verified` | verified results of the main experiment (copy of the main run with SHA256 manifest of 406 files) |
| `run_004_posthoc` | ablation, Nadeau-Bengio analysis, conflicting-label analysis |
| `run_005_replication` | replication on the improved dataset |
| `run_006_baselines` | impurity-based and random feature selection |
| `figures`, `figures_v2`, `tables` | figures (PNG and PDF) and tables |

Each `run_*` folder carries a `MANIFEST_SHA256.csv`.

## Not included

Raw and cleaned datasets, split files (`*.parquet`), trained models (`*.joblib`), predictions and SHAP arrays (`*.npy`) are too large for this repository. The splits can be recreated with `03_create_splits.py` and `14_corrected_splits.py` and checked with the hashes in `split_manifest.json`. [Archive with predictions and split files: Zenodo DOI to be added.]

## Licence and citation

Code: MIT licence (see `LICENSE`). The datasets keep the licences of their providers. See `CITATION.cff`.

Contact: Akrom Hamiyev, hamiyevakrom@samdu.uz
