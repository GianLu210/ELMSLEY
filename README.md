# ELMSLEY

ELMSLEY is an open‑source, YAML‑configurable ECG analysis framework for heartbeat classification that supports intra‑patient, inter‑patient, and Leave‑One‑Subject‑Out (LOSO) training/evaluation, with built‑in preprocessing, feature extraction/selection, class balancing, multiple ML models, and SHAP‑based explanations.

## Quick start

- Prerequisites
  - Python 3.9–3.12, Git, and a virtual environment tool (venv or conda).[1]
- Setup
```bash
git clone https://github.com/GianLu210/ELMSLEY.git
cd ELMSLEY

# create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate    # Linux/macOS
# or on Windows
py -m venv .venv && .venv\Scripts\activate

# install from source (editable for local development)
pip install -e .
# optional if present:
# pip install -r requirements.txt
```
- Minimal run
```bash
# CLI help (replace with your main entry module/script if different)
python -m elmsley --help

# Example pipeline using a config file (adapt paths to your repo)
python -m elmsley run \
  --config configs/default.yaml \
  --input /path/to/wfdb_datasets/ \
  --output out/
```
- Verify install
```bash
python -c "import elmsley; print('ELMSLEY OK:', getattr(elmsley,'__version__','dev'))"
```
This quick start targets source installation and a Python environment for fast, reproducible runs.

## Overview

ELMSLEY provides an end‑to‑end ECG workflow: data loading via WFDB, robust preprocessing and heartbeat segmentation, a novel morphological/time‑domain feature set, optional RFECV‑based feature elimination, flexible splitting (inter/intra/LOSO), class balancing (Random Undersampling, SMOTE, ADASYN), multi‑model training with grid search, cross‑dataset evaluation, and SHAP explanations for local interpretability.

## Features

- Configurable experiments via a single YAML file defining datasets, preprocessing, splitting, balancing, models, and outputs.
- Training/evaluation strategies: intra‑patient, inter‑patient, cross‑dataset, and LOSO with majority voting across subject‑held‑out folds.
- Preprocessing with NeuroKit2 for cleaning and delineation; retains only correctly segmented beats to reduce noise and enforce physiological peak order.
- Feature set spanning segment lengths, angles, slopes, segment ratios, peak values, and RR intervals; normalized across sampling rates to support multiple datasets.
- Optional feature elimination via RFECV (stand‑alone or combined with estimator grid search) with customizable estimator/scaler.
- Built‑in models: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, and MLP, each with grid‑search hyperparameters; extensible to custom models.
- Explainability using SHAP to quantify feature contributions per prediction on test and external datasets, improving transparency for clinical use.

## Installation

- From source (recommended)
```bash
git clone https://github.com/GianLu210/ELMSLEY.git
cd ELMSLEY
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional:
# pip install -r requirements.txt
```
- Future PyPI (planned)
```bash
pip install elmsley
```
Installing from source exposes all configuration options with minimal setup for users working in Python environments.

## Usage

- Command line
```bash
# Discover available commands
python -m elmsley --help

# Typical run (adapt to your repo’s CLI and configuration keys)
python -m elmsley run \
  --config configs/default.yaml \
  --input /path/to/wfdb_datasets/ \
  --output out/ \
  --explain true
```
- Python API
```python
# Illustrative API sketch; adjust to actual public API names
from elmsley import load_config, Pipeline

cfg = load_config("configs/default.yaml")
pipe = Pipeline(cfg)
metrics = pipe.run(input_path="/path/to/wfdb_datasets", output_path="out")
print(metrics)  # accuracy, recall, precision, f1
```
- Outputs
  - Metrics: accuracy, recall, precision, F1, and confusion matrices saved as .xlsx for post‑analysis.
  - Models: best model and scaler persisted (including per‑fold for LOSO) for downstream inference.
  - Explanations: SHAP summaries per model/dataset for interpretability and feature auditing.

## Configuration

Provide a single YAML file describing datasets, preprocessing, splitting, balancing, feature elimination, models, and evaluation. A typical template includes dataset paths, lead selection, annotation extensions, cleaning method, segmentation, feature elimination (RFECV), excluded labels, splitting strategy, resampling, grid search, and explainability flags.
Example (adapt keys to your implementation):
```yaml
dataset:
  train:
    - name: MIT-ARR
      root: /data/mit_arr
      ann_ext: atr
      lead: MLII
    - name: MIT-NSR
      root: /data/mit_nsr
      ann_ext: atr
      lead: MLII
  test_external:
    - name: INCART
      root: /data/incart
      ann_ext: atr
      lead: II

preprocessing:
  segment_length_minutes: 30
  cleaning_method: vg          # neurokit2 ecg_clean method (e.g., neurokit, pantompkins1985, hamilton2002, elgendi2010, engzeemod2012, vg)
  keep_only_valid_beats: true  # enforce complete/ordered peaks
  min_samples_per_patient: 300

features:
  extract: true                 # 27 features across lengths, angles, slopes, ratios, peaks, RR
  normalize_by_fs: true

feature_selection:
  method: rfecv
  with_grid_search: false
  estimator: random_forest
  scaler: robust
  exclude_labels: [F, Q]

splitting:
  paradigm: inter_patient       # inter_patient | intra_patient | loso
  desired_train_test_ratio: 0.8 # framework iteratively adjusts to achieve ratio

balancing:
  method: adasyn                # none | undersample | smote | adasyn

models:
  use: [logreg, rf, xgb, lgbm, mlp, svm]
  grid: default                 # use built-in grids; custom grids supported

evaluation:
  cross_dataset: true           # train on train sets, evaluate on external
  metrics: [accuracy, recall, precision, f1]
  confusion_matrix_excel: true

explain:
  shap: true
```
This mirrors the manuscript’s configuration sections, including dataset selection, NeuroKit2 cleaning/delineation, RFECV, inter/intra/LOSO, class balancing, and SHAP explanations.

## Datasets

ELMSLEY supports multiple ECG datasets and cross‑dataset evaluation. Commonly used sets include:
- MIT‑BIH Arrhythmia (MIT‑ARR): 48 two‑channel ambulatory ECG records, 30 minutes each, 360 Hz, arrhythmic and normal rhythms with expert annotations.
- MIT‑BIH Normal Sinus Rhythm (MIT‑NSR): 18 long‑term recordings from healthy adults, ~24 hours each, 128 Hz, rhythm and beat annotations.
- INCART: 75 annotated recordings derived from Holter monitors, 12 leads, 257 Hz, ~30 minutes each, >175k manually corrected beat annotations; external evaluation target in examples.
Use WFDB‑compatible folder structures and select the target lead per dataset in the YAML file for consistent processing and feature extraction.

## Preprocessing and segmentation

- Cleaning: NeuroKit2 ecg_clean methods are configurable (e.g., neurokit, pantompkins1985, hamilton2002, elgendi2010, engzeemod2012, vg) for noise reduction and robust R‑peak detection.
- Delineation: NeuroKit2 ecg_delineate identifies onsets/offsets around provided R‑peaks; only fully segmented, physiologically ordered beats are kept to ensure feature correctness.
- AAMI mapping: dataset annotations are mapped to AAMI classes (N, SVEB, VEB, F, Q) or a custom dictionary; rare classes (F, Q) can be excluded in configs due to low numerosity.

## Features

ELMSLEY extracts 27 sampling‑rate‑aware features:
- Segment lengths: P, PQ, QRS, QR, QT, RS, ST, T, PT.
- Angles: PonPQ, RoffQR, QRS, RST, STToff.
- Slopes: PQ, QR, RS, ST.
- Ratios: QR/QS, RS/QS.
- Peaks: P, Q, R, S, T.
- RR intervals: RR Back, RR Forw, RR mean.
These morphological/time‑domain features generalize across acquisition setups and datasets; RFECV can reduce the set to the most discriminative subset per experiment.

## Splitting and balancing

- Splitting: intra‑patient and inter‑patient paradigms are supported; LOSO is available for per‑subject holdout with majority‑vote aggregation at inference time.
- Ratio control: inter‑patient splits iteratively adjust patient partitions to reach a target train/test ratio while ensuring minimum test coverage (≥10%).
- Balancing: Random Undersampling, SMOTE, and ADASYN address class imbalance; ADASYN is highlighted in example configurations.

## Models and hyperparameters

Built‑in models with default grid‑search ranges:
- Logistic Regression: C ∈ [1, 0.8, 0.5], solvers ∈ {newton‑cg, sag, saga, lbfgs}.
- Random Forest: n_estimators ∈ {50,100,150,200}, max_features ∈ [1..n_features], max_depth ∈ {4,6,8}.
- XGBoost: same tree grids plus learning_rate ∈ {0.001, 0.01, 0.1, 1}.
- LightGBM: analogous to XGBoost grid.
- MLP: hidden_layer_sizes={(100,)}, activation ∈ {logistic,relu}, solver ∈ {adam,sgd}, batch_size ∈ {256,512}, lr_init ∈ {0.001,0.01,0.1}, max_iter ∈ {200,400,800}.
- SVM: C ∈ {0.5, 0.8, 1.0}, kernel ∈ {rbf, linear, sigmoid}.
Custom estimators and grids can be integrated via configuration hooks.

## Evaluation and metrics

- Standard and cross‑dataset evaluation are supported; train on one/more datasets, evaluate on held‑out test and external datasets (e.g., INCART).
- Metrics: accuracy, recall, precision, F1, plus per‑experiment confusion matrices saved as .xlsx for further analysis.
- LOSO: trains one model per subject‑held‑out fold; final predictions via majority voting; best fold models selected by recall to minimize false negatives in clinical contexts.
- Empirically, Random Forest and XGBoost achieved ≥95% across metrics on INCART in example settings, showing strong cross‑dataset generalization with the proposed feature set (see manuscript results).

## Explainability

SHAP values are computed on test and external sets to quantify each feature’s marginal contribution to predictions, enabling local explanations, auditing of morphological feature importance (e.g., QRS length/angle, slopes, RR intervals), and improved clinical trust in model outputs.

## Example configurations

The manuscript includes three YAML configurations:
- Config 1: Inter‑patient split, RFECV‑reduced features, exclude F and Q, train all six models, evaluate on internal test and external INCART.
- Config 2: As above with ADASYN oversampling in training; evaluate internal test and INCART.
- Config 3: LOSO strategy across subjects, train on MIT‑ARR+MIT‑NSR, evaluate on INCART only with majority‑vote predictions.
Use these as templates in configs/ to replicate experiments and adapt them to new datasets.

## Project structure

```
ELMSLEY/
  ├─ elmsley/            # core modules: io, preprocessing, features, selection, split, train, eval, explain
  ├─ scripts/            # runnable scripts and CLIs (e.g., run_pipeline.py)
  ├─ configs/            # YAML templates (e.g., default.yaml, config1.yaml, config2.yaml, config_loso.yaml)
  ├─ data/               # optional sample/WFDB-format placeholders
  ├─ tests/              # test suite
  ├─ pyproject.toml      # or setup.cfg/setup.py (build metadata)
  ├─ requirements.txt    # dependencies (if present)
  ├─ README.md
  └─ LICENSE
```
Adjust to the exact layout of your repository and keep internal links relative for reliability across branches and forks.

## Testing

- Run tests
```bash
pytest -q
```
- Coverage (if configured)
```bash
pytest --maxfail=1 --disable-warnings -q --cov=elmsley
```
Including quick tests helps users validate their environment before running larger pipelines.

## Troubleshooting

- Incomplete/incorrect peak sequences: use a different NeuroKit2 cleaning method (e.g., switch to vg, hamilton2002, or pantompkins1985) and re‑run; ELMSLEY keeps only fully segmented beats to preserve feature quality.
- Class imbalance: enable ADASYN or SMOTE in the YAML to improve minority class metrics without manual resampling.
- Train/test ratio with inter‑patient splits: adjust desired ratio in YAML; the splitter iteratively refines partitions while guaranteeing a minimum test portion.

## Contributing


## License


## Citation
