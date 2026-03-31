# In-Hospital Mortality Prediction

A machine learning and deep learning pipeline for predicting in-hospital mortality using ICU time-series data from the **MIMIC-III** benchmark dataset.

---

## Overview

This project tackles the clinical task of predicting patient mortality during an ICU stay, using the first 48 hours of physiological measurements. Three modelling approaches are compared:

- **Logistic Regression** — baseline linear model
- **Random Forest** — ensemble tree-based model
- **XGBoost** — gradient boosted trees with class imbalance handling
- **LSTM** — sequence model that operates directly on the raw time-series

Models are evaluated using **ROC-AUC** and **PR-AUC** (area under the precision-recall curve), with the latter being particularly informative given the class imbalance (~13.5% positive mortality rate).

---

## 📁 Project Structure

```
mortality_pred.ipynb       # Main notebook with full pipeline
```

---

## Dataset

This project uses the **MIMIC-III Clinical Database** processed via the [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) framework.

**Data is not included in this repository.** You must request access through PhysioNet and generate the benchmark datasets yourself.

- **Task:** In-hospital mortality (binary classification)
- **Input:** ICU time-series CSV files (one per patient episode)
- **Labels:** `listfile.csv` (maps episode filenames to mortality labels)
- **Split:** Pre-defined train/test split from the benchmark

### Features Used

| Feature | Feature |
|---|---|
| Capillary refill rate | Heart Rate |
| Diastolic blood pressure | Height / Weight |
| Fraction inspired oxygen | Mean blood pressure |
| Glasgow Coma Scale (Eye, Motor, Verbal, Total) | Oxygen saturation |
| Glucose | Respiratory rate |
| Systolic blood pressure | Temperature |
| pH | — |

---

## Pipeline

### 1. Data Loading
- Loads all `_timeseries.csv` episode files from the MIMIC-III benchmark directory
- Merges with mortality labels from `listfile.csv`
- Processes train and test sets separately

### 2. Preprocessing
- **GCS encoding:** Maps free-text Glasgow Coma Scale responses to integer scores (1–6 for motor, 1–4 for eye, 1–5 for verbal)
- **GCS total:** Computed as the sum of the three sub-scores
- **Missing value imputation:** Forward-fill → backward-fill within each episode, then fallback to clinically normal reference values
- **Aggregation:** Median of each feature across the episode's time-series (for tabular models)

### 3. Modelling

#### Tabular Models (Logistic Regression, Random Forest, XGBoost)
- Input: per-episode median feature vector
- 80/20 stratified train/test split
- Logistic Regression: `StandardScaler` + `class_weight='balanced'`
- Random Forest: 500 trees, `class_weight='balanced'`
- XGBoost: 2000 estimators, `scale_pos_weight` tuned for class imbalance, early stopping on validation AUC

#### LSTM (Sequence Model)
- Input: raw time-series padded/truncated to **48 timesteps × 17 features**
- Architecture: `Masking → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(16) → Sigmoid`
- `StandardScaler` applied across the flattened time dimension
- `class_weight` computed via `sklearn.utils.class_weight.compute_class_weight`
- Early stopping on validation AUROC with `ReduceLROnPlateau`

### 4. Text Export
- Median feature values are also serialised as natural-language text strings per patient, saved to `mort_text.csv` (for potential LLM-based experiments)

---

## Requirements

```
numpy
pandas
scikit-learn
xgboost
tensorflow / keras
seaborn
matplotlib
```

Install with:

```bash
pip install numpy pandas scikit-learn xgboost tensorflow seaborn matplotlib
```

---

## Usage

1. Obtain MIMIC-III access from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) and generate the benchmark data using [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks).

2. Update the data paths in the notebook:
   ```python
   data_dir = '/path/to/mimic3-benchmarks/data/in-hospital-mortality/train'
   label_file = '/path/to/mimic3-benchmarks/data/in-hospital-mortality/train/listfile.csv'
   ```

3. Run all cells in `mortality_pred.ipynb`.

---

## Evaluation Metrics

Models are evaluated on:

- **ROC-AUC** — overall discrimination ability
- **PR-AUC** (Average Precision) — preferred for imbalanced datasets, reflects precision-recall trade-off at all thresholds
- **Brier Score** — for the LSTM model (probabilistic calibration)

---

## Notes

- The dataset is **imbalanced** (~13.5% positive class). All models include explicit strategies to handle this (class weighting or `scale_pos_weight`).
- The LSTM uses zero-padding with a `Masking` layer to handle variable-length episodes.
- GCS sub-scores and the composite total are both retained as features.

---

## References

- Johnson, A. et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*.
- Harutyunyan, H. et al. (2019). Multitask learning and benchmarking with clinical time series data. *Scientific Data*. [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
# mimic3_analysis
Analysis of Health Care Data from MIMIC 3 dataset
