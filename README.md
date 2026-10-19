# Uterine EMG Analysis: Time-Frequency Based Preterm Birth Detection

## Overview
This repository implements an end-to-end biomedical signal processing and machine learning pipeline to classify term vs. preterm deliveries using electrohysterogram (EHG) records from the **PhysioNet Term-Preterm EHG Database (TPEHG) v1.0.1**.

## Methodology
1. **Signal Preprocessing:**
   - 4th-order Butterworth bandpass filter ($0.34\text{ Hz} - 3.0\text{ Hz}$) targeting physiological uterine contraction bursts on Channel 3 ($S_3$).
   - 120-second epoch segmentation ($2400\text{ samples}$ per window at $20\text{ Hz}$).
2. **Time-Frequency Decomposition:**
   - 4-level Discrete Wavelet Transform (DWT) using `db4` mother wavelet to extract sub-band energy and dispersion.
   - Continuous Wavelet Transform (CWT) Morlet scalogram energy analysis.
   - Spectral metrics: Peak Frequency ($F_{\text{peak}}$), Median Frequency ($F_{\text{med}}$), and Spectral Entropy via Welch's PSD.
3. **Machine Learning & Validation:**
   - Patient-level **GroupKFold (5-fold)** cross-validation to prevent intra-patient data leakage across epochs.
   - Class imbalance correction using **SMOTE** on training folds.
   - Classifiers: Random Forest and XGBoost evaluated primarily via **AUC-ROC**.

## Repository Structure
```
├── dataset/                    # Ignored in git (PhysioNet TPEHG v1.0.1)
├── notebooks/
│   └── tpehg_pipeline.ipynb    # Complete pipeline executed in Google Colab
├── results_roc_curve.png       # 5-Fold Cross-Validation ROC curves
├── shap_summary.png            # Feature importance and explainability plot
├── tpehg_time_frequency_features.csv # Extracted feature matrix
├── requirements.txt            # Dependency specification
└── README.md
```
