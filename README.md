# Uterine EHG Analysis for Preterm Birth Detection
Time-frequency analysis of uterine EHG signals for term and preterm pregnancy classification
Exploratory machine-learning analysis of uterine electrohysterogram (EHG) recordings to separate term from preterm deliveries, using the public TPEHG database from PhysioNet.

**Author:** [S.M.Shahriar Shadhin] Department of Biomedical Engineering, Islamic University
            * [Sidratul Muntaha] (Department of Biomedical Engineering, Islamic University)
            * [Toufique Ahmed] (Department of Biomedical Engineering, Islamic University)
            * [Unit Chakma Shuvro] (Department of Biomedical Engineering, Islamic University)
            * [Nishad Al Hasan] (Department of Biomedical Engineering, Islamic University)
            * [Moudud Ahmed] (Department of Biomedical Engineering, Islamic University)
**Status:** Completed as a learning and research project. Results are exploratory.

## Dataset

- **Term-Preterm ElectroHysteroGram (TPEHG) database**, PhysioNet: https://physionet.org/content/tpehgdb/1.0.1/
- 300 recordings: 262 term, 38 preterm
- 20 Hz sampling, about 30 minutes per recording, 3 physical channels (each with three pre-filtered versions)
- Preterm label: gestation under 37 weeks, read from each recording's header (not guessed from record names)

The data are downloaded directly from PhysioNet inside the notebooks, so they are not stored in this repo.

## Pipeline

| Notebook | Purpose |
|---|---|
| `00_setup` | Install and check the Python libraries |
| `01_dataset_exploration` | Explore the database structure |
| `02_metadata_and_signal_analysis` | Metadata and basic signal statistics |
| `03_ehg_signal_preprocessing` | Butterworth filtering and preprocessing |
| `04_time_domain_feature_extraction` | Time-domain features |
| `05_frequency_domain_analysis` | Welch power spectrum features |
| `06_time_frequency_analysis` | Spectrogram and wavelet analysis |
| `07_full_dataset_feature_extraction` | Features for all 300 recordings, labels, model comparison, statistics and final figures (all in one notebook) |

Later steps in the same notebook workflow:

1. **Labels:** gestation weeks read from record headers, then split at 37 weeks.
2. **Version 2 features:** 21 features from all three channels, using the 0.3-3 Hz filtered signal (RMS, peak, median and mean frequency, spectral entropy, relative band power).
3. **Model comparison:** logistic regression, random forest and SVM with balanced class weights, using 5-fold cross-validation repeated 5 times. Duration and label-defining columns were excluded to avoid leakage.
4. **Statistics:** Mann-Whitney U test per feature with Bonferroni correction, and a check of recording time (early vs late pregnancy).

## Results

| Features | Model | AUC (mean ± SD) |
|---|---|---|
| Version 1 (S1, 17 features) | Logistic regression | 0.447 ± 0.095 |
| | Random forest | 0.563 ± 0.103 |
| | SVM (RBF) | 0.417 ± 0.099 |
| Version 2 (3 channels, 21 features) | Logistic regression | 0.510 ± 0.072 |
| | Random forest | 0.586 ± 0.092 |
| | SVM (RBF) | 0.527 ± 0.093 |

- Performance is close to chance for all models. The random forest predicted every recording as term.
- No single feature stayed significant after Bonferroni correction.
- **Exploratory observation:** relative power in the 0.3-1 Hz band of channel 3 was higher in preterm recordings made before week 27 (AUC 0.71, uncorrected p = 0.0035, 19 preterm cases), but not in later recordings (AUC 0.58, p = 0.27). Recording time was balanced between groups, so it does not explain this pattern. The finding was not corrected for multiple comparisons and needs confirmation in independent data.

Figures are in `results/`: cross-validated ROC curves, SVM confusion matrix, and the channel-3 boxplots.

## Limitations

- Only 38 preterm recordings, so all estimates are very uncertain.
- The channel-3 observation was found after examining many features and subgroups and has not been validated on independent data.
- Only simple spectral and time-domain features were used.
- Single-centre dataset; results may not generalize.

## Repository structure

```
notebooks/   Jupyter/Colab notebooks (00-07 plus later analysis)
data/        Feature tables with labels, model comparison and test tables (CSV)
results/     Figures (PNG)
README.md
```

## How to run

1. Open the notebooks in Google Colab (File → Open notebook → GitHub).
2. Run them in order. Libraries used: `wfdb`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `PyWavelets`, `matplotlib`, `tqdm`.
3. Notebook 07 and the later analysis save outputs to Google Drive.

## References

- Fele-Žorž G. et al. (2008). A comparison of various linear and non-linear signal processing techniques to separate uterine EMG records of term and pre-term delivery groups. *Medical & Biological Engineering & Computing*.
- Goldberger A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*.
