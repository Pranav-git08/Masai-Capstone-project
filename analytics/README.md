# Module 2 — Analytics Pipeline (/analytics)

## Overview
This module builds a predictive modeling and analysis pipeline using Seaborn's Titanic dataset, covering dataset profiling, missing value handling, exploratory analysis, classification, hyperparameter tuning, and regression.

## Key Decisions & Findings
- **Missing Data:** Dropped `deck` (>70% missing), imputed `age` (19.8% missing) using median, and dropped rows with missing `embarked` values.
- **Univariate Skewness:** `fare` is strongly right-skewed (Mean: 32.20 > Median: 14.45 > Mode: 8.05).
- **Strongest Correlations:** `pclass` vs `fare` (-0.55) and `survived` vs `pclass` (-0.34).
- **Stratified Split:** Used an 80/20 train/test split preserving the ~38.4% survival balance.

## Model Results
- **Random Forest (Tuned):** Accuracy ~83.2%, F1 ~0.769, AUC ~0.871.
- **Regression (Fare Prediction):** MAE ~18.42, R2 ~0.44. Residuals exhibit heteroscedasticity due to high variance in luxury fares.

## Saved Artifact
The fitted pipeline (preprocessing + classifier) is saved to `analytics/best_titanic_pipeline.joblib` and reloadable for end-to-end inference on raw data.
