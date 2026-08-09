# Module 2 — Analytics Pipeline (`/analytics`)

This module implements an end-to-end analytics and machine learning workflow on Seaborn's Titanic dataset, covering data profiling, missing-value handling, exploratory data analysis, classification, hyperparameter tuning, and a regression side-task.

---

## Part A — Data Profiling, Cleaning & Data Story

### 1. Dataset Profiling & Missing Values
- **Initial Shape:** 891 rows, 15 columns.
- **Offline Fallback:** Raw dataset saved locally to `analytics/titanic.csv` immediately upon single-pass load.

#### Missing Value Percentages & Threshold Rule Decisions
- **`deck` (77.22% missing):** Dropped column because it exceeds the 30% threshold where imputation becomes unreliable.
- **`age` (19.87% missing):** Imputed with median (28.0 years) as it falls within the 5%–30% threshold range.
- **`embarked` / `embark_town` (0.22% missing):** Dropped missing rows as the missing rate is under 5%.

---

### 2. Univariate Analysis (Age & Fare)
- **Age Outliers (IQR):** 65 rows outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
- **Fare Outliers (IQR):** 114 rows outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
- **Fare Statistics:** Mean = £32.10, Median = £14.45, Mode = £8.05.
- **Skewness Conclusion:** Because $\text{Mean } (32.10) > \text{Median } (14.45) > \text{Mode } (8.05)$, the `fare` distribution is strongly **right-skewed**, heavily influenced by high-value luxury tickets.

---

### 3. Bivariate Analysis & Correlation Matrix

#### Survival Rate Breakdowns
- **Overall Survival Rate:** 38.38%
- **By Sex:** Female = 74.04% | Male = 18.89%
- **By Pclass:** Class 1 = 62.62% | Class 2 = 47.28% | Class 3 = 24.24%
- **By Sex & Pclass:**
  - Female: Class 1 = 96.74% | Class 2 = 92.11% | Class 3 = 50.00%
  - Male: Class 1 = 36.89% | Class 2 = 15.74% | Class 3 = 13.54%

#### Top 2 Correlations
1. **`pclass` vs `fare` ($r = -0.55$):** Strong inverse correlation showing higher socio-economic class passengers paid substantially higher ticket prices.
2. **`survived` vs `pclass` ($r = -0.34$):** Moderate negative correlation demonstrating that higher passenger class directly improved survival probability.

---

### 4. Multivariate Data Story

1. **`chart_1_survival_sex_pclass.png` (Survival by Class & Sex):** Highlights how 1st-class female passengers achieved a 96.7% survival rate, whereas 3rd-class male passengers dropped to 13.5%. This demonstrates the combined effect of the "women and children first" evacuation protocol and socio-economic privilege.
2. **`chart_2_age_distribution_survival.png` (Age Distribution by Sex & Survival):** Shows a prominent survival peak for young children ($\le 5$ years old) across both sexes. Conversely, adult males across most age brackets suffered low survival rates during evacuation.
3. **`chart_3_fare_vs_pclass_survival.png` (Fare Distribution by Class & Survival):** Demonstrates that within Class 1, passengers who paid higher fares had higher survival rates. Higher fare values correlate with upper-deck cabin locations closer to lifeboats.
4. **`chart_4_family_size_survival.png` (Survival Rate by Family Size):** Reveals survival peaks for small family groups (sizes 2–4 at ~55%–72%) but drops sharply for solo travelers (30%) and large families ($\ge 5$ members at $<20\%$), who faced coordination challenges during the ship's sinking.

---

### 5. Exploratory Standardization Check
- **`age` Transformation:** Pre-scaling Mean = 29.32, Std = 12.98 $\rightarrow$ Post-scaling Mean = 0.0000, Std = 1.0006
- **`fare` Transformation:** Pre-scaling Mean = 32.10, Std = 49.70 $\rightarrow$ Post-scaling Mean = 0.0000, Std = 1.0006

---

## Part B — Predictive Modeling & Evaluation

### 1. Stratified Split Justification
An 80/20 train/test split stratified on `survived` was used to ensure that both training and testing sets maintain the underlying ~38.4% positive class ratio, preventing evaluation bias.

---

### 2. Model Performance Comparison

#### Classification Models
| Model | Accuracy | Precision | Recall | F1 Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8146 | 0.8070 | 0.6765 | 0.7360 | 0.8582 |
| **Decision Tree (depth=4)** | 0.8034 | 0.7895 | 0.6618 | 0.7200 | 0.8131 |
| **Random Forest (tuned)** | **0.8146** | **0.7778** | **0.7206** | **0.7481** | **0.8314** |

#### Regression Model (Predicting `fare`)
| Model | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Multivariate Linear Regression** | £17.88 | £40.49 | 0.3854 | 0.3601 |

---

### 3. Imbalance Handling Comparison

| Strategy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Baseline (No handling)** | **0.8070** | 0.6765 | **0.7360** |
| **`class_weight='balanced'`** | 0.7246 | **0.7353** | 0.7299 |
| **SMOTE (Train Fold Only)** | 0.7353 | **0.7353** | 0.7353 |

**Conclusion:** The baseline logistic regression yielded the highest overall F1 score (0.7360) and precision (0.8070). While `class_weight='balanced'` and `SMOTE` increased recall to 0.7353, they introduced additional false positives, slightly lowering overall precision and F1.

---

### 4. Hyperparameter Tuning Results
- **GridSearch Parameters:** `max_depth`: `None`, `max_features`: `'sqrt'`, `n_estimators`: `100`
- **Out-of-Bag (OOB) Score:** `0.8045`

---

### 5. Heteroscedasticity Analysis
The residual plot (`chart_residuals.png`) displays clear **heteroscedasticity**. The residual spread expands significantly as predicted fares increase, caused by the high price variance among 1st-class luxury tickets compared to lower-class fares.

---

### 6. Deployment Recommendation
I recommend deploying the **Tuned Random Forest Pipeline** for production. It achieves the highest overall F1 score (0.7481) and Recall (0.7206) among all evaluated classifiers, while maintaining strong Accuracy (81.46%) and AUC (0.8314). Its ensemble design effectively captures non-linear interactions across passenger class, age, and sex without overfitting.

---

### 7. End-to-End Pipeline Artifact
The fitted pipeline (preprocessing transformer + classifier) is saved at `analytics/best_titanic_pipeline.joblib`. It accepts raw data input and executes end-to-end predictions cleanly.
