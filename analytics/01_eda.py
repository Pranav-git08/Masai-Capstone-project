import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("titanic")

os.makedirs(".", exist_ok=True)
df.to_csv("analytics/titanic.csv", index=False)

print(df.info())
print(df.describe())
print(df.shape)

missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

df_cleaned = df.drop(columns=["deck"]).copy()
df_cleaned["age"] = df_cleaned["age"].fillna(df_cleaned["age"].median())
df_cleaned = df_cleaned.dropna(subset=["embarked", "embark_town"])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df_cleaned["age"], kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution")
sns.boxplot(x=df_cleaned["age"], ax=axes[0, 1])
axes[0, 1].set_title("Age Boxplot")
sns.histplot(df_cleaned["fare"], kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Fare Distribution")
sns.boxplot(x=df_cleaned["fare"], ax=axes[1, 1])
axes[1, 1].set_title("Fare Boxplot")
plt.tight_layout()
plt.savefig("analytics/chart_univariate.png")
plt.close()

def count_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return len(series[(series < lower_bound) | (series > upper_bound)])

print("Age outliers count:", count_iqr_outliers(df_cleaned["age"]))
print("Fare outliers count:", count_iqr_outliers(df_cleaned["fare"]))

mean_fare = df_cleaned["fare"].mean()
median_fare = df_cleaned["fare"].median()
mode_fare = df_cleaned["fare"].mode()[0]
print(f"Fare Mean: {mean_fare:.2f}, Median: {median_fare:.2f}, Mode: {mode_fare:.2f}")

print("Sex breakdown:\n", df_cleaned.groupby("sex")["survived"].mean())
print("Pclass breakdown:\n", df_cleaned.groupby("pclass")["survived"].mean())
print("Sex & Pclass breakdown:\n", df_cleaned.groupby(["sex", "pclass"])["survived"].mean())

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_cleaned[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("analytics/chart_corr_heatmap.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.barplot(x="pclass", y="survived", hue="sex", data=df_cleaned, ci=None)
plt.title("Survival Rate by Class and Sex")
plt.savefig("analytics/chart_1_survival_sex_pclass.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.violinplot(x="sex", y="age", hue="survived", data=df_cleaned, split=True, inner="quart")
plt.title("Age vs Survival by Sex")
plt.savefig("analytics/chart_2_age_distribution_survival.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="pclass", y="fare", hue="survived", data=df_cleaned)
plt.ylim(0, 200)
plt.title("Fare vs Pclass and Survival")
plt.savefig("analytics/chart_3_fare_vs_pclass_survival.png")
plt.close()

df_cleaned["family_size"] = df_cleaned["sibsp"] + df_cleaned["parch"] + 1
plt.figure(figsize=(8, 5))
sns.barplot(x="family_size", y="survived", data=df_cleaned, ci=None)
plt.title("Survival Rate by Family Size")
plt.savefig("analytics/chart_4_family_size_survival.png")
plt.close()

scaler = StandardScaler()
scaled_vals = scaler.fit_transform(df_cleaned[["age", "fare"]])
df_scaled = pd.DataFrame(scaled_vals, columns=["age_z", "fare_z"])

print("Original Stats:")
print(df_cleaned[["age", "fare"]].agg(["mean", "std"]))

print("Standardized Stats:")
print(df_scaled.agg(["mean", "std"]).round(4))
