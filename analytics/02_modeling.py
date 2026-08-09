import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("analytics/titanic.csv")

if "deck" in df.columns:
    df = df.drop(columns=["deck"])
df = df.dropna(subset=["embarked"])

X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = ["age", "fare", "sibsp", "parch"]
cat_features = ["sex", "embarked", "pclass"]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ]
)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"\n{name} Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

dt_pipe = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]
)
dt_pipe.fit(X_train, y_train)

plt.figure(figsize=(14, 8))
plot_tree(
    dt_pipe.named_steps["clf"],
    filled=True,
    class_names=["Died", "Survived"],
    feature_names=dt_pipe.named_steps["prep"].get_feature_names_out(),
)
plt.savefig("analytics/chart_decision_tree.png")
plt.close()

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

log_base = LogisticRegression(random_state=42).fit(X_train_prep, y_train)
y_pred_base = log_base.predict(X_test_prep)

log_bal = LogisticRegression(class_weight="balanced", random_state=42).fit(
    X_train_prep, y_train
)
y_pred_bal = log_bal.predict(X_test_prep)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_prep, y_train)
log_smote = LogisticRegression(random_state=42).fit(X_res, y_res)
y_pred_smote = log_smote.predict(X_test_prep)

print("\nBaseline -> Precision:", precision_score(y_test, y_pred_base), "Recall:", recall_score(y_test, y_pred_base), "F1:", f1_score(y_test, y_pred_base))
print("Balanced -> Precision:", precision_score(y_test, y_pred_bal), "Recall:", recall_score(y_test, y_pred_bal), "F1:", f1_score(y_test, y_pred_bal))
print("SMOTE -> Precision:", precision_score(y_test, y_pred_smote), "Recall:", recall_score(y_test, y_pred_smote), "F1:", f1_score(y_test, y_pred_smote))

param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [4, 6, 8, None],
    "clf__max_features": ["sqrt", "log2"],
}

rf_pipe = Pipeline(
    [
        ("prep", preprocessor),
        (
            "clf",
            RandomForestClassifier(
                oob_score=True, random_state=42, bootstrap=True
            ),
        ),
    ]
)

grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
print("\nBest Params:", grid.best_params_)
print("OOB Score:", best_pipeline.named_steps["clf"].oob_score_)

X_reg = df[["pclass", "age", "sibsp", "parch", "survived", "sex", "embarked"]]
y_reg = df["fare"]

X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_prep = ColumnTransformer(
    transformers=[
        ("num", num_transformer, ["age", "sibsp", "parch", "survived"]),
        ("cat", cat_transformer, ["sex", "embarked", "pclass"]),
    ]
)

reg_pipe = Pipeline([("prep", reg_prep), ("reg", LinearRegression())])
reg_pipe.fit(X_reg_tr, y_reg_tr)
y_reg_pred = reg_pipe.predict(X_reg_te)

mae_val = mean_absolute_error(y_reg_te, y_reg_pred)
rmse_val = np.sqrt(mean_squared_error(y_reg_te, y_reg_pred))
r2_val = r2_score(y_reg_te, y_reg_pred)

n_samples = len(y_reg_te)
n_features = X_reg_tr.shape[1]
adj_r2_val = 1 - (1 - r2_val) * (n_samples - 1) / (n_samples - n_features - 1)

print(
    f"\nRegression -> MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}, R2: {r2_val:.4f}, Adj R2: {adj_r2_val:.4f}"
)

residuals = y_reg_te - y_reg_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("analytics/chart_residuals.png")
plt.close()

joblib.dump(best_pipeline, "analytics/best_titanic_pipeline.joblib")

loaded_model = joblib.load("analytics/best_titanic_pipeline.joblib")
sample_data = pd.DataFrame(
    [
        {
            "pclass": 1,
            "sex": "female",
            "age": 29.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.33,
            "embarked": "S",
        }
    ]
)

print("\nSample Prediction:", loaded_model.predict(sample_data))
print("Sample Probability:", loaded_model.predict_proba(sample_data)[:, 1])
