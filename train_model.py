"""Train the model the web app uses.

Usage:  python train_model.py tpehg_features_raw.csv
The CSV needs one row per recording, the feature columns, and a label column
(Term / Preterm, or 0 / 1). Output: model.joblib (put it next to app.py).
"""
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

path = sys.argv[1] if len(sys.argv) > 1 else "tpehg_features_raw.csv"
df = pd.read_csv(path)
label_col = next(c for c in df.columns if c.lower() in ("label", "class", "target", "group", "preterm"))
y = df[label_col].astype(str).str.lower().map(lambda v: 1 if v in ("1", "preterm", "p", "true") else 0)
X = df.drop(columns=[label_col]).select_dtypes("number")
X = X.fillna(X.median())

model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
print("5-fold CV AUC:", cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean().round(3))
model.fit(X, y)
joblib.dump({"model": model, "features": list(X.columns)}, "model.joblib")
print("Saved model.joblib with", X.shape[1], "features")
