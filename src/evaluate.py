
from pathlib import Path
import pandas as pd
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

ROOT = Path(__file__).resolve().parent.parent

test_df = pd.read_csv(
    ROOT / "data" / "processed" / "test.csv"
)

model = joblib.load(ROOT / "models" / "iris_model.joblib")

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

predictions = model.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "precision": round(float(precision_score(y_test, predictions, average="weighted")), 4),
    "recall": round(float(recall_score(y_test, predictions, average="weighted")), 4),
    "f1_score": round(float(f1_score(y_test, predictions, average="weighted")), 4)
}

report_dir = ROOT / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

with open(report_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model evaluation completed!")
print(json.dumps(metrics, indent=4))
