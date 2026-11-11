
from pathlib import Path
import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_df = pd.read_csv(
    ROOT / "data" / "processed" / "train.csv"
)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["model"]["random_state"]
)

model.fit(X_train, y_train)

model_dir = ROOT / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "iris_model.joblib")

print("Model training completed!")
print("Model saved successfully!")
