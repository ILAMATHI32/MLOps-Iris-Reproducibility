
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "params.yaml", "r") as f:
    params = yaml.safe_load(f)

df = pd.read_csv(ROOT / "data" / "raw" / "iris.csv")

train_df, test_df = train_test_split(
    df,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_state"],
    stratify=df["target"]
)

output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("Data preparation completed!")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
