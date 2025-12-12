import datetime
import os
import shutil
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import pickle

from config import (
    TRAINING_GOLD_DATA_FILE,
    DATA_VERSION,
    TRAIN_TEST_DATA_FILE,
    MODELS_DIR,
)

###################################
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
args = parser.parse_args()
print("Run name:", args.run_name)
experiment_name = args.run_name
###################################


# Constants used:
data_gold_path = TRAINING_GOLD_DATA_FILE
data_version = DATA_VERSION
#experiment_name = current_date

##### Should look into where these are created and if they can be referenced #####
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)
os.makedirs(str(MODELS_DIR), exist_ok=True)
####################################################

mlflow.set_experiment(experiment_name)


def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


data = pd.read_csv(data_gold_path)
print(f"Training data length: {len(data)}")
data.head(5)


data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)


for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")


y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)
y_train

with open(TRAIN_TEST_DATA_FILE, 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)


