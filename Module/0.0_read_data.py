import pandas as pd
import json
import datetime
import os
import subprocess
import warnings
from config import (
    DATA_FILTERED_FILE, 
    RAW_DATA_FILE, 
    DATE_LIMITS_FILE, 
    PROJ_ROOT,
    MAX_DATE,
    MIN_DATE,
    ARTIFACTS_DIR
    )

######## Constants used
max_date = MAX_DATE
min_date = MIN_DATE
########################

os.makedirs(ARTIFACTS_DIR,exist_ok=True)
print("Created artifacts directory")

warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x: "%.3f" % x)


# 2. Run DVC command
subprocess.run(["dvc", "update", "data/raw/raw_data.csv.dvc"], check=True, cwd=PROJ_ROOT)
subprocess.run(["dvc", "pull"], check=True, cwd=PROJ_ROOT)


print("Loading training data")
data = pd.read_csv(RAW_DATA_FILE)



if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

# Time limit data
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open(DATE_LIMITS_FILE, "w") as f:
    json.dump(date_limits, f) # Save date intervals

filtered_data = data
filtered_data.to_csv(DATA_FILTERED_FILE, index=False)
    