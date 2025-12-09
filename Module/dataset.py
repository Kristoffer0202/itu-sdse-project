from pathlib import Path

import typer
import pandas as pd
import json
import datetime
import os
import subprocess
import warnings

from config import DATE_LIMITS_FILE, PROJ_ROOT, RAW_DATA_FILE, DATA_FILTERED_FILE


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_FILE,
    date_limits_path: Path = DATE_LIMITS_FILE,
    output_path: Path = DATA_FILTERED_FILE,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    os.makedirs("artifacts",exist_ok=True)
    print("Created artifacts directory")

    warnings.filterwarnings('ignore')
    pd.set_option('display.float_format',lambda x: "%.3f" % x)

    # 2. Run DVC command
    subprocess.run(["dvc", "update", "data/raw/raw_data.csv.dvc"], check=True, cwd=PROJ_ROOT)
    subprocess.run(["dvc", "pull"], check=True, cwd=PROJ_ROOT)

    print("Loading training data")
    data = pd.read_csv(input_path)

    max_date = "2024-01-31"
    min_date = "2024-01-01"

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
    with open(date_limits_path, "w") as f:
        json.dump(date_limits, f)

    filtered_data = data
    filtered_data.to_csv(output_path)
    # -----------------------------------------


if __name__ == "__main__":
    app()
