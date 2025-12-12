import time
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient
import argparse
import pandas as pd
from mlflow.tracking import MlflowClient
import json

from config import MODEL_RESULTS_PATH, ARTIFACTS_DIR

#  Constants used: #################################
model_results_path = MODEL_RESULTS_PATH
with open(model_results_path, 'r') as results_file:
    model_results = json.load(results_file)

artifact_path = "model"
model_name = "lead_model"
#################################

###################################
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
args = parser.parse_args()
print("Run name:", args.run_name)
experiment_name = args.run_name
###################################



def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
experiment_ids

experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1
).iloc[0]
experiment_best



results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T
results_df

best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
print(f"Best model: {best_model}")


client = MlflowClient()
prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
prod_model_exists = len(prod_model)>0

if prod_model_exists:
    prod_model_version = dict(prod_model[0])['version']
    prod_model_run_id = dict(prod_model[0])['run_id']
    
    print('Production model name: ', model_name)
    print('Production model version:', prod_model_version)
    print('Production model run id:', prod_model_run_id)
    
else:
    print('No model in production')

train_model_score = experiment_best["metrics.f1_score"]
model_details = {}
model_status = {}
run_id = None

if prod_model_exists:
    data, details = mlflow.get_run(prod_model_run_id)
    prod_model_score = data[1]["metrics.f1_score"]

    model_status["current"] = train_model_score
    model_status["prod"] = prod_model_score

    if train_model_score>prod_model_score:
        print("Registering new model")
        run_id = experiment_best["run_id"]
else:
    print("No model in production")
    run_id = experiment_best["run_id"]

print(f"Registered model: {run_id}")

if run_id is not None:
    print(f'Best model found: {run_id}')

    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=run_id,
        artifact_path=artifact_path
    )
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    wait_until_ready(model_details.name, model_details.version)
    model_details = dict(model_details)
    print(model_details)