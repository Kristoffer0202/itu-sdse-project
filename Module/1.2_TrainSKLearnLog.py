import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import joblib
import argparse
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pprint import pprint
import pandas as pd
import json
import warnings

from config import (
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    LEAD_MODEL_LR_PATH,
    MODEL_RESULTS_PATH,
    COLUMNS_LIST_FILE,
)

warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x: "%.3f" % x)

# with open('train_test_data.pkl', 'rb') as f:
#     X_train, X_test, y_train, y_test = pickle.load(f)

################## Constants used
column_list_path = COLUMNS_LIST_FILE
X_train = X_TRAIN_FILE,
X_test = X_TEST_FILE,
y_train = Y_TRAIN_FILE,
y_test = Y_TEST_FILE
lr_model_path = LEAD_MODEL_LR_PATH
model_results_path = MODEL_RESULTS_PATH
##########################


###################################
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
args = parser.parse_args()
print("Run name:", args.run_name)
experiment_name = args.run_name
###################################


class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["none", "l1", "l2", "elasticnet"],
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)


    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=model, filename=lr_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))


model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

print("Best lr params")
pprint(best_model_lr_params)

print("Accuracy train:", accuracy_score(y_pred_train, y_train ))
print("Accuracy test:", accuracy_score(y_pred_test, y_test))

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Test actual/predicted\n")
print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_test, y_pred_test),'\n')

conf_matrix = confusion_matrix(y_train, y_pred_train)
print("Train actual/predicted\n")
print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_train, y_pred_train),'\n')


model_results_path[lr_model_path] = model_classification_report
print(model_classification_report["weighted avg"]["f1-score"])

with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    pprint(columns)
    json.dump(columns, columns_file)

print('Saved column list to ', column_list_path)

with open(model_results_path, 'w+') as results_file:
    json.dump(model_results_path, results_file)



