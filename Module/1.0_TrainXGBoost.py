from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
import pickle
import pandas as pd
import argparse
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pprint import pprint
import json
import warnings

from config import (
    TRAIN_TEST_DATA_FILE,
    XGBOOST_MODEL_PATH,
    MODEL_RESULTS_PATH,
)

warnings.filterwarnings('ignore')
pd.set_option('display.float_format',lambda x: "%.3f" % x)

################## Constants used
xgboost_model_path = XGBOOST_MODEL_PATH
model_results_path = MODEL_RESULTS_PATH

with open(TRAIN_TEST_DATA_FILE, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)
##########################




###################################
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
args = parser.parse_args()
print("Run name:", args.run_name)
experiment_name = args.run_name
###################################



model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)

model_grid.fit(X_train, y_train)



best_model_xgboost_params = model_grid.best_params_
print("Best xgboost params")
pprint(best_model_xgboost_params)

y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)
print("Accuracy train", accuracy_score(y_pred_train, y_train ))
print("Accuracy test", accuracy_score(y_pred_test, y_test))




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

xgboost_model = model_grid.best_estimator_
xgboost_model.save_model(xgboost_model_path)

model_results = {
    str(xgboost_model_path): classification_report(y_train, y_pred_train, output_dict=True)
}

#saving model scores
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)
