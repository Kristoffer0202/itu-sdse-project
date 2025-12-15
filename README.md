# ITU-sdse-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for MLOps exam

## Project Organization

```

├── .dvc 
│   └──config          <- Contains reference to remote for dvc data pull
│
├── .github/workflows  <- Github Action Workflows
│   └── dagger.yml         <- Workflow file that trains, tests and outputs models 
│
├── artifacts          <- Contains model artifacts
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── Module             <- Source code for use in this project.
│    │
│    ├── __init__.py                             <- Makes Module a Python module
│    │
│    ├── 0.0_read_data.py                        <- Pulls dvc data and filters range by date
│    │
│    ├── 0.1_data_preprocessing.py               <- Preprocessing of data and generates gold data file
│    │
│    ├── 0.3_generate_train_test_data.py         <- Splits processed data into train and test
│    │
│    ├── 1.0_TrainXGBoost.py                     <- Trains and tests XGBoost model
│    │
│    ├── 1.1_TrainSKLearnLR.py                   <- Trains and tests Logistic Regression model
│    │
│    ├── 2.0_SelectBestModelAndRegister.py       <- Selects and registers the best performning model
│    │
│    ├── 3.0_Deploy.py                           <- Deploys model (Not necessary, but used since it was in main.ipynb)
│    │
│    ├── config.py                               <- Script containing file and path references
│    │
│    └── util.py                                 <- Utility helper functions used by scripts
│
├── go.mod             <- file to define the module and required dependencies
│
├── go.sum             <- Locks versions with checksums to ensure integrity
│
├── LICENSE            <- Open-source license
│
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
│
├── pipeline.go        <- Dagger workflow pipeline
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Module and configuration for tools like black
│
├── README.md          <- The top-level README for developers using this project.
│
└── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                        generated with `pip freeze > requirements.txt`
```

--------

