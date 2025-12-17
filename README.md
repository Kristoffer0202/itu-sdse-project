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
├── docs              
│   ├── diagrams.excalidraw                      <- Excalidraw diagram visualisation   
│   ├── project-architecture.png                
│   └── projectDescription.md                    <- Original description of project
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

# How to Run the Project
The project can be made to generate the model-artifact in two ways. The Github workflow is the recommended method and only generates the model itself. The project can be run locally if all data, artifacs and models are desired. 

## Github Workflow (Recommended)
The github worklow only stores the artifact for the ```lead_model_lr.pkl```  as ```model.pkl``` saved in a zip called ```model.zip```. To generate the zip do the follwing:
- Navigate to the repo on the github website (https://github.com/Kristoffer0202/itu-sdse-project)
- Enter the "Actions" tab
- Go to "Run Dagger Pipeline"
- Press "Run Workflow"
- Once finished, the model can be dowloaded from the Artifacts field under the finshed job summary



## Locally 
The python dependencies themself will be taken care of automatically via the Go Pipline. But to succesfully run the pipeline these are requried:
- Dagger: 0.19.8
- go: 1.25.0
- Python 3.12.2
- Docker 28.5.1

To then run the pipline and generate the artifact you must simply navigate to the project reposetory root, and execute the following line of code:

```dagger run go run pipeline.go```

All artifacts, models and data will then be generated and stored in their respective folders. 
