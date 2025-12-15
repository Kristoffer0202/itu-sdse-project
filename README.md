# ITU-sdse-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Repository for MLOps exam

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── .dvc 
│   └──config              <- Contains reference to remote for dvc data pull
│
├── .github/workflows           <- Github Action Workflows
│   └── dagger.yml              <- Workflow file that trains, tests and outputs models 
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Module and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── Module   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Module a Python module
    │
    ├── 0.1_read_data.py               <- Store useful variables and configuration
    │
    ├── 0.2_data_preprocessing.py          <- Scripts to download or generate data
    │
    ├── 1.0_Train.py             <- Code to create features for modeling
    │
    ├── 1.1_TrainXGBoost.py               <- Store useful variables and configuration
    │
    ├── 1.2_TrainSKLearnLog.py               <- Store useful variables and configuration
    │
    ├── 2.0_SelectBestModelAndRegister.py               <- Store useful variables and configuration
    │
    ├── 3.0_Deploy.py               <- Store useful variables and configuration
    │
    │
    ├── config.py             <- Code to create features for modeling
    │
    └── util.py  
```

--------

