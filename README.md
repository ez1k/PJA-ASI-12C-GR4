# laptop-prediction

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.4`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Dataset

Used dataset: https://www.kaggle.com/datasets/ara001/laptop-prices-based-on-its-specifications

## Pipelines

# Automl Pipeline (automl):

    data_processing_pipeline -> modeling_pipeline -> automl_pipeline -> deployment_pipeline

# Machine Learning Pipeline (ml):

    data_processing_pipeline -> modeling_pipeline -> machine_learning_pipeline -> deployment_pipeline

# data_processing_pipeline: 

This pipeline processes laptop data. It merges new data with existing data, removes unnecessary columns, transforms values in columns like Ram and Weight, and extracts additional information related to screen resolution, memory, CPU, and operating system.

# modeling_pipeline:

This pipeline is responsible for modeling laptop data. It consists of two main stages:
    1. Data preprocessing (preprocess_data): Converts categorical columns to numerical values using a label encoder, fills missing values in memory-related columns with zeros.
    2. Data splitting (split_data): Splits the data into training, validation, and test sets, separating features and labels.

# machine_learning_pipeline

This pipeline is responsible for training, optimizing, and evaluating a Random Forest model for laptop price prediction. It consists of three main stages:
    1. Model training (run_model): Trains a Random Forest model on the training dataset.
    2. Model optimization (optimize_model): Optimizes the model's hyperparameters using RandomizedSearchCV.
    3. Model evaluation (evaluate_model): Evaluates the optimized model on the validation dataset, and in case of retraining, compares it with the best model saved in the files.

# automl_pipeline

This pipeline is responsible for automatic training and evaluation of a model using AutoGluon. It consists of two main stages:
    1. Model training (train_model_challenger): Trains an AutoGluon model on the training data with specified hyperparameters.
    2. Model evaluation (evaluate_model): Evaluates the AutoGluon model on test and validation data, and in case of retraining, compares it with the best model saved in the files.

# deployment_pipeline

This pipeline compares the new model (challenger) with the best existing model (champion). If the new model has better metrics, it is saved as the new champion model. The results are also logged to Weights & Biases (wandb).

## Configuration parameters - parameters.yml

    1. forest_n: Number of trees in the Random Forest model. Set to 100.
    2. cv: Number of folds in cross-validation. Set to 3.
    3. verbose: Level of detail in output during model training. Set to 2.
    4. n_jobs: Number of parallel threads used for model training. Set to -1 (use all available CPU cores).
    5. test_size: Proportion of data allocated to the test set. Set to 0.2 (20% of data).
    6. val_size: Proportion of data allocated to the validation set from the temporary set. Set to 0.5 (50% of the temporary set).
    7. retraining: Flag indicating whether to retrain the model. Set to false.

## Data configuration description - catalog.yml

This YAML file defines the data catalog used in a Kedro project. Each entry specifies a dataset, its type, file location, and layer in the Kedro visualization system. Here are the main data categories:

1. Raw Data:
    - laptops: Contains input laptop data in CSV format, located at data/01_raw/laptop_data_input_file.csv.
    - laptops_new: New laptop data, also in CSV format, located at data/01_raw/laptop_new_data_input_file.csv.
2. Intermediate Data:
    - laptops_for_model: Processed data ready for modeling, saved at data/02_intermediate/laptop_data_output_file.csv.
3. Model Input Data:
    - X_train, X_test, X_val: Input features for training, test, and validation sets.
    - y_train, y_test, y_val: Labels (prices) for training, test, and validation sets.
    - train_data, test_data: Combined input features and labels for training and test sets.
4. Models:
    - model, model_challenger, best_model: Models saved in pickle format.
5. Model Output Results:
    - score_best_model, score_challenger: Model results saved in JSON format.

## API

This API, built using FastAPI, allows for predicting the price of a laptop based on input features. Key components:
    1. InputJson: A class representing the structure of input data, with attributes corresponding to laptop features.
    2. FastAPI: The main FastAPI application, which defines the prediction endpoint.
    3. Predict Endpoint: The POST / endpoint that accepts input data in JSON format, loads the best model saved in the best_model.pickle file, processes the input data into a DataFrame, and generates a price prediction.

## Architecture of MLOps Pipelines - MLOps_diagram.png

In the root directory of the project, there is a diagram illustrating the architecture of MLOps pipelines. This diagram depicts the data flow and processing steps involved in the machine learning (ml) and automated modeling (automl) processes.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to create and activate kedro env in conda

You can create Kedro env with:

```
conda create --name kedro-environment python=3.10 -y
```
You can activate Kedro env with:

```
conda activate kedro-environment
```

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
pip install autogluon
```

### Weights & Biases
You need to login Weights & Biases with api key:

```
wandb login
```
### Running with docker

First build an image with

```
docker build -t IMAGE_NAME .
```

Then run docker with

```
docker run --rm --name CONTAINER_NAME -v .:/home/kedro_docker IMAGE_NAME:latest
```

### FastAPI

Run base conda and install dependencies

Run this command

```
fastapi dev src/endpoint/endpoint.py
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to retraining:

You need to add:

```
--params=retraining=true

Example:
kedro run --params=retraining=true
```

## How to run pipline visualization

You can run Kedro pipline visualization with:

```
kedro viz --autoreload
```

## How to run your Kedro pipeline automl or ml

Ml is deafult. You can add --pipeline=automl for automl or --pipeline=ml for ml:

```
kedro run --pipeline=automl or kedro viz --autoreload --pipeline=automl

kedro run --pipeline=ml or kedro viz --autoreload --pipeline=ml
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
`

