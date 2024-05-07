"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""
import pandas as pd
import numpy as np

import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import wandb

def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame, forest_n: int) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=forest_n, random_state=0)
    model.fit(X_train, y_train.values.ravel())
    return model

def optimize_model(X_train: pd.DataFrame, y_train: pd.DataFrame, cv: any, verbose: int, n_jobs: int) -> RandomForestRegressor:
    params = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'max_depth': [int(x) for x in np.linspace(2, 30, num=5)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestRegressor(random_state=0, n_jobs=n_jobs)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=cv, verbose=verbose, random_state=0, n_jobs=n_jobs)
    random_search.fit(X_train, y_train.values.ravel())
    best_model = random_search.best_estimator_
    return best_model

def model_metrics(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> str:
    model_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, model_predict)
    mse = mean_squared_error(y_test, model_predict)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, model_predict)

    wandb.init(project='PJA-ASI-12C-GR4')
    wandb.log({
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    return str(mae), str(mse), str(rmse), str(r2)

def best_model_metrics(best_model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> str:
    best_predictions = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, best_predictions)
    best_mse = mean_squared_error(y_test, best_predictions)
    best_rmse = math.sqrt(best_mse)
    best_r2 = r2_score(y_test, best_predictions)

    wandb.init(project='PJA-ASI-12C-GR4')
    wandb.log({
        "Best MAE": best_mae,
        "Best MSE": best_mse,
        "Best RMSE": best_rmse,
        "Best R2": best_r2
    })

    return str(best_mae), str(best_mse), str(best_rmse), str(best_r2)

def validate_model_metrics(model: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.Series):
    y_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_pred)
    val_mse = mean_squared_error(y_val, y_pred)
    val_rmse = math.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_pred)

    wandb.init(project='PJA-ASI-12C-GR4')
    wandb.log({
        "Validation MAE": val_mae,
        "Validation MSE": val_mse,
        "Validation RMSE": val_rmse,
        "Validation R2": val_r2
    })

    return str(val_mae), str(val_mse), str(val_rmse), str(val_r2)