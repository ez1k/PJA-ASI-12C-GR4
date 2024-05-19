"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""
import pandas as pd
import numpy as np
import json

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

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    metrics = {}

    y_pred_test = model.predict(X_test)
    metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
    metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
    metrics['test_rmse'] = math.sqrt(metrics['test_mse'])
    metrics['test_r2'] = r2_score(y_test, y_pred_test)

    y_pred_val = model.predict(X_val)
    metrics['val_mae'] = mean_absolute_error(y_val, y_pred_val)
    metrics['val_mse'] = mean_squared_error(y_val, y_pred_val)
    metrics['val_rmse'] = math.sqrt(metrics['val_mse'])
    metrics['val_r2'] = r2_score(y_val, y_pred_val)

    wandb.init(project='PJA-ASI-12C-GR4')
    wandb.log({
        "Test MAE": metrics['test_mae'],
        "Test MSE": metrics['test_mse'],
        "Test RMSE": metrics['test_rmse'],
        "Test R2": metrics['test_r2'],
        "Validation MAE": metrics['val_mae'],
        "Validation MSE": metrics['val_mse'],
        "Validation RMSE": metrics['val_rmse'],
        "Validation R2": metrics['val_r2']
    })

    return str(metrics)