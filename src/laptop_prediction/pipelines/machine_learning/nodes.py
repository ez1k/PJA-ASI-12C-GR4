"""
This is a boilerplate pipeline 'ml'
generated using Kedro 0.19.2
"""
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import pickle
import json

def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame, forest_n: int) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=forest_n, random_state=0)
    model.fit(X_train, y_train.values.ravel())
    return model

def optimize_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame, cv: any, verbose: int, n_jobs: int) -> RandomForestRegressor:
    params = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'max_depth': [int(x) for x in np.linspace(2, 30, num=5)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=cv, verbose=verbose, random_state=0, n_jobs=n_jobs)
    random_search.fit(X_test, y_test.values.ravel())
    model_challenger = random_search.best_estimator_
    return model_challenger

def evaluate_model(model_challenger: RandomForestRegressor, X_val: pd.DataFrame, y_val: pd.DataFrame, retraining: bool, test_data: pd.DataFrame) -> dict:
    if (retraining):
        best_model_path = 'data/04_model/best_model.pickle'
        best_score_path = "data/05_model_output/score_best_model.json"
        if os.path.exists(best_model_path) and os.path.exists(best_score_path):
            with open(best_model_path, 'rb') as file:
                best_model = pickle.load(file)
            with open(best_score_path, "r") as f:
                best_model_score = json.load(f)
            champion_model = best_model_score['model']
            print('Retraining')
            if (champion_model == 'ml'):
                y_pred_test = best_model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred_test)
                mse = mean_squared_error(y_val, y_pred_test)
                rmse = math.sqrt(mse)
                r2 = r2_score(y_val, y_pred_test)

                print(champion_model)
                print(mae)
                print(mse)
                print(rmse)
                print(r2)

                retraining_results = {
                    "model": champion_model,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": rmse,
                    "val_r2": r2
                }
            else:
                evaluation_results = best_model.evaluate(test_data)
                mae = evaluation_results.get('mean_absolute_error') * -1
                mse = evaluation_results.get('mean_squared_error') * -1
                rmse = evaluation_results.get('root_mean_squared_error') * -1
                r2 = evaluation_results.get('r2')
            
                print(champion_model)
                print(mae)
                print(mse)
                print(rmse)
                print(r2)
            
                retraining_results = {
                    "model": champion_model,
                    "val_mae": mae,
                    "val_mse": mse,
                    "val_rmse": rmse,
                    "val_r2": r2
                }
            with open(best_score_path, "w") as f:
                json.dump(retraining_results, f)



    metrics = {}

    y_pred_test = model_challenger.predict(X_val)
    metrics['model'] = 'ml'
    metrics['val_mae'] = mean_absolute_error(y_val, y_pred_test)
    metrics['val_mse'] = mean_squared_error(y_val, y_pred_test)
    metrics['val_rmse'] = math.sqrt(metrics['val_mse'])
    metrics['val_r2'] = r2_score(y_val, y_pred_test)   

    return metrics