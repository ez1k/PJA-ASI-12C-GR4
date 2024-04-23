"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from typing import Tuple

def run_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model

def calculate_mae(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.DataFrame) -> str:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return str(mae)

def optimize_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    params = {
        'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
        'max_depth': [int(x) for x in np.linspace(2, 30, num=5)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestRegressor(random_state=0)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=3, verbose=2, random_state=0, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    return best_model

def calculate_best_mae(best_model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    best_predictions = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, best_predictions)
    return str(best_mae)