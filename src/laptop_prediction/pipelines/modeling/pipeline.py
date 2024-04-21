"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.2
"""
import numpy as np 
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from typing import Tuple
from kedro.pipeline import Pipeline, node, pipeline


from .nodes import preprocess_data, run_model, optimize_model, run_price_prediction_model

def create_pipeline(** kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs="laptops_for_model",
                outputs=["X", "y"],
                name="preprocess_data_node",
            ), 
            node(
                func=run_model,
                inputs=["X", "y"],
                outputs=["model", "mae"],
                name="run_model_node",
            ),
            node(
                func=optimize_model,
                inputs=["X", "y"],
                outputs=["best_model", "best_mae"],
                name="optimize_model_node",
            ),
            node(
                func=run_price_prediction_model,
                inputs="laptops_for_model",
                outputs=None,
                name="run_price_prediction_model_node",
            ),
        ]
    )
