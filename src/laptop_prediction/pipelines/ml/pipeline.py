"""
This is a boilerplate pipeline 'ml'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import transform_data, preprocess_data, split_data, run_model, optimize_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=transform_data,
                inputs="laptops",
                outputs="laptops_for_model",
                name="transform_laptops_node"
            ),
        node(
                func=preprocess_data,
                inputs="laptops_for_model",
                outputs="data",
                name="preprocess_data_node",
            ),
        node(
                func=split_data,
                inputs=["data", "params:test_size", "params:val_size"],
                outputs=["X_train", "X_test", "X_val", "y_train", "y_test", "y_val"],
                name="split_data_node",
            ),
        node(
                func=run_model,
                inputs=["X_train", "y_train", "params:forest_n"],
                outputs="model",
                name="run_model_node",
            ),
        node(
                func=optimize_model,
                inputs=["model", "X_test", "y_test", "params:cv", "params:verbose", "params:n_jobs"],
                outputs="best_model",
                name="optimize_model_node",
            ),
        node(
                func=evaluate_model,
                inputs=["best_model", "X_val", "y_val"],
                outputs="metrics",
                name="evaluate_model_node",
            )
    ])
