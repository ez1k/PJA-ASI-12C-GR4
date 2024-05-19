"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_model,
                inputs=["X_train", "y_train", "params:forest_n"],
                outputs="model",
                name="run_model_node",
            ), node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test", "X_val", "y_val"],
                outputs="metrics",
                name="evaluate_model_node",
            )
        ]
    )

