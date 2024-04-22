"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_model, calculate_mae, optimize_model, calculate_best_mae

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="run_model_node",
            ),
            node(
                func=calculate_mae,
                inputs=["model", "X_test", "y_test"],
                outputs="mae",
                name="calculate_mae_node",
            ),
            node(
                func=optimize_model,
                inputs=["X_train", "y_train"],
                outputs="best_model",
                name="optimize_model_node",
            ), node(
                func=calculate_best_mae,
                inputs=["best_model","X_test", "y_test"],
                outputs="best_mae",
                name="calculate_best_mae_node",
            )
        ]
    )

