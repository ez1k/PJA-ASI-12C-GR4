"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict, run_model,optimize_model

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
                func=predict,
                inputs=["model", "X_test", "y_test"],
                outputs="mae",
                name="predict_node",
            ),
            node(
                func=optimize_model,
                inputs=["X_train", "y_train", "X_test", "y_test"],
                outputs=["best_model", "best_mae"],
                name="optimize_model_node",
            )

        ]
    )

