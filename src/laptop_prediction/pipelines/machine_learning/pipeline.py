"""
This is a boilerplate pipeline 'machine_learning'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_model, model_metrics, optimize_model, best_model_metrics, validate_model_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_model,
                inputs=["X_train", "y_train", "params:forest_n"],
                outputs="model",
                name="run_model_node",
            ),
            node(
                func=optimize_model,
                inputs=["X_train", "y_train", "params:cv",  "params:verbose",  "params:n_jobs"],
                outputs="best_model",
                name="optimize_model_node",
            ), node(
                func=model_metrics,
                inputs=["model", "X_test", "y_test"],
                outputs=["mae","mse", "rmse", "r2"],
                name="model_metrics_node",
            ),node(
                func=best_model_metrics,
                inputs=["best_model","X_test", "y_test"],
                outputs=["best_mae","best_mse", "best_rmse", "best_r2"],
                name="best_model_metrics_node",
            ), node(
                func=validate_model_metrics,
                inputs=["model", "X_val", "y_val"],
                outputs=["val_mae","val_mse", "val_rmse", "val_r2"],
                name="validate_model_metrics_node"
            )
        ]
    )

