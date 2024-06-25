from kedro.pipeline import Pipeline, pipeline, node
from .nodes import run_model, optimize_model, evaluate_model, compare_with_old_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
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
                func=compare_with_old_model,
                inputs=["best_model", "X_val", "y_val", "params:retraining"],
                outputs="new_or_old",
                name="compare_with_old_model_node",
            ),
        node(
                func=evaluate_model,
                inputs=["new_or_old", "X_val", "y_val"],
                outputs="metrics",
                name="evaluate_model_node",
            )
    ])
