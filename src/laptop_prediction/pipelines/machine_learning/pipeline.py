from kedro.pipeline import Pipeline, pipeline, node
from .nodes import run_model, optimize_model, evaluate_model

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
                inputs=["model", "X_val", "y_val", "params:cv", "params:verbose", "params:n_jobs"],
                outputs="model_challenger",
                name="optimize_model_node",
            ),
        node(
                func=evaluate_model,
                inputs=["model_challenger", "X_test", "y_test", "params:retraining", "test_data"],
                outputs="score_challenger",
                name="evaluate_model_node",
            )
    ])
