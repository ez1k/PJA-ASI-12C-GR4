"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, pipeline, node

from laptop_prediction.pipelines.automl.nodes import (
    train_model_challenger, 
    evaluate_model
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model_challenger,
            inputs="train_data",
            outputs="model_challenger",
            name="train_model_challenger_node",
        ),
        node(
            func=evaluate_model,
            inputs=["model_challenger", "test_data"],
            outputs="score_challenger",
            name="evaluate_model_challenger_node",
        )
    ])