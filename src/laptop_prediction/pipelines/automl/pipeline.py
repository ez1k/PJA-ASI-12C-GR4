"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, pipeline, node

from laptop_prediction.pipelines.automl.nodes import evaluate_models, train_model_challenger, train_model_champion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=train_model_champion,
                inputs="train_data",
                outputs="model_champion",
                name="train_model_champion_node",
            ),
        node(
                func=train_model_challenger,
                inputs="train_data",
                outputs="model_challenger",
                name="train_model_challenger_node",
            ),
        node(
                func=evaluate_models,
                inputs=["model_champion", "model_challenger", "test_data"],
                outputs="best_model_ml",
                name="evaluate_models_node",
            )
    ])
