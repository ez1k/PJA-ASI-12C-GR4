"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, pipeline, node

from laptop_prediction.pipelines.automl.nodes import (
    train_model_champion, 
    evaluate_model, 
    compare_with_challenger
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model_champion,
            inputs="train_data",
            outputs="model_champion",
            name="train_model_champion_node",
        ),
        node(
            func=evaluate_model,
            inputs=["model_champion", "test_data"],
            outputs="score_champion",
            name="evaluate_model_champion_node",
        ),
        node(
            func=compare_with_challenger,
            inputs=["score_champion", "model_champion"],
            outputs=["best_model_aml", "best_model_aml_score"],
            name="compare_with_challenger_node",
        )
    ])