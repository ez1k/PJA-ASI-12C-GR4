"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 0.19.2
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_with_champion

def create_pipeline(** kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compare_with_champion,
            inputs=["model_challenger", "score_challenger"],
            outputs=["best_model", "score_best_model"],
            name="compare_with_challenger_node",
        )
    ])