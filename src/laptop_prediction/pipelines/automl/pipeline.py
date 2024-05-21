"""
This is a boilerplate pipeline 'automl'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, pipeline, node

from laptop_prediction.pipelines.automl.nodes import evaluate_models, preprocess_data, split_data, train_model_challenger, train_model_champion, transform_data



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=transform_data,
                inputs="laptops",
                outputs="laptops_for_model",
                name="transform_laptops_node"
            ),
        node(
                func=preprocess_data,
                inputs="laptops_for_model",
                outputs="data",
                name="preprocess_data_node",
            ),
        node(
                func=split_data,
                inputs=["data", "params:test_size"],
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
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
                outputs="best_model",
                name="evaluate_models_node",
            )
    ])
