"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.2
"""
from kedro.pipeline import Pipeline, node, pipeline


from .nodes import preprocess_data,split_data

def create_pipeline(** kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs="laptops_for_model",
                outputs="data",
                name="preprocess_data_node",
            ),
            node(
                func=split_data,
                inputs=["data", "params:test_size", "params:val_size"],
                outputs=["X_train", "X_test", "X_val", "y_train", "y_test", "y_val", "train_data", "test_data"],
                name="split_data_node",
            ),
        ]
    )