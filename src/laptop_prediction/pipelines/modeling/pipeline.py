"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.2
"""
from kedro.pipeline import Pipeline, node, pipeline


from .nodes import preprocess_data,split_data

def create_pipeline(** kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs="laptops_for_model",
                outputs="data",
                name="preprocess_data_node",
            ),
            node(
                func=split_data,
                inputs=["data"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_data_node",
            )
        ]
    )
