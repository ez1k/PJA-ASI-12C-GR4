"""Project pipelines."""
from __future__ import annotations
 
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline
from laptop_prediction.pipelines.automl import pipeline as automl_pipeline
from laptop_prediction.pipelines. data_processing import pipeline as  data_processing_pipeline
from laptop_prediction.pipelines.machine_learning import pipeline as machine_learning_pipeline
from laptop_prediction.pipelines.modeling import pipeline as modeling_pipeline
from laptop_prediction.pipelines.deployment import pipeline as deployment_pipeline
 
def register_pipelines() -> dict[str, Pipeline]:
    automl = pipeline(
        [
            data_processing_pipeline.create_pipeline(),
            modeling_pipeline.create_pipeline(),
            automl_pipeline.create_pipeline(),
            deployment_pipeline.create_pipeline(),
        ]
    )

    ml = pipeline(
        [
            data_processing_pipeline.create_pipeline(),
            modeling_pipeline.create_pipeline(),
            machine_learning_pipeline.create_pipeline(),
            deployment_pipeline.create_pipeline(),
        ]
    )

    return {
        "__default__": ml,
        "automl": automl,
        "ml": ml,
    }