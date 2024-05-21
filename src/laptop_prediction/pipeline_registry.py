"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from laptop_prediction.pipelines.automl import pipeline as automl_pipeline
from laptop_prediction.pipelines.ml import pipeline as ml_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "automl": automl_pipeline.create_pipeline(),
        "ml": ml_pipeline.create_pipeline(),
        "__default__": ml_pipeline.create_pipeline()
    }
