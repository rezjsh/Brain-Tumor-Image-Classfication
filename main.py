from brainMRI.config.configuration import ConfigHandler
from brainMRI.logging import logger
from brainMRI.pipeline.fetch_data_pipeline import FetchDataPipeline
from brainMRI.pipeline.analze_data_pipeline import AnalyzeDataPipeline
from brainMRI.pipeline.prepare_datasets_pipeline import PrepareDatasetsPipeline

config = ConfigHandler()

pipelines = {
    "Fetch Data stage": FetchDataPipeline(config),
    "Analyze Data stage": AnalyzeDataPipeline(config),
    "Prepare Datasets stage": PrepareDatasetsPipeline(config),
}

def run_pipeline(stage_name, pipeline_instance):
    """
    Run a specific stage of the pipeline and log the start and completion messages.

    Args:
        stage_name: Name of the pipeline stage
        pipeline_instance: Instance of the pipeline to be executed

    Returns:
        None

    Raises:
        Any exceptions that occur during the pipeline execution

    """
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")  # Log the start of the pipeline stage
        pipeline_instance.main()  # Execute the main method of the pipeline instance
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")  # Log the completion of the pipeline stage
    except Exception as e:
        logger.exception(e)  # Log the exception if an error occurs
        raise e  # Raise the exception to propagate it further

if __name__ == '__main__':
    for stage_name, pipeline_instance in pipelines.items():
        run_pipeline(stage_name, pipeline_instance)  # Execute each pipeline stage using the run_pipeline function