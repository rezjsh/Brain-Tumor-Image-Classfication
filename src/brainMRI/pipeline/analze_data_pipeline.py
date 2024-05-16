from brainMRI.config.configuration import ConfigHandler
from brainMRI.logging import logger


class AnalyzeDataPipeline:
    def __init__(self, config) -> None:
        self.config = config


    def main(self) -> None:
        analyzer_config = self.config.get_analyze_image_data_config()
        analyzer_config.analyzer()

if __name__ == '__main__':
    try:
        config = ConfigHandler()
        stage_name = 'Analyze Data stage'
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")  # Log the start of the pipeline stage
        pipeline = AnalyzeDataPipeline(config)
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")  # Log the completion of the pipeline stage

    except Exception as e:
        logger.exception(e)  # Log the exception if an error occurs
        raise e