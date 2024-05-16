from brainMRI.config.configuration import ConfigHandler
from brainMRI.logging import logger



class FetchDataPipeline:
    def __init__(self, config) -> None:
            self.config = config

    def main(self):
        fetch_data_config = self.config.get_fetch_data_config()
        fetch_data_config.download_file()
        fetch_data_config.unzip_file()

if __name__ == '__main__':
    try:
        config = ConfigHandler()
        stage_name = 'Fetch Data stage'
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")  # Log the start of the pipeline stage
        pipeline = FetchDataPipeline(config)
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")  # Log the completion of the pipeline stage

    except Exception as e:
        logger.exception(e)  # Log the exception if an error occurs
        raise e