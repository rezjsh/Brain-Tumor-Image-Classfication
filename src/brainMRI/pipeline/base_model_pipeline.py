from brainMRI.config.configuration import ConfigHandler
from brainMRI.logging import logger



class BaseModelPipeline:
    def __init__(self, config) -> None:
            self.config = config

    def main(self):
        base_model_config = self.config.get_base_model_config()
        base_model_config.build_model()

if __name__ == '__main__':
    try:
        config = ConfigHandler()
        stage_name = 'Base Model stage'
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")  # Log the start of the pipeline stage
        pipeline = BaseModelPipeline(config)
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")  # Log the completion of the pipeline stage

    except Exception as e:
        logger.exception(e)  # Log the exception if an error occurs
        raise e