from brainMRI.config.configuration import ConfigHandler
from brainMRI.logging import logger



class CallbacksPipeline:
    def __init__(self, config) -> None:
            self.config = config

    def main(self):
        callbacks_config = self.config.get_callbacks_config()
        f = callbacks_config.get_callbacks()
   
        print(f)

if __name__ == '__main__':
    try:
        config = ConfigHandler()
        stage_name = 'Callbacks stage'
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")  # Log the start of the pipeline stage
        pipeline = CallbacksPipeline(config)
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")  # Log the completion of the pipeline stage

    except Exception as e:
        logger.exception(e)  # Log the exception if an error occurs
        raise e