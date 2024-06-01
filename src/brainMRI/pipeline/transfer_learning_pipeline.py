from brainMRI.config.configuration import ConfigHandler
from brainMRI.logging import logger



class TransferLearningPipeline:
    def __init__(self, config) -> None:
            self.config = config

    def main(self):
        transfer_learning_config = config.get_transfer_learning_config()
        transfer_learning_config.train()
        transfer_learning_config.save_plots()

if __name__ == '__main__':
    try:
        config = ConfigHandler()
        stage_name = 'Transfer Learning stage'
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")  # Log the start of the pipeline stage
        pipeline = TransferLearningPipeline(config)
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")  # Log the completion of the pipeline stage

    except Exception as e:
        logger.exception(e)  # Log the exception if an error occurs
        raise e