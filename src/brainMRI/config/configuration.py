from brainMRI.constants import *
from brainMRI.utils.helpers import load_config, create_directories
from brainMRI.components.analyze_data import AnalyzeImageData
from brainMRI.components.augmentation import DataAugmentation
from brainMRI.components.base_model import BaseModel
from brainMRI.components.callbacks import Callbacks
from brainMRI.components.fetch_data import FetchData
from brainMRI.components.prepare_datasets import PrepareDatasets
from brainMRI.components.transfer_learning import TransferLearning


class ConfigHandler:
    def __init__(self, file_path=CONFIG_FILE_PATH, params_path = PARAMS_FILE_PATH):
        self.config = load_config(file_path)
        self.params = load_config(params_path)
        create_directories([self.config.root_dir])

    
    def get_fetch_data_config(self) -> FetchData:
        config = self.config.data
        fetch_data_config = FetchData(
            root_dir = config.root_dir,
            filepath = config.filepath,
            extract_path= config.extract_path,
            data_url= config.data_url

        )
        return fetch_data_config
    

    def get_analyze_image_data_config(self) -> AnalyzeImageData:
        config = self.config.info
        create_directories([config.root_dir])
        analyze_image_data_config = AnalyzeImageData(
            data_folder=config.data_folder,
            image_quality_and_format=config.image_quality_and_format,
            image_counts_path=config.image_counts_path,
            allowed_formats=config.allowed_formats,
            image_metadata_path=config.image_metadata_path,
            image_samples_path=config.image_samples_path,
            image_stats_results_path=config.image_stats_results_path,
            plots_path=config.plots_path,
        )
        return analyze_image_data_config


    
    def get_prepare_datasets_config(self) -> PrepareDatasets:
        config = self.config.prepare_datasets
        params = self.params.prepare_datasets
        create_directories([config.save_dir])
        
        prepare_datasets_config = PrepareDatasets(
            data_dir= config.data_dir,
            save_dir= config.save_dir,
            validation_split= params.validation_split,
            image_size= params.image_size,
            batch_size= params.batch_size,
            labels= params.labels,
            subset= params.subset,
            seed= params.seed
        )

        return prepare_datasets_config
    
    def get_data_augmentation_config(self) -> DataAugmentation:
        params = self.params.data_augmentation
        config = self.config.data_augmentation
        data_augmentation_config = DataAugmentation(
            training_dir= config.training_dir,
            random_flip_horizontal= params.random_flip_horizontal,
            random_flip_vertical= params.random_flip_vertical,
            random_rotation= params.random_rotation,
            random_zoom_height= params.random_zoom_height,
            random_zoom_width= params.random_zoom_width,
            random_brightness= params.random_brightness,
            random_contrast= params.random_contrast,
            random_translation_height= params.random_translation_height,
            random_translation_width= params.random_translation_width,
            random_rotation_factor= params.random_rotation_factor,
            random_zoom_height_factor= params.random_zoom_height_factor,
            random_zoom_width_factor= params.random_zoom_width_factor,
            random_brightness_factor= params.random_brightness_factor,
            random_contrast_lower_factor= params.random_contrast_lower_factor,
            random_contrast_upper_factor= params.random_contrast_upper_factor,
            random_translation_height_factor= params.random_translation_height_factor,
            random_translation_width_factor= params.random_translation_width_factor
        )
        return data_augmentation_config
    
    def get_base_model_config(self) -> BaseModel:
        config = self.config.base_model
        params= self.params.base_model
        data_augmentation_config = self.get_data_augmentation_config()

        base_model_config = BaseModel(
            root_dir = config.root_dir,
            weights=params.weights,
            include_top=params.include_top,
            input_shape=params.input_shape,
            fine_tune_at=params.fine_tune_at,
            data_augmentation_config = data_augmentation_config
      )
        return base_model_config
    

    def get_callbacks_config(self) -> Callbacks:
        config = self.config.callbacks
        params = self.params.callbacks
        create_directories([config.root_dir])

        callbacks_config = Callbacks(
            root_dir=config.root_dir,
            patience=params.patience,
            factor=params.factor,
            min_lr=params.min_lr
        )
        return callbacks_config


    def get_transfer_learning_config(self) -> TransferLearning:
        config = self.config.transfer_learning
        params = self.params.transfer_learning

        create_directories([config.root_dir])
        transfer_learning_config = TransferLearning(
            root_dir=config.root_dir,
            train_dir=config.train_dir,
            val_dir=config.val_dir,
            base_model_path=config.base_model_path,
            callback_path=config.callback_path,
            epochs=params.epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate
        )
        return transfer_learning_config