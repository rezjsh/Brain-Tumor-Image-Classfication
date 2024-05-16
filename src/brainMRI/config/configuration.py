from brainMRI.components.analyze_data import AnalyzeImageData
from brainMRI.components.fetch_data import FetchData
from brainMRI.constants import *
from brainMRI.utils.helpers import load_config, create_directories


class ConfigHandler:
    def __init__(self, file_path=CONFIG_FILE_PATH, params_path = PARAMS_FILE_PATH):
        self.config = load_config(file_path)
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
        config = self.config['info']
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