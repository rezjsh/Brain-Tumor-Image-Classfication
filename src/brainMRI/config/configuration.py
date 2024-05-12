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