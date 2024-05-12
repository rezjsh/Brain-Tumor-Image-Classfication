from brainMRI.logging import logger
import yaml
import os
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path

@ensure_annotations
def load_config(config_file: Path) -> ConfigBox:
    """
    Load a YAML configuration file and return a ConfigBox object.

    Args:
        config_file (Path): The path to the YAML configuration file.

    Returns:
        ConfigBox: A ConfigBox object containing the loaded configuration.

    Raises:
        BoxValueError: If the YAML file is empty.
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        # Open the YAML file and load its contents
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"YAML file: {config_file} loaded successfully")

        # Wrap the loaded configuration in a ConfigBox object
        return ConfigBox(config)

    # Handle specific exceptions
    except BoxValueError:
       # Log an error message and re-raise the exception
        logger.error(f"Config file '{config_file}' is empty")
        raise ValueError('Yaml file is empty')
    except FileNotFoundError:
        # Log an error message and re-raise the exception
        logger.error(f"Config file '{config_file}' not found.")
        raise
    except yaml.YAMLError as e:
        # Log an error message and re-raise the exception
        logger.error(f"Error parsing YAML file: {e}")
        raise



@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create directories at the specified paths.

    Args:
        path_to_directories (list): A list of paths where directories will be created.
        verbose (bool, optional): Whether to log information about the created directories. Defaults to True.

    Raises:
        OSError: If there's an error creating any of the directories.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
    except OSError as e:
        # Log an error message and re-raise the exception
        logger.error(f"Error creating directory '{path_to_directories}': {e}")
        raise
