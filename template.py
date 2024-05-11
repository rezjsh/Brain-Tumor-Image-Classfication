import os
from pathlib import Path
import logging

FORMAT = '[%(asctime)s]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)


project_name = 'brainMRI'

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "notebook/trials.ipynb"
]



for filepath in list_of_files:
    filepath = Path(filepath)
    dir, filename = os.path.split(filepath)


    if dir != "":
        os.makedirs(dir, exist_ok=True)
        logging.info(f"directory {dir} is created.")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            logging.info(f"file{filepath} is created.")
    else:
        logging.info(f"{filename} is already exists.")