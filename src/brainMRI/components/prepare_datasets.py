from dataclasses import dataclass
from typing import Tuple
from brainMRI.logging import logger
import tensorflow as tf
from pathlib import Path
import os

@dataclass
class PrepareDatasets:
    data_dir:Path
    save_dir: Path
    validation_split: float
    image_size: tuple[int, int]
    batch_size: int
    labels: str
    subset: str
    seed: int

    def prepare_datasets(self):
        """
        Prepare the training and validation datasets for the machine learning model.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset]: The prepared training and validation datasets.

        Raises:
            None
        """

        AUTOTUNE = tf.data.AUTOTUNE
        logger.info("Loading image datasets from directory: %s", self.data_dir)
        train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            image_size=self.image_size,
            batch_size=self.batch_size,
            labels=self.labels,
            subset=self.subset,
            seed=self.seed,
        )
        self.class_names = train_dataset.class_names
        logger.info("Saving class names to file: %s/class_names.txt", self.save_dir)
        with open(self.save_dir + '/class_names.txt', 'w') as f:
            f.write('\n'.join(self.class_names))

        logger.info("Class names: %s", self.class_names)
        logger.info("Prefetching datasets")
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

        logger.info("Saving datasets to directory: %s", self.save_dir)
        train_dataset.save(self.save_dir + '/train_dataset')
        val_dataset.save(self.save_dir + '/val_dataset')

        logger.info("Datasets prepared successfully")
        return train_dataset, val_dataset