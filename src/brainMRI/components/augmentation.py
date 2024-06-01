from dataclasses import dataclass
from typing import Tuple
import tensorflow as tf
from brainMRI.logging import logger
import matplotlib.pyplot as plt
from pathlib import Path
@dataclass
class DataAugmentation:
    training_dir: Path
    random_flip_horizontal: bool = True
    random_flip_vertical: bool = False
    random_rotation: bool = True
    random_zoom_height: bool = True
    random_zoom_width: bool = False
    random_brightness: bool = True
    random_contrast: bool = True
    random_translation_height: bool = True
    random_translation_width: bool = True
    random_rotation_factor: float = .2
    random_zoom_height_factor: float = .2
    random_zoom_width_factor: float = 0
    random_brightness_factor: float = .2
    random_contrast_lower_factor: float = .8
    random_contrast_upper_factor: float = 1.2
    random_translation_height_factor: float = .2
    random_translation_width_factor: float = .2



    def show_aug(self, data_augmentation):
        plt.figure(figsize=(10, 10))
        train_ds = tf.data.Dataset.load(self.training_dir)
       
        for image, _ in train_ds.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')

    def augmentation(self) -> tf.keras.Sequential:
        """
        Builds a data augmentation pipeline using the configuration stored in the dataclass.

        Returns:
            tf.keras.Sequential: The data augmentation pipeline.
        """
        
        logger.info("Building data augmentation pipeline...")
        data_augmentation = tf.keras.Sequential([])

        if self.random_flip_horizontal or self.random_flip_vertical:
            mode = ''
            if self.random_flip_horizontal and self.random_flip_vertical:
                mode = 'horizontal_and_vertical'
            elif self.random_flip_horizontal:
                mode = "horizontal"
            else:
                mode = "vertical"
            data_augmentation.add(tf.keras.layers.RandomFlip(mode=mode))
            logger.info(f"Added random flip layer with mode: {mode}")

        if self.random_rotation:
            data_augmentation.add(tf.keras.layers.RandomRotation(self.random_rotation_factor))
            logger.info(f"Added random rotation layer with factor: {self.random_rotation_factor}")

        if self.random_zoom_height or self.random_zoom_width:
            data_augmentation.add(tf.keras.layers.RandomZoom(self.random_zoom_height_factor if self.random_zoom_height else 0, self.random_zoom_width_factor if self.random_zoom_width else 0))
            logger.info(f"Added random zoom layer with height factor: {self.random_zoom_height_factor} and width factor: {self.random_zoom_width_factor}")

        if self.random_brightness:
            data_augmentation.add(tf.keras.layers.RandomBrightness(self.random_brightness_factor))
            logger.info(f"Added random brightness layer with factor: {self.random_brightness_factor}")

        if self.random_contrast:
            data_augmentation.add(tf.keras.layers.RandomContrast((self.random_contrast_lower_factor, self.random_contrast_upper_factor)))
            logger.info(f"Added random contrast layer with lower factor: {self.random_contrast_lower_factor} and upper factor: {self.random_contrast_upper_factor}")

        if self.random_translation_height or self.random_translation_width:
            data_augmentation.add(tf.keras.layers.RandomTranslation(self.random_translation_height_factor if self.random_translation_height else 0,
                                                                    self.random_translation_width_factor if self.random_translation_width else 0))
            logger.info(f"Added random translation layer with height factor: {self.random_translation_height_factor} and width factor: {self.random_translation_width_factor}")

        logger.info("Data augmentation pipeline built successfully.")
        
        logger.info("Data augmentation pipeline summary:")
        data_augmentation.summary(print_fn=lambda x: logger.info(x))

        return data_augmentation