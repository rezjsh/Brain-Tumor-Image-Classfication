import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from brainMRI.components.augmentation import DataAugmentation
from brainMRI.logging import logger

@dataclass
class BaseModel:
    root_dir: Path
    weights: str
    include_top: bool
    input_shape: tuple
    fine_tune_at: int = 0
    use_augmentation: bool = True
    data_augmentation_config: DataAugmentation = None

    def __post_init__(self):
        self.base_model = tf.keras.applications.VGG16(weights=self.weights,
                                                     include_top=self.include_top, input_shape=self.input_shape)


    def build_model(self, data_augmentation):
        """
        Build the final model, including the base model and the classification layers.

        Parameters:
        data_augmentation (tensorflow.keras.Sequential): A data augmentation pipeline.

        Returns:
        tensorflow.keras.Model: The final model.
        """
        try:
          
            if self.use_augmentation:
                data_augmentation = self.data_augmentation_config.augmentation()
                self.data_augmentation_config.show_aug(data_augmentation)
            preprocess_input = tf.keras.applications.vgg16.preprocess_input
            inputs = tf.keras.Input(shape=self.input_shape)

            if self.use_augmentation:
                x = data_augmentation(inputs)
                x = preprocess_input(x)
            else:
                x = preprocess_input(inputs)

            x = self.base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.Model(inputs, outputs)

            model.summary()

            model_summary_path = self.root_dir + '/model_summary.txt'
            with open(model_summary_path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            logger.info(f"Model summary saved to: {model_summary_path}")


            tf.keras.utils.plot_model(model, show_shapes=True)
            # Save the model plot to a file
            model_plot_path = self.root_dir + '/model_plot.png'
            tf.keras.utils.plot_model(model, to_file=str(model_plot_path), show_shapes=True)
            logger.info(f"Model plot saved to: {model_plot_path}")


            logger.info(f"Number of trainable variables: {len(model.trainable_variables)}")
            logger.info(f"Number of layers in the base model: {len(self.base_model.layers)}")

            # Freeze the base model layers
            for layer in self.base_model.layers:
                layer.trainable = False

            # Unfreeze the specified number of layers for fine-tuning
            if self.fine_tune_at > 0:
                # Freeze all the layers before the `fine_tune_at` layer
                for layer in self.base_model.layers[:self.fine_tune_at]:
                    layer.trainable = False

            # Save the model
            model_path = self.root_dir + '/base_model.keras'
            # tf.keras.models.save_model(model, model_path)
            model.save(model_path)
            logger.info(f"Model saved to: {model_path}")

            return model
        except Exception as e:
            raise e