from dataclasses import dataclass
import os
from brainMRI.logging import logger
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import pickle

@dataclass
class Callbacks:
    root_dir: Path
    patience: int = 5
    factor: float = 0.1
    min_lr: float = 1e-6


    def get_callbacks(self) -> list:
        """
        A class that generates a set of callbacks for use in Keras model training.

        Attributes:
            log_dir (str): The directory to store the TensorBoard logs.
            checkpoint_dir (str): The directory to store the model checkpoint files.
            patience (int): The number of epochs with no improvement after which training will be stopped.
            factor (float): The factor by which the learning rate will be reduced on plateau.
            min_lr (float): The minimum learning rate.
    """
        
        checkpoint_dir = self.root_dir + '/ckpt'
        # Create the ModelCheckpoint callback
        model_checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model-{epoch:02d}-{val_accuracy:.2f}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
        log_dir = self.root_dir + '/logs'
        # Create the TensorBoard callback
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )

        # Create the EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            verbose=1
        )

        # Create the ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.factor,
            patience=self.patience // 2,
            min_lr=self.min_lr,
            verbose=1
        )

        logger.info("Callbacks initialized successfully.")

        callbacks = [model_checkpoint, tensorboard, early_stopping]
        callbacks_path = os.path.join(self.root_dir, "callbacks.pickle")
        with open(callbacks_path, 'wb') as handle:
            pickle.dump(callbacks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return [model_checkpoint, tensorboard, early_stopping, reduce_lr]
