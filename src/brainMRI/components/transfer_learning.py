from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from pathlib import Path
import pickle

@dataclass
class TransferLearning:
    root_dir: Path
    train_dir: Path
    val_dir: Path
    base_model_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    callback_path: Path


    def __post_init__(self):
        self.train_dataset = tf.data.Dataset.load(self.train_dir)
        self.val_dataset = tf.data.Dataset.load(self.val_dir)
        self.base_model = tf.keras.models.load_model(self.base_model_path, safe_mode=False)
        with open(self.callback_path, 'rb') as handle:
            self.callbacks = pickle.load(handle)

    def train(self):
        self.base_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])
        
        self.history = self.base_model.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.val_dataset,
            callbacks=self.callbacks
        )
        self.save_model(self.base_model)

    def save_plots(self):
        """
        Saves the accuracy and loss plots for the training and validation sets.
        
        Args:
            history (tf.keras.callbacks.History): The history object returned by the model.fit() method.
        """
        
        # Plot the accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(self.root_dir, 'accuracy.png'))
        
        # Plot the loss
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(os.path.join(self.root_dir, 'loss.png'))

    def save_model(self, model):
        """
        Saves the trained model to the specified output directory.
        
        Args:
            model (tf.keras.Model): The trained model to be saved.
        """
        # Save the model
        model.save(os.path.join(self.root_dir, 'model.keras'))
