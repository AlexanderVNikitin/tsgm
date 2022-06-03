import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import typing

import seaborn as sns
import matplotlib.pyplot as plt

import tsgm


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_samples: int, latent_dim: int, labels: tsgm.types.Tensor,
                 save: bool = True, save_path: typing.Optional[str] = None, mode: str = "clf"):
        self._num_samples = num_samples
        self._latent_dim = latent_dim
        self._save = save
        self._save_path = save_path
        self._mode = mode
        if self._mode not in ["clf", "reg"]:
            raise ValueError("The mode should be in ['clf', 'reg']")

        self._labels = labels

        if self._save and self._save_path is None:
            self._save_path = "/tmp/"
            print("[WARNING]: save_path is not specified. Using `/tmp` as the default save_path")

        if self._save is False and self._save_path is not None:
            print("[WARNING]: save_path is specified, but save is False.")
            os.makedirs(self._save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self._mode in ["clf", "reg"]:
            random_latent_vectors = tf.random.normal(shape=(self._num_samples, self._latent_dim))
        else:
            raise ValueError("Invalid `mode` in GANMonitor: ", self._mode)
 
        labels = self._labels[:self._num_samples]
    
        generator_input = tf.concat([random_latent_vectors, labels], 1)
        generated_samples = self.model.generator(generator_input)

        for i in range(generated_samples.shape[0]):
            label = np.argmax(labels[i])
            tsgm.utils.visualize_ts_lineplot(
                generated_samples[i][None, :],
                np.argmax(labels[i][None, :], axis=1), 1)  # TODO: update visualize_ts API

            if self._save:
                plt.savefig(os.path.join(self._save_path, "epoch_{}_sample_{}".format(epoch, i)))
            else:
                plt.show()


class VAEMonitor(keras.callbacks.Callback):
    def __init__(self, num_samples=6, latent_dim=128, num_classes=2):
        self._num_samples = num_samples
        self._latent_dim = latent_dim
        self._num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self._num_classes * self._num_samples, self._latent_dim))

        labels = []
        for i in range(self._num_classes):
            if not len(labels):
                labels = keras.utils.to_categorical([i], self._num_classes)
            else:
                labels = tf.concat((labels, keras.utils.to_categorical([i], self._num_classes)), 0)

        labels = tf.repeat(labels, self._num_samples, axis=0)
        generated_images = self.model.decoder(tf.concat([random_latent_vectors, labels], 1))

        for i in range(self._num_classes * self._num_samples):
            sns.lineplot(x=range(0, generated_images[i].shape[0]), y=tf.squeeze(generated_images[i]))
            plt.show()
