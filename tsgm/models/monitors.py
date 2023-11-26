import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow import keras
import typing as T

import seaborn as sns
import matplotlib.pyplot as plt

import tsgm.types
import tsgm.utils


logger = logging.getLogger('monitors')
logger.setLevel(logging.DEBUG)


class GANMonitor(keras.callbacks.Callback):
    """
    GANMonitor is a Keras callback for monitoring and visualizing generated samples during training.
    :param num_samples: The number of samples to generate and visualize.
    :type num_samples: int

    :param latent_dim: The dimensionality of the latent space. Defaults to 128.
    :type latent_dim: int

    :param output_dim: The dimensionality of the output space. Defaults to 2.
    :type output_dim: int

    :param save: Whether to save the generated samples. Defaults to True.
    :type save: bool

    :param save_path: The path to save the generated samples. Defaults to None.
    :type save_path: str
    :raises ValueError: If the mode is not one of ['clf', 'reg']

    :note: If `save` is True and `save_path` is not specified, the default save path is "/tmp/".

    :warning: If `save_path` is specified but `save` is False, a warning is issued.
    """
    def __init__(self, num_samples: int, latent_dim: int, labels: tsgm.types.Tensor,
                 save: bool = True, save_path: T.Optional[str] = None, mode: str = "clf") -> None:
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
            logger.warning("save_path is not specified. Using `/tmp` as the default save_path")

        if self._save_path is not None:
            if self._save is False:
                logger.warning("save_path is specified, but save is False.")
            os.makedirs(self._save_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: T.Optional[T.Dict] = None) -> None:
        """
        Callback function called at the end of each training epoch.

        :param epoch: Current epoch number.
        :type epoch: int

        :param logs: Dictionary containing the training loss values.
        :type logs: dict
        """
        if self._mode in ["clf", "reg"]:
            random_latent_vectors = tf.random.normal(shape=(self._num_samples, self._latent_dim))
        elif self._mode == "temporal":
            raise NotImplementedError
            # random_latent_vectors = tf.random.normal(shape=(self._output_dim * self._num_samples, self._latent_dim))
        else:
            raise ValueError("Invalid `mode` in GANMonitor: ", self._mode)

        labels = self._labels[:self._num_samples]

        generator_input = tf.concat([random_latent_vectors, labels], 1)
        generated_samples = self.model.generator(generator_input)

        for i in range(generated_samples.shape[0]):
            label = np.argmax(labels[i][None, :], axis=1)
            tsgm.utils.visualize_ts_lineplot(
                generated_samples[i][None, :],
                label, 1)  # TODO: update visualize_ts API

            if self._save:
                plt.savefig(os.path.join(self._save_path, "epoch_{}_sample_{}".format(epoch, i)))
            else:
                plt.show()


class VAEMonitor(keras.callbacks.Callback):
    """
    VAEMonitor is a Keras callback for monitoring and visualizing generated samples from a Variational Autoencoder (VAE) during training.

    :param num_samples: The number of samples to generate and visualize. Defaults to 6.
    :type num_samples: int

    :param latent_dim: The dimensionality of the latent space. Defaults to 128.
    :type latent_dim: int

    :param output_dim: The dimensionality of the output space. Defaults to 2.
    :type output_dim: int

    :param save: Whether to save the generated samples. Defaults to True.
    :type save: bool

    :param save_path: The path to save the generated samples. Defaults to None.
    :type save_path: str

    :raises ValueError: If `output_dim` is less than or equal to 0.

    :note: If `save` is True and `save_path` is not specified, the default save path is "/tmp/".

    :warning: If `save_path` is specified but `save` is False, a warning is issued.
    """
    def __init__(self, num_samples: int = 6, latent_dim: int = 128, output_dim: int = 2,
                 save: bool = True, save_path: T.Optional[str] = None) -> None:
        self._num_samples = num_samples
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._save = save
        self._save_path = save_path

        if self._save and self._save_path is None:
            self._save_path = "/tmp/"
            logger.warning("save_path is not specified. Using `/tmp` as the default save_path")

        if self._save_path is not None:
            if self._save is False:
                logger.warning("save_path is specified, but save is False.")
            os.makedirs(self._save_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: T.Optional[T.Dict] = None) -> None:
        """
        Callback function called at the end of each training epoch.

        :param epoch: The current epoch number.
        :type epoch: int

        :param logs: Dictionary containing the training loss values.
        :type logs: dict
        """
        labels = []
        for i in range(self._output_dim):
            if not len(labels):
                labels = keras.utils.to_categorical([i], self._output_dim)
            else:
                labels = tf.concat((labels, keras.utils.to_categorical([i], self._output_dim)), 0)

        labels = tf.repeat(labels, self._num_samples, axis=0)
        generated_images, _ = self.model.generate(labels)

        for i in range(self._output_dim * self._num_samples):
            sns.lineplot(
                x=range(0, generated_images[i].shape[0]),
                y=tf.squeeze(generated_images[i]).numpy()
            )
            if self._save:
                plt.savefig(os.path.join(self._save_path, "epoch_{}_sample_{}".format(epoch, i)))
            else:
                plt.show()
