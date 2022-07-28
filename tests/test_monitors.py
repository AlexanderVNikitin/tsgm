import pytest
from unittest.mock import MagicMock

import random

import tensorflow as tf
from tensorflow import keras

import tsgm


def _get_labels(num_samples, num_classes):
    labels = None
    for i in range(num_samples):
        sample = random.randint(0, num_classes - 1)
        if labels is None:
            labels = keras.utils.to_categorical([sample], num_classes)
        else:
            labels = tf.concat((labels, keras.utils.to_categorical([sample], num_classes)), 0)
    return labels


def test_ganmonitor():
    n_samples, n_classes = 3, 2
    labels = _get_labels(n_samples, n_classes)
    gan_monitor = tsgm.models.monitors.GANMonitor(
        num_samples=3, latent_dim=123, labels=labels, mode="clf", save=True)
    gan_monitor.model =  MagicMock()  # mock the model
    gan_monitor.model.generator.side_effect = lambda x: x[:, None]

    gan_monitor.on_epoch_end(epoch=100)


def test_exceptions():
    n_samples, n_classes = 3, 2
    labels = _get_labels(n_samples, n_classes)

    with pytest.raises(ValueError):
        tsgm.models.monitors.GANMonitor(
            num_samples=3, latent_dim=123, labels=labels, mode="abcde123", save=True)

    with pytest.raises(ValueError):
        gan_monitor = tsgm.models.monitors.GANMonitor(
            num_samples=3, latent_dim=123, labels=labels, mode="clf", save=True)
        gan_monitor._mode = "abcde123"
        gan_monitor.on_epoch_end(epoch=100)
