import pytest
from unittest.mock import MagicMock

import random

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 

import tsgm


def _get_labels(num_samples, output_dim):
    labels = None
    for i in range(num_samples):
        sample = random.randint(0, output_dim - 1)
        if labels is None:
            labels = keras.utils.to_categorical([sample], output_dim)
        else:
            labels = tf.concat((labels, keras.utils.to_categorical([sample], output_dim)), 0)
    return keras.ops.cast(labels, "float32")


@pytest.mark.parametrize("save", [
    True, False
])
def test_ganmonitor(save, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    n_samples, n_classes = 3, 2
    labels = _get_labels(n_samples, n_classes)
    gan_monitor = tsgm.models.monitors.GANMonitor(
        num_samples=3, latent_dim=12, labels=labels, mode="clf", save=save)
    gan_monitor._model =  MagicMock()  # mock the model
    gan_monitor._model.generator.side_effect = lambda x: x[:, None]

    gan_monitor.on_epoch_end(epoch=2)


@pytest.mark.parametrize("save", [
    True, False
])
def test_vaemonitor(save, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    n_samples, n_classes = 3, 2
    vae_monitor = tsgm.models.monitors.VAEMonitor(
        num_samples=n_samples, latent_dim=12, save=save)
    vae_monitor._model =  MagicMock()  # mock the model
    vae_monitor._model.generate = lambda x: (x[:, 0][:, None], None)

    vae_monitor.on_epoch_end(epoch=2)


def test_exceptions():
    n_samples, n_classes = 3, 2
    labels = _get_labels(n_samples, n_classes)

    with pytest.raises(ValueError):
        tsgm.models.monitors.GANMonitor(
            num_samples=3, latent_dim=12, labels=labels, mode="abcde123", save=True)

    with pytest.raises(ValueError):
        gan_monitor = tsgm.models.monitors.GANMonitor(
            num_samples=3, latent_dim=12, labels=labels, mode="clf", save=True)
        gan_monitor._mode = "abcde123"
        gan_monitor.on_epoch_end(epoch=2)
