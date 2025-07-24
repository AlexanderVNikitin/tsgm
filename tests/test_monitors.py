import pytest
from unittest.mock import MagicMock, patch

import random

import keras
from keras import ops
import matplotlib.pyplot as plt 

import tsgm


def _get_labels(num_samples, output_dim):
    labels = None
    for i in range(num_samples):
        sample = random.randint(0, output_dim - 1)
        if labels is None:
            labels = keras.utils.to_categorical([sample], output_dim)
        else:
            labels = ops.concatenate((labels, keras.utils.to_categorical([sample], output_dim)), 0)
    return labels


@pytest.mark.parametrize("save", [
    True, False
])
def test_ganmonitor(save, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    n_samples, n_classes = 3, 2
    labels = _get_labels(n_samples, n_classes)
    gan_monitor = tsgm.models.monitors.GANMonitor(
        num_samples=3, latent_dim=12, labels=labels, mode="clf", save=save)
    
    # Create mock model and patch the model property for Keras 3.0 compatibility
    mock_model = MagicMock()
    mock_model.generator.side_effect = lambda x: x[:, None]
    
    with patch.object(type(gan_monitor), 'model', new=mock_model):
        gan_monitor.on_epoch_end(epoch=2)


@pytest.mark.parametrize("save", [
    True, False
])
def test_vaemonitor(save, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    n_samples, n_classes = 3, 2
    vae_monitor = tsgm.models.monitors.VAEMonitor(
        num_samples=3, latent_dim=12, save=save)
    
    # Create mock model and patch the model property for Keras 3.0 compatibility
    mock_model = MagicMock()
    mock_model.generate = lambda x: (x[:, 0][:, None], None)
    
    with patch.object(type(vae_monitor), 'model', new=mock_model):
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
