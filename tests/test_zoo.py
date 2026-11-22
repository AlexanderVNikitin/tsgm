import pytest

import numpy as np
import keras
from keras import layers

import sklearn.metrics.pairwise

import tsgm


@pytest.mark.parametrize("model_type", [
    tsgm.models.architectures.zoo["cvae_conv5"],
])
def test_zoo_cvae(model_type):
    seq_len = 10
    feat_dim = 2
    latent_dim = 1
    output_dim = 1
    
    arch = model_type(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim, output_dim=output_dim)
    arch_dict = arch.get()

    assert arch.encoder == arch_dict["encoder"] and arch.decoder == arch_dict["decoder"]


@pytest.mark.parametrize("model_type", [
    tsgm.models.architectures.zoo["cgan_base_c4_l1"],
    tsgm.models.architectures.zoo["cgan_lstm_n"],
    tsgm.models.architectures.zoo["cgan_lstm_3"],
])
def test_zoo_cgan(model_type):
    seq_len = 10
    feat_dim = 2
    latent_dim = 1
    output_dim = 1
    
    arch = model_type(
        seq_len=seq_len, feat_dim=feat_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    arch_dict = arch.get()

    assert arch.generator == arch_dict["generator"] and arch.discriminator == arch_dict["discriminator"]


@pytest.mark.parametrize("model_type_name", [
    "clf_cn",
    "clf_cl_n",
    "clf_block",
    "clf_transformer",]
)
def test_zoo_clf(model_type_name):
    seq_len = 10
    feat_dim = 2
    output_dim = 1
    model_type = tsgm.models.architectures.zoo[model_type_name]
    if model_type_name == "clf_block":
        arch = model_type(
            seq_len=seq_len, feat_dim=feat_dim, output_dim=output_dim, blocks=[layers.Conv1D(filters=64, kernel_size=3, activation="relu")])
    else:
        arch = model_type(
            seq_len=seq_len, feat_dim=feat_dim, output_dim=output_dim)
    arch_dict = arch.get()

    assert arch.model == arch_dict["model"]


@pytest.mark.parametrize("network_type", [
    "gru",
    "lstm",
])
def test_basic_rec(network_type):
    seq_len = 10
    feat_dim = 2
    output_dim = 1
    
    arch = tsgm.models.zoo["recurrent"](
        hidden_dim=2,
        output_dim=output_dim,
        n_layers=1,
        network_type=network_type)
    model = arch.build()
    assert model is not None


def test_summary(capfd):
    tsgm.models.zoo.summary()
    out, err = capfd.readouterr()
    assert "cgan_base_c4_l1" in out and \
           "id" in out and "type" in out


def test_zoo_lstm_n():
    seq_len = 10
    feat_dim = 2
    latent_dim = 1
    output_dim = 1
    model_type = tsgm.models.architectures.zoo["cgan_lstm_n"]
    arch = model_type(
        seq_len=seq_len, feat_dim=feat_dim,
        latent_dim=latent_dim, output_dim=output_dim, n_blocks=2)
    arch_dict = arch.get()

    assert arch.generator == arch_dict["generator"] and arch.discriminator == arch_dict["discriminator"]


def _dropout_rates(model):
    return [layer.rate for layer in model.layers if isinstance(layer, layers.Dropout)]


@pytest.mark.parametrize("model_name, kwargs", [
    ("cgan_base_c4_l1", {}),
    ("t-cgan_c4", {}),
    ("cgan_lstm_n", {"n_blocks": 2}),
    ("cgan_lstm_3", {}),
])
def test_zoo_gan_dropout_rate(model_name, kwargs):
    # Dropout is off unless it is requested with `dropout_rate`
    arch_kwargs = dict(seq_len=16, feat_dim=2, latent_dim=4, output_dim=1, **kwargs)

    arch = tsgm.models.architectures.zoo[model_name](**arch_kwargs)
    rates = _dropout_rates(arch.generator) + _dropout_rates(arch.discriminator)
    assert rates and set(rates) == {0.0}

    arch = tsgm.models.architectures.zoo[model_name](dropout_rate=0.3, **arch_kwargs)
    rates = _dropout_rates(arch.generator) + _dropout_rates(arch.discriminator)
    assert rates and set(rates) == {0.3}


@pytest.mark.parametrize("model_name, kwargs", [
    ("vae_conv5", {}),
    ("cvae_conv5", {"output_dim": 2}),
])
def test_zoo_vae_dropout_rate(model_name, kwargs):
    # `dropout_rate` configures the encoder; a decoder trained with Dropout generates distorted samples without it
    arch_kwargs = dict(seq_len=16, feat_dim=2, latent_dim=4, **kwargs)

    arch = tsgm.models.architectures.zoo[model_name](**arch_kwargs)
    assert set(_dropout_rates(arch.encoder)) == {0.0}
    assert _dropout_rates(arch.decoder) == []

    arch = tsgm.models.architectures.zoo[model_name](dropout_rate=0.3, **arch_kwargs)
    assert set(_dropout_rates(arch.encoder)) == {0.3}
    assert _dropout_rates(arch.decoder) == []


def test_zoo_dropout_rate_controls_training_mode_noise():
    seq_len = 16
    feat_dim = 1
    output_dim = 1
    x = np.random.normal(size=(8, seq_len, feat_dim + output_dim)).astype(np.float32)

    def training_mode_outputs(**kwargs):
        arch = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
            seq_len=seq_len, feat_dim=feat_dim, latent_dim=4, output_dim=output_dim, **kwargs)
        return [keras.ops.convert_to_numpy(arch.discriminator(x, training=True)) for _ in range(2)]

    first, second = training_mode_outputs()
    np.testing.assert_array_equal(first, second)

    first, second = training_mode_outputs(dropout_rate=0.5)
    assert not np.allclose(first, second)


def test_zoo_lstm_n_discriminator_is_convolutional():
    # A discriminator made of LSTM blocks and average pooling over time stays at chance level in training
    arch = tsgm.models.architectures.zoo["cgan_lstm_n"](
        seq_len=16, feat_dim=2, latent_dim=4, output_dim=1, n_blocks=2)

    assert not any(isinstance(layer, layers.RNN) for layer in arch.discriminator.layers)
    assert sum(isinstance(layer, layers.Conv1D) for layer in arch.discriminator.layers) == 4
    assert arch.discriminator.output_shape == (None, 1)

    # `n_blocks` configures the generator
    assert sum(isinstance(layer, layers.LSTM) for layer in arch.generator.layers) == 2
