import pytest

import functools
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers

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
    "clf_block"],
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


