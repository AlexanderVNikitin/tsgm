import pytest
import tsgm

import tensorflow as tf
import numpy as np
from tensorflow import keras


def test_vae():
    seq_len = 32
    feat_dim = 1
    latent_dim = 4

    model_type = tsgm.models.architectures.zoo["vae_conv5"]
    architecture = model_type(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim)

    encoder, decoder = architecture.encoder, architecture.decoder

    X = tsgm.utils.gen_sine_dataset(50, seq_len, feat_dim, max_value=20)

    scaler = tsgm.utils.TSFeatureWiseScaler((0, 1))
    X = scaler.fit_transform(X).astype(np.float64)

    vae = tsgm.models.cvae.BetaVAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(0.0003))
    vae.fit(X, epochs=1, batch_size=128)
    x_decoded = vae.predict([X])
    assert x_decoded.shape == X.shape

    x_samples = vae.generate(7)
    assert x_samples.shape == (7, seq_len, feat_dim)

    x_decoded = vae([X])
    assert x_decoded.shape == X.shape


def test_cvae():
    seq_len = 32
    feat_dim = 1
    output_dim = 2
    latent_dim = 4

    model_type = tsgm.models.architectures.zoo["cvae_conv5"]
    architecture = model_type(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim, output_dim=2)

    encoder, decoder = architecture.encoder, architecture.decoder

    X, y_i = tsgm.utils.gen_sine_vs_const_dataset(50, seq_len, feat_dim, max_value=20, const=10)

    scaler = tsgm.utils.TSFeatureWiseScaler((0, 1))
    X = scaler.fit_transform(X).astype(np.float64)
    y = keras.utils.to_categorical(y_i, output_dim).astype(np.float64)

    cbk = tsgm.models.monitors.VAEMonitor(
        num_samples=1, latent_dim=latent_dim, output_dim=2)

    vae = tsgm.models.cvae.cBetaVAE(encoder, decoder, latent_dim=latent_dim, temporal=False)
    vae.compile(optimizer=keras.optimizers.Adam(0.0003))

    vae.fit(X, y, epochs=1, batch_size=128, callbacks=[cbk])
    x_decoded = vae.predict([X, y])
    assert x_decoded.shape == X.shape

    x_samples, y_samples = vae.generate(y[:7])
    assert x_samples.shape == (7, seq_len, feat_dim)

    x_decoded = vae([X, y])
    assert x_decoded.shape == X.shape


def test_temp_cvae():
    seq_len = 32
    feat_dim = 1
    output_dim = 1
    latent_dim = 4
    batch_size = 128

    model_type = tsgm.models.architectures.zoo["cvae_conv5"]
    architecture = model_type(seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim, output_dim=output_dim)

    X, y = tsgm.utils.gen_sine_const_switch_dataset(50, seq_len, 1, max_value=20, const=10)

    scaler = tsgm.utils.TSFeatureWiseScaler((0, 1))
    X_train = scaler.fit_transform(X)

    X_train = X_train.astype(np.float32)
    y = y.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    encoder, decoder = architecture.encoder, architecture.decoder

    vae = tsgm.models.cvae.cBetaVAE(encoder, decoder,  latent_dim=latent_dim, temporal=True)
    vae.compile(optimizer=keras.optimizers.Adam(0.0003))

    vae.fit(X_train, y, epochs=1, batch_size=128)
    x_decoded = vae.predict([X_train, y])
    assert x_decoded.shape == X.shape
