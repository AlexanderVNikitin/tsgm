import pytest
import tsgm

import numpy as np
import keras


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

    encoder, decoder = architecture.encoder, architecture.decoder

    vae = tsgm.models.cvae.cBetaVAE(encoder, decoder,  latent_dim=latent_dim, temporal=True)
    vae.compile(optimizer=keras.optimizers.Adam(0.0003))

    vae.fit(X_train, y, epochs=1, batch_size=batch_size)
    x_decoded = vae.predict([X_train, y])
    assert x_decoded.shape == X.shape


def _shifted_encoder(encoder: keras.Model, shift: float) -> keras.Model:
    """Wraps an encoder so that the sampled z is deterministic and differs from z_mean."""
    inputs = keras.Input(shape=encoder.input_shape[1:])
    z_mean, z_log_var, _ = encoder(inputs)
    z = keras.layers.Lambda(lambda t: t + shift)(z_mean)
    return keras.Model(inputs, [z_mean, z_log_var, z])


def _vae_and_data(conditional: bool):
    seq_len, feat_dim, latent_dim, output_dim = 32, 1, 4, 2
    if conditional:
        architecture = tsgm.models.architectures.zoo["cvae_conv5"](
            seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim, output_dim=output_dim)
        X, y_i = tsgm.utils.gen_sine_vs_const_dataset(16, seq_len, feat_dim, max_value=20, const=10)
        y = keras.utils.to_categorical(y_i, output_dim).astype(np.float32)
    else:
        architecture = tsgm.models.architectures.zoo["vae_conv5"](
            seq_len=seq_len, feat_dim=feat_dim, latent_dim=latent_dim)
        X = tsgm.utils.gen_sine_dataset(16, seq_len, feat_dim, max_value=20)
        y = None
    X = tsgm.utils.TSFeatureWiseScaler((0, 1)).fit_transform(X).astype(np.float32)
    return architecture, X, y, latent_dim


@pytest.mark.parametrize("conditional", [False, True])
def test_vae_decodes_sampled_z(conditional):
    # Regression test: the training loss must decode the sampled z, not z_mean
    architecture, X, y, latent_dim = _vae_and_data(conditional)
    shift = 5.0
    encoder = _shifted_encoder(architecture.encoder, shift)
    decoder = architecture.decoder
    if conditional:
        vae = tsgm.models.cvae.cBetaVAE(encoder, decoder, latent_dim=latent_dim, temporal=False)

        def decode(z):
            return decoder(vae._get_decoder_input(z, y))
    else:
        vae = tsgm.models.cvae.BetaVAE(encoder, decoder)
        decode = decoder

    X = keras.ops.convert_to_tensor(X)
    if conditional:
        y = keras.ops.convert_to_tensor(y)
        data = (X, y)
    else:
        data = X
    _, reconstruction_loss, _ = vae._compute_losses(data)
    encoder_input = vae._get_encoder_input(X, y) if conditional else X
    z_mean, _, z = encoder(encoder_input)

    def to_np(t):
        return keras.ops.convert_to_numpy(t)

    expected = vae._get_reconstruction_loss(X, decode(z))
    from_mean = vae._get_reconstruction_loss(X, decode(z_mean))
    assert np.allclose(to_np(reconstruction_loss), to_np(expected), rtol=1e-5)
    assert not np.allclose(to_np(reconstruction_loss), to_np(from_mean), rtol=1e-5)


@pytest.mark.parametrize("conditional", [False, True])
def test_vae_fit_updates_weights(conditional):
    architecture, X, y, latent_dim = _vae_and_data(conditional)
    encoder, decoder = architecture.encoder, architecture.decoder
    if conditional:
        vae = tsgm.models.cvae.cBetaVAE(encoder, decoder, latent_dim=latent_dim, temporal=False)
    else:
        vae = tsgm.models.cvae.BetaVAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(0.01))

    def snapshot(model):
        return [keras.ops.convert_to_numpy(w).copy() for w in model.trainable_weights]

    encoder_before, decoder_before = snapshot(encoder), snapshot(decoder)
    if conditional:
        history = vae.fit(X, y, epochs=1, batch_size=8, verbose=0)
    else:
        history = vae.fit(X, epochs=1, batch_size=8, verbose=0)

    def changed(before, model):
        return any(not np.allclose(b, a) for b, a in zip(before, snapshot(model)))

    assert changed(encoder_before, encoder)
    assert changed(decoder_before, decoder)
    assert np.isfinite(history.history["loss"][-1])


class TrainingSpy(keras.layers.Layer):
    """Identity layer that records the `training` argument of every call."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def call(self, inputs, training=None):
        self.calls.append(training)
        return inputs


def _spied(model: keras.Model) -> tuple:
    """Wraps a model so that the `training` argument it is called with is recorded."""
    spy = TrainingSpy()
    inputs = keras.Input(shape=model.input_shape[1:])
    wrapped = keras.Model(inputs, model(spy(inputs)))
    # forget the symbolic call made while the wrapper was built
    spy.calls.clear()
    return wrapped, spy


@pytest.mark.parametrize("conditional", [False, True])
def test_vae_trains_submodels_in_training_mode(conditional):
    # Regression test: layers such as Dropout and BatchNormalization are active only when called with training=True
    architecture, X, y, latent_dim = _vae_and_data(conditional)
    encoder, encoder_spy = _spied(architecture.encoder)
    decoder, decoder_spy = _spied(architecture.decoder)
    if conditional:
        vae = tsgm.models.cvae.cBetaVAE(encoder, decoder, latent_dim=latent_dim, temporal=False)
    else:
        vae = tsgm.models.cvae.BetaVAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(0.0003))

    if conditional:
        vae.fit(X, y, epochs=1, batch_size=8, verbose=0)
    else:
        vae.fit(X, epochs=1, batch_size=8, verbose=0)

    # a training step runs the encoder and the decoder once each
    assert True in encoder_spy.calls
    assert True in decoder_spy.calls

    # reconstruction and generation run in the inference mode
    encoder_spy.calls.clear()
    decoder_spy.calls.clear()
    if conditional:
        vae([X, y])
        vae.generate(keras.ops.convert_to_tensor(y[:3]))
    else:
        vae(X)
        vae.generate(3)
    assert encoder_spy.calls and True not in encoder_spy.calls
    assert decoder_spy.calls and True not in decoder_spy.calls
