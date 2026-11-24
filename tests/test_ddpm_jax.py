"""DDPM training regressions, shared across JAX, TensorFlow, and PyTorch."""

import numpy as np

import tsgm
import keras


def _snapshot(variables):
    return [keras.ops.convert_to_numpy(variable).copy() for variable in variables]


def _model(network, ema=0.5, learning_rate=0.01):
    ema_network = keras.models.clone_model(network)
    model = tsgm.models.ddpm.DDPM(network, ema_network, timesteps=4, ema=ema)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate))
    return model


def test_jax_ddpm_fit_updates_weights_and_generates():
    keras.utils.set_random_seed(17)
    architecture = tsgm.models.zoo["ddpm_denoiser"]
    network = architecture(seq_len=6, feat_dim=1, n_filters=4, n_conv_layers=1).model
    ema_network = architecture(seq_len=6, feat_dim=1, n_filters=4, n_conv_layers=1).model
    model = tsgm.models.ddpm.DDPM(network, ema_network, timesteps=4, ema=0.5)
    model.compile(optimizer=keras.optimizers.Adam(0.01))
    before = _snapshot(model.network.trainable_weights)
    ema_before = _snapshot(model.ema_network.weights)
    x = np.linspace(-1, 1, 30, dtype=np.float32).reshape(5, 6, 1)

    history = model.fit(x, epochs=2, batch_size=2, verbose=0)

    assert np.all(np.isfinite(history.history["loss"]))
    assert all(value > 0 for value in history.history["loss"])
    assert any(not np.array_equal(a, b) for a, b in zip(before, _snapshot(model.network.trainable_weights)))
    assert any(not np.array_equal(a, b) for a, b in zip(ema_before, _snapshot(model.ema_network.weights)))
    samples = keras.ops.convert_to_numpy(model.generate(3))
    assert samples.shape == (3, 6, 1)
    assert np.all(np.isfinite(samples))


def test_jax_ddpm_updates_ema_and_preserves_non_trainable_state():
    keras.utils.set_random_seed(23)
    inputs = keras.Input((6, 1))
    time = keras.Input((1,))
    time_features = keras.layers.RepeatVector(6)(time)
    x = keras.layers.Concatenate()([inputs, time_features])
    x = keras.layers.Conv1D(4, 1, kernel_initializer="ones")(x)
    batch_norm = keras.layers.BatchNormalization(momentum=0.5)
    x = batch_norm(x)
    dropout = keras.layers.Dropout(0.25, seed=13)
    x = dropout(x)
    outputs = keras.layers.Conv1D(1, 1)(x)
    network = keras.Model([inputs, time], outputs, name="stateful_denoiser")
    model = _model(network)
    before = _snapshot(model.ema_network.weights)
    dropout_seed_before = keras.ops.convert_to_numpy(dropout.seed_generator.state).copy()
    diffusion_seed_before = keras.ops.convert_to_numpy(model.seed_generator.state).copy()

    model.fit(np.ones((4, 6, 1), dtype=np.float32), epochs=1, batch_size=4, verbose=0)

    for previous, current, averaged in zip(
        before, _snapshot(model.network.weights), _snapshot(model.ema_network.weights),
    ):
        np.testing.assert_allclose(averaged, 0.5 * previous + 0.5 * current, rtol=1e-5, atol=1e-6)
    assert np.any(keras.ops.convert_to_numpy(batch_norm.moving_mean) != 0)
    assert not np.array_equal(dropout_seed_before, keras.ops.convert_to_numpy(dropout.seed_generator.state))
    assert not np.array_equal(diffusion_seed_before, keras.ops.convert_to_numpy(model.seed_generator.state))
    assert not model.ema_network.trainable_variables
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == 1


def test_jax_ddpm_draws_fresh_noise_each_step():
    keras.utils.set_random_seed(31)
    architecture = tsgm.models.zoo["ddpm_denoiser"]
    network = architecture(seq_len=6, feat_dim=1, n_filters=4, n_conv_layers=1).model
    ema_network = architecture(seq_len=6, feat_dim=1, n_filters=4, n_conv_layers=1).model
    for variable in network.weights:
        variable.assign(keras.ops.zeros(variable.shape, dtype=variable.dtype))
    model = tsgm.models.ddpm.DDPM(network, ema_network, timesteps=4)
    model.compile(optimizer=keras.optimizers.SGD(0.0))
    seed_before = keras.ops.convert_to_numpy(model.seed_generator.state).copy()

    # With fixed zero predictions and a zero learning rate, a different loss
    # requires different diffusion noise in the second compiled training step.
    history = model.fit(
        np.ones((4, 6, 1), dtype=np.float32), epochs=2, batch_size=4, shuffle=False, verbose=0,
    )

    assert not np.isclose(history.history["loss"][0], history.history["loss"][1])
    seed_after = keras.ops.convert_to_numpy(model.seed_generator.state)
    assert seed_after[1] - seed_before[1] == 4
