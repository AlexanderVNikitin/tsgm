"""Regression coverage for GAN training through Keras' stateless JAX trainer."""

import numpy as np
import pytest

import tsgm
import keras


pytestmark = pytest.mark.skipif(keras.backend.backend() != "jax", reason="JAX trainer regressions")


def _make_gan(kind):
    seq_len, latent_dim = 4, 2
    temporal = kind in ("temporal", "temporal_multifeature")
    conditional = kind == "conditional" or temporal
    label_features = (1 if kind == "temporal" else 2) if conditional else 0
    noise_shape = (seq_len, latent_dim + label_features) if temporal else (latent_dim + label_features,)
    noise = keras.Input(noise_shape)
    hidden = keras.layers.Dense(8, activation="tanh")(noise)
    hidden = keras.layers.BatchNormalization()(hidden)
    hidden = keras.layers.Dropout(0.1)(hidden)
    output = keras.layers.Dense(1 if temporal else seq_len)(hidden)
    generated = output if temporal else keras.layers.Reshape((seq_len, 1))(output)
    generator = keras.Model(noise, generated)

    samples = keras.Input((seq_len, 1 + label_features))
    hidden = keras.layers.Flatten()(samples)
    hidden = keras.layers.Dense(8, activation="tanh")(hidden)
    hidden = keras.layers.BatchNormalization()(hidden)
    hidden = keras.layers.Dropout(0.1)(hidden)
    predictions = keras.layers.Dense(1, activation=None if kind == "wgan" else "sigmoid")(hidden)
    discriminator = keras.Model(samples, predictions)

    if conditional:
        model = tsgm.models.cgan.ConditionalGAN(discriminator, generator, latent_dim, temporal=temporal)
    else:
        model = tsgm.models.cgan.GAN(discriminator, generator, latent_dim, use_wgan=kind == "wgan")
    model.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.002),
        g_optimizer=keras.optimizers.SGD(learning_rate=0.003, momentum=0.5),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    return model


def _weights(model):
    return [np.asarray(variable).copy() for variable in model.trainable_variables]


def _assert_changed(before, model):
    assert any(not np.array_equal(old, new) for old, new in zip(before, _weights(model)))
    assert all(np.isfinite(value).all() for value in _weights(model))


@pytest.mark.parametrize("kind", ["gan", "conditional", "temporal", "temporal_multifeature", "wgan"])
def test_jax_gan_fit_preserves_alternating_state(kind):
    keras.utils.set_random_seed(42)
    model = _make_gan(kind)
    data = np.random.default_rng(42).normal(loc=0.5, size=(7, 4, 1)).astype("float32")
    labels = None
    if kind == "conditional":
        labels = np.eye(2, dtype="float32")[np.arange(len(data)) % 2]
    elif kind == "temporal":
        labels = (np.arange(len(data) * 4).reshape(len(data), 4) % 2).astype("float32")
    elif kind == "temporal_multifeature":
        labels = np.eye(2, dtype="float32")[np.arange(len(data) * 4).reshape(len(data), 4) % 2]

    before_generator, before_discriminator = _weights(model.generator), _weights(model.discriminator)
    bn_layers = [
        layer for network in (model.generator, model.discriminator)
        for layer in network.layers if isinstance(layer, keras.layers.BatchNormalization)
    ]
    means_before = [np.asarray(layer.moving_mean).copy() for layer in bn_layers]
    seed_before = np.asarray(model.seed_generator.state).copy()
    dropout_layers = [
        layer for network in (model.generator, model.discriminator)
        for layer in network.layers if isinstance(layer, keras.layers.Dropout)
    ]
    dropout_seeds_before = [np.asarray(layer.seed_generator.state).copy() for layer in dropout_layers]
    history = model.fit(data, labels, batch_size=3, epochs=2, shuffle=False, verbose=0)

    assert len(history.history) == 2
    assert all(len(values) == 2 and np.isfinite(values).all() for values in history.history.values())
    _assert_changed(before_generator, model.generator)
    _assert_changed(before_discriminator, model.discriminator)
    assert int(model.d_optimizer.iterations) == 6
    assert int(model.g_optimizer.iterations) == 6
    assert not np.array_equal(seed_before, np.asarray(model.seed_generator.state))
    # Both adversarial passes consume noise; WGAN also samples the penalty interpolation.
    assert int(model.seed_generator.state[1]) == int(seed_before[1]) + 6 * (3 if kind == "wgan" else 2)
    for before, layer in zip(dropout_seeds_before, dropout_layers):
        assert int(layer.seed_generator.state[1]) > int(before[1])
    for optimizer in (model.d_optimizer, model.g_optimizer):
        slots = [np.asarray(variable) for variable in optimizer.variables[2:]]
        assert slots and any(np.any(slot) for slot in slots)
        assert all(np.isfinite(slot).all() for slot in slots)
    for before, layer in zip(means_before, bn_layers):
        assert not np.array_equal(before, np.asarray(layer.moving_mean))
        assert np.isfinite(np.asarray(layer.moving_variance)).all()

    # A second fit must continue both optimizer slot sets and random streams.
    before_generator, before_discriminator = _weights(model.generator), _weights(model.discriminator)
    seed_before = np.asarray(model.seed_generator.state).copy()
    history = model.fit(data, labels, batch_size=3, epochs=1, shuffle=False, verbose=0)
    assert all(np.isfinite(values).all() for values in history.history.values())
    assert int(model.d_optimizer.iterations) == 9
    assert int(model.g_optimizer.iterations) == 9
    _assert_changed(before_generator, model.generator)
    _assert_changed(before_discriminator, model.discriminator)
    assert not np.array_equal(seed_before, np.asarray(model.seed_generator.state))

    generation_input = labels if labels is not None else len(data)
    samples = np.asarray(model.generate(generation_input))
    next_samples = np.asarray(model.generate(generation_input))
    assert samples.shape == next_samples.shape == data.shape
    assert np.isfinite(samples).all() and np.isfinite(next_samples).all()
    assert not np.array_equal(samples, next_samples)


def test_jax_conditional_labels_are_constant_across_time():
    model = _make_gan("conditional")
    data = np.zeros((2, 4, 1), dtype="float32")
    labels = np.eye(2, dtype="float32")
    _, generator_labels, discriminator_labels = model._prepare_jax_data((data, labels))
    np.testing.assert_array_equal(generator_labels, labels)
    np.testing.assert_array_equal(discriminator_labels, np.repeat(labels[:, None, :], 4, axis=1))


def test_jax_gan_step_matches_alternating_gradients():
    """The generator must use the newly updated discriminator and its own learning rate."""
    import jax
    import jax.numpy as jnp

    noise = keras.Input((2,))
    generator_layer = keras.layers.Dense(4, use_bias=False)
    generator = keras.Model(noise, keras.layers.Reshape((4, 1))(generator_layer(noise)))
    samples = keras.Input((4, 1))
    discriminator_layer = keras.layers.Dense(1, use_bias=False, activation="sigmoid")
    discriminator = keras.Model(samples, discriminator_layer(keras.layers.Flatten()(samples)))
    model = tsgm.models.cgan.GAN(discriminator, generator, 2)
    model.seed_generator.state.assign([12345, 0])
    model.compile(keras.optimizers.SGD(0.02), keras.optimizers.SGD(0.03), keras.losses.BinaryCrossentropy())
    generator_kernel = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 20
    discriminator_kernel = jnp.arange(4, dtype=jnp.float32).reshape(4, 1) / 10
    generator_layer.kernel.assign(generator_kernel)
    discriminator_layer.kernel.assign(discriminator_kernel)
    data = np.arange(12, dtype="float32").reshape(3, 4, 1) / 10
    reference_seed = keras.random.SeedGenerator(12345)
    d_noise = keras.random.normal((3, 2), seed=reference_seed)
    g_noise = keras.random.normal((3, 2), seed=reference_seed)

    def discriminator_loss(kernel):
        fake_logits = d_noise @ generator_kernel @ kernel
        real_logits = data.reshape(3, 4) @ kernel
        return (jnp.mean(jax.nn.softplus(-fake_logits)) + jnp.mean(jax.nn.softplus(real_logits))) / 2

    expected_d_loss, gradient = jax.value_and_grad(discriminator_loss)(discriminator_kernel)
    expected_discriminator = discriminator_kernel - 0.02 * gradient

    def generator_loss(kernel):
        return jnp.mean(jax.nn.softplus(g_noise @ kernel @ expected_discriminator))

    expected_g_loss, gradient = jax.value_and_grad(generator_loss)(generator_kernel)
    expected_generator = generator_kernel - 0.03 * gradient
    history = model.fit(data, batch_size=3, epochs=1, verbose=0)

    np.testing.assert_allclose(discriminator_layer.kernel, expected_discriminator, atol=1e-6)
    np.testing.assert_allclose(generator_layer.kernel, expected_generator, atol=1e-6)
    np.testing.assert_allclose(history.history["d_loss"], [expected_d_loss], atol=1e-6)
    np.testing.assert_allclose(history.history["g_loss"], [expected_g_loss], atol=1e-6)


def test_jax_wgan_penalty_differentiates_critic_weights():
    import jax

    noise = keras.Input((2,))
    generator = keras.Model(noise, keras.layers.Reshape((4, 1))(keras.layers.Dense(4)(noise)))
    samples = keras.Input((4, 1))
    critic = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")
    discriminator = keras.Model(samples, critic(keras.layers.Flatten()(samples)))
    model = tsgm.models.cgan.GAN(discriminator, generator, 2, use_wgan=True)
    real_data, fake_data = np.zeros((2, 4, 1), dtype="float32"), np.ones((2, 4, 1), dtype="float32")

    def penalty(kernel):
        with keras.StatelessScope(state_mapping=[(critic.kernel, kernel)]):
            return model._jax_gradient_penalty(jax, real_data, fake_data)

    value, gradient = jax.value_and_grad(penalty)(critic.kernel.value)
    np.testing.assert_allclose(value, 1.0)
    np.testing.assert_allclose(gradient, np.ones((4, 1)), rtol=1e-6)
