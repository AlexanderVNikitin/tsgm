import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tsgm

try:
    import tensorflow_privacy as tf_privacy
    __tf_privacy_available = True
except ModuleNotFoundError:
    __tf_privacy_available = False
import numpy as np
import keras


from tsgm.backend import get_backend


def _gen_dataset(seq_len: int, feature_dim: int, batch_size: int):
    data = tsgm.utils.gen_sine_dataset(50, seq_len, feature_dim)

    scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
    X_train = scaler.fit_transform(data).astype(np.float32)

    backend = get_backend()
    if os.environ.get("KERAS_BACKEND") == "tensorflow":
        dataset = backend.data.Dataset.from_tensor_slices(X_train)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    elif os.environ.get("KERAS_BACKEND") == "torch":
        X_train_tensor = backend.from_numpy(X_train)
        dataset = backend.utils.data.TensorDataset(X_train_tensor)
        dataset = backend.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset


def _gen_cond_dataset(seq_len: int, batch_size: int):
    X, y_i = tsgm.utils.gen_sine_vs_const_dataset(50, seq_len, 1, max_value=20, const=10)

    scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
    X_train = scaler.fit_transform(X).astype(np.float32)
    y = keras.utils.to_categorical(y_i, 2).astype(np.float32)

    backend = get_backend()
    if os.environ.get("KERAS_BACKEND") == "tensorflow":
        dataset = backend.data.Dataset.from_tensor_slices((X_train, y))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    elif os.environ.get("KERAS_BACKEND") == "torch":
        X_train_tensor = backend.from_numpy(X_train)
        y_tensor = backend.from_numpy(y)
        dataset = backend.utils.data.TensorDataset(X_train_tensor, y_tensor)
        dataset = backend.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, y


def _gen_t_cond_dataset(seq_len: int, batch_size: int):
    X, y = tsgm.utils.gen_sine_const_switch_dataset(50, seq_len, 1, max_value=20, const=10)

    scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
    X_train = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)

    backend = get_backend()
    if os.environ.get("KERAS_BACKEND") == "tensorflow":
        dataset = backend.data.Dataset.from_tensor_slices((X_train, y))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    elif os.environ.get("KERAS_BACKEND") == "torch":
        X_train_tensor = backend.from_numpy(X_train)
        y_tensor = backend.from_numpy(y)
        dataset = backend.utils.data.TensorDataset(X_train_tensor, y_tensor)
        dataset = backend.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, y


def test_gan():
    latent_dim = 4
    feature_dim = 1
    seq_len = 32
    batch_size = 30

    dataset = _gen_dataset(seq_len, feature_dim, batch_size)
    architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=0)
    discriminator, generator = architecture.discriminator, architecture.generator

    gan = tsgm.models.cgan.GAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    gan.fit(dataset, epochs=1)

    assert gan.generator is not None
    assert gan.discriminator is not None
    # Check generation
    generated_samples = gan.generate(10)
    assert generated_samples.shape == (10, seq_len, 1)


def test_cgan():
    latent_dim = 8
    output_dim = 2
    feature_dim = 1
    seq_len = 32
    batch_size = 48

    dataset, labels = _gen_cond_dataset(seq_len, batch_size)
    architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    cbk = tsgm.models.monitors.GANMonitor(
        num_samples=3, latent_dim=latent_dim, labels=labels, save=True)
    cond_gan.fit(dataset, epochs=1, callbacks=[cbk])

    assert cond_gan.generator is not None
    assert cond_gan.discriminator is not None

    # Check generation
    # Handle PyTorch DataLoader vs TensorFlow dataset
    if hasattr(dataset, 'as_numpy_iterator'):
        # TensorFlow dataset
        labels = next(dataset.as_numpy_iterator())[1][:10]
    else:
        # PyTorch DataLoader
        batch = next(iter(dataset))
        labels = batch[1][:10].cpu().numpy() if hasattr(batch[1], 'cpu') else batch[1][:10]

    generated_samples = cond_gan.generate(labels)
    assert generated_samples.shape == (10, seq_len, 1)


def test_cgan_seq_len_33():
    latent_dim = 4
    output_dim = 2
    feature_dim = 1
    seq_len = 33
    batch_size = 16

    dataset, labels = _gen_cond_dataset(seq_len, batch_size)
    architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim,
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    cbk = tsgm.models.monitors.GANMonitor(
        num_samples=3, latent_dim=latent_dim, labels=labels, save=True, save_path="./tmp/gan_test_log")
    cond_gan.fit(dataset, epochs=1, callbacks=[cbk])

    assert cond_gan.generator is not None
    assert cond_gan.discriminator is not None

    # Check generation
    # Handle PyTorch DataLoader vs TensorFlow dataset
    if hasattr(dataset, 'as_numpy_iterator'):
        # TensorFlow dataset
        labels = next(dataset.as_numpy_iterator())[1][:10]
    else:
        # PyTorch DataLoader
        batch = next(iter(dataset))
        labels = batch[1][:10].cpu().numpy() if hasattr(batch[1], 'cpu') else batch[1][:10]

    generated_samples = cond_gan.generate(labels)

    assert generated_samples.shape == (10, seq_len, 1)


def test_temporal_cgan():
    latent_dim = 2
    output_dim = 1
    feature_dim = 1
    seq_len = 32
    batch_size = 12
    dataset, labels = _gen_t_cond_dataset(seq_len, batch_size)
    architecture = tsgm.models.architectures.zoo["t-cgan_c4"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim, temporal=True,
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    cond_gan.fit(dataset, epochs=1)
    assert cond_gan.generator is not None
    assert cond_gan.discriminator is not None

    # Check generation
    # Handle PyTorch DataLoader vs TensorFlow dataset
    if hasattr(dataset, 'as_numpy_iterator'):
        # TensorFlow dataset
        labels = next(dataset.as_numpy_iterator())[1][:10]
    else:
        # PyTorch DataLoader
        batch = next(iter(dataset))
        labels = batch[1][:10].cpu().numpy() if hasattr(batch[1], 'cpu') else batch[1][:10]

    generated_samples = cond_gan.generate(labels)
    assert generated_samples.shape == (10, seq_len, 1)


def test_temporal_cgan_seq_len_55():
    latent_dim = 2
    output_dim = 1
    feature_dim = 1
    seq_len = 55
    batch_size = 48

    dataset, labels = _gen_t_cond_dataset(seq_len, batch_size)
    architecture = tsgm.models.architectures.zoo["t-cgan_c4"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim, temporal=True,
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    cond_gan.fit(dataset, epochs=1)
    assert cond_gan.generator is not None
    assert cond_gan.discriminator is not None

    # Check generation
    # Handle PyTorch DataLoader vs TensorFlow dataset
    if hasattr(dataset, 'as_numpy_iterator'):
        # TensorFlow dataset
        labels = next(dataset.as_numpy_iterator())[1][:10]
    else:
        # PyTorch DataLoader
        batch = next(iter(dataset))
        labels = batch[1][:10].cpu().numpy() if hasattr(batch[1], 'cpu') else batch[1][:10]

    generated_samples = cond_gan.generate(labels)
    assert generated_samples.shape == (10, seq_len, 1)


@pytest.mark.skipif(not __tf_privacy_available, reason="TF privacy is required for this test")
def test_dp_compiler():
    latent_dim = 2
    output_dim = 1
    feature_dim = 1
    seq_len = 64
    batch_size = 48

    dataset, labels = _gen_t_cond_dataset(seq_len, batch_size)
    architecture = tsgm.models.architectures.zoo["t-cgan_c4"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim, temporal=True,
    )

    # DP optimizers
    l2_norm_clip = 1.5
    noise_multiplier = 1.3
    num_microbatches = 1
    learning_rate = 0.25

    d_optimizer = tf_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )

    g_optimizer = tf_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )
    cond_gan.compile(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    assert cond_gan.dp is True

    cond_gan.fit(dataset, epochs=1)
    assert cond_gan.generator is not None
    assert cond_gan.discriminator is not None

    # Check generation
    # Handle PyTorch DataLoader vs TensorFlow dataset
    if hasattr(dataset, 'as_numpy_iterator'):
        # TensorFlow dataset
        labels = next(dataset.as_numpy_iterator())[1][:10]
    else:
        # PyTorch DataLoader
        batch = next(iter(dataset))
        labels = batch[1][:10].cpu().numpy() if hasattr(batch[1], 'cpu') else batch[1][:10]

    generated_samples = cond_gan.generate(labels)
    assert generated_samples.shape == (10, 64, 1)    


def test_wavegan():
    latent_dim = 2
    output_dim = 1
    feature_dim = 1
    seq_len = 64
    batch_size = 48

    dataset = _gen_dataset(seq_len, feature_dim, batch_size)
    architecture = tsgm.models.architectures.zoo["wavegan"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator
    gan = tsgm.models.cgan.GAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim, use_wgan=True
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    
    gan.fit(dataset, epochs=1)

    assert gan.generator is not None
    assert gan.discriminator is not None
    # Check generation
    generated_samples = gan.generate(10)
    assert generated_samples.shape == (10, seq_len, 1)


class TrainingSpy(keras.layers.Layer):
    """Identity layer that records the `training` argument of every call."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def call(self, inputs, training=None):
        self.calls.append(training)
        return inputs


def _spied_gan_models(seq_len: int, feature_dim: int, generator_input_dim: int, discriminator_feature_dim: int):
    generator_spy, discriminator_spy = TrainingSpy(), TrainingSpy()

    g_input = keras.Input((generator_input_dim,))
    x = keras.layers.Dense(seq_len * feature_dim, activation="tanh")(g_input)
    x = keras.layers.Reshape((seq_len, feature_dim))(generator_spy(x))
    generator = keras.Model(g_input, x)

    d_input = keras.Input((seq_len, discriminator_feature_dim))
    x = keras.layers.Flatten()(discriminator_spy(d_input))
    discriminator = keras.Model(d_input, keras.layers.Dense(1)(x))

    # forget the symbolic calls made while the models were built
    generator_spy.calls.clear()
    discriminator_spy.calls.clear()
    return generator, discriminator, generator_spy, discriminator_spy


@pytest.mark.parametrize("use_wgan", [False, True])
def test_gan_trains_submodels_in_training_mode(use_wgan):
    # Regression test: layers such as Dropout and BatchNormalization are active only when called with training=True
    latent_dim = 4
    feature_dim = 1
    seq_len = 32
    batch_size = 30

    dataset = _gen_dataset(seq_len, feature_dim, batch_size)
    generator, discriminator, generator_spy, discriminator_spy = _spied_gan_models(
        seq_len, feature_dim, generator_input_dim=latent_dim, discriminator_feature_dim=feature_dim)

    gan = tsgm.models.cgan.GAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim, use_wgan=use_wgan
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    gan.fit(dataset, epochs=1, verbose=0)

    assert generator_spy.calls and set(generator_spy.calls) == {True}
    assert discriminator_spy.calls and set(discriminator_spy.calls) == {True}

    # generation runs in the inference mode
    generator_spy.calls.clear()
    gan.generate(3)
    assert generator_spy.calls and True not in generator_spy.calls


def test_cgan_trains_submodels_in_training_mode():
    latent_dim = 4
    output_dim = 2
    feature_dim = 1
    seq_len = 32
    batch_size = 30

    dataset, labels = _gen_cond_dataset(seq_len, batch_size)
    generator, discriminator, generator_spy, discriminator_spy = _spied_gan_models(
        seq_len, feature_dim,
        generator_input_dim=latent_dim + output_dim, discriminator_feature_dim=feature_dim + output_dim)

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    cond_gan.fit(dataset, epochs=1, verbose=0)

    assert generator_spy.calls and set(generator_spy.calls) == {True}
    assert discriminator_spy.calls and set(discriminator_spy.calls) == {True}

    # generation runs in the inference mode
    generator_spy.calls.clear()
    cond_gan.generate(labels[:3])
    assert generator_spy.calls and True not in generator_spy.calls


def test_gan_fit_updates_batchnorm_statistics():
    # BatchNormalization updates its moving statistics only when it is called in the training mode
    latent_dim = 4
    feature_dim = 1
    seq_len = 32
    batch_size = 30

    dataset = _gen_dataset(seq_len, feature_dim, batch_size)

    generator_bn, discriminator_bn = keras.layers.BatchNormalization(), keras.layers.BatchNormalization()
    g_input = keras.Input((latent_dim,))
    x = generator_bn(keras.layers.Dense(seq_len * feature_dim)(g_input))
    generator = keras.Model(g_input, keras.layers.Reshape((seq_len, feature_dim))(x))

    d_input = keras.Input((seq_len, feature_dim))
    x = discriminator_bn(keras.layers.Flatten()(d_input))
    discriminator = keras.Model(d_input, keras.layers.Dense(1)(x))

    gan = tsgm.models.cgan.GAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    gan.fit(dataset, epochs=1, verbose=0)

    for batchnorm in (generator_bn, discriminator_bn):
        moving_mean = keras.ops.convert_to_numpy(batchnorm.moving_mean)
        assert not np.allclose(moving_mean, 0.0)


class BatchSpy(keras.layers.Layer):
    """Identity layer that records the values of every batch it is called with. Works only in the eager mode."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batches = []

    def compute_output_shape(self, input_shape):
        # lets Keras build the functional model without tracing `call` with a symbolic tensor
        return input_shape

    def call(self, inputs):
        self.batches.append(keras.ops.convert_to_numpy(inputs))
        return inputs


@pytest.mark.parametrize("conditional", [False, True])
def test_gan_discriminator_sees_real_and_generated_batches_separately(conditional):
    # BatchNormalization computes its statistics per batch. If the discriminator step used one batch that mixes real
    # and generated samples, they would be normalized differently from the generated-only batch of the generator step.
    latent_dim = 4
    output_dim = 2 if conditional else 0
    feature_dim = 1
    seq_len = 32
    batch_size = 10
    real_value = 5.0  # outside of the range of the generator, which ends with tanh

    X = np.full((30, seq_len, feature_dim), real_value, dtype=np.float32)
    y = keras.utils.to_categorical(np.arange(len(X)) % 2, 2).astype(np.float32)
    arrays = (X, y) if conditional else (X,)
    backend = get_backend()
    if os.environ.get("KERAS_BACKEND") == "tensorflow":
        dataset = backend.data.Dataset.from_tensor_slices(arrays if conditional else X).batch(batch_size)
    elif os.environ.get("KERAS_BACKEND") == "torch":
        dataset = backend.utils.data.TensorDataset(*[backend.from_numpy(a) for a in arrays])
        dataset = backend.utils.data.DataLoader(dataset, batch_size=batch_size)

    g_input = keras.Input((latent_dim + output_dim,))
    x = keras.layers.Dense(seq_len * feature_dim, activation="tanh")(g_input)
    generator = keras.Model(g_input, keras.layers.Reshape((seq_len, feature_dim))(x))

    discriminator_spy = BatchSpy()
    d_input = keras.Input((seq_len, feature_dim + output_dim))
    x = keras.layers.Flatten()(discriminator_spy(d_input))
    discriminator = keras.Model(d_input, keras.layers.Dense(1)(x))

    if conditional:
        gan = tsgm.models.cgan.ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    else:
        gan = tsgm.models.cgan.GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    gan.run_eagerly = True  # the spy needs concrete values
    gan.fit(dataset, epochs=1, verbose=0)

    is_real = [np.isclose(batch[..., :feature_dim], real_value) for batch in discriminator_spy.batches]
    assert any(batch.all() for batch in is_real)
    assert any(not batch.any() for batch in is_real)
    assert all(batch.all() or not batch.any() for batch in is_real)


def test_cgan_discriminator_sees_constant_labels_across_time():
    latent_dim = 4
    output_dim = 3
    feature_dim = 1
    seq_len = 8
    batch_size = 4

    X = np.zeros((batch_size, seq_len, feature_dim), dtype=np.float32)
    y = np.eye(output_dim, dtype=np.float32)[np.arange(batch_size) % output_dim]
    backend = get_backend()
    if os.environ.get("KERAS_BACKEND") == "tensorflow":
        dataset = backend.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
    elif os.environ.get("KERAS_BACKEND") == "torch":
        dataset = backend.utils.data.TensorDataset(backend.from_numpy(X), backend.from_numpy(y))
        dataset = backend.utils.data.DataLoader(dataset, batch_size=batch_size)

    g_input = keras.Input((latent_dim + output_dim,))
    x = keras.layers.Dense(seq_len * feature_dim, activation="tanh")(g_input)
    generator = keras.Model(g_input, keras.layers.Reshape((seq_len, feature_dim))(x))

    discriminator_spy = BatchSpy()
    d_input = keras.Input((seq_len, feature_dim + output_dim))
    x = keras.layers.Flatten()(discriminator_spy(d_input))
    discriminator = keras.Model(d_input, keras.layers.Dense(1)(x))

    cond_gan = tsgm.models.cgan.ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    cond_gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    cond_gan.run_eagerly = True
    cond_gan.fit(dataset, epochs=1, verbose=0)

    expected_labels = np.repeat(y[:, None, :], seq_len, axis=1)
    assert discriminator_spy.batches
    for batch in discriminator_spy.batches:
        np.testing.assert_array_equal(batch[..., feature_dim:], expected_labels)


@pytest.mark.skipif(os.environ.get("KERAS_BACKEND") != "tensorflow",
                    reason="Keras computes the crossentropy of a sigmoid output from its logits only with the TensorFlow backend")
@pytest.mark.parametrize("conditional", [False, True])
def test_gan_saturated_discriminator_receives_gradients(conditional):
    # A discriminator that is confidently wrong has to be able to recover. Keras computes the crossentropy of a
    # sigmoid output from its logits, but only if the predictions reach the loss unchanged: after, e.g., a concatenation
    # the probabilities are clipped instead, and a saturated discriminator stops receiving gradients.
    latent_dim = 4
    output_dim = 2 if conditional else 0
    feature_dim = 1
    seq_len = 32
    batch_size = 30
    saturated_logit = -30.0  # sigmoid(-30) ~ 1e-13: "certainly real", for generated samples as well

    g_input = keras.Input((latent_dim + output_dim,))
    x = keras.layers.Dense(seq_len * feature_dim, activation="tanh")(g_input)
    generator = keras.Model(g_input, keras.layers.Reshape((seq_len, feature_dim))(x))

    d_output_layer = keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=keras.initializers.Constant(saturated_logit))
    d_input = keras.Input((seq_len, feature_dim + output_dim))
    discriminator = keras.Model(d_input, d_output_layer(keras.layers.Flatten()(d_input)))

    if conditional:
        dataset, _ = _gen_cond_dataset(seq_len, batch_size)
        gan = tsgm.models.cgan.ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    else:
        dataset = _gen_dataset(seq_len, feature_dim, batch_size)
        gan = tsgm.models.cgan.GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.01),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    gan.fit(dataset, epochs=1, verbose=0)

    bias = float(keras.ops.convert_to_numpy(d_output_layer.bias)[0])
    assert abs(bias - saturated_logit) > 1e-3
