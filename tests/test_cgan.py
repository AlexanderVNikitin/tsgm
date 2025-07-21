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
        dataset = backend.utils.data.TensorDataset(X_train)
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
        dataset = backend.utils.data.TensorDataset(X_train, y)
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
        dataset = backend.utils.data.TensorDataset(X_train, y)
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
    generated_samples = cond_gan.generate(next(dataset.as_numpy_iterator())[1][:10])
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
    generated_samples = cond_gan.generate(next(dataset.as_numpy_iterator())[1][:10])

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
    generated_samples = cond_gan.generate(next(dataset.as_numpy_iterator())[1][:10])
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
    generated_samples = cond_gan.generate(next(dataset.as_numpy_iterator())[1][:10])
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
    generated_samples = cond_gan.generate(next(dataset.as_numpy_iterator())[1][:10])
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
