import pytest
from unittest.mock import Mock
import tsgm

import tensorflow as tf
import numpy as np
from tensorflow import keras


def test_timegan():
    latent_dim = 24
    feature_dim = 6
    seq_len = 24
    batch_size = 2

    dataset = _gen_dataset(batch_size, seq_len, feature_dim)
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=3,
        batch_size=batch_size,
    )
    timegan.compile()
    timegan.fit(dataset, epochs=1)

    _check_internals(timegan)

    # Check generation
    generated_samples = timegan.generate(1)
    assert generated_samples.shape == (1, seq_len, feature_dim)


def test_timegan_on_dataset():
    latent_dim = 24
    feature_dim = 6
    seq_len = 24
    batch_size = 16

    dataset = _gen_tf_dataset(batch_size, seq_len, feature_dim)  # tf.data.Dataset
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=3,
        batch_size=batch_size,
    )
    timegan.compile()
    timegan.fit(dataset, epochs=1)

    _check_internals(timegan)

    # Check generation
    generated_samples = timegan.generate(1)
    assert generated_samples.shape == (1, seq_len, feature_dim)


def _gen_dataset(no, seq_len, dim):
    """Sine data generation.
    Args:
        - no: the number of samples
        - seq_len: sequence length of the time-series
        - dim: feature dimensions
    Returns:
        - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def _gen_tf_dataset(no, seq_len, dim):
    dataset = _gen_dataset(no, seq_len, dim)
    dataset = tf.convert_to_tensor(dataset, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensors(dataset).unbatch().batch(no)

    return dataset


def _check_internals(timegan):
    # Check internal nets
    assert timegan.generator is not None
    assert timegan.discriminator is not None
    assert timegan.embedder is not None
    assert timegan.autoencoder is not None
    assert timegan.adversarial_embedded is not None
    assert timegan.adversarial_supervised is not None
    assert timegan.generator_aux is not None

    # Check loss
    assert timegan._mse is not None
    assert timegan._bce is not None

    # Check optimizers
    assert timegan.generator_opt is not None
    assert timegan.discriminator_opt is not None
    assert timegan.embedder_opt is not None
    assert timegan.autoencoder_opt is not None
    assert timegan.adversarialsup_opt is not None


def test_losstracker():
    losstracker = tsgm.models.timeGAN.LossTracker()
    losstracker["foo"] = 0.1
    assert isinstance(losstracker.to_numpy(), np.ndarray)
    assert isinstance(losstracker.labels(), list)


@pytest.fixture
def mocked_gradienttape(mocker):
    mock = Mock()
    mock.gradient.return_value = [1.0, 1.0, 1.0]
    return mock


def test_train_timegan(mocked_gradienttape):
    latent_dim = 24
    feature_dim = 6
    seq_len = 24
    batch_size = 2

    dataset = _gen_dataset(batch_size, seq_len, feature_dim)
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=3,
        batch_size=batch_size,
    )
    timegan.compile()
    timegan.fit(dataset, epochs=1)
    batches = timegan._get_data_batch(dataset, n_windows=len(dataset))
    assert timegan._train_autoencoder(next(batches), timegan.autoencoder_opt)
    assert timegan._train_supervisor(next(batches), timegan.adversarialsup_opt)
    assert timegan._train_generator(
        next(batches), next(timegan.get_noise_batch()), timegan.generator_opt
    )
    assert timegan._train_embedder(next(batches), timegan.embedder_opt)
    assert timegan._train_discriminator(
        next(batches), next(timegan.get_noise_batch()), timegan.discriminator_opt
    )


@pytest.fixture
def mock_optimizer():
    yield tf.keras.optimizers.Adam(learning_rate=0.001)


@pytest.fixture
def mocked_data():
    feature_dim = 6
    seq_len = 24
    batch_size = 16
    yield _gen_tf_dataset(batch_size, seq_len, feature_dim)


@pytest.fixture
def mocked_timegan(mocked_data):
    latent_dim = 24
    feature_dim = 6
    seq_len = 24
    batch_size = 16

    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=3,
        batch_size=batch_size,
    )
    timegan.compile()
    timegan.fit(mocked_data, epochs=1)
    yield timegan


def test_timegan_train_autoencoder(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    loss = mocked_timegan._train_autoencoder(X_, mocked_timegan.autoencoder_opt)

    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_embedder(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    _, loss = mocked_timegan._train_embedder(X_, mocked_timegan.embedder_opt)

    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_generator(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    Z_ = next(mocked_timegan.get_noise_batch())
    (
        step_g_loss_u,
        step_g_loss_u_e,
        step_g_loss_s,
        step_g_loss_v,
        step_g_loss,
    ) = mocked_timegan._train_generator(X_, Z_, mocked_timegan.generator_opt)

    # Assert that the loss is a float
    for loss in (
        step_g_loss_u,
        step_g_loss_u_e,
        step_g_loss_s,
        step_g_loss_v,
        step_g_loss,
    ):
        assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_check_discriminator_loss(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    Z_ = next(mocked_timegan.get_noise_batch())
    loss = mocked_timegan._check_discriminator_loss(X_, Z_)

    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_discriminator(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    Z_ = next(mocked_timegan.get_noise_batch())
    loss = mocked_timegan._train_discriminator(X_, Z_, mocked_timegan.discriminator_opt)

    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]
