import pytest
from unittest.mock import Mock
import tsgm

import tensorflow as tf
import numpy as np
from tensorflow import keras


def test_timegan():
    latent_dim = 4
    feature_dim = 3
    seq_len = 24
    batch_size = 2

    dataset = _gen_dataset(batch_size, seq_len, feature_dim)
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=2,
        batch_size=batch_size,
    )
    timegan.compile()

    try:
        tf.config.experimental_run_functions_eagerly(True)
        timegan.fit(dataset, epochs=1)

        _check_internals(timegan)

        generated_samples = timegan.generate(1)
    finally:
        tf.config.experimental_run_functions_eagerly(False)
    assert generated_samples.shape == (1, seq_len, feature_dim)


def test_timegan_fit():
    latent_dim = 4
    feature_dim = 3
    seq_len = 24
    batch_size = 2

    dataset = _gen_dataset(batch_size, seq_len, feature_dim)
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=2,
        batch_size=batch_size,
    )
    timegan.compile()
    try:
        tf.config.experimental_run_functions_eagerly(True)
        timegan.fit(dataset, epochs=3, checkpoints_interval=2, generate_synthetic=(1,))

        _check_internals(timegan)
    finally:
        tf.config.experimental_run_functions_eagerly(False)

    # Check intermediate generation
    assert timegan.synthetic_data_generated_in_training
    assert len(timegan.synthetic_data_generated_in_training[1].shape) == 3


def test_timegan_on_dataset():
    latent_dim = 4
    feature_dim = 3
    seq_len = 24
    batch_size = 16

    dataset = _gen_tf_dataset(batch_size, seq_len, feature_dim)  # tf.data.Dataset
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=2,
        batch_size=batch_size,
    )
    timegan.compile()
    try:
        tf.config.experimental_run_functions_eagerly(True)
        timegan.fit(dataset, epochs=1)

        _check_internals(timegan)

        generated_samples = timegan.generate(1)
    finally:
        tf.config.experimental_run_functions_eagerly(False)
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
    latent_dim = 4
    feature_dim = 3
    seq_len = 24
    batch_size = 2

    dataset = _gen_dataset(batch_size, seq_len, feature_dim)
    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=2,
        batch_size=batch_size,
    )
    timegan.compile()
    try:
        tf.config.experimental_run_functions_eagerly(True)
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
    finally:
        tf.config.experimental_run_functions_eagerly(False)


@pytest.fixture
def mock_optimizer():
    yield tf.keras.optimizers.Adam(learning_rate=0.001)


@pytest.fixture
def mocked_data():
    feature_dim = 3
    seq_len = 24
    batch_size = 16
    yield _gen_tf_dataset(batch_size, seq_len, feature_dim)


@pytest.fixture
def mocked_timegan(mocked_data):
    latent_dim = 4
    feature_dim = 3
    seq_len = 24
    batch_size = 16

    timegan = tsgm.models.timeGAN.TimeGAN(
        seq_len=seq_len,
        module="gru",
        hidden_dim=latent_dim,
        n_features=feature_dim,
        n_layers=2,
        batch_size=batch_size,
    )
    timegan.compile()
    timegan.fit(mocked_data, epochs=1)
    yield timegan


def test_timegan_train_autoencoder(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    try:
        tf.config.experimental_run_functions_eagerly(True)
        loss = mocked_timegan._train_autoencoder(X_, mocked_timegan.autoencoder_opt)
    finally:
        tf.config.experimental_run_functions_eagerly(False)

    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_supervisor(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    try:
        tf.config.experimental_run_functions_eagerly(True)
        _, loss = mocked_timegan._train_embedder(X_, mocked_timegan.embedder_opt)
    finally:
        tf.config.experimental_run_functions_eagerly(False)
    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_embedder(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    try:
        tf.config.experimental_run_functions_eagerly(True)
        _, loss = mocked_timegan._train_embedder(X_, mocked_timegan.embedder_opt)
    finally:
        tf.config.experimental_run_functions_eagerly(False)
    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_generator(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())

    mocked_timegan._define_timegan()
    X_ = next(batches)
    Z_ = next(mocked_timegan.get_noise_batch())
    try:
        tf.config.experimental_run_functions_eagerly(True)
        (
            step_g_loss_u,
            step_g_loss_u_e,
            step_g_loss_s,
            step_g_loss_v,
            step_g_loss,
        ) = mocked_timegan._train_generator(X_, Z_, mocked_timegan.generator_opt)
    finally:
        tf.config.experimental_run_functions_eagerly(False)

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
    try:
        tf.config.experimental_run_functions_eagerly(True)
        loss = mocked_timegan._check_discriminator_loss(X_, Z_)
    finally:
        tf.config.experimental_run_functions_eagerly(False)

    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_timegan_train_discriminator(mocked_data, mocked_timegan):
    batches = iter(mocked_data.repeat())
    mocked_timegan._define_timegan()
    X_ = next(batches)
    Z_ = next(mocked_timegan.get_noise_batch())
    try:
        tf.config.experimental_run_functions_eagerly(True)
        loss = mocked_timegan._train_discriminator(X_, Z_, mocked_timegan.discriminator_opt)
    finally:
        tf.config.experimental_run_functions_eagerly(False)
    # Assert that the loss is a float
    assert loss.dtype in [tf.float32, tf.float64]


def test_generate_noise(mocked_timegan):
    # Set finite values for sequence length and dimension for testing
    mocked_timegan.seq_len = 10
    mocked_timegan.dim = 5

    # Generate noise using the method
    generator = mocked_timegan._generate_noise()

    # Generate a finite number of noise samples
    num_samples = 3  # Define the number of samples to generate
    for _ in range(num_samples):
        generated_noise = next(generator)
        assert isinstance(generated_noise, np.ndarray)
        assert generated_noise.shape == (mocked_timegan.seq_len, mocked_timegan.dim)
        assert np.all(generated_noise >= 0) and np.all(generated_noise <= 1)


def test_compute_generator_moments_loss(mocked_timegan):
    # Generate some test data
    y_true_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred_data = np.array([[1.2, 2.3, 3.1], [4.2, 4.8, 6.2]])

    # Calculate the expected loss manually
    _eps = 1e-6
    y_true_mean, y_true_var = tf.nn.moments(x=y_true_data, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred_data, axes=[0])

    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(
        tf.abs(tf.sqrt(y_true_var + _eps) - tf.sqrt(y_pred_var + _eps))
    )
    expected_loss = g_loss_mean + g_loss_var

    try:
        tf.config.experimental_run_functions_eagerly(True)
        # Calculate the loss using the method
        computed_loss = mocked_timegan._compute_generator_moments_loss(
            y_true_data, y_pred_data
        )
    finally:
        tf.config.experimental_run_functions_eagerly(False)

    # Assert that the computed loss matches the expected loss
    np.testing.assert_almost_equal(computed_loss, expected_loss, decimal=5)
