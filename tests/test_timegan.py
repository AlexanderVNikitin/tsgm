import pytest
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
        seq_len=seq_len, module="gru", hidden_dim=latent_dim, n_features=feature_dim, n_layers=3, batch_size=batch_size
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
        seq_len=seq_len, module="gru", hidden_dim=latent_dim, n_features=feature_dim, n_layers=3, batch_size=batch_size
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
