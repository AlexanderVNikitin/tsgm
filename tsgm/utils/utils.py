import random
import numpy as np
import tensorflow as tf


def reconstruction_loss_by_axis(original, reconstructed, axis=0):
    # axis=0 all (sum of squared diffs)
    # axis=1 features (MSE)
    # axis=2 times (MSE)
    if axis == 0:
        return tf.reduce_sum(tf.math.squared_difference(original, reconstructed))
    else:
        return tf.losses.mean_squared_error(tf.reduce_mean(original, axis=axis), tf.reduce_mean(reconstructed, axis=axis))


def generate_slices(X, slice_len=10):
    new_X = []
    for el in X:
        for i in range(0, len(el) - slice_len, slice_len):
            new_X.append(el[i : i + slice_len])
    return np.array(new_X)


def fix_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
