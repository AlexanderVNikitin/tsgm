import tensorflow as tf


def reconstruction_loss_by_axis(original, reconstructed, axis=0):
    # axis=0 all (sum of squared diffs)
    # axis=1 features (MSE)
    # axis=2 times (MSE)
    if axis == 0:
        return tf.reduce_sum(tf.math.squared_difference(original, reconstructed))
    else:
        return tf.losses.mean_squared_error(tf.reduce_mean(original, axis=axis), tf.reduce_mean(reconstructed, axis=axis))
