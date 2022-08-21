import tensorflow as tf
from tensorflow import keras

import logging

import tsgm


logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


class TimeGAN(keras.Model):
    """
    Time-series Generative Adversarial Networks (TimeGAN)

    Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
    "Time-series Generative Adversarial Networks,"
    Neural Information Processing Systems (NeurIPS), 2019.

    Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
    """
    def __init__(self):
        pass

    @tf.function
    def _train_autoencoder(self, X, optimizer: keras.optimizers.Optimizer):
        """
        minimize E_loss0
        """
        with tf.GradientTape() as tape:
            X_tilde = self.autoencoder(X)
            E_loss_T0 = keras.losses.MeanSquaredError(X, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars

        gradients = tape.gradient(E_loss0, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss0  # or tf.sqrt(E_loss_T0), it should be the same

    @tf.function
    def _train_supervisor(self, X, optimizer: keras.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            pass

    @tf.function
    def _train_embedder(self, X, optimizer: keras.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            pass

    @tf.function
    def _train_generator(self, X, Z, optimizer: keras.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            pass

    @tf.function
    def _train_discrimiator(self, X, Z, optimizer: keras.optimizers.Optimizer):
        with tf.GradientTape() as tape:
            pass