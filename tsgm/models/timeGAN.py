import tensorflow as tf
from tensorflow import keras
import numpy as np

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
        1. Embedding network training: minimize E_loss0
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
        """
        2. Training with supervised loss only: minimize G_loss_S
        """
        with tf.GradientTape() as tape:
            H = self.embedder(X)
            H_hat_supervised = self.supervisor(H)
            G_loss_S = keras.losses.MeanSquaredError(H[:, 1:, :], H_hat_supervised[:, :-1, :])

        g_vars = self.generator.trainable_variables
        s_vars = self.supervisor.trainable_variables
        all_trainable = g_vars + s_vars
        gradients = tape.gradient(G_loss_S, all_trainable)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, all_trainable) if grad is not None]
        optimizer.apply_gradients(apply_grads)
        return G_loss_S

    @tf.function
    def _train_generator(self, X, Z, optimizer: keras.optimizers.Optimizer):
        """
        3. Joint training (Generator training twice more than discriminator training): minimize G_loss
        """
        with tf.GradientTape() as tape:
            # 1. Adversarial loss
            Y_fake = self.adversarial_supervised(Z)
            G_loss_U = keras.losses.BinaryCrossEntropy(y_true=tf.ones_like(Y_fake),
                                                       y_pred=Y_fake)

            Y_fake_e = self.adversarial_embedded(Z)
            G_loss_U_e = keras.losses.BinaryCrossEntropy(y_true=tf.ones_like(Y_fake_e),
                                                         y_pred=Y_fake_e)
            # 2. Supervised loss
            H = self.embedder(X)
            H_hat_supervised = self.supervisor(H)
            G_loss_S = keras.losses.MeanSquaredError(H[:, 1:, :], H_hat_supervised[:, :-1, :])

            # 3. Two Moments
            X_hat = self.generator(Z)
            G_loss_V = self.compute_generator_moments_loss(X, X_hat)

            # 4. Summation
            G_loss = (G_loss_U + self.gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V)

        g_vars = self.generator.trainable_variables
        s_vars = self.supervisor.trainable_variables
        all_trainable = g_vars + s_vars
        gradients = tape.gradient(G_loss, all_trainable)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, all_trainable) if grad is not None]
        optimizer.apply_gradients(apply_grads)
        return G_loss, G_loss_U, G_loss_S, G_loss_V

    @tf.function
    def _train_embedder(self, X, optimizer: keras.optimizers.Optimizer):
        """
        Train embedder during joint training: minimize E_loss
        """
        with tf.GradientTape() as tape:
            # Supervised Loss
            H = self.embedder(X)
            H_hat_supervised = self.supervisor(H)
            G_loss_S = keras.losses.MeanSquaredError(H[:, 1:, :], H_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            X_tilde = self.autoencoder(X)
            E_loss_T0 = keras.losses.MeanSquaredError(X, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

            E_loss = E_loss0 + 0.1 * G_loss_S

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars
        gradients = tape.gradient(E_loss, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss, E_loss_T0

    @tf.function
    def _train_discriminator(self, X, Z, optimizer: keras.optimizers.Optimizer):
        """
        minimize D_loss
        """
        with tf.GradientTape() as tape:
            D_loss = self.check_discriminator_loss(X, Z)

        d_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(D_loss, d_vars)
        optimizer.apply_gradients(zip(gradients, d_vars))
        return D_loss

    @staticmethod
    def compute_generator_moments_loss(y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        :return G_loss_V:
        """
        _eps = 1e-6
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        # G_loss_V2
        g_loss_mean = tf.reduce_mean(abs(y_true_mean - y_pred_mean))
        # G_loss_V1
        g_loss_var = tf.reduce_mean(abs(tf.sqrt(y_true_var + _eps) - tf.sqrt(y_pred_var + _eps)))
        # G_loss_V = G_loss_V1 + G_loss_V2
        return g_loss_mean + g_loss_var

    def check_discriminator_loss(self, X, Z):
        """
        :param X:
        :param Z:
        :return D_loss:
        """
        # Loss on false negatives
        Y_real = self.discriminator(X)
        D_loss_real = keras.losses.BinaryCrossEntropy(y_true=tf.ones_like(Y_real),
                                                      y_pred=Y_real)

        # Loss on false positives
        Y_fake = self.adversarial_supervised(Z)
        D_loss_fake = keras.losses.BinaryCrossEntropy(y_true=tf.zeros_like(Y_fake),
                                                      y_pred=Y_fake)

        Y_fake_e = self.adversarial_embedded(Z)
        D_loss_fake_e = keras.losses.BinaryCrossEntropy(y_true=tf.zeros_like(Y_fake_e),
                                                        y_pred=Y_fake_e)

        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
        return D_loss

    def _generate_noise(self):
        """
        Random vector generation
        :return Z, generated random vector
        """
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.dim))

    def get_noise_batch(self):
        """
        Return an iterator of random noise vectors
        """
        return iter(tf.data.Dataset.from_generator(self._generate_noise, output_types=tf.float32)
                    .batch(self.mini_batch_size)
                    .repeat())
