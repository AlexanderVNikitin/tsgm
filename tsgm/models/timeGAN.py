import tensorflow as tf
from tensorflow import keras
from tensorflow.python.types.core import TensorLike
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange
import typing

import logging

from tsgm.models.architectures.zoo import BasicRecurrentArchitecture

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


class TimeGAN:
    """
    Time-series Generative Adversarial Networks (TimeGAN)

    Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
    "Time-series Generative Adversarial Networks,"
    Neural Information Processing Systems (NeurIPS), 2019.

    Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
    """

    def __init__(
        self,
        seq_len: int = 24,
        module: str = "gru",
        hidden_dim: int = 24,
        n_features: int = 6,
        n_layers: int = 3,
        batch_size: int = 256,
        gamma: float = 1.0,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dim = n_features

        assert module in ["gru", "lstm", "lstmLN"]
        self.module = module

        self.n_layers = n_layers

        self.batch_size = batch_size

        self.gamma = gamma

        # ----------------------------
        # Basic Architectures
        # ----------------------------
        self.embedder = BasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Embedder",
        ).build()

        self.recovery = BasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Recovery",
        ).build()

        self.supervisor = BasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Supervisor",
        ).build()

        self.discriminator = BasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=1,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Discriminator",
        ).build()

        self.generator_aux = BasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Generator",
        ).build()

        # ----------------------------
        # Optimizers: call .compile() to set them
        # ----------------------------
        DEFAULT_ADAM = keras.optimizers.Adam()
        self.autoencoder_opt = deepcopy(DEFAULT_ADAM)
        self.adversarialsup_opt = deepcopy(DEFAULT_ADAM)
        self.generator_opt = deepcopy(DEFAULT_ADAM)
        self.embedder_opt = deepcopy(DEFAULT_ADAM)
        self.discriminator_opt = deepcopy(DEFAULT_ADAM)
        # ----------------------------
        # Loss functions: call .compile() to set them
        # ----------------------------
        DEFAULT_MSE = keras.losses.MeanSquaredError()
        DEFAULT_BCE = keras.losses.BinaryCrossentropy()
        self._mse = DEFAULT_MSE
        self._bce = DEFAULT_BCE

        # --------------------------
        # All losses: will be populated in .fit()
        # --------------------------
        self.training_losses = []
        self.losses_labels = []

    def compile(
        self,
        d_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        g_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        emb_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        supgan_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        ae_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        emb_loss: keras.losses.Loss = keras.losses.MeanSquaredError(),
        clf_loss: keras.losses.Loss = keras.losses.BinaryCrossentropy(),
    ):
        """
        Assign optimizers and loss functions.

        :param d_optimizer: An optimizer for the GAN's discriminator
        :param g_optimizer: An optimizer for the GAN's generator
        :param emb_optimizer: An optimizer for the GAN's embedder
        :param supgan_optimizer: An optimizer for the adversarial supervised network
        :param ae_optimizer: An optimizer for the autoencoder network
        :param emb_loss: A loss function for the embedding recovery
        :param clf_loss: A loss function for the discriminator task
        :return: None
        """
        # ----------------------------
        # Optimizers
        # ----------------------------
        self.autoencoder_opt = ae_optimizer
        self.adversarialsup_opt = supgan_optimizer
        self.generator_opt = g_optimizer
        self.embedder_opt = emb_optimizer
        self.discriminator_opt = d_optimizer
        # ----------------------------
        # Loss functions
        # ----------------------------
        self._mse = emb_loss
        self._bce = clf_loss

    def _define_timegan(self):
        # --------------------------------
        # Data and Noise Inputs
        # --------------------------------
        X = keras.layers.Input(
            shape=[self.seq_len, self.dim], batch_size=self.batch_size, name="RealData"
        )

        Z = keras.layers.Input(
            shape=[self.seq_len, self.dim],
            batch_size=self.batch_size,
            name="RandomNoise",
        )

        # --------------------------------
        # Autoencoder: Embedder + Recovery
        # --------------------------------
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = keras.models.Model(
            inputs=X, outputs=X_tilde, name="Autoencoder"
        )
        self.autoencoder.summary()

        # ---------------------------------
        # Adversarial Supervised
        # ---------------------------------
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)

        self.adversarial_supervised = keras.models.Model(
            inputs=Z, outputs=Y_fake, name="AdversarialSupervised"
        )
        self.adversarial_supervised.summary()

        # ---------------------------------
        # Adversarial embedded in latent space
        # ---------------------------------
        Y_fake_e = self.discriminator(E_Hat)

        self.adversarial_embedded = keras.models.Model(
            inputs=Z, outputs=Y_fake_e, name="AdversarialEmbedded"
        )
        self.adversarial_embedded.summary()

        # ---------------------------------
        # Synthetic data generator
        # ---------------------------------
        X_hat = self.recovery(H_hat)
        self.generator = keras.models.Model(
            inputs=Z, outputs=X_hat, name="FinalGenerator"
        )
        self.generator.summary()

        # --------------------------------
        # Discriminator
        # --------------------------------
        Y_real = self.discriminator(H)
        self.discriminator_model = keras.models.Model(
            inputs=X, outputs=Y_real, name="FinalDiscriminator"
        )
        self.discriminator_model.summary()

    @tf.function
    def _train_autoencoder(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        """
        1. Embedding network training: minimize E_loss0
        """
        with tf.GradientTape() as tape:
            X_tilde = self.autoencoder(X)
            E_loss_T0 = self._mse(X, X_tilde)
            E_loss0 = 10.0 * tf.sqrt(E_loss_T0)

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars

        gradients = tape.gradient(E_loss0, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss0

    @tf.function
    def _train_supervisor(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        """
        2. Training with supervised loss only: minimize G_loss_S
        """
        with tf.GradientTape() as tape:
            H = self.embedder(X)
            H_hat_supervised = self.supervisor(H)
            G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])

        g_vars = self.generator.trainable_variables
        s_vars = self.supervisor.trainable_variables
        all_trainable = g_vars + s_vars
        gradients = tape.gradient(G_loss_S, all_trainable)
        apply_grads = [
            (grad, var)
            for (grad, var) in zip(gradients, all_trainable)
            if grad is not None
        ]
        optimizer.apply_gradients(apply_grads)
        return G_loss_S

    @tf.function
    def _train_generator(
        self, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> typing.Tuple[float, float, float, float, float]:
        """
        3. Joint training (Generator training twice more than discriminator training): minimize G_loss
        """
        with tf.GradientTape() as tape:
            # 1. Adversarial loss
            Y_fake = self.adversarial_supervised(Z)
            G_loss_U = self._bce(y_true=tf.ones_like(Y_fake), y_pred=Y_fake)

            Y_fake_e = self.adversarial_embedded(Z)
            G_loss_U_e = self._bce(y_true=tf.ones_like(Y_fake_e), y_pred=Y_fake_e)
            # 2. Supervised loss
            H = self.embedder(X)
            H_hat_supervised = self.supervisor(H)
            G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])

            # 3. Two Moments
            X_hat = self.generator(Z)
            G_loss_V = self._compute_generator_moments_loss(X, X_hat)

            # 4. Summation
            G_loss = (
                G_loss_U
                + self.gamma * G_loss_U_e
                + 100 * tf.sqrt(G_loss_S)
                + 100 * G_loss_V
            )

        g_vars = self.generator_aux.trainable_variables
        s_vars = self.supervisor.trainable_variables
        all_trainable = g_vars + s_vars
        gradients = tape.gradient(G_loss, all_trainable)
        apply_grads = [
            (grad, var)
            for (grad, var) in zip(gradients, all_trainable)
            if grad is not None
        ]
        optimizer.apply_gradients(apply_grads)
        return G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, G_loss

    @tf.function
    def _train_embedder(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> typing.Tuple[float, float]:
        """
        Train embedder during joint training: minimize E_loss
        """
        with tf.GradientTape() as tape:
            # Supervised Loss
            H = self.embedder(X)
            H_hat_supervised = self.supervisor(H)
            G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            X_tilde = self.autoencoder(X)
            E_loss_T0 = self._mse(X, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

            E_loss = E_loss0 + 0.1 * G_loss_S

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars
        gradients = tape.gradient(E_loss, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss, E_loss_T0

    @tf.function
    def _train_discriminator(
        self, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        """
        minimize D_loss
        """
        with tf.GradientTape() as tape:
            D_loss = self._check_discriminator_loss(X, Z)

        d_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(D_loss, d_vars)
        optimizer.apply_gradients(zip(gradients, d_vars))
        return D_loss

    @staticmethod
    def _compute_generator_moments_loss(
        y_true: TensorLike, y_pred: TensorLike
    ) -> float:
        """
        :param y_true: TensorLike
        :param y_pred: TensorLike
        :return G_loss_V: float
        """
        _eps = 1e-6
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        # G_loss_V2
        g_loss_mean = tf.reduce_mean(abs(y_true_mean - y_pred_mean))
        # G_loss_V1
        g_loss_var = tf.reduce_mean(
            abs(tf.sqrt(y_true_var + _eps) - tf.sqrt(y_pred_var + _eps))
        )
        # G_loss_V = G_loss_V1 + G_loss_V2
        return g_loss_mean + g_loss_var

    def _check_discriminator_loss(self, X: TensorLike, Z: TensorLike) -> float:
        """
        :param X: TensorLike
        :param Z: TensorLike
        :return D_loss: float
        """
        # Loss on false negatives
        Y_real = self.discriminator_model(X)
        D_loss_real = self._bce(y_true=tf.ones_like(Y_real), y_pred=Y_real)

        # Loss on false positives
        Y_fake = self.adversarial_supervised(Z)
        D_loss_fake = self._bce(y_true=tf.zeros_like(Y_fake), y_pred=Y_fake)

        Y_fake_e = self.adversarial_embedded(Z)
        D_loss_fake_e = self._bce(y_true=tf.zeros_like(Y_fake_e), y_pred=Y_fake_e)

        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
        return D_loss

    def _generate_noise(self) -> TensorLike:
        """
        Random vector generation
        :return Z, generated random vector
        """
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.dim))

    def get_noise_batch(self) -> typing.Iterator:
        """
        Return an iterator of random noise vectors
        """
        return iter(
            tf.data.Dataset.from_generator(
                self._generate_noise, output_types=tf.float32
            )
            .batch(self.batch_size)
            .repeat()
        )

    def _get_data_batch(self, data, n_windows: int) -> typing.Iterator:
        """
        Return an iterator of shuffled input data
        """
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        return iter(
            tf.data.Dataset.from_tensor_slices(data)
            .shuffle(buffer_size=n_windows)
            .batch(self.batch_size)
            .repeat()
        )

    def fit(self, data: TensorLike, epochs: int, checkpoints_interval: int = None):
        """
        :param data: TensorLike, the training data
        :param epochs: int, the number of epochs for the training loops
        :param checkpoints_interval: int, the interval for printing out loss values
            (loss values will be print out every 'checkpoints_interval' epochs)
            Default: None
        """
        assert not (
            self.autoencoder_opt is None
            or self.adversarialsup_opt is None
            or self.generator_opt is None
            or self.embedder_opt is None
            or self.discriminator_opt is None
        ), "One of the optimizers is not defined. Please call .compile() to set them"
        assert not (
            self._mse is None or self._bce is None
        ), "One of the loss functions is not defined. Please call .compile() to set them"

        checkpoints_interval = (
            epochs if checkpoints_interval is None else checkpoints_interval
        )

        # Define the model
        self._define_timegan()

        # 1. Embedding network training
        print("Start Embedding Network Training")

        autoencoder_losses = []
        for epoch in tqdm(range(epochs), desc="Autoencoder - training"):
            X_ = next(self._get_data_batch(data, n_windows=len(data)))
            step_e_loss_0 = self._train_autoencoder(X_, self.autoencoder_opt)

            # Checkpoint
            if epoch % checkpoints_interval == 0:
                print(f"step: {epoch}/{epochs}, e_loss: {step_e_loss_0}")
            autoencoder_losses.append(float(step_e_loss_0))

        print("Finished Embedding Network Training")

        # 2. Training only with supervised loss
        print("Start Training with Supervised Loss Only")

        # Adversarial Supervised network training
        adversarial_s_losses = []
        for epoch in tqdm(range(epochs), desc="Adversarial Supervised - training"):
            X_ = next(self._get_data_batch(data, n_windows=len(data)))
            step_g_loss_s = self._train_supervisor(X_, self.adversarialsup_opt)

            # Checkpoint
            if epoch % checkpoints_interval == 0:
                print(
                    f"step: {epoch}/{epochs}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}"
                )
            adversarial_s_losses.append(float(np.sqrt(step_g_loss_s)))
        print("Finished Training with Supervised Loss Only")

        # 3. Joint Training
        print("Start Joint Training")

        # GAN with embedding network training
        g_loss_u = []
        g_loss_u_e = []
        g_loss_s = []
        g_loss_v = []
        g_loss = []
        e_loss_t0 = []
        d_loss = []
        step_g_loss_u = 0
        step_g_loss_u_e = 0
        step_g_loss_s = 0
        step_g_loss_v = 0
        step_g_loss = 0
        step_e_loss_t0 = 0
        step_d_loss = 0
        for epoch in tqdm(range(epochs), desc="GAN with embedding - training"):

            # Generator training (twice more than discriminator training)
            for kk in range(2):
                X_ = next(self._get_data_batch(data, n_windows=len(data)))
                Z_ = next(self.get_noise_batch())
                # --------------------------
                # Train the generator
                # --------------------------
                (
                    step_g_loss_u,
                    step_g_loss_u_e,
                    step_g_loss_s,
                    step_g_loss_v,
                    step_g_loss,
                ) = self._train_generator(X_, Z_, self.generator_opt)

                # --------------------------
                # Train the embedder
                # --------------------------
                _, step_e_loss_t0 = self._train_embedder(X_, self.embedder_opt)

            X_ = next(self._get_data_batch(data, n_windows=len(data)))
            Z_ = next(self.get_noise_batch())
            step_d_loss = self._check_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                print("Train Discriminator (discriminator does not work well yet)")
                step_d_loss = self._train_discriminator(X_, Z_, self.discriminator_opt)

            # Print multiple checkpoints
            if epoch % checkpoints_interval == 0:
                print(
                    f"""step: {epoch}/{epochs},
                    d_loss: {np.round(step_d_loss, 4)},
                    g_loss_u: {np.round(step_g_loss_u, 4)},
                    g_loss_u_e: {np.round(step_g_loss_u_e, 4)},
                    g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)},
                    g_loss_v: {np.round(step_g_loss_v, 4)},
                    g_loss_v: {np.round(step_g_loss, 4)},
                    e_loss_t0: {np.round(np.sqrt(step_e_loss_t0), 4)}"""
                )
            d_loss.append(float(step_d_loss))
            g_loss_u.append(float(step_g_loss_u))
            g_loss_u_e.append(float(step_g_loss_u_e))
            g_loss_s.append(float(np.sqrt(step_g_loss_s)))
            g_loss_v.append(float(step_g_loss_v))
            g_loss.append(float(step_g_loss))
            e_loss_t0.append(float(np.sqrt(step_e_loss_t0)))

        print("Finished Joint Training")

        # Record training losses
        self.training_losses = np.array(
            [
                np.array(autoencoder_losses),
                np.array(adversarial_s_losses),
                np.array(d_loss),
                np.array(g_loss_u),
                np.array(g_loss_u_e),
                np.array(g_loss_s),
                np.array(g_loss_v),
                np.array(g_loss),
                np.array(e_loss_t0),
            ]
        )

        self.losses_labels = [
            "autoencoder",
            "adversarial_supervised",
            "discriminator",
            "generator_u",
            "generator_u_e",
            "generator_v",
            "generator",
            "embedder",
        ]

    def generate(self, n_samples: int) -> TensorLike:
        """
        Generate synthetic time series
        """
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in trange(steps, desc="Synthetic data generation"):
            Z_ = next(self.get_noise_batch())
            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))[:n_samples]
