import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm, trange

import logging

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
        epochs: int = 10,
        checkpoint: int = 2,
        batch_size: int = 256,
        gamma: float = 1.0,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dim = n_features

        assert module in ["gru", "lstm", "lstmLN"]
        self.module = module

        self.n_layers = n_layers

        self.epochs = epochs
        self.checkpoint = checkpoint
        self.batch_size = batch_size

        self.gamma = gamma

        # ----------------------------
        # Basic Architectures
        # ----------------------------
        self.embedder = TimeGanBasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Embedder",
        ).build()

        self.recovery = TimeGanBasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Recovery",
        ).build()

        self.supervisor = TimeGanBasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Supervisor",
        ).build()

        self.discriminator = TimeGanBasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=1,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Discriminator",
        ).build()

        self.generator_aux = TimeGanBasicRecurrentArchitecture(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            n_layers=self.n_layers,
            network_type=self.module,
            name="Generator",
        ).build()

        # ----------------------------
        # Basic Losses
        # ----------------------------
        self._mse = keras.losses.MeanSquaredError()
        self._bce = keras.losses.BinaryCrossentropy()

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
    def _train_autoencoder(self, X, optimizer: keras.optimizers.Optimizer):
        """
        1. Embedding network training: minimize E_loss0
        """
        with tf.GradientTape() as tape:
            X_tilde = self.autoencoder(X)
            E_loss_T0 = self._mse(X, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars

        gradients = tape.gradient(E_loss0, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss0

    @tf.function
    def _train_supervisor(self, X, optimizer: keras.optimizers.Optimizer):
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
    def _train_generator(self, X, Z, optimizer: keras.optimizers.Optimizer):
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
        return G_loss_U, G_loss_S, G_loss_V

    @tf.function
    def _train_embedder(self, X, optimizer: keras.optimizers.Optimizer):
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
    def _train_discriminator(self, X, Z, optimizer: keras.optimizers.Optimizer):
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
    def _compute_generator_moments_loss(y_true, y_pred):
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
        g_loss_var = tf.reduce_mean(
            abs(tf.sqrt(y_true_var + _eps) - tf.sqrt(y_pred_var + _eps))
        )
        # G_loss_V = G_loss_V1 + G_loss_V2
        return g_loss_mean + g_loss_var

    def _check_discriminator_loss(self, X, Z):
        """
        :param X:
        :param Z:
        :return D_loss:
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
        return iter(
            tf.data.Dataset.from_generator(
                self._generate_noise, output_types=tf.float32
            )
            .batch(self.batch_size)
            .repeat()
        )

    def _get_data_batch(self, data, n_windows: int):
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

    def fit(self, data):
        # Define the model
        self._define_timegan()

        # 1. Embedding network training
        print("Start Embedding Network Training")

        autoencoder_opt = keras.optimizers.Adam()
        for epoch in tqdm(range(self.epochs), desc="Autoencoder - training"):
            X_ = next(self._get_data_batch(data, n_windows=len(data)))
            step_e_loss_0 = self._train_autoencoder(X_, autoencoder_opt)
            # Checkpoint
            if epoch % self.checkpoint == 0:
                print(f"step: {epoch}/{self.epochs}, e_loss: {step_e_loss_0}")

        print("Finish Embedding Network Training")

        # 2. Training only with supervised loss
        print("Start Training with Supervised Loss Only")

        # Adversarial Supervised network training
        adversarialsup_opt = keras.optimizers.Adam()
        for epoch in tqdm(range(self.epochs), desc="Adversarial Supervised - training"):
            X_ = next(self._get_data_batch(data, n_windows=len(data)))
            step_g_loss_s = self._train_supervisor(X_, adversarialsup_opt)
            # Checkpoint
            if epoch % self.checkpoint == 0:
                print(
                    f"step: {epoch}/{self.epochs}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}"
                )

        print("Finish Training with Supervised Loss Only")

        # 3. Joint Training
        print("Start Joint Training")

        # GAN with embedding network training
        generator_opt = keras.optimizers.Adam()
        embedder_opt = keras.optimizers.Adam()
        discriminator_opt = keras.optimizers.Adam()

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for epoch in tqdm(range(self.epochs), desc="GAN with embedding - training"):

            # Generator training (twice more than discriminator training)
            for kk in range(2):
                X_ = next(self._get_data_batch(data, n_windows=len(data)))
                Z_ = next(self.get_noise_batch())
                # --------------------------
                # Train the generator
                # --------------------------
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self._train_generator(
                    X_, Z_, generator_opt
                )

                # --------------------------
                # Train the embedder
                # --------------------------
                step_e_loss_t0 = self._train_embedder(X_, embedder_opt)

            X_ = next(self._get_data_batch(data, n_windows=len(data)))
            Z_ = next(self.get_noise_batch())
            step_d_loss = self._check_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                print("Train discriminator (discriminator does not work well)")
                step_d_loss = self._train_discriminator(X_, Z_, discriminator_opt)

            # Print multiple checkpoints
            if epoch % self.checkpoint == 0:
                print(
                    f"""step: {epoch}/{self.epochs},
                    d_loss: {np.round(step_d_loss, 4)},
                    g_loss_u: {np.round(step_g_loss_u, 4)},
                    g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)},
                    g_loss_v: {np.round(step_g_loss_v, 4)},
                    e_loss_t0: {np.round(np.sqrt(step_e_loss_t0), 4)}"""
                )
            print("Finish Joint Training")

    def generate(self, n_samples: int):
        """
        Generate synthetic time series
        """
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in trange(steps, desc="Synthetic data generation"):
            Z_ = next(self.get_noise_batch())
            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))


class TimeGanBasicRecurrentArchitecture(keras.models.Model):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        network_type: str,
        name: str = "Sequential",
    ):
        """
        :param hidden_dim: int, the number of units (e.g. 24)
        :param output_dim: int, the number of output units (e.g. 1)
        :param n_layers: int, the number of layers (e.g. 3)
        :param network_type: str, one of 'gru', 'lstm', or 'lstmLN'
        :param name: str, model name
            Default: "Sequential"
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        assert network_type in ["gru", "lstm", "lstmLN"]
        self.network_type = network_type

        self._name = name

    def _rnn_cell(self) -> keras.layers.Layer:
        """
        Basic RNN Cell
        :return cell: keras.layers.Layer
        """
        cell = None
        # GRU
        if self.network_type == "gru":
            cell = keras.layers.GRUCell(self.hidden_dim, activation="tanh")
        # LSTM
        elif self.network_type == "lstm":
            cell = keras.layers.LSTMCell(self.hidden_dim, activation="tanh")
        # LSTM Layer Normalization
        elif self.network_type == "lstmLN":
            cell = keras.layers.LayerNormLSTMCell(
                num_units=self.hidden_dim, activation="tanh"
            )
        return cell

    def _make_network(self, model: keras.models.Model) -> keras.models.Model:
        _cells = tf.keras.layers.StackedRNNCells(
            [self._rnn_cell() for _ in range(self.n_layers)],
            name=f"{self.network_type}_x{self.n_layers}",
        )
        model.add(keras.layers.RNN(_cells, return_sequences=True))
        model.add(
            keras.layers.Dense(units=self.output_dim, activation="sigmoid", name="OUT")
        )
        return model

    def build(self):
        model = keras.models.Sequential(name=f"{self._name}")
        model = self._make_network(model)
        return model
