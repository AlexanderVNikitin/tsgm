import os
import keras
from keras import ops
from tsgm.types import Tensor as TensorLike
#  a keras_dataset can be tf.data.Dataset or Torch DataLoader
from tsgm.backend import tf_function_decorator, Keras_Dataset, get_backend
import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange
from collections import OrderedDict
import typing as T

import logging

from tsgm.models.architectures.zoo import BasicRecurrentArchitecture

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


class LossTracker(OrderedDict):
    """
    Dictionary of lists, extends python OrderedDict.
    Example: Given {'loss_a': [1], 'loss_b': [2]}, adding key='loss_a' with value=0.7
            gives {'loss_a': [1, 0.7], 'loss_b': [2]}, and adding key='loss_c' with value=1.2
            gives {'loss_a': [1, 0.7], 'loss_b': [2], 'loss_c': [1.2]}
    """

    def __setitem__(self, key: T.Any, value: T.Any) -> None:
        try:
            # Assumes the key already exists
            # and the value is a list [oldest_value, another_old, ...]
            # key -> [oldest_value, another_old, ..., new_value]
            self[key].append(value)
        # If there is no key, add key -> [new_value]
        except KeyError:
            # key -> [new_value]
            super(LossTracker, self).__setitem__(key, [value])

    def to_numpy(self) -> npt.NDArray:
        """
        :return 2d vector of losses
        """
        _losses = np.array([np.array(v) for v in self.values() if isinstance(v, list)])
        return _losses

    def labels(self) -> T.List:
        """
        :return list of keys
        """
        return list(self.keys())


class TimeGAN(keras.Model):
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
    ) -> None:
        super().__init__()
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
        self.autoencoder_opt = keras.optimizers.Adam()
        self.adversarialsup_opt = keras.optimizers.Adam()
        self.generator_opt = keras.optimizers.Adam()
        self.embedder_opt = keras.optimizers.Adam()
        self.discriminator_opt = keras.optimizers.Adam()
        # ----------------------------
        # Loss functions: call .compile() to set them
        # ----------------------------
        self._mse = keras.losses.MeanSquaredError()
        self._bce = keras.losses.BinaryCrossentropy()

        # --------------------------
        # All losses: will be populated in .fit()
        # --------------------------
        self.training_losses_history = LossTracker()

        # --------------------------
        # Synthetic data generation during training: will be populated in .fit()
        # --------------------------
        self.synthetic_data_generated_in_training = dict()

    def compile(
        self,
        d_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        g_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        emb_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        supgan_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        ae_optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(),
        emb_loss: keras.losses.Loss = keras.losses.MeanSquaredError(),
        clf_loss: keras.losses.Loss = keras.losses.BinaryCrossentropy(),
    ) -> None:
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

    def _define_timegan(self) -> None:
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
    
    @tf_function_decorator  
    def _train_autoencoder_tf(
        self, tf, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        """
        1. Embedding network training: minimize E_loss0
        """
        with tf.GradientTape() as tape:
            X_tilde = self.autoencoder(X)
            E_loss_T0 = self._mse(X, X_tilde)
            E_loss0 = 10.0 * ops.sqrt(E_loss_T0)

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars

        gradients = tape.gradient(E_loss0, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss0

    def _train_autoencoder_torch(
        self, torch, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        X_tilde = self.autoencoder(X)
        E_loss_T0 = self._mse(X, X_tilde)
        E_loss0 = 10.0 * ops.sqrt(E_loss_T0)
        self.embedder.zero_grad()
        self.recovery.zero_grad()
        E_loss0.backward()

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars
        gradients = [v.value.grad for v in all_trainable] 

        with torch.no_grad():
            optimizer.apply(zip(gradients, all_trainable))
        return E_loss0

    
    def _train_autoencoder(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            return self._train_autoencoder_tf(backend, X, optimizer)
        elif os.environ["KERAS_BACKEND"] == "torch":
            return self._train_autoencoder_torch(backend, X, optimizer)

    @tf_function_decorator
    def _train_supervisor_tf(
        self, tf, X: TensorLike, optimizer: keras.optimizers.Optimizer
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
    
    def _train_supervisor_torch(
        self, torch, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        H = self.embedder(X)
        H_hat_supervised = self.supervisor(H)
        G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])
        self.generator.zero_grad()
        self.supervisor.zero_grad()
        G_loss_S.backward()

        g_vars = self.generator.trainable_variables
        s_vars = self.supervisor.trainable_variables
        all_trainable = g_vars + s_vars
        gradients = [v.value.grad for v in all_trainable]
        apply_grads = [
            (grad, var)
            for (grad, var) in zip(gradients, all_trainable)
            if grad is not None
        ]

        with torch.no_grad():
            optimizer.apply(apply_grads)
        return G_loss_S

    def _train_supervisor(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        """
        2. Training with supervised loss only: minimize G_loss_S
        """
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            return self._train_supervisor_tf(backend, X, optimizer)
        elif os.environ["KERAS_BACKEND"] == "torch":
            return self._train_supervisor_torch(backend, X, optimizer)

    @tf_function_decorator
    def _train_generator_tf(
        self, tf, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float, float, float, float]:
        """
        3. Joint training (Generator training twice more than discriminator training): minimize G_loss
        """
        with tf.GradientTape() as tape:
            # 1. Adversarial loss
            Y_fake = self.adversarial_supervised(Z)
            G_loss_U = self._bce(y_true=ops.ones_like(Y_fake), y_pred=Y_fake)

            Y_fake_e = self.adversarial_embedded(Z)
            G_loss_U_e = self._bce(y_true=ops.ones_like(Y_fake_e), y_pred=Y_fake_e)
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
                + 100 * ops.sqrt(G_loss_S)
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
    
    def _train_generator_torch(
        self, torch, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float, float, float, float]:
        Y_fake = self.adversarial_supervised(Z)
        G_loss_U = self._bce(y_true=ops.ones_like(Y_fake), y_pred=Y_fake)

        Y_fake_e = self.adversarial_embedded(Z)
        G_loss_U_e = self._bce(y_true=ops.ones_like(Y_fake_e), y_pred=Y_fake_e)

        H = self.embedder(X)
        H_hat_supervised = self.supervisor(H)
        G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])

        X_hat = self.generator(Z)
        G_loss_V = self._compute_generator_moments_loss(X, X_hat)

        G_loss = (
            G_loss_U
            + self.gamma * G_loss_U_e
            + 100 * ops.sqrt(G_loss_S)
            + 100 * G_loss_V
        )

        g_vars = self.generator_aux.trainable_variables
        s_vars = self.supervisor.trainable_variables
        all_trainable = g_vars + s_vars
        gradients = [v.value.grad for v in all_trainable]
        apply_grads = [
            (grad, var)
            for (grad, var) in zip(gradients, all_trainable)
            if grad is not None
        ]
        
        with torch.no_grad():
            optimizer.apply(apply_grads)
        return G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, G_loss 
    
    
    def _train_generator(
        self, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float, float, float, float]:
        """
        3. Joint training (Generator training twice more than discriminator training): minimize G_loss
        """
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            return self._train_generator_tf(backend, X, Z, optimizer)
        elif os.environ["KERAS_BACKEND"] == "torch":
            return self._train_generator_torch(backend, X, Z, optimizer)

    @tf_function_decorator
    def _train_embedder_tf(
        self, tf, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float]:
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
            E_loss0 = 10 * ops.sqrt(E_loss_T0)

            E_loss = E_loss0 + 0.1 * G_loss_S

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars
        gradients = tape.gradient(E_loss, all_trainable)
        optimizer.apply_gradients(zip(gradients, all_trainable))
        return E_loss, E_loss_T0
    
    def _train_embedder_torch(
        self, torch, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float]:
        H = self.embedder(X)
        H_hat_supervised = self.supervisor(H)
        G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])

        X_tilde = self.autoencoder(X)
        E_loss_T0 = self._mse(X, X_tilde)
        E_loss0 = 10 * ops.sqrt(E_loss_T0)

        E_loss = E_loss0 + 0.1 * G_loss_S

        e_vars = self.embedder.trainable_variables
        r_vars = self.recovery.trainable_variables
        all_trainable = e_vars + r_vars
        gradients = [v.value.grad for v in all_trainable]

        with torch.no_grad():
            optimizer.apply(zip(gradients, all_trainable))
        return E_loss, E_loss_T0
        
    def _train_embedder(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float]:
        """
        Train embedder during joint training: minimize E_loss
        """
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            return self._train_embedder_tf(backend, X, optimizer)
        elif os.environ["KERAS_BACKEND"] == "torch":
            return self._train_embedder_torch(backend, X, optimizer)

    @tf_function_decorator
    def _train_discriminator_tf(
        self, tf, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
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
    
    def _train_discriminator_torch(
        self, torch, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        D_loss = self._check_discriminator_loss(X, Z)
        self.discriminator.zero_grad()
        D_loss.backward()
        
        d_vars = [v for v in self.discriminator.trainable_variables]
        gradients = [v.value.grad for v in d_vars]

        with torch.no_grad():
            optimizer.apply(zip(gradients, d_vars))
        return D_loss
    
    def _train_discriminator(
        self, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        """
        minimize D_loss
        """
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            return self._train_discriminator_tf(backend, X, Z, optimizer)
        elif os.environ["KERAS_BACKEND"] == "torch":
            return self._train_discriminator_torch(backend, X, Z, optimizer)

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
        y_true_mean, y_true_var = ops.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = ops.nn.moments(x=y_pred, axes=[0])
        # G_loss_V2
        g_loss_mean = ops.mean(abs(y_true_mean - y_pred_mean))
        # G_loss_V1
        g_loss_var = ops.mean(
            abs(ops.sqrt(y_true_var + _eps) - ops.sqrt(y_pred_var + _eps))
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
        D_loss_real = self._bce(y_true=ops.ones_like(Y_real), y_pred=Y_real)

        # Loss on false positives
        Y_fake = self.adversarial_supervised(Z)
        D_loss_fake = self._bce(y_true=ops.zeros_like(Y_fake), y_pred=Y_fake)

        Y_fake_e = self.adversarial_embedded(Z)
        D_loss_fake_e = self._bce(y_true=ops.zeros_like(Y_fake_e), y_pred=Y_fake_e)

        D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
        return D_loss

    def _generate_noise(self) -> TensorLike:
        """
        Random vector generation
        :return Z, generated random vector
        """
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.dim))

    def get_noise_batch(self) -> T.Iterator:
        """
        Return an iterator of random noise vectors
        """
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            tf = backend
            return iter(
                tf.data.Dataset.from_generator(
                    self._generate_noise, output_types=tf.float32
                )
                .batch(self.batch_size)
                .repeat()
            )
        elif os.environ["KERAS_BACKEND"] == "torch":
            torch = backend
            def noise_generator():
                while True:
                    yield torch.tensor(
                        np.random.uniform(low=0, high=1, size=(self.seq_len, self.dim)),
                        dtype=torch.float32
                    )

            noise_dataset = torch.utils.data.IterableDataset.from_generator(noise_generator)
            return iter(torch.utils.data.DataLoader(noise_dataset, batch_size=self.batch_size))

    def _get_data_batch(self, data: TensorLike, n_windows: int) -> T.Iterator:
        """
        Return an iterator of shuffled input data
        """
        data = ops.convert_to_tensor(data, dtype="float32")
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            tf = backend
            return iter(
                tf.data.Dataset.from_tensor_slices(data)
                .shuffle(buffer_size=n_windows)
                .batch(self.batch_size)
                .repeat()
            )
        elif os.environ["KERAS_BACKEND"] == "torch":
            torch = backend
            data = torch.tensor(data, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(data)
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fit(
        self,
        data: T.Union[TensorLike, Keras_Dataset],
        epochs: int,
        checkpoints_interval: T.Optional[int] = None,
        generate_synthetic: T.Tuple = (),
        *args,
        **kwargs,
    ):
        """
        :param data: TensorLike, the training data
        :param epochs: int, the number of epochs for the training loops
        :param checkpoints_interval: int, the interval for printing out loss values
            (loss values will be print out every 'checkpoints_interval' epochs)
            Default: None (no print out)
        :param generate_synthetic: list of int, a list of epoch numbers when synthetic data samples are generated
            Default: [] (no generation)
        :return None
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

        # take tf.data.Dataset | torch.utils.data.Dataloader | TensorLike  
        if os.environ["KERAS_BACKEND"] == "tensorflow" and isinstance(data, Keras_Dataset):
            batches = iter(data.repeat())
        elif os.environ["KERAS_BACKEND"] == "torch" and isinstance(data, Keras_Dataset):
            batches = iter(data)
        else:
            data_loader = self._get_data_batch(data, n_windows=len(data))
            batches = iter(data_loader)

        # Define the model
        self._define_timegan()

        # 1. Embedding network training
        logger.info("Start Embedding Network Training")

        for epoch in tqdm(range(epochs), desc="Autoencoder - training"):
            X_ = next(batches)
            step_e_loss_0 = self._train_autoencoder(X_, self.autoencoder_opt)

            # Checkpoint
            if checkpoints_interval is not None and epoch % checkpoints_interval == 0:
                logger.info(f"step: {epoch}/{epochs}, e_loss: {step_e_loss_0}")
            self.training_losses_history["autoencoder"] = float(step_e_loss_0)

        logger.info("Finished Embedding Network Training")

        # 2. Training only with supervised loss
        logger.info("Start Training with Supervised Loss Only")

        # Adversarial Supervised network training
        for epoch in tqdm(range(epochs), desc="Adversarial Supervised - training"):
            X_ = next(batches)
            step_g_loss_s = self._train_supervisor(X_, self.adversarialsup_opt)

            # Checkpoint
            if checkpoints_interval is not None and epoch % checkpoints_interval == 0:
                logger.info(
                    f"step: {epoch}/{epochs}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}"
                )
            self.training_losses_history["adversarial_supervised"] = float(
                np.sqrt(step_g_loss_s)
            )

        logger.info("Finished Training with Supervised Loss Only")

        # 3. Joint Training
        logger.info("Start Joint Training")

        # GAN with embedding network training
        for epoch in tqdm(range(epochs), desc="GAN with embedding - training"):
            # Generator training (twice more than discriminator training)
            for kk in range(2):
                X_ = next(batches)
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

            X_ = next(batches)
            Z_ = next(self.get_noise_batch())
            step_d_loss = self._check_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                logger.info(
                    "Train Discriminator (discriminator does not work well yet)"
                )
                step_d_loss = self._train_discriminator(X_, Z_, self.discriminator_opt)

            # Print multiple checkpoints
            if checkpoints_interval is not None and epoch % checkpoints_interval == 0:
                logger.info(
                    f"""step: {epoch}/{epochs},
                    d_loss: {np.round(step_d_loss, 4)},
                    g_loss_u: {np.round(step_g_loss_u, 4)},
                    g_loss_u_e: {np.round(step_g_loss_u_e, 4)},
                    g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)},
                    g_loss_v: {np.round(step_g_loss_v, 4)},
                    g_loss_v: {np.round(step_g_loss, 4)},
                    e_loss_t0: {np.round(np.sqrt(step_e_loss_t0), 4)}"""
                )
            self.training_losses_history["discriminator"] = float(step_d_loss)
            self.training_losses_history["generator_u"] = float(step_g_loss_u)
            self.training_losses_history["generator_u_e"] = float(step_g_loss_u_e)
            self.training_losses_history["generator_v"] = float(step_g_loss_v)
            self.training_losses_history["generator_s"] = float(np.sqrt(step_g_loss_s))
            self.training_losses_history["generator"] = float(step_g_loss)
            self.training_losses_history["embedder"] = float(np.sqrt(step_e_loss_t0))

            # Synthetic data generation
            if epoch in generate_synthetic:
                _sample = self.generate(n_samples=len(data))
                self.synthetic_data_generated_in_training[epoch] = _sample

        logger.info("Finished Joint Training")
        return

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
