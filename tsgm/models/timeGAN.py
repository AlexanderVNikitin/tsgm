import os
import keras
from keras import ops
from tsgm.types import Tensor as TensorLike
# a keras_dataset can be tf.data.Dataset or Torch DataLoader
from tsgm.backend import tf_function_decorator, Keras_Dataset, get_backend
import numpy as np
import numpy.typing as npt
from tqdm import tqdm, trange
from collections import OrderedDict
from contextlib import nullcontext
from functools import wraps
import typing as T

import logging

from tsgm.models.architectures.zoo import BasicRecurrentArchitecture

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


def _torch_device_scope(method):
    """Keep Keras ops and optimizer state on this model's PyTorch device."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        device = None
        if keras.backend.backend() == "torch":
            variables = self.embedder.trainable_variables
            device = str(variables[0].value.device) if variables else self._device
        with keras.device(device) if device is not None else nullcontext():
            return method(self, *args, **kwargs)
    return wrapped


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

    :param device: Optional PyTorch device, such as "cpu" or "cuda".
        Defaults to the device selected by Keras.
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
        device: T.Optional[str] = None,
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

        # An explicit PyTorch device is local to this model. By default, retain
        # the device selected by Keras rather than changing Torch global state.
        self._device = device
        if keras.backend.backend() == "jax":
            self.seed_generator = keras.random.SeedGenerator()

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
        # Keras wraps assigned dicts as TrackedDict, which would discard the
        # append semantics and methods of LossTracker. This stores only Python
        # loss values, so it does not need layer/variable tracking.
        object.__setattr__(self, "training_losses_history", LossTracker())

        # --------------------------
        # Synthetic data generation during training: will be populated in .fit()
        # --------------------------
        self.synthetic_data_generated_in_training = dict()

    def compile(
        self,
        d_optimizer: keras.optimizers.Optimizer = None,
        g_optimizer: keras.optimizers.Optimizer = None,
        emb_optimizer: keras.optimizers.Optimizer = None,
        supgan_optimizer: keras.optimizers.Optimizer = None,
        ae_optimizer: keras.optimizers.Optimizer = None,
        emb_loss: keras.losses.Loss = None,
        clf_loss: keras.losses.Loss = None,
    ) -> None:
        """
        Assign optimizers and loss functions.

        :param d_optimizer: An optimizer for the GAN's discriminator.
        :type d_optimizer: keras.optimizers.Optimizer
        :param g_optimizer: An optimizer for the GAN's generator.
        :type g_optimizer: keras.optimizers.Optimizer
        :param emb_optimizer: An optimizer for the GAN's embedder.
        :type emb_optimizer: keras.optimizers.Optimizer
        :param supgan_optimizer: An optimizer for the adversarial supervised network.
        :type supgan_optimizer: keras.optimizers.Optimizer
        :param ae_optimizer: An optimizer for the autoencoder network.
        :type ae_optimizer: keras.optimizers.Optimizer
        :param emb_loss: A loss function for the embedding recovery.
        :type emb_loss: keras.losses.Loss
        :param clf_loss: A loss function for the discriminator task.
        :type clf_loss: keras.losses.Loss
        """
        # ----------------------------
        # Optimizers - create fresh instances for Keras 3.0 compatibility
        # ----------------------------
        self.autoencoder_opt = ae_optimizer if ae_optimizer is not None else keras.optimizers.Adam()
        self.adversarialsup_opt = supgan_optimizer if supgan_optimizer is not None else keras.optimizers.Adam()
        self.generator_opt = g_optimizer if g_optimizer is not None else keras.optimizers.Adam()
        self.embedder_opt = emb_optimizer if emb_optimizer is not None else keras.optimizers.Adam()
        self.discriminator_opt = d_optimizer if d_optimizer is not None else keras.optimizers.Adam()
        # ----------------------------
        # Loss functions - create fresh instances for Keras 3.0 compatibility
        # ----------------------------
        self._mse = emb_loss if emb_loss is not None else keras.losses.MeanSquaredError()
        self._bce = clf_loss if clf_loss is not None else keras.losses.BinaryCrossentropy()

    def _move_models_to_device(self) -> None:
        """Apply an explicitly requested PyTorch device to the built networks."""
        if keras.backend.backend() == "torch" and self._device is not None:
            for model in (self.embedder, self.recovery, self.generator_aux,
                          self.supervisor, self.discriminator):
                model.to(self._device)

    def _torch_batch(self, data: TensorLike) -> TensorLike:
        """Unwrap a single-input DataLoader batch and match model dtype/device."""
        if isinstance(data, (tuple, list)) and len(data) == 1:
            data = data[0]
        variable = self.embedder.trainable_variables[0].value
        torch = get_backend()
        return torch.as_tensor(data, dtype=variable.dtype, device=variable.device)

    @staticmethod
    def _apply_torch_gradients(torch, loss, variables, optimizer) -> None:
        # autograd.grad only returns this phase's gradients. In particular, it
        # neither reads stale .grad buffers nor accumulates into frozen networks.
        gradients = torch.autograd.grad(loss, [variable.value for variable in variables])
        with torch.no_grad():
            if not optimizer.built:
                optimizer.build(variables)
            # Optimizers may have been constructed outside the model device
            # scope. Keras does not relocate their learning rate, iteration
            # counter or existing slots when a model changes devices.
            device = variables[0].value.device
            for state in optimizer.variables:
                if state.value.device != device:
                    state.value.data = state.value.data.to(device)
            optimizer.apply_gradients(zip(gradients, variables))

    def _apply_jax_gradients(self, jax, loss_fn, variables, optimizer):
        """Update one phase without mutating Keras state inside JAX autodiff."""
        if not optimizer.built:
            optimizer.build(variables)
        non_trainable = self.non_trainable_variables

        def stateless_loss(values, non_trainable_values):
            mapping = list(zip(variables, values)) + list(zip(non_trainable, non_trainable_values))
            with keras.StatelessScope(state_mapping=mapping) as scope:
                loss, outputs = loss_fn()
            return loss, (outputs, [scope.get_current_value(v) for v in non_trainable])

        (_, (outputs, non_trainable_values)), gradients = jax.value_and_grad(
            stateless_loss, has_aux=True
        )([v.value for v in variables], [v.value for v in non_trainable])
        values, optimizer_values = optimizer.stateless_apply(
            [v.value for v in optimizer.variables], gradients, [v.value for v in variables]
        )
        for variable, value in zip(variables, values):
            variable.assign(value)
        for variable, value in zip(non_trainable, non_trainable_values):
            variable.assign(value)
        for variable, value in zip(optimizer.variables, optimizer_values):
            variable.assign(value)
        return outputs

    @_torch_device_scope
    def _define_timegan(self) -> None:
        if hasattr(self, "autoencoder"):
            return
        # --------------------------------
        # Data and Noise Inputs
        # --------------------------------
        X = keras.layers.Input(
            shape=[self.seq_len, self.dim], name="RealData"
        )

        Z = keras.layers.Input(
            shape=[self.seq_len, self.dim],
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
        self._move_models_to_device()

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

    @_torch_device_scope
    def _train_autoencoder_torch(
        self, torch, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        X = self._torch_batch(X)
        X_tilde = self.autoencoder(X, training=True)
        E_loss_T0 = self._mse(X, X_tilde)
        E_loss0 = 10.0 * ops.sqrt(E_loss_T0)
        variables = self.embedder.trainable_variables + self.recovery.trainable_variables
        self._apply_torch_gradients(torch, E_loss0, variables, optimizer)
        return E_loss0

    def _train_autoencoder(
        self, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            return self._train_autoencoder_tf(backend, X, optimizer)
        elif os.environ["KERAS_BACKEND"] == "torch":
            return self._train_autoencoder_torch(backend, X, optimizer)
        elif os.environ["KERAS_BACKEND"] == "jax":
            def loss_fn():
                loss = 10.0 * ops.sqrt(self._mse(X, self.autoencoder(X, training=True)))
                return loss, loss

            variables = self.embedder.trainable_variables + self.recovery.trainable_variables
            return self._apply_jax_gradients(backend, loss_fn, variables, optimizer)

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

        all_trainable = self.supervisor.trainable_variables
        gradients = tape.gradient(G_loss_S, all_trainable)
        apply_grads = [
            (grad, var)
            for (grad, var) in zip(gradients, all_trainable)
            if grad is not None
        ]
        optimizer.apply_gradients(apply_grads)
        return G_loss_S

    @_torch_device_scope
    def _train_supervisor_torch(
        self, torch, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        X = self._torch_batch(X)
        with torch.no_grad():
            H = self.embedder(X, training=False)
        H_hat_supervised = self.supervisor(H, training=True)
        G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])
        self._apply_torch_gradients(
            torch, G_loss_S, self.supervisor.trainable_variables, optimizer
        )
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
        elif os.environ["KERAS_BACKEND"] == "jax":
            H = self.embedder(X, training=False)

            def loss_fn():
                H_hat = self.supervisor(H, training=True)
                loss = self._mse(H[:, 1:, :], H_hat[:, :-1, :])
                return loss, loss

            return self._apply_jax_gradients(
                backend, loss_fn, self.supervisor.trainable_variables, optimizer
            )

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

    @_torch_device_scope
    def _train_generator_torch(
        self, torch, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float, float, float, float]:
        X, Z = self._torch_batch(X), self._torch_batch(Z)
        E_hat = self.generator_aux(Z, training=True)
        H_hat = self.supervisor(E_hat, training=True)
        # Discriminator/recovery are frozen for this update, but their forward
        # computations must remain differentiable with respect to G and S.
        Y_fake = self.discriminator(H_hat, training=False)
        G_loss_U = self._bce(y_true=ops.ones_like(Y_fake), y_pred=Y_fake)
        Y_fake_e = self.discriminator(E_hat, training=False)
        G_loss_U_e = self._bce(y_true=ops.ones_like(Y_fake_e), y_pred=Y_fake_e)

        with torch.no_grad():
            H = self.embedder(X, training=False)
        H_hat_supervised = self.supervisor(H, training=True)
        G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])
        X_hat = self.recovery(H_hat, training=False)
        G_loss_V = self._compute_generator_moments_loss(X, X_hat)
        G_loss = (
            G_loss_U
            + self.gamma * G_loss_U_e
            + 100 * ops.sqrt(G_loss_S)
            + 100 * G_loss_V
        )
        variables = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        self._apply_torch_gradients(torch, G_loss, variables, optimizer)
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
        elif os.environ["KERAS_BACKEND"] == "jax":
            H = self.embedder(X, training=False)

            def loss_fn():
                E_hat = self.generator_aux(Z, training=True)
                H_hat = self.supervisor(E_hat, training=True)
                Y_fake = self.discriminator(H_hat, training=False)
                G_loss_U = self._bce(ops.ones_like(Y_fake), Y_fake)
                Y_fake_e = self.discriminator(E_hat, training=False)
                G_loss_U_e = self._bce(ops.ones_like(Y_fake_e), Y_fake_e)
                H_hat_supervised = self.supervisor(H, training=True)
                G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])
                X_hat = self.recovery(H_hat, training=False)
                G_loss_V = self._compute_generator_moments_loss(X, X_hat)
                G_loss = (G_loss_U + self.gamma * G_loss_U_e
                          + 100 * ops.sqrt(G_loss_S) + 100 * G_loss_V)
                return G_loss, (G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, G_loss)

            variables = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
            return self._apply_jax_gradients(backend, loss_fn, variables, optimizer)

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

    @_torch_device_scope
    def _train_embedder_torch(
        self, torch, X: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> T.Tuple[float, float]:
        X = self._torch_batch(X)
        H = self.embedder(X, training=True)
        H_hat_supervised = self.supervisor(H, training=False)
        G_loss_S = self._mse(H[:, 1:, :], H_hat_supervised[:, :-1, :])
        X_tilde = self.recovery(H, training=True)
        E_loss_T0 = self._mse(X, X_tilde)
        E_loss = 10 * ops.sqrt(E_loss_T0) + 0.1 * G_loss_S
        variables = self.embedder.trainable_variables + self.recovery.trainable_variables
        self._apply_torch_gradients(torch, E_loss, variables, optimizer)
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
        elif os.environ["KERAS_BACKEND"] == "jax":
            def loss_fn():
                H = self.embedder(X, training=True)
                H_hat = self.supervisor(H, training=False)
                G_loss_S = self._mse(H[:, 1:, :], H_hat[:, :-1, :])
                X_tilde = self.recovery(H, training=True)
                E_loss_T0 = self._mse(X, X_tilde)
                E_loss = 10 * ops.sqrt(E_loss_T0) + 0.1 * G_loss_S
                return E_loss, (E_loss, E_loss_T0)

            variables = self.embedder.trainable_variables + self.recovery.trainable_variables
            return self._apply_jax_gradients(backend, loss_fn, variables, optimizer)

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

    @_torch_device_scope
    def _train_discriminator_torch(
        self, torch, X: TensorLike, Z: TensorLike, optimizer: keras.optimizers.Optimizer
    ) -> float:
        D_loss = self._check_discriminator_loss(X, Z)
        self._apply_torch_gradients(
            torch, D_loss, self.discriminator.trainable_variables, optimizer
        )
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
        elif os.environ["KERAS_BACKEND"] == "jax":
            def loss_fn():
                loss = self._check_discriminator_loss(X, Z)
                return loss, loss

            return self._apply_jax_gradients(
                backend, loss_fn, self.discriminator.trainable_variables, optimizer
            )

    @staticmethod
    def _compute_generator_moments_loss(
        y_true: TensorLike, y_pred: TensorLike
    ) -> float:
        """
        :param y_true: True values.
        :type y_true: TensorLike
        :param y_pred: Predicted values.
        :type y_pred: TensorLike

        :returns: Generator moments loss.
        :rtype: float
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

    @_torch_device_scope
    def _check_discriminator_loss(self, X: TensorLike, Z: TensorLike) -> float:
        """Compute discriminator loss without updating any network."""
        if keras.backend.backend() == "torch":
            torch = get_backend()
            X, Z = self._torch_batch(X), self._torch_batch(Z)
            with torch.no_grad():
                H = self.embedder(X, training=False)
                E_hat = self.generator_aux(Z, training=False)
                H_hat = self.supervisor(E_hat, training=False)
            Y_real = self.discriminator(H, training=True)
            Y_fake = self.discriminator(H_hat, training=True)
            Y_fake_e = self.discriminator(E_hat, training=True)
        else:
            Y_real = self.discriminator_model(X)
            Y_fake = self.adversarial_supervised(Z)
            Y_fake_e = self.adversarial_embedded(Z)
        D_loss_real = self._bce(y_true=ops.ones_like(Y_real), y_pred=Y_real)
        D_loss_fake = self._bce(y_true=ops.zeros_like(Y_fake), y_pred=Y_fake)
        D_loss_fake_e = self._bce(y_true=ops.zeros_like(Y_fake_e), y_pred=Y_fake_e)
        return D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e

    def _generate_noise(self) -> TensorLike:
        """
        Random vector generation
        :return Z, generated random vector
        """
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.dim))

    def get_noise_batch(self, batch_size: T.Optional[int] = None) -> T.Iterator:
        """
        Return an iterator of random noise vectors
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            tf = backend
            return iter(
                tf.data.Dataset.from_generator(
                    self._generate_noise, output_types=tf.float32
                )
                .batch(batch_size)
                .repeat()
            )
        elif os.environ["KERAS_BACKEND"] == "torch":
            torch = backend

            def noise_generator():
                while True:
                    variables = self.embedder.trainable_variables
                    device = variables[0].value.device if variables else self._device
                    yield torch.rand(
                        batch_size, self.seq_len, self.dim,
                        dtype=torch.float32, device=device,
                    )

            return noise_generator()
        elif os.environ["KERAS_BACKEND"] == "jax":
            def noise_generator():
                while True:
                    yield keras.random.uniform(
                        (batch_size, self.seq_len, self.dim), seed=self.seed_generator
                    )

            return noise_generator()

    def _get_data_batch(self, data: TensorLike, n_windows: int) -> T.Iterator:
        """
        Return an iterator of shuffled input data
        """
        backend = get_backend()
        if os.environ["KERAS_BACKEND"] == "tensorflow":
            tf = backend
            data = ops.convert_to_tensor(data, dtype="float32")
            return iter(
                tf.data.Dataset.from_tensor_slices(data)
                .shuffle(buffer_size=n_windows)
                .batch(self.batch_size)
                .repeat()
            )
        elif os.environ["KERAS_BACKEND"] == "torch":
            torch = backend
            data = torch.as_tensor(data, dtype=torch.float32, device="cpu").detach()
            dataset = torch.utils.data.TensorDataset(data)
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        elif os.environ["KERAS_BACKEND"] == "jax":
            if n_windows == 0 or self.batch_size <= 0:
                raise ValueError("Training data and batch_size must be non-empty and positive")
            data = ops.convert_to_tensor(data, dtype="float32")

            def repeating_batches():
                while True:
                    indices = np.random.permutation(n_windows)
                    for start in range(0, n_windows, self.batch_size):
                        yield data[indices[start:start + self.batch_size]]

            return repeating_batches()

    def fit(  # noqa: C901 - suppress cyclomatic complexity warning for this function
        self,
        data: T.Union[TensorLike, Keras_Dataset],
        epochs: int,
        checkpoints_interval: T.Optional[int] = None,
        generate_synthetic: T.Tuple = (),
        *args,
        **kwargs,
    ):
        """
        :param data: The training data.
        :type data: TensorLike
        :param epochs: The number of epochs for the training loops.
        :type epochs: int
        :param checkpoints_interval: The interval for printing out loss values
            (loss values will be printed every `checkpoints_interval` epochs).
            Default: None (no print out).
        :type checkpoints_interval: int or None
        :param generate_synthetic: A list of epoch numbers when synthetic data samples are generated.
            Default: () (no generation).
        :type generate_synthetic: tuple
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
        if os.environ["KERAS_BACKEND"] == "tensorflow" and Keras_Dataset is not None and isinstance(data, Keras_Dataset):
            batches = iter(data.repeat())
        elif os.environ["KERAS_BACKEND"] == "torch" and Keras_Dataset is not None and isinstance(data, Keras_Dataset):
            # Create repeating iterator for PyTorch DataLoader
            def repeating_iter():
                while True:
                    for batch in data:
                        yield batch
            batches = repeating_iter()
        else:
            data_loader = self._get_data_batch(data, n_windows=len(data))
            if os.environ["KERAS_BACKEND"] == "torch":
                # Create repeating iterator for PyTorch DataLoader
                def repeating_iter():
                    while True:
                        for batch in data_loader:
                            yield batch
                batches = repeating_iter()
            else:
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
            self.training_losses_history["autoencoder"] = float(ops.convert_to_numpy(step_e_loss_0))

        logger.info("Finished Embedding Network Training")

        # 2. Training only with supervised loss
        logger.info("Start Training with Supervised Loss Only")

        # Adversarial Supervised network training
        for epoch in tqdm(range(epochs), desc="Adversarial Supervised - training"):
            X_ = next(batches)
            step_g_loss_s = self._train_supervisor(X_, self.adversarialsup_opt)

            step_g_loss_s = float(ops.convert_to_numpy(step_g_loss_s))
            # Checkpoint
            if checkpoints_interval is not None and epoch % checkpoints_interval == 0:
                logger.info(
                    f"step: {epoch}/{epochs}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}"
                )
            self.training_losses_history["adversarial_supervised"] = float(np.sqrt(step_g_loss_s))

        logger.info("Finished Training with Supervised Loss Only")

        # 3. Joint Training
        logger.info("Start Joint Training")

        # GAN with embedding network training
        for epoch in tqdm(range(epochs), desc="GAN with embedding - training"):
            # Generator training (twice more than discriminator training)
            for kk in range(2):
                X_ = next(batches)
                if keras.backend.backend() == "torch":
                    X_ = self._torch_batch(X_)
                Z_ = next(self.get_noise_batch(batch_size=int(ops.shape(X_)[0])))
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
            if keras.backend.backend() == "torch":
                X_ = self._torch_batch(X_)
            Z_ = next(self.get_noise_batch(batch_size=int(ops.shape(X_)[0])))
            step_d_loss = self._check_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                logger.info(
                    "Train Discriminator (discriminator does not work well yet)"
                )
                step_d_loss = self._train_discriminator(X_, Z_, self.discriminator_opt)

            (step_d_loss, step_g_loss_u, step_g_loss_u_e, step_g_loss_s,
             step_g_loss_v, step_g_loss, step_e_loss_t0) = (
                float(ops.convert_to_numpy(loss))
                for loss in (step_d_loss, step_g_loss_u, step_g_loss_u_e,
                             step_g_loss_s, step_g_loss_v, step_g_loss, step_e_loss_t0)
            )

            # Print multiple checkpoints
            if checkpoints_interval is not None and epoch % checkpoints_interval == 0:
                logger.info(
                    f"""step: {epoch}/{epochs},
                    d_loss: {np.round(step_d_loss, 4)},
                    g_loss_u: {np.round(step_g_loss_u, 4)},
                    g_loss_u_e: {np.round(step_g_loss_u_e, 4)},
                    g_loss_s: {np.round(np.sqrt(step_g_loss_s), 4)},
                    g_loss_v: {np.round(step_g_loss_v, 4)},
                    g_loss: {np.round(step_g_loss, 4)},
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
                n_samples = (len(data.dataset)
                             if keras.backend.backend() == "torch"
                             and Keras_Dataset is not None
                             and isinstance(data, Keras_Dataset) else len(data))
                _sample = self.generate(n_samples=n_samples)
                self.synthetic_data_generated_in_training[epoch] = _sample

        logger.info("Finished Joint Training")

    @_torch_device_scope
    def generate(self, n_samples: int) -> TensorLike:
        """Generate synthetic time series as a NumPy array."""
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if n_samples == 0:
            return np.empty((0, self.seq_len, self.dim), dtype="float32")
        steps = (n_samples + self.batch_size - 1) // self.batch_size
        data = []
        context = get_backend().no_grad() if keras.backend.backend() == "torch" else nullcontext()
        with context:
            for _ in trange(steps, desc="Synthetic data generation"):
                Z_ = next(self.get_noise_batch())
                records = self.generator(Z_, training=False)
                data.append(ops.convert_to_numpy(records))
        return np.concatenate(data, axis=0)[:n_samples]
