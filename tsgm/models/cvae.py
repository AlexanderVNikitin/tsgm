import os
import keras
from keras import ops
import typing as T

from tsgm.backend import get_backend

import tsgm.utils


class _BaseVAE(keras.Model):
    """
    Shared training logic for :class:`BetaVAE` and :class:`cBetaVAE`.

    Subclasses implement :meth:`_compute_losses`, the forward pass and loss computation.
    The backend-specific train steps below only differ in how gradients are computed and applied.
    """
    def __init__(self, beta: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> T.List:
        """
        :returns: A list of metrics trackers (total loss, reconstruction loss, and KL loss).
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def _compute_losses(self, data: tsgm.types.Tensor) -> T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor, tsgm.types.Tensor]:
        """
        Runs the forward pass on a batch and returns `(total_loss, reconstruction_loss, kl_loss)`.
        """
        raise NotImplementedError

    def _total_loss(self, reconstruction_loss: tsgm.types.Tensor, z_mean: tsgm.types.Tensor,
                    z_log_var: tsgm.types.Tensor) -> T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor, tsgm.types.Tensor]:
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta * kl_loss
        # Gradient computation (torch backward(), jax.value_and_grad) requires a scalar loss
        if len(ops.shape(total_loss)) > 0:
            total_loss = ops.mean(total_loss)
        return total_loss, reconstruction_loss, kl_loss

    def _update_trackers(self, total_loss, reconstruction_loss, kl_loss) -> T.Dict:
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def _apply_gradients(self, gradients: T.List, variables: T.List) -> None:
        # Variables that do not contribute to the loss get no gradient; skip them
        grads_and_vars = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)

    def train_step_tf(self, tf, data: tsgm.types.Tensor) -> T.Dict:
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self._apply_gradients(grads, self.trainable_weights)
        return self._update_trackers(total_loss, reconstruction_loss, kl_loss)

    def train_step_torch(self, torch, data: tsgm.types.Tensor) -> T.Dict:
        total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)
        self.zero_grad()
        total_loss.backward()

        trainable_weights = list(self.trainable_weights)
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self._apply_gradients(gradients, trainable_weights)
        return self._update_trackers(total_loss, reconstruction_loss, kl_loss)

    def train_step_jax(self, jax, state: T.Tuple, data: tsgm.types.Tensor) -> T.Tuple[T.Dict, T.Tuple]:
        # The JAX trainer is stateless: variable values come in via `state` and
        # updated values must be returned, together with the logs.
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state

        def loss_fn(trainable_values, non_trainable_values):
            state_mapping = list(zip(self.trainable_variables, trainable_values)) + \
                list(zip(self.non_trainable_variables, non_trainable_values))
            with keras.StatelessScope(state_mapping=state_mapping) as scope:
                total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)
            # Non-trainable state (e.g. seed generators, batch-norm statistics) may change in the forward pass
            new_non_trainable_values = [
                scope.get_current_value(v) for v in self.non_trainable_variables
            ]
            return total_loss, (reconstruction_loss, kl_loss, new_non_trainable_values)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (reconstruction_loss, kl_loss, non_trainable_variables)), grads = grad_fn(
            trainable_variables, non_trainable_variables
        )
        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        with keras.StatelessScope(
            state_mapping=list(zip(self.metrics_variables, metrics_variables))
        ) as scope:
            logs = self._update_trackers(total_loss, reconstruction_loss, kl_loss)
        metrics_variables = [
            scope.get_current_value(v) for v in self.metrics_variables
        ]
        return logs, (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )

    def train_step(self, *args) -> T.Any:
        """
        Performs a training step using a batch of data.

        With TensorFlow and PyTorch backends, it takes a batch `data` and returns a dict with losses.
        With the JAX backend, Keras calls it as `train_step(state, data)` and it returns `(logs, state)`.

        :returns: A dict with losses (or `(logs, state)` for JAX)
        """
        backend = get_backend()
        keras_backend = os.environ.get("KERAS_BACKEND")
        if keras_backend == "tensorflow":
            return self.train_step_tf(backend, *args)
        elif keras_backend == "torch":
            return self.train_step_torch(backend, *args)
        elif keras_backend == "jax":
            return self.train_step_jax(backend, *args)
        raise ValueError(f"Unsupported KERAS_BACKEND: {keras_backend}")


class BetaVAE(_BaseVAE):
    """
    beta-VAE implementation for unlabeled time series.
    """
    def __init__(self, encoder: keras.Model, decoder: keras.Model, beta: float = 1.0, **kwargs) -> None:
        """
        :param encoder: An encoder model which takes a time series as input.
        :type encoder: keras.Model
        :param decoder: Takes as input a random noise vector and returns a simulated time-series.
        :type decoder: keras.Model
        :param beta: The weight of the KL divergence term. Default is 1.0.
        :type beta: float
        """
        super(BetaVAE, self).__init__(beta=beta, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._seq_len = self.decoder.output_shape[1]
        self.latent_dim = self.decoder.input_shape[1]

    def call(self, X: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Encodes and decodes time series dataset X.

        :param X: The input time series tensor.
        :type X: tsgm.types.Tensor

        :returns: Generated samples
        :rtype: tsgm.types.Tensor
        """
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def _get_reconstruction_loss(self, X: tsgm.types.Tensor, Xr: tsgm.types.Tensor) -> float:
        reconst_loss = tsgm.utils.reconstruction_loss_by_axis(X, Xr, axis=0) +\
            tsgm.utils.reconstruction_loss_by_axis(X, Xr, axis=1) +\
            tsgm.utils.reconstruction_loss_by_axis(X, Xr, axis=2)
        return reconst_loss

    def _compute_losses(self, data: tsgm.types.Tensor) -> T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor, tsgm.types.Tensor]:
        z_mean, z_log_var, z = self.encoder(data)
        # Decode the sampled z (reparameterization trick), not z_mean
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
        return self._total_loss(reconstruction_loss, z_mean, z_log_var)

    def generate(self, n: int) -> tsgm.types.Tensor:
        """
        Generates new data from the model.

        :param n: the number of samples to be generated.
        :type n: int

        :returns: A tensor with generated samples.
        :rtype: tsgm.types.Tensor
        """
        #  keras 3.0 support
        z = keras.random.normal((n, self.latent_dim))
        return self.decoder(z)


class cBetaVAE(_BaseVAE):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, latent_dim: int, temporal: bool, beta: float = 1.0, **kwargs) -> None:
        super(cBetaVAE, self).__init__(beta=beta, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._temporal = temporal
        self._seq_len = self.decoder.output_shape[1]
        self.latent_dim = latent_dim

    def generate(self, labels: tsgm.types.Tensor) -> T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor]:
        """
        Generates new data from the model.

        :param labels: The labels for which to generate conditional samples.
        :type labels: tsgm.types.Tensor

        :returns: A tuple of synthetically generated data and labels.
        :rtype: T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor]
        """
        #  keras 3.0 support
        batch_size = ops.shape(labels)[0]
        dtype = 'float32' if os.environ.get("KERAS_BACKEND") == "torch" else labels.dtype
        z = keras.random.normal((batch_size, self._seq_len, self.latent_dim), dtype=dtype)
        decoder_input = self._get_decoder_input(z, labels)
        return (self.decoder(decoder_input), labels)

    def call(self, data: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Encodes and decodes time series dataset.

        :param data: The input data, either a tensor or a tuple of (X, labels).
        :type data: tsgm.types.Tensor

        :returns: Generated samples.
        :rtype: tsgm.types.Tensor
        """
        # Handle both single tensor and tuple of (X, labels)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            X, labels = data
        else:
            # During model building, just return the input
            return data
        encoder_input = self._get_encoder_input(X, labels)
        z_mean, _, _ = self.encoder(encoder_input)
        decoder_input = self._get_decoder_input(z_mean, labels)
        x_decoded = self.decoder(decoder_input)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def _get_reconstruction_loss(self, X: tsgm.types.Tensor, Xr: tsgm.types.Tensor) -> float:
        #  keras 3.0 support
        reconst_loss = ops.sum(ops.square(X - Xr)) +\
            ops.sum(ops.square(ops.mean(X, axis=1) - ops.mean(Xr, axis=1))) +\
            ops.sum(ops.square(ops.mean(X, axis=2) - ops.mean(Xr, axis=2)))
        return reconst_loss

    def _get_encoder_input(self, X: tsgm.types.Tensor, labels: tsgm.types.Tensor) -> tsgm.types.Tensor:
        #  keras 3.0 support
        if os.environ.get("KERAS_BACKEND") == "torch" and hasattr(labels, 'dtype'):
            labels = ops.cast(labels, 'float32')
        if self._temporal:
            return ops.concatenate([X, labels[:, :, None]], axis=2)
        else:
            rep_labels = ops.repeat(labels[:, None, :], self._seq_len, axis=1)
            return ops.concatenate([X, rep_labels], axis=2)

    def _get_decoder_input(self, z: tsgm.types.Tensor, labels: tsgm.types.Tensor) -> tsgm.types.Tensor:
        #  keras 3.0 support
        if os.environ.get("KERAS_BACKEND") == "torch" and hasattr(labels, 'dtype'):
            labels = ops.cast(labels, 'float32')
        if self._temporal:
            rep_labels = labels[:, :, None]
        else:
            rep_labels = ops.repeat(labels[:, None, :], self._seq_len, axis=1)
        z = ops.reshape(z, [-1, self._seq_len, self.latent_dim])
        return ops.concatenate([z, rep_labels], axis=2)

    def _compute_losses(self, data: tsgm.types.Tensor) -> T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor, tsgm.types.Tensor]:
        X, labels = data
        encoder_input = self._get_encoder_input(X, labels)
        z_mean, z_log_var, z = self.encoder(encoder_input)
        # Decode the sampled z (reparameterization trick), not z_mean
        decoder_input = self._get_decoder_input(z, labels)
        reconstruction = self.decoder(decoder_input)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)
        return self._total_loss(reconstruction_loss, z_mean, z_log_var)
