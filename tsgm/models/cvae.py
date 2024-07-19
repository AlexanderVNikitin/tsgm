import os
import keras
from keras import ops
import typing as T

from tsgm.backend import get_backend
        
import tsgm.utils


class BetaVAE(keras.Model):
    """
    beta-VAE implementation for unlabeled time series.
    """
    def __init__(self, encoder: keras.Model, decoder: keras.Model, beta: float = 1.0, **kwargs) -> None:
        """
        :param encoder: An encoder model which takes a time series as input and check
            whether the image is real or fake.
        :type encoder: keras.Model
        :param decoder: Takes as input a random noise vector of `latent_dim` length and returns
            a simulated time-series.
        :type decoder: keras.Model
        :param latent_dim: The size of the noise vector.
        :type latent_dim: int
        """
        super(BetaVAE, self).__init__(**kwargs)
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self._seq_len = self.decoder.output_shape[1]
        self.latent_dim = self.decoder.input_shape[1]

    @property
    def metrics(self) -> T.List:
        """
        :returns: A list of metrics trackers (e.g., generator's loss and discriminator's loss).
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, X: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Encodes and decodes time series dataset X.

        :param X: The size of the noise vector.
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
    
    def tf_train_step(self, tf, data: tsgm.types.Tensor) -> T.Dict:
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        #  I am not sure if this should be self.optimizer.apply(grads, model.trainable_weights)
        #  see https://keras.io/guides/writing_a_custom_training_loop_in_tensorflow/
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def torch_train_step(self, torch, data: tsgm.types.Tensor) -> T.Dict:
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss = torch.mean(torch.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def train_step(self, data: tsgm.types.Tensor) -> T.Dict:
        """
        Performs a training step using a batch of data, stored in data.

        :param data: A batch of data in a format batch_size x seq_len x feat_dim
        :type data: tsgm.types.Tensor

        :returns: A dict with losses
        :rtype: T.Dict
        """
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.tf_train_step(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.torch_train_step(backend, data)

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


class cBetaVAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, latent_dim: int, temporal: bool, beta: float = 1.0, **kwargs) -> None:
        super(cBetaVAE, self).__init__(**kwargs)
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self._temporal = temporal
        self._seq_len = self.decoder.output_shape[1]
        self.latent_dim = latent_dim

    @property
    def metrics(self) -> T.List:
        """
        Returns the list of loss tracker:  `[loss, reconstruction_loss, kl_loss]`.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def generate(self, labels: tsgm.types.Tensor) -> T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor]:
        """
        Generates new data from the model.

        :param labels: the number of samples to be generated.
        :type labels: tsgm.types.Tensor

        :returns: a tuple of synthetically generated data and labels.
        :rtype: T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor]
        """
        #  keras 3.0 support
        batch_size = ops.shape(labels)[0]
        z = keras.random.normal((batch_size, self._seq_len, self.latent_dim), dtype=labels.dtype)
        decoder_input = self._get_decoder_input(z, labels)
        return (self.decoder(decoder_input), labels)

    def call(self, data: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Encodes and decodes time series dataset X.

        :param X: The size of the noise vector.
        :type X: tsgm.types.Tensor

        :returns: Generated samples
        :rtype: tsgm.types.Tensor
        """
        X, labels = data
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
        if self._temporal:
            return ops.concatenate([X, labels[:, :, None]], axis=2)
        else:
            rep_labels = ops.repeat(labels[:, None, :], [self._seq_len], axis=1)
            return ops.concatenate([X, rep_labels], axis=2)

    def _get_decoder_input(self, z: tsgm.types.Tensor, labels: tsgm.types.Tensor) -> tsgm.types.Tensor:
        #  keras 3.0 support
        if self._temporal:
            rep_labels = labels[:, :, None]
        else:
            rep_labels = ops.repeat(labels[:, None, :], [self._seq_len], axis=1)
        z = ops.reshape(z, [-1, self._seq_len, self.latent_dim])
        return ops.concatenate([z, rep_labels], axis=2)

    def tf_train_step(self, tf, data: tsgm.types.Tensor) -> T.Dict[str, float]:
        X, labels = data
        with tf.GradientTape() as tape:
            encoder_input = self._get_encoder_input(X, labels)
            z_mean, z_log_var, z = self.encoder(encoder_input)

            decoder_input = self._get_decoder_input(z_mean, labels)
            reconstruction = self.decoder(decoder_input)
            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def torch_train_step(self, torch, data: tsgm.types.Tensor) -> T.Dict[str, float]:
        X, labels = data
        encoder_input = self._get_encoder_input(X, labels)
        z_mean, z_log_var, z = self.encoder(encoder_input)

        decoder_input = self._get_decoder_input(z_mean, labels)
        reconstruction = self.decoder(decoder_input)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        kl_loss = torch.mean(torch.sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta * kl_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }    



    def train_step(self, data: tsgm.types.Tensor) -> T.Dict[str, float]:
        """
        Performs a training step using a batch of data, stored in data.

        :param data: A batch of data in a format batch_size x seq_len x feat_dim
        :type data: tsgm.types.Tensor

        :returns: A dict with losses
        :rtype: T.Dict[str, float]
        """
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.tf_train_step(backend, data)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.torch_train_step(backend, data)