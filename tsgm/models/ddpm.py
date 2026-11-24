"""
The implementation is based on Keras DDPM implementation: https://keras.io/examples/generative/ddpm/
"""
import os
import numpy as np
import keras
from keras import ops
from tsgm.types import Tensor as TensorLike
from tsgm.backend import get_backend

import typing as T


class GaussianDiffusion:
    """
    Gaussian diffusion utility for generating samples using a diffusion process.

    This class implements a Gaussian diffusion process, where a sample is gradually
    perturbed by adding Gaussian noise over a series of timesteps. It also includes
    methods to reverse the diffusion process, predicting the original data from
    the noisy samples.

    :param beta_start: Start value of the scheduled variance for the diffusion process.
    :type beta_start: float
    :param beta_end: End value of the scheduled variance for the diffusion process.
    :type beta_end: float
    :param timesteps: Number of timesteps in the forward process.
    :type timesteps: int
    """

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        timesteps: int = 1000,
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        # Define the linear variance schedule
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float32,  # Using float32 for MPS compatibility
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = ops.convert_to_tensor(betas, dtype="float32")
        self.alphas_cumprod = ops.convert_to_tensor(alphas_cumprod, dtype="float32")
        self.alphas_cumprod_prev = ops.convert_to_tensor(alphas_cumprod_prev, dtype="float32")

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = ops.convert_to_tensor(
            np.sqrt(alphas_cumprod), dtype="float32"
        )

        self.sqrt_one_minus_alphas_cumprod = ops.convert_to_tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype="float32"
        )

        self.log_one_minus_alphas_cumprod = ops.convert_to_tensor(
            np.log(1.0 - alphas_cumprod), dtype="float32"
        )

        self.sqrt_recip_alphas_cumprod = ops.convert_to_tensor(
            np.sqrt(1.0 / alphas_cumprod), dtype="float32"
        )
        self.sqrt_recipm1_alphas_cumprod = ops.convert_to_tensor(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype="float32"
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = ops.convert_to_tensor(posterior_variance, dtype="float32")

        # Log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = ops.convert_to_tensor(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype="float32"
        )

        self.posterior_mean_coef1 = ops.convert_to_tensor(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype="float32",
        )

        self.posterior_mean_coef2 = ops.convert_to_tensor(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype="float32",
        )

    def _extract(self, a: TensorLike, t: int, x_shape) -> TensorLike:
        """
        Extracts coefficients for a specific timestep and reshapes them for broadcasting.

        :param a: Tensor to extract from.
        :type a: TensorLike
        :param t: Timestep for which the coefficients are to be extracted.
        :type t: int
        :param x_shape: Shape of the current batched samples.
        :type x_shape: tuple

        :returns: Tensor reshaped to [batch_size, 1, 1] for broadcasting.
        :rtype: TensorLike
        """
        batch_size = x_shape[0]
        # Handle both (batch_size,) and (batch_size, 1) shapes
        t_flat = ops.reshape(t, [-1])
        out = ops.take(a, t_flat, axis=0)
        return ops.reshape(out, [batch_size, 1, 1])

    def q_mean_variance(self, x_start: TensorLike, t: float) -> T.Tuple:
        """
        Extracts the mean and variance at a specific timestep in the forward diffusion process.

        :param x_start: Initial sample (before the first diffusion step).
        :type x_start: TensorLike
        :param t: A timestep.
        :type t: float

        :returns: Tensors representing the mean, variance, and log variance of the distribution at `t`.
        :rtype: tuple
        """
        x_start_shape = ops.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start: TensorLike, t: float, noise: float) -> T.Tuple:
        """
        Performs the forward diffusion step by adding Gaussian noise to the sample.

        :param x_start: Initial sample (before the first diffusion step).
        :type x_start: TensorLike
        :param t: Current timestep.
        :type t: float
        :param noise: Gaussian noise to be added at timestep `t`.
        :type noise: float

        :returns: Diffused samples at timestep `t`.
        :rtype: TensorLike
        """
        x_start_shape = ops.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, ops.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t: TensorLike, t, noise):
        """
        Predicts the initial sample from the noisy sample at timestep `t`.

        :param x_t: Noisy sample at timestep `t`.
        :type x_t: TensorLike
        :param t: Current timestep.
        :type t: int
        :param noise: Gaussian noise added at timestep `t`.
        :type noise: TensorLike

        :returns: Predicted initial sample.
        :rtype: TensorLike
        """

        x_t_shape = ops.shape(x_t)

        # Extract coefficients
        coeff1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape)
        coeff2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape)

        # Perform operations step by step to avoid device mismatch
        term1 = ops.multiply(coeff1, x_t)
        term2 = ops.multiply(coeff2, noise)
        result = ops.subtract(term1, term2)

        return result

    def q_posterior(self, x_start, x_t, t):
        """
        Computes the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0).

        :param x_start: Initial sample (x_0) for the posterior computation.
        :type x_start: TensorLike
        :param x_t: Sample at timestep `t`.
        :type x_t: TensorLike
        :param t: Current timestep.
        :type t: int

        :returns: Posterior mean, variance, and clipped log variance at the current timestep.
        :rtype: tuple
        """

        x_t_shape = ops.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t):
        """
        Predicts the mean and variance for the reverse diffusion step.

        :param pred_noise: Noise predicted by the diffusion model.
        :type pred_noise: TensorLike
        :param x: Samples at a given timestep for which the noise was predicted.
        :type x: TensorLike
        :param t: Current timestep.
        :type t: int

        :returns: Model mean, posterior variance, and posterior log variance tensors.
        :rtype: tuple
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t):
        """
        Generates a sample from the diffusion model by reversing the diffusion process.

        :param pred_noise: Noise predicted by the diffusion model.
        :type pred_noise: TensorLike
        :param x: Samples at a given timestep for which the noise was predicted.
        :type x: TensorLike
        :param t: Current timestep.
        :type t: int

        :returns: Sample generated by reversing the diffusion process at timestep `t`.
        :rtype: TensorLike
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t
        )
        noise = keras.random.normal(shape=ops.shape(x), dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = ops.reshape(
            1 - ops.cast(ops.equal(t, 0), "float32"), [ops.shape(x)[0], 1, 1]
        )
        return model_mean + nonzero_mask * ops.exp(0.5 * model_log_variance) * noise


class DDPM(keras.Model):
    """
    Denoising Diffusion Probabilistic Model.

    :param network: A Keras model that predicts the noise added to the images.
    :type network: keras.Model
    :param ema_network: EMA model, a clone of `network`.
    :type ema_network: keras.Model
    :param timesteps: The number of timesteps in the diffusion process.
    :type timesteps: int
    :param ema: The decay factor for the EMA, default is 0.999.
    :type ema: float
    """
    def __init__(self, network: keras.Model, ema_network: keras.Model, timesteps: int, ema: float = 0.999) -> None:
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = GaussianDiffusion(timesteps=timesteps)
        self.ema = ema

        self.ema_network.set_weights(network.get_weights())  # Initially the weights are the same
        # EMA parameters are updated by averaging, never by the optimizer.
        self.ema_network.trainable = False
        self.seed_generator = keras.random.SeedGenerator()
        self.loss_tracker = keras.metrics.Mean(name="loss")

        # Filled in during training
        self.seq_len = None
        self.feat_dim = None

    @property
    def metrics(self) -> T.List:
        return [self.loss_tracker]

    def _diffusion_loss(self, images: TensorLike) -> TensorLike:
        batch_size = ops.shape(images)[0]
        t = keras.random.randint(
            minval=0, maxval=self.timesteps, shape=(batch_size, 1),
            dtype="int32", seed=self.seed_generator,
        )
        noise = keras.random.normal(
            shape=ops.shape(images), dtype=images.dtype, seed=self.seed_generator,
        )
        images_t = self.gdf_util.q_sample(images, t, noise)
        pred_noise = self.network([images_t, t], training=True)
        return ops.mean(ops.square(noise - pred_noise))

    def _update_ema(self) -> None:
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

    def train_step_jax(self, jax, state: T.Tuple, data: TensorLike) -> T.Tuple[T.Dict, T.Tuple]:
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state
        images, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        self.seq_len, self.feat_dim = images.shape[1], images.shape[2]

        def loss_fn(trainable_values, non_trainable_values):
            state_mapping = list(zip(self.trainable_variables, trainable_values)) + \
                list(zip(self.non_trainable_variables, non_trainable_values))
            with keras.StatelessScope(state_mapping=state_mapping) as scope:
                loss = self._diffusion_loss(images)
            # Preserve random-generator state and updates such as batch-norm statistics.
            new_non_trainable_values = [
                scope.get_current_value(v) for v in self.non_trainable_variables
            ]
            return loss, new_non_trainable_values

        (loss, non_trainable_variables), gradients = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(trainable_variables, non_trainable_variables)
        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, gradients, trainable_variables,
        )

        # Average the updated network weights, including non-trainable statistics,
        # and return all changes through the JAX trainer's explicit state.
        state_mapping = list(zip(self.trainable_variables, trainable_variables)) + \
            list(zip(self.non_trainable_variables, non_trainable_variables)) + \
            list(zip(self.metrics_variables, metrics_variables))
        with keras.StatelessScope(state_mapping=state_mapping) as scope:
            self._update_ema()
            self.loss_tracker.update_state(loss)
            logs = {"loss": self.loss_tracker.result()}
        non_trainable_variables = [
            scope.get_current_value(v) for v in self.non_trainable_variables
        ]
        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]
        return logs, (
            trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables,
        )

    def train_step(self, *args) -> T.Any:
        """
        Performs a single training step on a batch of time series data.

        TensorFlow and PyTorch pass a batch of data. JAX passes ``(state, data)``
        and expects ``(logs, state)`` instead of a dictionary of losses.
        """
        if os.environ.get("KERAS_BACKEND") == "jax":
            return self.train_step_jax(get_backend(), *args)
        images, _, _ = keras.utils.unpack_x_y_sample_weight(args[0])
        self.seq_len, self.feat_dim = images.shape[1], images.shape[2]

        # Standard Keras 3 gradient computation approach
        if os.environ.get("KERAS_BACKEND") == "torch":
            # PyTorch backend approach (based on CVAE implementation)
            # Clear gradients
            for var in self.network.trainable_weights:
                if hasattr(var.value, 'grad') and var.value.grad is not None:
                    var.value.grad.zero_()

            loss = self._diffusion_loss(images)

            # Backward pass
            loss.backward(retain_graph=False)

            # Extract gradients
            gradients = [v.value.grad for v in self.network.trainable_weights]

            # Apply gradients using Keras optimizer
            with get_backend().no_grad():
                grads_and_vars = list(zip(gradients, self.network.trainable_weights))
                self.optimizer.apply_gradients(grads_and_vars)
        else:
            # TensorFlow backend approach
            backend = get_backend()
            with backend.GradientTape() as tape:
                loss = self._diffusion_loss(images)

            # Get the gradients
            gradients = tape.gradient(loss, self.network.trainable_weights)

            # Update the weights of the network
            self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # Update the EMA network after the optimizer step.
        self._update_ema()

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def generate(self, n_samples: int = 16) -> TensorLike:
        """
        Generates new samples by running the reverse diffusion process.

        :param n_samples: The number of samples to generate.
        :type n_samples: int

        :returns: Generated samples after running the reverse diffusion process.
        :rtype: TensorLike
        """

        if self.seq_len is None or self.feat_dim is None:
            raise ValueError("DDPM is not trained")

        # 1. Randomly sample noise (starting point for reverse process)
        samples = keras.random.normal(
            shape=(n_samples, self.seq_len, self.feat_dim), dtype="float32"
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = ops.cast(ops.full([n_samples], t), dtype="int32")
            tt = ops.expand_dims(tt, axis=-1)  # Match expected shape (batch_size, 1)

            # Ensure all tensors are on the same device (MPS compatibility)
            if os.environ.get("KERAS_BACKEND") == "torch":
                device = samples.device if hasattr(samples, 'device') else 'cpu'
                if hasattr(tt, 'device') and tt.device != device:
                    tt = tt.to(device)

            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=n_samples
            )

            # Convert pred_noise back to tensor if needed for device compatibility
            if os.environ.get("KERAS_BACKEND") == "torch":
                if isinstance(pred_noise, np.ndarray):
                    pred_noise = ops.convert_to_tensor(pred_noise, dtype="float32")
                if hasattr(pred_noise, 'device') and hasattr(samples, 'device') and pred_noise.device != samples.device:
                    pred_noise = pred_noise.to(samples.device)

            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt
            )
        # 3. Return generated samples
        return samples

    def call(self, n_samples) -> TensorLike:
        """
        Calls the generate method to produce samples.

        :param n_samples: The number of samples to generate or tensor for building.
        :type n_samples: int

        :returns: Generated samples or input tensor for building.
        :rtype: TensorLike
        """
        # Handle Keras 3 model building - if n_samples is a tensor, return it
        if hasattr(n_samples, 'shape'):
            return n_samples
        return self.generate(n_samples)
