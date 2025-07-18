"""
The implementation is based on Keras DDPM implementation: https://keras.io/examples/generative/ddpm/
"""
import numpy as np
import keras
from keras import ops
from tsgm.types import Tensor as TensorLike
from tsgm.backend import get_backend

import typing as T


class GaussianDiffusion:
    """Gaussian diffusion utility for generating samples using a diffusion process.

    This class implements a Gaussian diffusion process, where a sample is gradually
    perturbed by adding Gaussian noise over a series of timesteps. It also includes
    methods to reverse the diffusion process, predicting the original data from
    the noisy samples.

    Args:
        beta_start (float): Start value of the scheduled variance for the diffusion process.
        beta_end (float): End value of the scheduled variance for the diffusion process.
        timesteps (int): Number of timesteps in the forward process.
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
            dtype=np.float64,  # Using float64 for better precision
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

        Args:
            a: Tensor to extract from.
            t: Timestep for which the coefficients are to be extracted.
            x_shape: Shape of the current batched samples.

        Returns:
            Tensor reshaped to [batch_size, 1, 1] for broadcasting.
        """
        batch_size = x_shape[0]
        out = ops.take(a, t, axis=0)
        return ops.reshape(out, [batch_size, 1, 1])

    def q_mean_variance(self, x_start: TensorLike, t: float) -> T.Tuple:
        """Extracts the mean and variance at a specific timestep in the forward diffusion process.

        Args:
            x_start: Initial sample (before the first diffusion step).
            t: A timestep.

        Returns:
            mean, variance, log_variance: Tensors representing the mean, variance,
            and log variance of the distribution at `t`.
        """
        x_start_shape = ops.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start: TensorLike, t: float, noise: float) -> T.Tuple:
        """Performs the forward diffusion step by adding Gaussian noise to the sample.

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at timestep `t`

        Returns:
            Diffused samples at timestep `t`
        """
        x_start_shape = ops.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, ops.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t: TensorLike, t, noise):
        """Predicts the initial sample from the noisy sample at timestep `t`.

        Args:
            x_t: Noisy sample at timestep `t`.
            t: Current timestep.
            noise: Gaussian noise added at timestep `t`.

        Returns:
            Predicted initial sample.
        """

        x_t_shape = ops.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Computes the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Initial sample (x_0) for the posterior computation.
            x_t: Sample at timestep `t`.
            t: Current timestep.

        Returns:
            Posterior mean, variance, and clipped log variance at the current timestep.
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
        """Predicts the mean and variance for the reverse diffusion step.

        Args:
            pred_noise: Noise predicted by the diffusion model.
            x: Samples at a given timestep for which the noise was predicted.
            t: Current timestep.

        Returns:
            model_mean, posterior_variance, posterior_log_variance: Tensors
            representing the mean and variance of the model at the current timestep.
        """
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t):
        """Generates a sample from the diffusion model by reversing the diffusion process.

        Args:
            pred_noise: Noise predicted by the diffusion model.
            x: Samples at a given timestep for which the noise was predicted.
            t: Current timestep.

        Returns:
            Sample generated by reversing the diffusion process at timestep `t`.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t
        )
        noise = ops.random.normal(shape=ops.shape(x), dtype=x.dtype)
        # No noise when t == 0
        nonzero_mask = ops.reshape(
            1 - ops.cast(ops.equal(t, 0), "float32"), [ops.shape(x)[0], 1, 1]
        )
        return model_mean + nonzero_mask * ops.exp(0.5 * model_log_variance) * noise


class DDPM(keras.Model):
    """
    Denoising Diffusion Probabilistic Model

    Args:
        network (keras.Model): A Keras model that predicts the noise added to the images.
        ema_network (keras.Model): EMA model, a clone of `network`
        timesteps (int): The number of timesteps in the diffusion process.
        ema (float): The decay factor for the EMA, default is 0.999.
    """
    def __init__(self, network: keras.Model, ema_network: keras.Model, timesteps: int, ema: float = 0.999) -> None:
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = GaussianDiffusion(timesteps=timesteps)
        self.ema = ema

        self.ema_network.set_weights(network.get_weights())  # Initially the weights are the same

        # Filled in during training
        self.seq_len = None
        self.feat_dim = None

    def train_step(self, images: TensorLike) -> T.Dict:
        """
        Performs a single training step on a batch of images.

        Args:
            images: A batch of images to train on.

        Returns:
            A dictionary containing the loss value for the training step.
        """
        self.seq_len, self.feat_dim = images.shape[1], images.shape[2]

        # 1. Get the batch size
        batch_size = ops.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = ops.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype="int64"
        )

        with ops.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = ops.random.normal(shape=ops.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t], training=True)

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    def generate(self, n_samples: int = 16) -> TensorLike:
        """
        Generates new samples by running the reverse diffusion process.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            Generated samples after running the reverse diffusion process.
        """

        if self.seq_len is None or self.feat_dim is None:
            raise ValueError("DDPM is not trained")

        # 1. Randomly sample noise (starting point for reverse process)
        samples = ops.random.normal(
            shape=(n_samples, self.seq_len, self.feat_dim), dtype="float32"
        )
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = ops.cast(ops.full([n_samples], t), dtype="int64")
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=n_samples
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt
            )
        # 3. Return generated samples
        return samples

    def call(self, n_samples: int) -> TensorLike:
        """
        Calls the generate method to produce samples.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            Generated samples.
        """
        return self.generate(n_samples)
