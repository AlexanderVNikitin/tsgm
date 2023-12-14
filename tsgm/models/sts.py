import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow_probability import sts

import tsgm


DEFAULT_TREND = sts.LocalLinearTrend()
DEFAULT_SEASONAL = tfp.sts.Seasonal(num_seasons=12)
DEFAULT_MODEL = sts.Sum([DEFAULT_TREND, DEFAULT_SEASONAL])


class STS:
    def __init__(self, model=None):
        self._model = model or DEFAULT_MODEL
        self._dist = None
        self._elbo_loss = None

    def train(self, ds: tsgm.dataset.Dataset, num_variational_steps: int = 200,
              steps_forw: int = 10) -> None:
        assert ds.shape[0] == 1  # now works only with 1 TS
        X = ds.X.astype(np.float32)
        variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
            model=self._model)

        self._elbo_loss = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self._model.joint_distribution(observed_time_series=X).log_prob,
            surrogate_posterior=variational_posteriors,
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            num_steps=num_variational_steps,
            jit_compile=True)

        q_samples = variational_posteriors.sample(50)

        self._dist = tfp.sts.forecast(
            self._model, observed_time_series=X,
            parameter_samples=q_samples, num_steps_forecast=steps_forw)

    def elbo_loss(self) -> float:
        return self._elbo_loss

    def generate(self, num_samples: int) -> tsgm.types.Tensor:
        assert self._dist is not None

        return self._dist.sample(num_samples).numpy()[..., 0]
