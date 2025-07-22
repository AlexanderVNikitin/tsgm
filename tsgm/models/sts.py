import tsgm
import keras
import numpy as np


try:
    import tensorflow_probability as tfp
    from tensorflow_probability import sts

    DEFAULT_TREND = sts.LocalLinearTrend()
    DEFAULT_SEASONAL = tfp.sts.Seasonal(num_seasons=12)
    DEFAULT_MODEL = sts.Sum([DEFAULT_TREND, DEFAULT_SEASONAL])
    has_tensorflow = True

    class STSTensorFlow():
        """
        Class for training and generating from a structural time series model.
        """

        def __init__(self, model: tfp.sts.StructuralTimeSeries = None) -> None:
            """
            Initializes a new instance of the STS class.

            :param model: Structural time series model to use. If None, default model is used.
            :type model: tfp.sts.StructuralTimeSeriesModel or None
            """
            self._model = model or DEFAULT_MODEL
            self._dist = None
            self._elbo_loss = None

        def train(self, ds: tsgm.dataset.Dataset, num_variational_steps: int = 200,
                  steps_forw: int = 10) -> None:
            """
            Trains the structural time series model.

            :param ds: Dataset containing time series data.
            :type ds: tsgm.dataset.Dataset
            :param num_variational_steps: Number of variational optimization steps, defaults to 200.
            :type num_variational_steps: int
            :param steps_forw: Number of steps to forecast, defaults to 10.
            :type steps_forw: int
            """
            assert ds.shape[0] == 1  # now works only with 1 TS
            X = ds.X.astype(np.float32)
            variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
                model=self._model)

            self._elbo_loss = tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=self._model.joint_distribution(observed_time_series=X).log_prob,
                surrogate_posterior=variational_posteriors,
                optimizer=keras.optimizers.Adam(learning_rate=0.1),
                num_steps=num_variational_steps,
                jit_compile=True)

            q_samples = variational_posteriors.sample(50)

            self._dist = tfp.sts.forecast(
                self._model, observed_time_series=X,
                parameter_samples=q_samples, num_steps_forecast=steps_forw)

        def elbo_loss(self) -> float:
            """
            Returns the evidence lower bound (ELBO) loss from training.

            :returns: The value of the ELBO loss.
            :rtype: float
            """
            return self._elbo_loss

        def generate(self, num_samples: int) -> tsgm.types.Tensor:
            """
            Generates samples from the trained model.

            :param num_samples: Number of samples to generate.
            :type num_samples: int

            :returns: Generated samples.
            :rtype: tsgm.types.Tensor
            """
            assert self._dist is not None

            return self._dist.sample(num_samples).numpy()[..., 0]
except ImportError:
    has_tensorflow = False
    print("TensorFlow not installed. STS model is not available.")

    class STSTorch():
        def __init__(self, *args, **kwargs):
            raise EnvironmentError("This is the PyTorch environment. STS is only available in TensorFlow backend.")

# Dynamically select the appropriate STS class
if has_tensorflow:
    STS = STSTensorFlow
else:
    STS = STSTorch
