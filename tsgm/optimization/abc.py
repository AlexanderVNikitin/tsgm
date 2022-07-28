import itertools
import typing
import tqdm

import numpy as np
import tensorflow_probability as tfp

import tsgm


DEFAULT_PRIOR = tfp.distributions.Normal(0, 1)


class ABCAlgorithm:
    """
    A base class for ABC algorithms.
    """
    def sample_parameters(self, n_samples: int) -> list:
        raise NotImplementedError


class RejectionSampler(ABCAlgorithm):
    """
    Rejection sampling algorithm for approximate Bayesian computation.
    """
    def __init__(self, simulator: tsgm.simulator.ModelBasedSimulator, data: tsgm.dataset.Dataset,
                 statistics: list, epsilon: float, discrepancy: typing.Callable, priors: dict = None,
                 **kwargs):
        """
        :param simulator: A model based simulator
        :type simulator: class `tsgm.simulator.ModelBasedSimulator`
        :param data: Historical dataset storage
        :type data: class `tsgm.dataset.Dataset`
        :param statistics: contains a list of summary statistics
        :type statistics: list
        :param epsilon: tolerance of synthetically generated data to a set of summary statistics
        :type epsilon: float
        :param discrepancy: discrepancy measure function
        :type discrepancy: Callable
        :param priors: set of priors for each of the simulator parametors, defaults to DEFAULT_PRIOR
        :type prior: dict
        """
        self._epsilon = epsilon
        self._simulator = simulator.clone()
        self._data = data
        self._statistics = statistics
        self._discrepancy = discrepancy
        self._priors = priors or dict()

        self._data_stats = self._calc_statistics(self._data)

    def _calc_statistics(self, data: tsgm.dataset.Dataset) -> tsgm.types.Tensor:
        # TODO measure both X & y
        return np.array(list(itertools.chain.from_iterable(s(data.X) for s in self._statistics)))

    def sample_parameters(self, n_samples: int) -> list:
        """
        Samples parameters from the rejection sampler.
        :param n_samples: Number of samples
        :type simulator: int
        ...
        ...
        :return: A list of samples. Each sample is represent as dict.
        :rtype: typing.List[typing.Dict]
        """
        cur_sim = self._simulator.clone()

        samples: typing.List[typing.Dict] = []
        for i in tqdm.tqdm(range(n_samples)):
            err, params = None, None
            while err is None or err > self._epsilon:
                params = prior_samples(self._priors, cur_sim.params())

                cur_sim.set_params(**params)
                sampled_data = cur_sim.generate(self._data.N)

                cur_stats = self._calc_statistics(sampled_data)
                err = self._discrepancy(self._data_stats, cur_stats)
            samples.append(params)
        return samples


def prior_samples(priors, params):
    samples = {}
    for var in params:
        distr = priors.get(var, DEFAULT_PRIOR)
        samples[var] = distr.sample()
    return samples
