import itertools
import typing as T
import tqdm

import numpy as np

import tsgm
from tsgm.backend import get_distributions

# Lazy loading of distributions
distributions = None


def _get_distributions():
    global distributions
    if distributions is None:
        distributions = get_distributions()
    return distributions


def _get_default_prior():
    return _get_distributions().Normal(0, 1)


class ABCAlgorithm:
    """
    A base class for ABC algorithms.
    """
    def sample_parameters(self, n_samples: int) -> T.List:
        raise NotImplementedError


class RejectionSampler(ABCAlgorithm):
    """
    Rejection sampling algorithm for approximate Bayesian computation.
    """
    def __init__(self, simulator: tsgm.simulator.ModelBasedSimulator, data: tsgm.dataset.Dataset,
                 statistics: T.List, epsilon: float, discrepancy: T.Callable, priors: T.Dict = None,
                 **kwargs) -> None:
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

    def sample_parameters(self, n_samples: int) -> T.List:
        """
        Samples parameters from the rejection sampler.

        :param n_samples: Number of samples
        :type simulator: int
        :returns: A list of samples. Each sample is represent as dict.
        :rtype: T.List[T.Dict]
        """
        cur_sim = self._simulator.clone()

        samples: T.List[T.Dict] = []
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


def prior_samples(priors: T.Dict, params: T.List) -> T.Dict:
    """
    Generate prior samples for the specified parameters.

    :param priors: A dictionary containing probability distributions for each parameter.
                   Keys are parameter names, and values are instances of probability distribution classes.
                   If a parameter is not present in the dictionary, a default prior distribution is used.
    :type priors: T.Dict

    :param params: A list of parameter names for which prior samples are to be generated.
    :type params: T.List

    :returns: A dictionary where keys are parameter names and values are samples drawn from their respective prior distributions.
    :rtype: T.Dict

    Example:

    .. code-block:: python

        priors = {'mean': NormalDistribution(0, 1), 'std_dev': UniformDistribution(0, 2)}
        params = ['mean', 'std_dev']
        samples = prior_samples(priors, params)

    """
    samples = {}
    for var in params:
        distr = priors.get(var, _get_default_prior())
        samples[var] = distr.sample()
    return samples
