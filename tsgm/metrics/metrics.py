import abc
import typing
import logging
import numpy as np
import itertools
import sklearn
from tqdm import tqdm

import tsgm

logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)


DEFAULT_SPLIT_STRATEGY = sklearn.model_selection.KFold(
    n_splits=3, random_state=42, shuffle=True)


def _dataset_or_tensor_to_tensor(D1):
    if isinstance(D1, tsgm.dataset.Dataset):
        return D1.X
    else:
        return D1


class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


class SimilarityMetric(Metric):
    """
    Metric that measures similarity between synthetic and real time series
    """
    def __init__(self, statistics: list, discrepancy: typing.Callable):
        """
        :param statistics: A list of summary statistics (callable)
        :type statistics: list
        :param discrepancy: Discrepancy function, measures the distance between the vectors of summary statistics.
        :type discrepancy: typing.Callable
        """
        self._statistics = statistics
        self._discrepancy = discrepancy

    def stats(self, X: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        :param X: A time series dataset.
        :type X: tsgm.types.Tensor.

        :returns: a tensor with calculated summary statistics.
        """
        return np.array(list(itertools.chain.from_iterable(s(X) for s in self._statistics))) if X is not None else None

    def discrepancy(self, stats1: tsgm.types.Tensor, stats2: tsgm.types.Tensor) -> float:
        """
        :param stats1: A vector of summary statistics.
        :type stats1: tsgm.types.Tensor.
        :param stats2: A vector of summary statistics.
        :type stats2: tsgm.types.Tensor.

        :returns: the distance between two vectors calculated by self._discrepancy.
        """
        return self._discrepancy(stats1, stats2)

    def __call__(self, D1: tsgm.dataset.DatasetOrTensor, D2: tsgm.dataset.DatasetOrTensor) -> float:
        """
        :param D1: A time series dataset.
        :type D1: tsgm.dataset.DatasetOrTensor.
        :param D2: A time series dataset.
        :type D2: tsgm.dataset.DatasetOrTensor.

        :returns: similarity metric between D1 & D2.
        """

        #  TODO: check compatibility of this metric in different versions of python
        #  typing.get_args() can be used instead
        #  assert isinstance(D1, tsgm.dataset.Dataset) and isinstance(D2, tsgm.dataset.Dataset) or\
        #      isinstance(D1, tsgm.types.Tensor.__args__) and isinstance(D2, tsgm.types.Tensor.__args__)
        if isinstance(D1, tsgm.dataset.Dataset) and isinstance(D2, tsgm.dataset.Dataset):
            X1, X2 = D1.Xy_concat, D2.Xy_concat
        else:
            X1, X2 = D1, D2

        stats1, stats2 = self.stats(X1), self.stats(X2)

        return self.discrepancy(stats1, stats2)


class ConsistencyMetric(Metric):
    """
    Predictive consistency metric measures whether a set of evaluators yield consistent results on real and synthetic data.
    """
    def __init__(self, evaluators: list):
        """
        :param evaluators: A list of evaluators (each item should implement method `.evaluate(D)`)
        :type evaluators: list
        """
        self._evaluators = evaluators

    def _apply_models(self, D: tsgm.dataset.DatasetOrTensor, D_test: tsgm.dataset.DatasetOrTensor) -> list:
        return [e.evaluate(D, D_test) for e in self._evaluators]

    def __call__(self, D1: tsgm.dataset.DatasetOrTensor, D2: tsgm.dataset.DatasetOrTensor, D_test: tsgm.dataset.DatasetOrTensor) -> float:
        """
        :param D1: A time series dataset.
        :type D1: tsgm.dataset.DatasetOrTensor.
        :param D2: A time series dataset.
        :type D2: tsgm.dataset.DatasetOrTensor.

        :returns: consistency metric between D1 & D2.
        """
        evaluations1 = self._apply_models(D1, D_test)
        evaluations2 = self._apply_models(D2, D_test)
        consistencies_cnt = 0
        n_evals = len(evaluations1)
        for i1 in tqdm(range(n_evals)):
            for i2 in range(i1 + 1, n_evals):
                if evaluations1[i1] > evaluations1[i2] and evaluations2[i1] > evaluations2[i2] or \
                        evaluations1[i1] < evaluations1[i2] and evaluations2[i1] < evaluations2[i2] or \
                        evaluations1[i1] == evaluations1[i2] and evaluations2[i1] == evaluations2[i2]:
                    consistencies_cnt += 1

        total_pairs = n_evals * (n_evals - 1) / 2.0
        return consistencies_cnt / total_pairs


class BaseDownstreamEvaluator(abc.ABC):
    def evaluate(self, *args, **kwargs):
        pass


class DownstreamPerformanceMetric(Metric):
    """
    The downstream performance metric evaluates the performance of a model on a downstream task.
    It returns performance gains achieved with the addition of synthetic data.
    """
    def __init__(self, evaluator: BaseDownstreamEvaluator):
        """
        :param evaluator: An evaluator,  should implement method `.evaluate(D)`
        :type evaluator: BaseDownstreamEvaluator
        """
        self._evaluator = evaluator

    def __call__(self, D1: tsgm.dataset.DatasetOrTensor, D2: tsgm.dataset.DatasetOrTensor, D_test: typing.Optional[tsgm.dataset.DatasetOrTensor], return_std=False) -> float:
        """
        :param D1: A time series dataset.
        :type D1: tsgm.dataset.DatasetOrTensor.
        :param D2: A time series dataset.
        :type D2: tsgm.dataset.DatasetOrTensor.

        :returns: downstream performance metric between D1 & D2.
        """
        if isinstance(D1, tsgm.dataset.Dataset):
            D1D2 = D1 | D2
        else:
            if isinstance(D2, tsgm.dataset.Dataset):
                D1D2 = np.concatenate((D1, D2.X))
            else:
                D1D2 = np.concatenate((D1, D2))
        evaluations1 = self._evaluator.evaluate(D1, D_test)
        evaluations2 = self._evaluator.evaluate(D1D2, D_test)
        if return_std:
            diff = evaluations2 - evaluations1
            return np.mean(diff), np.std(diff)
        else:
            return np.mean(evaluations2 - evaluations1)


class PrivacyMembershipInferenceMetric(Metric):
    """
    The metric that measures the possibility of membership inference attacks.
    """
    def __init__(self, attacker: typing.Any, metric: typing.Callable = None):
        """
        :param attacker: An attacker, one class classififier (OCC) that implements methods `.fit` and `.predict`
        :type attacker: typing.Any
        :param metric: Measures quality of attacker (precision by default)
        :type attacker: typing.Callable
        """
        self._attacker = attacker
        self._metric = metric or sklearn.metrics.precision_score

    def __call__(self, d_tr: tsgm.dataset.Dataset, d_syn: tsgm.dataset.Dataset, d_test: tsgm.dataset.Dataset) -> float:
        """
        :param d_tr: Training dataset (the dataset that was used to produce `d_dyn`).
        :type d_tr: tsgm.dataset.DatasetOrTensor.
        :param d_syn: Training dataset (the dataset that was used to produce `d_dyn`).
        :type d_syn: tsgm.dataset.DatasetOrTensor.
        :param d_test: Training dataset (the dataset that was used to produce `d_dyn`).
        :type d_test: tsgm.dataset.DatasetOrTensor.

        :returns: how well the attacker can distinguish `d_tr` & `d_test` when it is trained on `d_syn`.
        """
        self._attacker.fit(d_syn.Xy_concat)
        labels = self._attacker.predict((d_tr + d_test).Xy_concat)
        correct_labels = [1] * len(d_tr) + [-1] * len(d_test)
        return 1 - self._metric(labels, correct_labels)


class MMDMetric(Metric):
    """
    This metric calculated MMD between real and synthetic samples
    """

    def __call__(self, D1: tsgm.dataset.DatasetOrTensor, D2: tsgm.dataset.DatasetOrTensor) -> float:
        if isinstance(D1, tsgm.dataset.Dataset) and D1.y is not None or isinstance(D2, tsgm.dataset.Dataset) and D2.y is not None:
            logger.warning("It is currently impossible to run MMD for labeled time series. Labels will be ignored!")
        X1, X2 = _dataset_or_tensor_to_tensor(D1), _dataset_or_tensor_to_tensor(D2)
        return tsgm.utils.mmd.MMD(X1, X2)
