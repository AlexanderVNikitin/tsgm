import abc
import antropy
import typing as T
import logging
import numpy as np
import itertools
import sklearn
import scipy
from tqdm import tqdm
from tensorflow.python.types.core import TensorLike

import tsgm

logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)


DEFAULT_SPLIT_STRATEGY = sklearn.model_selection.KFold(
    n_splits=3, random_state=42, shuffle=True)


def _dataset_or_tensor_to_tensor(D1: tsgm.dataset.DatasetOrTensor) -> tsgm.types.Tensor:
    if isinstance(D1, tsgm.dataset.Dataset):
        return D1.X
    else:
        return D1


class Metric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass


class DistanceMetric(Metric):
    """
    Metric that measures similarity between synthetic and real time series
    """
    def __init__(self, statistics: list, discrepancy: T.Callable) -> None:
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
    def __init__(self, evaluators: T.List) -> None:
        """
        :param evaluators: A list of evaluators (each item should implement method `.evaluate(D)`)
        :type evaluators: list
        """
        self._evaluators = evaluators

    def _apply_models(self, D: tsgm.dataset.DatasetOrTensor, D_test: tsgm.dataset.DatasetOrTensor) -> T.List:
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
    def __init__(self, evaluator: BaseDownstreamEvaluator) -> None:
        """
        :param evaluator: An evaluator,  should implement method `.evaluate(D)`
        :type evaluator: BaseDownstreamEvaluator
        """
        self._evaluator = evaluator

    def __call__(self, D1: tsgm.dataset.DatasetOrTensor, D2: tsgm.dataset.DatasetOrTensor, D_test: T.Optional[tsgm.dataset.DatasetOrTensor], return_std: bool = False) -> float:
        """
        :param D1: A time series dataset.
        :type D1: tsgm.dataset.DatasetOrTensor.
        :param D2: A time series dataset.
        :type D2: tsgm.dataset.DatasetOrTensor.

        :returns: downstream performance metric between D1 & D2.
        """
        if isinstance(D1, tsgm.dataset.Dataset) and isinstance(D2, tsgm.dataset.Dataset):
            D1D2 = D1 | D2
        else:
            if isinstance(D1, tsgm.dataset.Dataset):
                D1D2 = np.concatenate((D1.X, D2))
            elif isinstance(D2, tsgm.dataset.Dataset):
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
    def __init__(self, attacker: T.Any, metric: T.Optional[T.Callable] = None) -> None:
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

    Args:
        d (tsgm.dataset.DatasetOrTensor): The input dataset or tensor.

    Returns:
        float: The computed spectral entropy.

    Example:
        >>> metric = MMDMetric(kernel)
        >>> dataset, synth_dataset = tsgm.dataset.Dataset(...), tsgm.dataset.Dataset(...)
        >>> result = metric(dataset)
        >>> print(result)
    """
    def __init__(self, kernel: T.Callable = tsgm.utils.mmd.exp_quad_kernel) -> None:
        self.kernel = kernel

    def __call__(self, D1: tsgm.dataset.DatasetOrTensor, D2: tsgm.dataset.DatasetOrTensor) -> float:
        if isinstance(D1, tsgm.dataset.Dataset) and D1.y is not None or isinstance(D2, tsgm.dataset.Dataset) and D2.y is not None:
            logger.warning("It is currently impossible to run MMD for labeled time series. Labels will be ignored!")
        X1, X2 = _dataset_or_tensor_to_tensor(D1), _dataset_or_tensor_to_tensor(D2)
        return tsgm.utils.mmd.MMD(X1, X2, kernel=self.kernel)


class DiscriminativeMetric(Metric):
    """
    The DiscriminativeMetric measures the discriminative performance of a model in distinguishing
    between synthetic and real datasets.

    This metric evaluates a discriminative model by training it on a combination of synthetic
    and real datasets and assessing its performance on a test set.

    :param d_hist: Real dataset.
    :type d_hist: tsgm.dataset.DatasetOrTensor
    :param d_syn: Synthetic dataset.
    :type d_syn: tsgm.dataset.DatasetOrTensor
    :param model: Discriminative model to be evaluated.
    :type model: T.Callable
    :param test_size: Proportion of the dataset to include in the test split
                     or the absolute number of test samples.
    :type test_size: T.Union[float, int]
    :param n_epochs: Number of training epochs for the model.
    :type n_epochs: int
    :param metric: Optional evaluation metric to use (default: accuracy).
    :type metric: T.Optional[T.Callable]
    :param random_seed: Optional random seed for reproducibility.
    :type random_seed: T.Optional[int]

    :return: Discriminative performance metric.
    :rtype: float

    Example:
    --------
    >>> from my_module import DiscriminativeMetric, MyDiscriminativeModel
    >>> import tsgm.dataset
    >>> import numpy as np
    >>> import sklearn
    >>>
    >>> # Create real and synthetic datasets
    >>> real_dataset = tsgm.dataset.Dataset(...)  # Replace ... with appropriate arguments
    >>> synthetic_dataset = tsgm.dataset.Dataset(...)  # Replace ... with appropriate arguments
    >>>
    >>> # Create a discriminative model
    >>> model = MyDiscriminativeModel()  # Replace with the actual discriminative model class
    >>>
    >>> # Create and use the DiscriminativeMetric
    >>> metric = DiscriminativeMetric()
    >>> result = metric(real_dataset, synthetic_dataset, model, test_size=0.2, n_epochs=10)
    >>> print(result)
    """
    def __call__(self, d_hist: tsgm.dataset.DatasetOrTensor, d_syn: tsgm.dataset.DatasetOrTensor, model: T.Callable, test_size: T.Union[float, int], n_epochs: int, metric: T.Optional[T.Callable] = None, random_seed: T.Optional[int] = None) -> float:
        X_hist, X_syn = _dataset_or_tensor_to_tensor(d_hist), _dataset_or_tensor_to_tensor(d_syn)
        X_all, y_all = np.concatenate([X_hist, X_syn]), np.concatenate([[1] * len(d_hist), [0] * len(d_syn)])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_all, y_all, test_size=test_size, random_state=random_seed)
        model.fit(X_train, y_train, epochs=n_epochs)
        pred = model.predict(X_test)
        # check the shape, 1D array or N-D arrary
        if len(pred.shape) == 1:  # binary classification with sigmoid activation
            y_pred = (pred > 0.5).astype(int)
        else:  # multiple classification with softmax activation
            y_pred = np.argmax(pred, axis=-1).astype(int)
        if metric is None:
            return sklearn.metrics.accuracy_score(y_test, y_pred)
        else:
            return metric(y_test, y_pred)


def _spectral_entropy_per_feature(X: TensorLike) -> TensorLike:
    return antropy.spectral_entropy(X.ravel(), sf=1, method='welch', normalize=True)


def _spectral_entropy_per_sample(X: TensorLike) -> TensorLike:
    if len(X.shape) == 1:
        X = X[:, None]
    return np.apply_along_axis(_spectral_entropy_per_feature, 0, X)


def _spectral_entropy_sum(X: TensorLike) -> TensorLike:
    return np.apply_along_axis(_spectral_entropy_per_sample, 1, X)


class EntropyMetric(Metric):
    """
    Calculates the spectral entropy of a dataset or tensor.

    This metric measures the randomness or disorder in a dataset or tensor
    using spectral entropy, which is a measure of the distribution of energy
    in the frequency domain.

    Args:
        d (tsgm.dataset.DatasetOrTensor): The input dataset or tensor.

    Returns:
        float: The computed spectral entropy.

    Example:
        >>> metric = EntropyMetric()
        >>> dataset = tsgm.dataset.Dataset(...)
        >>> result = metric(dataset)
        >>> print(result)
    """
    def __call__(self, d: tsgm.dataset.DatasetOrTensor) -> float:
        """
        Calculate the spectral entropy of the input dataset or tensor.

        Args:
            d (tsgm.dataset.DatasetOrTensor): The input dataset or tensor.

        Returns:
            float: The computed spectral entropy.
        """
        X = _dataset_or_tensor_to_tensor(d)
        return np.sum(_spectral_entropy_sum(X), axis=None)


class DemographicParityMetric(Metric):
    """
    Measuring demographic parity between two datasets.

    This metric assesses the disparity in the distributions of a target variable among different groups in two datasets.
    By default, it uses the Kolmogorov-Smirnov statistic to quantify the maximum vertical deviation between the cumulative distribution functions
    of the target variable for the historical and synthetic data within each group.

    Args:
        d_hist (tsgm.dataset.DatasetOrTensor): The historical input dataset or tensor.
        groups_hist (TensorLike): The group assignments for the historical data.
        d_synth (tsgm.dataset.DatasetOrTensor): The synthetic input dataset or tensor.
        groups_synth (TensorLike): The group assignments for the synthetic data.
        metric (callable, optional): The metric used to compare the target variable distributions within each group.
            Default is the Kolmogorov-Smirnov statistic.

    Returns:
        dict: A dictionary mapping each group to the computed demographic parity metric.

    Example:
        >>> metric = DemographicParityMetric()
        >>> dataset_hist = tsgm.dataset.Dataset(...)
        >>> dataset_synth = tsgm.dataset.Dataset(...)
        >>> groups_hist = [0, 1, 0, 1, 1, 0]
        >>> groups_synth = [1, 1, 0, 0, 0, 1]
        >>> result = metric(dataset_hist, groups_hist, dataset_synth, groups_synth)
        >>> print(result)
    """

    _DEFAULT_KS_METRIC = lambda data1, data2: scipy.stats.ks_2samp(data1, data2).statistic  # noqa: E731

    def __call__(self, d_hist: tsgm.dataset.DatasetOrTensor, groups_hist: TensorLike, d_synth: tsgm.dataset.DatasetOrTensor, groups_synth: TensorLike, metric: T.Callable = _DEFAULT_KS_METRIC) -> T.Dict:
        """
        Calculate the demographic parity metric for the input datasets.

        Args:
            d_hist (tsgm.dataset.DatasetOrTensor): The historical input dataset or tensor.
            groups_hist (TensorLike): The group assignments for the historical data.
            d_synth (tsgm.dataset.DatasetOrTensor): The synthetic input dataset or tensor.
            groups_synth (TensorLike): The group assignments for the synthetic data.
            metric (callable, optional): The metric used to compare the target variable distributions within each group.
                Default is the Kolmogorov-Smirnov statistic.

        Returns:
            dict: A dictionary mapping each group to the computed demographic parity metric.
        """

        y_hist, y_synth = d_hist.y, d_synth.y

        unique_groups_hist, unique_groups_synth = set(groups_hist), set(groups_synth)
        all_groups = unique_groups_hist | unique_groups_synth
        if len(all_groups) > len(unique_groups_hist) or len(all_groups) > len(unique_groups_synth):
            logger.warning("Groups in historical and synthetic data are not entirely identical.")

        result = {}
        for g in all_groups:
            y_g_hist, y_g_synth = y_hist[groups_hist == g], y_synth[groups_synth == g]
            if not len(y_g_synth):
                result[g] = np.inf
            elif not len(y_g_hist):
                result[g] = -np.inf
            else:
                result[g] = metric(y_g_hist, y_g_synth)
        return result
