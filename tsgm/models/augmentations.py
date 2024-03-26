import math
import numpy as np
import numpy.typing as npt
import random
import scipy.interpolate
from dtaidistance import dtw_barycenter
from typing import List, Dict, Any, Optional, Tuple, Union
from tensorflow.python.types.core import TensorLike

import logging


AugmentationOutput = Union[TensorLike, Tuple[TensorLike, TensorLike]]


logger = logging.getLogger("augmentations")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
        self,
        per_feature: bool,
    ) -> None:
        self.per_channel = per_feature

    def _get_seeds(self, total_num: int, n_seeds: int) -> TensorLike:
        seeds_idx = np.random.choice(range(total_num), size=n_seeds, replace=True)
        return seeds_idx

    def generate(
        self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1
    ) -> AugmentationOutput:
        raise NotImplementedError


class BaseCompose:
    def __init__(
        self,
        augmentations: List[BaseAugmenter],
    ) -> None:
        if isinstance(augmentations, (BaseCompose, BaseAugmenter)):
            augmentations = [augmentations]

        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.augmentations)

    def __call__(self, *args, **data) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> BaseAugmenter:
        return self.augmentations[item]


class GaussianNoise(BaseAugmenter):
    """Apply noise to the input time series.
    Args:
        variance ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_feature (bool): if set to True, noise will be sampled for each feature independently.
            Otherwise, the noise will be sampled once for all features. Default: True
    """

    def __init__(
        self,
        per_feature: bool = True,
    ) -> None:
        super(GaussianNoise, self).__init__(per_feature)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1,
                 mean: float = 0, variance: float = 1.0,) -> AugmentationOutput:
        """
        Generate synthetic data with Gaussian noise.

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: TensorLike

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[TensorLike]

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :param mean: The mean of the noise. Default is 0.
        :type mean: float

        :param variance: The variance of the noise. Default is 1.0.
        :type variance: float

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[TensorLike, Tuple[TensorLike, TensorLike]]
        """
        seeds_idx = self._get_seeds(total_num=X.shape[0], n_seeds=n_samples)

        sigma = variance**0.5
        has_labels = y is not None
        if self.per_channel:
            gauss = np.random.normal(
                mean, sigma, (n_samples, X.shape[1], X.shape[2])
            )
        else:
            gauss = np.random.normal(mean, sigma, (n_samples, X.shape[1]))
            gauss = np.expand_dims(gauss, -1)
        synthetic_X = X[seeds_idx] + gauss
        if has_labels:
            synthetic_y = y[seeds_idx]
            return np.array(synthetic_X), np.array(synthetic_y)
        else:
            return np.array(synthetic_X)


class SliceAndShuffle(BaseAugmenter):
    """Slice the time series in k pieces and create a new time series by shuffling.
    Args:
        per_feature (bool): if set to True, each time series is sliced independently.
            Otherwise, all features are sliced in the same way. Default: True
    """

    def __init__(
        self,
        per_feature: bool = False,
    ) -> None:
        super(SliceAndShuffle, self).__init__(per_feature)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1, n_segments: int = 2) -> AugmentationOutput:
        """
        Generate synthetic data using Slice-And-Shuffle strategy. Slices are randomly selected.

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: TensorLike

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[TensorLike]

        :param n_segments: The number of slices, default is 2.
        :type n_segments: int

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[TensorLike, Tuple[TensorLike, TensorLike]]
        """
        assert 0 < n_segments <= X.shape[1]

        seeds_idx = self._get_seeds(total_num=X.shape[0], n_seeds=n_samples)

        synthetic_data = []
        has_labels = y is not None
        if has_labels:
            new_labels = []
        for i in seeds_idx:
            sequence = X[i]
            if self.per_channel:
                raise NotImplementedError(
                    "SliceAndShuffle separately by feature is not supported yet."
                )
            else:
                # Randomly pick n_segments-1 points where to slice
                idxs = np.random.randint(0, sequence.shape[0], size=n_segments - 1)
                slices = []
                start_idx = 0
                for j in sorted(idxs):
                    s = sequence[start_idx:j]
                    start_idx = j
                    slices.append(s)
                slices.append(sequence[start_idx:])
                np.random.shuffle(slices)
            # concatenate the slices
            sequence = np.concatenate(slices)
            synthetic_data.append(sequence)
            if has_labels:
                new_labels.append(y[i])
        if has_labels:
            return np.array(synthetic_data), np.array(new_labels)
        else:
            return np.array(synthetic_data)


class Shuffle(BaseAugmenter):
    """
    Shuffles time series features.
    Shuffling is beneficial when each feature corresponds to interchangeable sensors.
    """

    def __init__(self) -> None:
        super(Shuffle, self).__init__(per_feature=False)

    def _n_repeats(self, n: int, total_num: int) -> int:
        return math.ceil(n / total_num)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> AugmentationOutput:
        """
        Generate synthetic data using Shuffle strategy.
        Features are randomly shuffled to generate novel samples.

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: TensorLike

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[TensorLike]

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[TensorLike, Tuple[TensorLike, TensorLike]]
        """
        seeds_idx = self._get_seeds(total_num=X.shape[0], n_seeds=n_samples)
        n_features = X.shape[2]
        n_repeats = self._n_repeats(n_samples, total_num=len(X))
        shuffle_ids = [
            np.random.choice(np.arange(n_features), n_features, replace=False)
            for _ in range(n_repeats)
        ]

        synthetic_data = []
        has_labels = y is not None
        if has_labels:
            new_labels = []
        for num, i in enumerate(seeds_idx):
            sequence = X[i]
            id_repeat = self._n_repeats(num + 1, total_num=len(X))
            synthetic_data.append(sequence[:, shuffle_ids[id_repeat - 1]])
            if has_labels:
                new_labels.append(y[i])
        if has_labels:
            return np.array(synthetic_data), np.array(new_labels)
        else:
            return np.array(synthetic_data)


class MagnitudeWarping(BaseAugmenter):
    """
    Magnitude warping changes the magnitude of each
    sample by convolving the data window with a smooth curve varying around one
    https://dl.acm.org/doi/pdf/10.1145/3136755.3136817
    """

    def __init__(self) -> None:
        super(MagnitudeWarping, self).__init__(per_feature=False)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1, sigma: float = 0.2, n_knots: int = 4) -> AugmentationOutput:
        """
        Generates augmented samples via MagnitudeWarping for (X, y)

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: TensorLike

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[TensorLike]

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :param sigma: Standard deviation for the random warping. Default is 0.2.
        :type sigma: float

        :param n_knots: Number of knots used for warping curve. Default is 4.
        :type n_knots: int

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[TensorLike, Tuple[TensorLike, TensorLike]]
        """
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        orig_steps = np.arange(n_timesteps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, n_knots + 2, n_features))
        warp_steps = (np.ones(
            (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=n_knots + 2))).T

        result = np.zeros((n_samples, n_timesteps, n_features))
        has_labels = y is not None

        if has_labels:
            result_y = np.zeros((n_samples, 1))

        for i in range(n_samples):
            random_sample_id = random.randint(0, n_data - 1)
            warper = np.array(
                [
                    scipy.interpolate.CubicSpline(
                        warp_steps[:, dim], random_warps[i, :, dim]
                    )(orig_steps)
                    for dim in range(n_features)
                ]
            ).T
            result[i] = X[random_sample_id] * warper
            if has_labels:
                result_y[i] = y[random_sample_id]
        if has_labels:
            return result, result_y
        else:
            return result


class WindowWarping(BaseAugmenter):
    """
    https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """

    def __init__(self) -> None:
        super(WindowWarping, self).__init__(per_feature=False)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, window_ratio: float = 0.2, scales: Tuple = (0.25, 1.0), n_samples: int = 1) -> AugmentationOutput:
        """
        Generates augmented samples via MagnitudeWarping for (X, y)

        :param X: Input data tensor of shape (n_data, n_timesteps, n_features).
        :type X: TensorLike

        :param y: Optional labels tensor. If provided, labels will also be returned
        :type y: Optional[TensorLike]

        :param window_ratio: The ratio of the window size relative to the total number of timesteps.
            Default is 0.2.
        :type window_ratio: float

        :param scale: A tuple specifying the scale range for warping.
            Default is (0.25, 1.0).
        :type scale: tuple

        :param n_samples: Number of augmented samples to generate. Default is 1.
        :type n_samples: int

        :return: Augmented data tensor of shape (n_samples, n_timesteps, n_features) and optionally augmented labels if 'y' is provided.
        :rtype: Union[TensorLike, Tuple[TensorLike, TensorLike]]
        """
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        scales_per_sample = np.random.choice(scales, n_samples)
        warp_size = max(np.round(window_ratio * n_timesteps).astype(np.int64), 1)

        result = np.zeros((n_samples, n_timesteps, n_features))
        result_y = np.zeros((n_samples, 1))
        has_labels = y is not None
        for i in range(n_samples):
            window_starts = np.random.randint(
                low=0, high=n_timesteps - warp_size,
                size=(n_samples))
            window_ends = window_starts + warp_size
            random_sample_id = random.randint(0, n_data - 1)
            random_sample = X[random_sample_id]

            for dim in range(n_features):
                start_seg = random_sample[:window_starts[i], dim]
                warp_ts_size = max(round(warp_size * scales_per_sample[i]), 1)
                window_seg = np.interp(
                    x=np.linspace(0, warp_size - 1, num=warp_ts_size),
                    xp=np.arange(warp_size),
                    fp=random_sample[window_starts[i] : window_ends[i], dim],
                )
                end_seg = random_sample[window_ends[i] :, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                result[i, :, dim] = np.interp(
                    np.arange(n_timesteps),
                    np.linspace(0, n_timesteps - 1.0, num=warped.size),
                    warped,
                ).T
                if has_labels:
                    result_y[i] = y[random_sample_id]

        if has_labels:
            return result, result_y
        else:
            return result


class DTWBarycentricAveraging(BaseAugmenter):
    """
    DTW Barycenter Averaging (DBA) [1] method estimated through
        Expectation-Maximization algorithm [2] as in https://github.com/tslearn-team/tslearn/
    ----------
    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """

    def __init__(self):
        super(DTWBarycentricAveraging, self).__init__(per_feature=False)

    def generate(
        self,
        X: TensorLike,
        y: Optional[TensorLike] = None,
        n_samples: int = 1,
        num_initial_samples: Optional[int] = None,
        initial_timeseries: Optional[List[TensorLike]] = None,
        initial_labels: Optional[List[int]] = None,
        **kwargs,
    ) -> AugmentationOutput:
        """
        Parameters
        ----------
        X : TensorLike, the timeseries dataset
        y : TensorLike or None, the classes
        n_samples : int, number of samples to generate (per class, if y is given)
        num_initial_samples : int or None (default: None)
            The number of timeseries to draw (per class) from the dataset before computing DTW_BA.
            If None, use the entire set (per class).
        initial_timeseries : array or None (default: None)
            Initial timesteries to start from for the optimization process, with shape (original_size, d).
            In case y is given, the shape of initial_timeseries is assumed to be (n_classes, original_size, d)
        initial_labels: array or None (default: None)
            Labels for samples from `initial_timeseries`
        Returns
        -------
        np.array of shape (n_samples, original_size, d) if y is None
            or (n_classes * n_samples, original_size, d),
            and np.array of labels (or None)
        """
        assert initial_timeseries is None or len(initial_timeseries) == n_samples
        has_labels = y is not None

        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(y, list):
            y = np.asarray(y)

        random_samples = random.choices(range(X.shape[0]), k=n_samples)
        if initial_timeseries is None:
            initial_timeseries = X[random_samples]
        if has_labels:
            if initial_labels is None:
                initial_labels = y[random_samples]

            y_new = []
            X_new = []
            unique_labels = np.unique(initial_labels)
            for i, label in enumerate(unique_labels):
                logger.debug(f"DTWBA Class {label}...")
                cur_initial_timeseries = initial_timeseries[np.ravel(initial_labels) == label]
                n_samples_per_label = len(cur_initial_timeseries)
                X_class = X[np.ravel(y) == label]
                y_new += [label] * n_samples_per_label
                X_new.append(
                    self._dtwba(
                        X_subset=X_class,
                        n_samples=n_samples_per_label,
                        num_initial_samples=num_initial_samples,
                        initial_timeseries=cur_initial_timeseries,
                        **kwargs,
                    )
                )
            return np.concatenate(X_new), np.array(y_new).reshape(-1, 1)
        else:
            return self._dtwba(
                X_subset=X,
                n_samples=n_samples,
                num_initial_samples=num_initial_samples,
                initial_timeseries=initial_timeseries,
                **kwargs,
            )

    def _dtwba(
        self,
        X_subset: TensorLike,
        n_samples: int,
        num_initial_samples: Optional[int],
        initial_timeseries: Optional[TensorLike],
        **kwargs,
    ) -> npt.NDArray:
        samples = []
        for i, st in enumerate(initial_timeseries):
            samples.append(
                dtw_barycenter.dba(
                    s=X_subset,
                    c=st,
                    nb_initial_samples=num_initial_samples,
                    # TODO: use_c=True,
                    **kwargs,
                )
            )
        return np.array(samples)
