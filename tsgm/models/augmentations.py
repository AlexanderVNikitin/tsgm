import math
import numpy as np
import random
import scipy.interpolate
from typing import List, Dict, Any, Optional
from tensorflow.python.types.core import TensorLike

import logging

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
        self,
        per_feature: bool,
    ):
        self.per_channel = per_feature
        self._data = None
        self._targets = None
        return

    def _get_seeds(self, n: int) -> TensorLike:
        seeds_idx = np.random.choice(range(self._data.shape[0]), size=n, replace=True)
        return seeds_idx

    def _check_fitted(self):
        if self._data is None:
            raise AttributeError(
                "This object is not fitted. Call .fit(your_dataset) first."
            )

    def fit(self, time_series: TensorLike, y: Optional[TensorLike] = None):
        assert len(time_series.shape) == 3
        self._data = time_series
        if y is not None:
            self._targets = y
        return self

    def generate(self, n_samples: int) -> TensorLike:
        raise NotImplementedError

    def fit_generate(
        self, time_series: TensorLike, y: Optional[TensorLike], n_samples: int
    ) -> TensorLike:
        self.fit(time_series=time_series, y=y)
        return self.generate(n_samples=n_samples)


class BaseCompose:
    def __init__(
        self,
        augmentations: List[BaseAugmenter],
    ):
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
        mean: TensorLike = 0,
        variance: TensorLike = (10.0, 50.0),
        per_feature: bool = True,
    ):
        super(GaussianNoise, self).__init__(per_feature)
        if isinstance(variance, (tuple, list)):
            if variance[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if variance[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.variance = variance
        elif isinstance(variance, (int, float)):
            if variance < 0:
                raise ValueError("var_limit should be non negative.")
            self.variance = (0, variance)
        else:
            raise TypeError(
                f"Expected var_limit type to be one of (int, float, tuple, list), got {type(variance)}"
            )

        self.mean = mean

    def generate(self, n_samples: int) -> TensorLike:
        self._check_fitted()
        seeds_idx = self._get_seeds(n_samples)

        synthetic_data = []
        labels = []
        for i in seeds_idx:
            sequence = self._data[i]
            variance = np.random.uniform(self.variance[0], self.variance[1])
            sigma = variance**0.5
            if self.per_channel:
                gauss = np.random.normal(self.mean, sigma, sequence.shape)
            else:
                gauss = np.random.normal(self.mean, sigma, sequence.shape[:2])
                gauss = np.expand_dims(gauss, -1)
            synthetic_data.append(sequence + gauss)
            if self._targets is not None:
                labels.append(self._targets[i])
        if self._targets is not None:
            return np.array(synthetic_data), np.array(labels)
        else:
            return np.array(synthetic_data)


class SliceAndShuffle(BaseAugmenter):
    """Slice the time series in k pieces and create a new time series by shuffling.
    Args:
        n_segments (int): the number of slices
        per_feature (bool): if set to True, each time series is sliced independently.
            Otherwise, all features are sliced in the same way. Default: True
    """

    def __init__(
        self,
        n_segments: int,
        per_feature: bool = True,
    ):
        super(SliceAndShuffle, self).__init__(per_feature)

        self.n_segments = n_segments

    def generate(self, n_samples: int) -> TensorLike:
        self._check_fitted()
        assert 0 < self.n_segments < self._data.shape[1]

        seeds_idx = self._get_seeds(n_samples)

        synthetic_data = []
        labels = []
        for i in seeds_idx:
            sequence = self._data[i]
            if self.per_channel:
                raise NotImplementedError(
                    "SliceAndShuffle separately by feature is not supported yet."
                )
            else:
                # Randomly pick n_segments-1 points where to slice
                idxs = np.random.randint(0, sequence.shape[0], size=self.n_segments - 1)
                slices = []
                start_idx = 0
                for j in sorted(idxs):
                    s = sequence[start_idx:j]
                    start_idx = j
                    slices.append(s)
                slices.append(sequence[start_idx:])
                np.random.shuffle(slices)
            synthetic_data.append(sequence)
            if self._targets is not None:
                labels.append(self._targets[i])
        if self._targets is not None:
            return np.array(synthetic_data), np.array(labels)
        else:
            return np.array(synthetic_data)


class Shuffle(BaseAugmenter):
    """Shuffles time series features.
    Shuffling is beneficial when each feature corresponds to interchangeable sensors.
    """

    def __init__(self):
        super(Shuffle, self).__init__(per_feature=False)

    def _n_repeats(self, n: int) -> int:
        return math.ceil(n / len(self._data))

    def generate(self, n_samples: int) -> TensorLike:
        self._check_fitted()

        seeds_idx = self._get_seeds(n_samples)
        n_features = self._data.shape[2]

        n_repeats = self._n_repeats(n_samples)
        shuffle_ids = [np.random.choice(np.arange(n_features), n_features, replace=False) for _ in range(n_repeats)]

        synthetic_data = []
        labels = []
        for num, i in enumerate(seeds_idx):
            sequence = self._data[i]
            id_repeat = self._n_repeats(num)
            synthetic_data.append(sequence[:, shuffle_ids[id_repeat]])
            if self._targets is not None:
                labels.append(self._targets[i])
        if self._targets is not None:
            return np.array(synthetic_data), np.array(labels)
        else:
            return np.array(synthetic_data)


class MagnitudeWarping(BaseAugmenter):
    """
    Magnitude warping changes the magnitude of each
    sample by convolving the data window with a smooth curve varying around one
    https://dl.acm.org/doi/pdf/10.1145/3136755.3136817
    """
    def __init__(self):
        super(MagnitudeWarping, self).__init__(per_feature=False)

    def generate(self, n_samples: int, sigma: float = 0.2, knot: int = 4):
        n_data = self._data.shape[0]
        n_timesteps = self._data.shape[1]
        n_features = self._data.shape[2]

        orig_steps = np.arange(n_timesteps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, knot + 2, n_features))
        warp_steps = (np.ones(
            (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=knot + 2))).T
        result = np.zeros((n_samples, n_timesteps, n_features))
        for i in range(n_samples):
            random_sample_id = random.randint(0, n_data - 1)
            warper = np.array([scipy.interpolate.CubicSpline(
                warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(n_features)]).T
            result[i] = self._data[random_sample_id] * warper

        return result


class WindowWarping(BaseAugmenter):
    """
    https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    def __init__(self):
        super(WindowWarping, self).__init__(per_feature=False)

    def generate(self, n_samples, window_ratio=0.2, scales=[0.25, 1.0]):
        n_data = self._data.shape[0]
        n_timesteps = self._data.shape[1]
        n_features = self._data.shape[2]

        scales_per_sample = np.random.choice(scales, n_samples)
        warp_size = max(np.round(window_ratio * n_timesteps).astype(np.int64), 1)
        window_starts = np.random.randint(
            low=0, high=n_timesteps - warp_size,
            size=(n_samples))
        window_ends = window_starts + warp_size

        result = np.zeros((n_samples, n_timesteps, n_features))
        for i in range(n_samples):
            for dim in range(n_features):
                random_sample_id = random.randint(0, n_data - 1)
                random_sample = self._data[random_sample_id]
                start_seg = random_sample[:window_starts[i], dim]
                warp_ts_size = max(round(warp_size * scales_per_sample[i]), 1)
                window_seg = np.interp(
                    x=np.linspace(0, warp_size - 1, num=warp_ts_size),
                    xp=np.arange(warp_size),
                    fp=random_sample[window_starts[i] : window_ends[i], dim])
                end_seg = random_sample[window_ends[i]:, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                result[i, :, dim] = np.interp(
                    np.arange(n_timesteps),
                    np.linspace(0, n_timesteps - 1., num=warped.size), warped).T
        return result
