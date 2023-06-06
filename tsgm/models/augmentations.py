import math
import numpy as np
import random
import scipy.interpolate
from typing import List, Dict, Any, Optional
from tensorflow.python.types.core import TensorLike

import logging

logger = logging.getLogger("augmentations")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
        self,
        per_feature: bool,
    ):
        self.per_channel = per_feature

    def _get_seeds(self, total_num: int, n_seeds: int) -> TensorLike:
        seeds_idx = np.random.choice(range(total_num), size=n_seeds, replace=True)
        return seeds_idx

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        raise NotImplementedError


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
        mean: float = 0,
        variance: float = 1.0,
        per_feature: bool = True,
    ):
        super(GaussianNoise, self).__init__(per_feature)
        self.variance = variance
        self.mean = mean

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        seeds_idx = self._get_seeds(total_num=X.shape[0], n_seeds=n_samples)

        sigma = self.variance**0.5
        if self.per_channel:
            gauss = np.random.normal(self.mean, sigma, (n_samples, X.shape[1], X.shape[2]))
        else:
            gauss = np.random.normal(self.mean, sigma, (n_samples, X.shape[1]))
            gauss = np.expand_dims(gauss, -1)
        synthetic_X = X[seeds_idx] + gauss
        if y is not None:
            synthetic_y = y[seeds_idx]
            return np.array(synthetic_X), np.array(synthetic_y)
        else:
            return np.array(synthetic_X)


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

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        assert 0 < self.n_segments < X.shape[1]

        seeds_idx = self._get_seeds(n_samples=X.shape[0], n_seeds=n_samples)

        synthetic_data = []
        labels = []
        for i in seeds_idx:
            sequence = X[i]
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
            if y is not None:
                labels.append(self._targets[i])
        if y is not None:
            return np.array(synthetic_data), np.array(labels)
        else:
            return np.array(synthetic_data)


class Shuffle(BaseAugmenter):
    """Shuffles time series features.
    Shuffling is beneficial when each feature corresponds to interchangeable sensors.
    """

    def __init__(self):
        super(Shuffle, self).__init__(per_feature=False)

    def _n_repeats(self, n: int, total_num: int) -> int:
        return math.ceil(n / total_num)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        seeds_idx = self._get_seeds(X.shape[0], n_samples)
        n_features = X.shape[2]

        n_repeats = self._n_repeats(n_samples, total_num=len(X))
        shuffle_ids = [np.random.choice(np.arange(n_features), n_features, replace=False) for _ in range(n_repeats)]

        synthetic_data = []
        labels = []
        for num, i in enumerate(seeds_idx):
            sequence = X[i]
            id_repeat = self._n_repeats(num, total_num=len(X))
            synthetic_data.append(sequence[:, shuffle_ids[id_repeat]])
            if y is not None:
                labels.append(y[i])
        if y is not None:
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

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1, sigma: float = 0.2, knot: int = 4) -> TensorLike:
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        orig_steps = np.arange(n_timesteps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, knot + 2, n_features))
        warp_steps = (np.ones(
            (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=knot + 2))).T
        result = np.zeros((n_samples, n_timesteps, n_features))
        for i in range(n_samples):
            random_sample_id = random.randint(0, n_data - 1)
            warper = np.array([scipy.interpolate.CubicSpline(
                warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(n_features)]).T
            result[i] = X[random_sample_id] * warper

        return result


class WindowWarping(BaseAugmenter):
    """
    https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    def __init__(self):
        super(WindowWarping, self).__init__(per_feature=False)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, window_ratio=0.2, scales=[0.25, 1.0], n_samples=1):
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

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
                random_sample = X[random_sample_id]
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
        if y is not None:
            return result, y
        else:
            return result
