import numpy as np
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
        seeds_idx = np.random.choice(
            range(self._data.shape[0]), size=n, replace=True
        )
        return  seeds_idx

    def _check_fitted(self):
        if self._data is None:
            raise AttributeError(
                "This object is not fitted. Call .fit(your_dataset) first."
            )

    def fit(self, time_series: TensorLike, y: Optional[TensorLike]):
        self._data = time_series
        if y is not None:
            self._targets = y
        return self

    def generate(self, n_samples: int) -> TensorLike:
        raise NotImplementedError

    def fit_generate(self, time_series: TensorLike, y: Optional[TensorLike], n_samples: int) -> TensorLike:
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
            sigma = variance ** 0.5
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
        k (int): the number of slices
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
                raise NotImplementedError()
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
