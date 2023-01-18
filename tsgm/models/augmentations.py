import numpy as np
from typing import List, Dict, Any, Optional
from tensorflow.python.types.core import TensorLike

import logging

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
        self,
    ):

    def fit(self, time_series: TensorLike, y: Optional[TensorLike]):
        raise NotImplementedError

    def generate(self, n_samples: int) -> TensorLike:
        raise NotImplementedError

    def fit_generate(self, time_series: TensorLike, y: Optional[TensorLike], n_samples: int) -> TensorLike:
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
        mean: TensorLike = 0,
        variance: TensorLike = (10.0, 50.0),
        per_feature: bool = True,
    ):
        super(GaussianNoise, self).__init__()
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
        self.per_channel = per_feature
        self._data = None
        self._targets = None

    def fit(self, time_series: TensorLike, y: Optional[TensorLike]) -> BaseAugmenter:
        self._data = time_series
        if y is not None:
            self._targets = y
        return self

    def generate(self, n_samples: int) -> TensorLike:
        if self._data is None:
            raise AttributeError(
                "This object is not fitted. Call .fit(your_dataset) first."
            )
        seeds_idx = np.random.choice(
            range(self._data.shape[0]), size=n_samples, replace=True
        )

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

    def fit_generate(self, time_series: TensorLike, y: Optional[TensorLike], n_samples: int) -> TensorLike:
        self.fit(time_series=time_series, y=y)
        return self.generate(n_samples=n_samples)
