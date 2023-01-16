import numpy as np
from typing import List, Dict, Any, Optional
from tensorflow.python.types.core import TensorLike
import warnings

import logging

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
        self, always_apply: bool = False, p: float = 0.5, seed: Optional[float] = None
    ):
        self.p = p
        self.always_apply = always_apply
        self.seed = seed

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = dict()
        self.replay_mode = False
        self.applied_in_replay = False

    def fit(self, time_series: np.ndarray):
        raise NotImplementedError

    def generate(self, n_samples: int) -> TensorLike:
        raise NotImplementedError

    def fit_generate(self, time_series: np.ndarray, n_samples: int) -> TensorLike:
        raise NotImplementedError

    def get_params(self) -> Dict:
        return self.params

    @classmethod
    def is_serializable(cls):
        return True

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"always_apply": self.always_apply, "p": self.p}

    def get_dict_with_id(self) -> Dict[str, Any]:
        d = self._to_dict()
        d["id"] = id(self)
        return d


class BaseCompose:
    def __init__(
        self,
        augmentations: List[BaseAugmenter],
        p: float,
        seed: Optional[float] = None,
    ):
        if isinstance(augmentations, (BaseCompose, BaseAugmenter)):
            warnings.warn(
                "augmentations is a single object, but a sequence is expected! It will be wrapped into list."
            )
            augmentations = [augmentations]

        self.augmentations = augmentations
        self.p = p
        self.seed = seed

        self.replay_mode = False
        self.applied_in_replay = False

    def __len__(self) -> int:
        return len(self.augmentations)

    def __call__(self, *args, **data) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> BaseAugmenter:
        return self.augmentations[item]

    def __repr__(self) -> str:
        args = {
            k: v
            for k, v in self._to_dict().items()
            if not (k.startswith("__") or k == "transforms")
        }
        repr_string = self.__class__.__name__ + "(["
        for a in self.augmentations:
            repr_string += "\n" + " " * 4 + repr(a) + ","
        repr_string += "\n" + f"], {args})"
        return repr_string

    @classmethod
    def is_serializable(cls) -> bool:
        return True


class GaussianNoise(BaseAugmenter):
    """Apply noise to the input time series.
    Args:
        variance ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_feature (bool): if set to True, noise will be sampled for each feature independently.
            Otherwise, the noise will be sampled once for all features. Default: True
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        mean: TensorLike = 0,
        variance: TensorLike = (10.0, 50.0),
        per_feature: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GaussianNoise, self).__init__(always_apply, p)
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

        # this will be populated with the seed datapoints after .fit()
        self._data = None
        # this will be populated with the synthetic data after .generate()
        self.synthetic = []

    def fit(self, dataset: TensorLike) -> BaseAugmenter:
        # reset
        self.synthetic = []
        self._data = dataset
        return self

    def generate(self, n_samples: int) -> TensorLike:
        if self._data is None:
            raise AttributeError(
                "This object is not fitted. Call .fit(your_dataset) first."
            )
        seeds_idx = np.random.choice(
            range(self._data.shape[0]), size=n_samples, replace=True
        )

        for i in seeds_idx:
            sequence = self._data[i]
            _draw = np.random.uniform()
            if _draw >= self.p:
                self.synthetic.append(sequence)
            else:
                variance = np.random.uniform(self.variance[0], self.variance[1])
                sigma = variance**0.5

                if self.per_channel:
                    gauss = np.random.normal(self.mean, sigma, sequence.shape)
                else:
                    gauss = np.random.normal(self.mean, sigma, sequence.shape[:2])
                    gauss = np.expand_dims(gauss, -1)
                self.synthetic.append(sequence + gauss)
        return np.array(self.synthetic)

    def fit_generate(self, dataset: TensorLike, n_samples: int) -> TensorLike:
        self.fit(dataset=dataset)
        return self.generate(n_samples=n_samples)
