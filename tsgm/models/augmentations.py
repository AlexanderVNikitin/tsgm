import numpy as np
from typing import List, Dict, Any, Optional
from tensorflow.python.types.core import TensorLike
import warnings

import tsgm

import logging

logger = logging.getLogger("models")
logger.setLevel(logging.DEBUG)


class BaseCompose:
    def __init__(
        self,
        augmentations: List[tsgm.tsgm.models.augmentations.BaseAugmenter],
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

    def __getitem__(self, item: int) -> tsgm.tsgm.models.augmentations.BaseAugmenter:
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

    def add_targets(self, additional_targets: Optional[Dict[str, str]]) -> None:
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> None:
        for t in self.transforms:
            t.set_deterministic(flag, save_key)


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

    def fit(self, img: np.ndarray):
        return

    def generate(self, n_samples: int) -> TensorLike:
        return
