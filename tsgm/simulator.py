import abc
import copy
import typing as T
import numpy as np
import tensorflow_probability as tfp
from tensorflow.python.types.core import TensorLike

import tsgm


class BaseSimulator(abc.ABC):
    @abc.abstractmethod
    def generate(self, num_samples: int, *args) -> tsgm.dataset.Dataset:
        pass

    @abc.abstractmethod
    def dump(self, path: str, format: str = "csv") -> None:
        pass


class Simulator(BaseSimulator):
    def __init__(self, data: tsgm.dataset.DatasetProperties, driver: T.Optional[tsgm.types.Model] = None):
        self._data = data
        self._driver = driver

    def fit(self, **kwargs) -> None:
        if self._data.y is not None:
            self._driver.fit(self._data.X, self._data.y, **kwargs)
        else:
            self._driver.fit(self._data.X, **kwargs)

    def generate(self, num_samples: int, *args) -> TensorLike:
        raise NotImplementedError

    def dump(self, path: str, format: str = "csv") -> None:
        raise NotImplementedError

    def clone(self) -> "Simulator":
        return Simulator(copy.deepcopy(self._data))


class ModelBasedSimulator(Simulator):
    def __init__(self, data: tsgm.dataset.DatasetProperties):
        super().__init__(data)

    def params(self) -> T.Dict[str, T.Any]:
        params = copy.deepcopy(self.__dict__)
        del params["_data"], params["_driver"]
        return params

    def set_params(self, params: T.Dict[str, T.Any]) -> None:
        for param_name, param_value in params.items():
            self.__dict__[param_name] = param_value

    @abc.abstractmethod
    def generate(self, num_samples: int, *args) -> None:
        raise NotImplementedError


class NNSimulator(Simulator):
    def clone(self) -> "NNSimulator":
        return NNSimulator(copy.deepcopy(self._data), self._driver.clone())


class SineConstSimulator(ModelBasedSimulator):
    def __init__(self, data: tsgm.dataset.DatasetProperties, max_scale: float = 10.0, max_const: float = 5.0) -> None:
        super().__init__(data)

        self.set_params(max_scale, max_const)

    def set_params(self, max_scale: float, max_const: float, *args, **kwargs):
        self._scale = tfp.distributions.Uniform(0, max_scale)
        self._const = tfp.distributions.Uniform(0, max_const)
        self._shift = tfp.distributions.Uniform(0, 2)

        super().set_params({"max_scale": max_scale, "max_const": max_const})

    def generate(self, num_samples: int, *args) -> tsgm.dataset.Dataset:
        result_X, result_y = [], []
        for i in range(num_samples):
            scales = self._scale.sample(self._data.D)
            consts = self._const.sample(self._data.D)
            shifts = self._shift.sample(self._data.D)
            if np.random.random() < 0.5:
                times = np.repeat(np.arange(0, self._data.T, 1)[:, None], self._data.D, axis=1) / 10
                result_X.append(np.sin(times + shifts) * scales)
                result_y.append(0)
            else:
                result_X.append(np.tile(consts, (self._data.T, 1)))
                result_y.append(1)
        return tsgm.dataset.Dataset(x=np.array(result_X), y=np.array(result_y))

    def clone(self) -> "SineConstSimulator":
        copy_simulator = SineConstSimulator(self._data)
        params = self.params()
        copy_simulator.set_params(max_scale=params["max_scale"], max_const=params["max_const"])
        return copy_simulator
