import abc
import copy
import os
import sklearn
from scipy import integrate
from tqdm import tqdm
import typing as T
import numpy as np

from tsgm.backend import get_distributions
from tsgm.types import Tensor as TensorLike
import tsgm


# Lazy loading of distributions
distributions = None


def _to_numpy(x):
    """Convert tensor to numpy array safely across backends."""
    if os.environ.get("KERAS_BACKEND") == "torch":
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except ImportError:
            pass
    elif hasattr(x, 'numpy'):
        try:
            return x.numpy()
        except TypeError:
            # Handle cases where .numpy() might fail
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
    return np.asarray(x)


def _get_distributions():
    global distributions
    if distributions is None:
        distributions = get_distributions()
    return distributions


class BaseSimulator(abc.ABC):
    """
    Abstract base class for simulators. This class defines the interface for simulators.
    """
    @abc.abstractmethod
    def generate(self, num_samples: int, *args) -> tsgm.dataset.Dataset:
        """
        Abstract method to generate a dataset.

        :param num_samples: Number of samples to generate.
        :type num_samples: int

        :returns: The generated dataset.
        :rtype: tsgm.dataset.Dataset
        """
        pass

    @abc.abstractmethod
    def dump(self, path: str, format: str = "csv") -> None:
        """
        Abstract method to save the generated dataset to a file.

        :param path: The file path where the dataset will be saved.
        :type path: str
        :param format: The format in which to save the dataset, by default "csv".
        :type format: str
        """
        pass


class Simulator(BaseSimulator):
    """
    Concrete class for a basic simulator. This class implements the basic methods for fitting a model and
    generating a dataset, but does not implement the generation and dump methods.
    """
    def __init__(self, data: tsgm.dataset.DatasetProperties, driver: T.Optional[tsgm.types.Model] = None):
        """
        :param data: Properties of the dataset to be used.
        :type data: tsgm.dataset.DatasetProperties
        :param driver: The model to be used for generating data, by default None.
        :type driver: typing.Optional[tsgm.types.Model]
        """
        self._data = data
        self._driver = driver

    def fit(self, **kwargs) -> None:
        """
        Fit the model using the dataset properties.

        :param kwargs: Additional keyword arguments to pass to the model's fit method.
        """
        if self._data.y is not None:
            self._driver.fit(self._data.X, self._data.y, **kwargs)
        else:
            self._driver.fit(self._data.X, **kwargs)

    def generate(self, num_samples: int, *args) -> TensorLike:
        """
        Method to generate a dataset. Not implemented in this class.

        :param num_samples: Number of samples to generate.
        :type num_samples: int

        :returns: The generated dataset.
        :rtype: TensorLike

        :raises NotImplementedError: This method is not implemented in this class.
        """
        raise NotImplementedError

    def dump(self, path: str, format: str = "csv") -> None:
        """
        Method to save the generated dataset to a file. Not implemented in this class.

        :param path: The file path where the dataset will be saved.
        :type path: str
        :param format: The format in which to save the dataset, by default "csv".
        :type format: str

        :raises NotImplementedError: This method is not implemented in this class.
        """
        raise NotImplementedError

    def clone(self) -> "Simulator":
        """
        Create a deep copy of the simulator.

        :returns: A deep copy of the current simulator instance.
        :rtype: Simulator
        """
        return Simulator(copy.deepcopy(self._data))


class ModelBasedSimulator(Simulator):
    """
    A simulator that is based on a model. This class extends the Simulator class and provides additional
    methods for handling model parameters.
    """
    def __init__(self, data: tsgm.dataset.DatasetProperties):
        """
        :param data: Properties of the dataset to be used.
        :type data: tsgm.dataset.DatasetProperties
        """
        super().__init__(data)

    def params(self) -> T.Dict[str, T.Any]:
        """
        Get a dictionary of the simulator's parameters.

        :returns: A dictionary containing the simulator's parameters.
        :rtype: dict
        """
        params = copy.deepcopy(self.__dict__)
        if "_data" in params:
            del params["_data"]
        if "_driver" in params:
            del params["_driver"]
        return params

    def set_params(self, params: T.Dict[str, T.Any]) -> None:
        """
        Set the simulator's parameters from a dictionary.

        :param params: A dictionary containing the parameters to set.
        :type params: dict
        """
        for param_name, param_value in params.items():
            self.__dict__[param_name] = param_value

    @abc.abstractmethod
    def generate(self, num_samples: int, *args) -> None:
        """
        Abstract method to generate a dataset. Must be implemented by subclasses.

        :param num_samples: Number of samples to generate.
        :type num_samples: int

        :raises NotImplementedError: This method is not implemented in this class and must be overridden by subclasses.
        """
        raise NotImplementedError


class NNSimulator(Simulator):
    def clone(self) -> "NNSimulator":
        return NNSimulator(copy.deepcopy(self._data), self._driver.clone())


class SineConstSimulator(ModelBasedSimulator):
    """
    Sine and Constant Function Simulator class that extends the ModelBasedSimulator base class.
    """
    def __init__(self, data: tsgm.dataset.DatasetProperties, max_scale: float = 10.0, max_const: float = 5.0) -> None:
        """
        :param data: Dataset properties for the simulator.
        :type data: tsgm.dataset.DatasetProperties
        :param max_scale: Maximum value for the scale parameter. Defaults to 10.0.
        :type max_scale: float
        :param max_const: Maximum value for the constant parameter. Defaults to 5.0.
        :type max_const: float
        """
        super().__init__(data)

        self.set_params(max_scale, max_const)

    def set_params(self, max_scale: float, max_const: float, *args, **kwargs):
        """
        Sets the parameters for scale, constant, and shift distributions.

        :param max_scale: Maximum value for the scale parameter.
        :type max_scale: float
        :param max_const: Maximum value for the constant parameter.
        :type max_const: float
        """
        #  change to pdists usage
        distributions = _get_distributions()
        self._scale = distributions.Uniform(0, max_scale)
        self._const = distributions.Uniform(0, max_const)
        self._shift = distributions.Uniform(0, 2)

        super().set_params({"max_scale": max_scale, "max_const": max_const})

    def generate(self, num_samples: int, *args) -> tsgm.dataset.Dataset:
        """
        Generates a dataset based on sine and constant functions.

        :param num_samples: Number of samples to generate.
        :type num_samples: int

        :returns: A dataset containing generated samples.
        :rtype: tsgm.dataset.Dataset
        """
        result_X, result_y = [], []
        for i in range(num_samples):
            D = self._data.D
            if isinstance(D, int):
                D = (D,)  # for PyTorch compatibility
            scales = _to_numpy(self._scale.sample(D))
            consts = _to_numpy(self._const.sample(D))
            shifts = _to_numpy(self._shift.sample(D))
            if np.random.random() < 0.5:
                times = np.repeat(np.arange(0, self._data.T, 1)[:, None], self._data.D, axis=1) / 10
                result_X.append(np.sin(times + shifts) * scales)
                result_y.append(0)
            else:
                result_X.append(np.tile(consts, (self._data.T, 1)))
                result_y.append(1)
        return tsgm.dataset.Dataset(x=np.array(result_X), y=np.array(result_y))

    def clone(self) -> "SineConstSimulator":
        """
        Creates a deep copy of the current SineConstSimulator instance.

        :returns: A new instance of SineConstSimulator with copied data and parameters.
        :rtype: SineConstSimulator
        """
        copy_simulator = SineConstSimulator(self._data)
        params = self.params()
        copy_simulator.set_params(max_scale=params["max_scale"], max_const=params["max_const"])
        return copy_simulator


class PredictiveMaintenanceSimulator(ModelBasedSimulator):
    """
    Predictive Maintenance Simulator class that extends the ModelBasedSimulator base class.
    The simulator is based on https://github.com/AaltoPML/human-in-the-loop-predictive-maintenance
    From publication:
    Nikitin, Alexander, and Samuel Kaski. "Human-in-the-loop large-scale predictive maintenance of
    workstations." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.
    """

    # categorical features
    CAT_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]

    def __init__(self, data: tsgm.dataset.DatasetProperties) -> None:
        """
        Initializes the PredictiveMaintenanceSimulator with dataset properties and sets encoders for categorical features.

        :param data: Dataset properties for the simulator.
        :type data: tsgm.dataset.DatasetProperties
        """
        self._data = data
        self.encoders = {d: sklearn.preprocessing.OneHotEncoder() for d in self.CAT_FEATURES}

        for d in self.CAT_FEATURES:
            self.encoders[d].fit([[d], [d + 2], [d + 4], [d + 1], [d + 3], [d + 5], [d + 7]])
        self.set_params()

    def S(self, lmbd, t):
        """
        Calculates the survival curve.

        :param lmbd: Lambda parameter for the exponential distribution.
        :type lmbd: float
        :param t: Time variable.
        :type t: float

        :returns: Survival probability at time t.
        :rtype: float
        """
        return np.exp(-lmbd * t)

    def R(self, rho, lmbd, t):
        """
        Calculates the recovery curve parameter.

        :param rho: Rho parameter for the recovery function.
        :type rho: float
        :param lmbd: Lambda parameter for the exponential distribution.
        :type lmbd: float
        :param t: Time variable.
        :type t: float

        :returns: Recovery curve parameter at time t.
        :rtype: float
        """
        s_ = self.S(lmbd, t)
        return (1 - s_) - rho

    def set_params(self, **kwargs):
        """
        Sets the parameters for the simulator.

        :param kwargs: Arbitrary keyword arguments for setting simulator parameters.
        """
        if "switches" in kwargs:
            self._switches = kwargs["switches"]
        else:
            self._switches = {d: np.random.gamma(4, 2) for d in range(self._data.D)}

        if "m_norms" in kwargs:
            self._m_norms = kwargs["m_norms"]
        else:
            self._m_norms = {d: lambda: np.random.gamma(2, 1) for d in range(self._data.D)}

        if "sigma_norms" in kwargs:
            self._sigma_norms = kwargs["sigma_norms"]
        else:
            self._sigma_norms = {d: lambda: np.random.gamma(1, 1) for d in range(self._data.D)}

        super().set_params({
            "switches": self._switches,
            "m_norms": self._m_norms,
            "sigma_norms": self._sigma_norms
        })

    def mixture_function(self, a, x):
        """
        Calculates the mixture function.

        :param a: Mixture parameter.
        :type a: float
        :param x: Input variable.
        :type x: float

        :returns: Mixture function value.
        :rtype: float
        """
        return (a**x - 1) / (a - 1)

    def sample_equipment(self, num_samples):
        """
        Samples equipment data and generates the dataset.

        :param num_samples: Number of samples to generate.
        :type num_samples: int

        :returns: A tuple containing the dataset and equipment information.
        :rtype: tuple
        """
        equipment, dataset = [], []
        for _ in tqdm(range(num_samples)):
            last_norm_tmp = 0
            lmbd = np.random.gamma(1, 0.005)
            rho = np.random.gamma(1, 0.1)
            equipment.append({
                "lambda": lmbd,
                "rho": rho
            })
            current_measurements = []
            ss = []
            fix_tmps = []
            rnd = np.random.uniform(0, 1)
            for t in range(self._data.T):
                measurements = []

                s_ = self.S(lmbd, t - last_norm_tmp)
                r_ = self.R(rho, lmbd, t - last_norm_tmp)
                ss.append(s_)

                if rnd < r_:
                    rnd = np.random.uniform(0, 1)
                    last_norm_tmp = t
                    fix_tmps.append(t)

                for d in range(self._data.D):
                    m_norm = self._m_norms[d]()
                    sigma_norm = self._sigma_norms[d]()

                    m_abnorm = m_norm + self._switches[d]
                    sigma_abnorm = 1.5 * sigma_norm

                    if d in self.CAT_FEATURES:
                        norm_functioning = np.random.choice([d, d + 2, d + 4], p=[0.7, 0.2, 0.1])
                        abnorm_functioning = np.random.choice([d + 1, d + 3, d + 5, d + 7], p=[0.2, 0.2, 0.4, 0.2])
                    else:
                        norm_functioning = np.random.normal(m_norm, sigma_norm)
                        abnorm_functioning = np.random.normal(m_abnorm, sigma_abnorm)

                    mixt = self.mixture_function(3, s_)
                    if d in self.CAT_FEATURES:
                        if rnd < 1 - s_:
                            measurements.extend(self.encoders[d].transform([[abnorm_functioning]]).toarray()[0])
                        else:
                            measurements.extend(self.encoders[d].transform([[norm_functioning]]).toarray()[0])
                    else:
                        measurements.extend([mixt * norm_functioning + (1 - mixt) * abnorm_functioning])

                if not len(current_measurements):
                    current_measurements.append([measurements])
                    current_measurements = np.array(current_measurements[0])
                else:
                    current_measurements = np.concatenate((current_measurements, np.array(measurements)[np.newaxis, :]), axis=0)
            equipment[-1]["fixes"] = fix_tmps
            equipment[-1]["ss"] = ss
            dataset.append(current_measurements)
        dataset = np.transpose(np.array(dataset), [0, 2, 1])
        return dataset, equipment

    def generate(self, num_samples: int):
        """
        Samples equipment data and generates the dataset.

        :param num_samples: Number of samples to generate.
        :type num_samples: int

        :returns: A tuple containing the dataset and equipment information.
        :rtype: tuple
        """
        return self.sample_equipment(num_samples)

    def clone(self) -> "PredictiveMaintenanceSimulator":
        """
        Creates a deep copy of the current PredictiveMaintenanceSimulator instance.

        :returns: A new instance of PredictiveMaintenanceSimulator with copied data and parameters.
        :rtype: PredictiveMaintenanceSimulator
        """
        copy_simulator = PredictiveMaintenanceSimulator(self._data)
        params = self.params()
        copy_simulator.set_params(
            switches=params["switches"],
            m_norms=params["m_norms"],
            sigma_norms=params["sigma_norms"])
        return copy_simulator


def _lv_derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])


class LotkaVolterraSimulator(ModelBasedSimulator):
    """
    Simulates the Lotka-Volterra equations, which model the dynamics of biological systems in which two species interact,
    one as a predator and the other as prey.

    For the details refer to https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    """
    def __init__(
            self, data: tsgm.dataset.DatasetProperties,
            alpha: float = 1, beta: float = 1, gamma: float = 1, delta: float = 1,
            x0: float = 1, y0: float = 1) -> None:
        """
        Initializes the Lotka-Volterra simulator with given parameters.

        :param data: The dataset properties.
        :type data: tsgm.dataset.DatasetProperties
        :param alpha: The maximum prey per capita growth rate. Default is 1.
        :type alpha: float
        :param beta: The effect of the presence of predators on the prey death rate. Default is 1.
        :type beta: float
        :param gamma: The predator's per capita death rate. Default is 1.
        :type gamma: float
        :param delta: The effect of the presence of prey on the predator's growth rate. Default is 1.
        :type delta: float
        :param x0: The initial population density of prey. Default is 1.
        :type x0: float
        :param y0: The initial population density of predator. Default is 1.
        :type y0: float
        """
        self._data = data

        self.set_params(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            x0=x0,
            y0=y0
        )

    def set_params(self, alpha, beta, gamma, delta, x0, y0, **kwargs):
        """
        Sets the parameters for the simulator.

        :param alpha: The maximum prey per capita growth rate.
        :type alpha: float
        :param beta: The effect of the presence of predators on the prey death rate.
        :type beta: float
        :param gamma: The predator's per capita death rate.
        :type gamma: float
        :param delta: The effect of the presence of prey on the predator's growth rate.
        :type delta: float
        :param x0: The initial population density of prey.
        :type x0: float
        :param y0: The initial population density of predator.
        :type y0: float
        """
        super().set_params({
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "x0": x0,
            "y0": y0,
        })

    def generate(self, num_samples: int, tmax: float = 1):
        """
        Generates the simulation data based on the Lotka-Volterra equations.

        :param num_samples: The number of sample points to generate.
        :type num_samples: int
        :param tmax: The maximum time value for the simulation. Default is 1.
        :type tmax: float

        :returns: An array containing the population densities of prey and predators over time.
        :rtype: np.ndarray
        """
        t = np.linspace(0., tmax, num_samples)
        X0 = [self.x0, self.y0]
        res = integrate.odeint(_lv_derivative, X0, t, args=(self.alpha, self.beta, self.delta, self.gamma))
        return res

    def clone(self) -> "LotkaVolterraSimulator":
        """
        Creates a deep copy of the current LotkaVolterraSimulator instance.

        :returns: A new instance of LotkaVolterraSimulator with copied data and parameters.
        :rtype: LotkaVolterraSimulator
        """
        copy_simulator = LotkaVolterraSimulator(self._data)
        params = self.params()
        copy_simulator.set_params(
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            delta=params["delta"],
            x0=params["x0"],
            y0=params["y0"])
        return copy_simulator
