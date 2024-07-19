import abc
import copy
import sklearn
from scipy import integrate
from tqdm import tqdm
import typing as T
import numpy as np

from tsgm.backend import get_distributions
distributions = get_distributions()

#  make TensorLike more flexible
from tsgm.types import Tensor as TensorLike

import tsgm


class BaseSimulator(abc.ABC):
    """
    Abstract base class for simulators. This class defines the interface for simulators.

    Methods
    -------
    generate(num_samples: int, *args) -> tsgm.dataset.Dataset
        Generate a dataset with the specified number of samples.

    dump(path: str, format: str = "csv") -> None
        Save the generated dataset to a file in the specified format.
    """
    @abc.abstractmethod
    def generate(self, num_samples: int, *args) -> tsgm.dataset.Dataset:
        """
        Abstract method to generate a dataset.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        *args
            Additional arguments to be passed to the method.

        Returns
        -------
        tsgm.dataset.Dataset
            The generated dataset.
        """
        pass

    @abc.abstractmethod
    def dump(self, path: str, format: str = "csv") -> None:
        """
        Abstract method to save the generated dataset to a file.

        Parameters
        ----------
        path : str
            The file path where the dataset will be saved.
        format : str, optional
            The format in which to save the dataset, by default "csv".
        """
        pass


class Simulator(BaseSimulator):
    """
    Concrete class for a basic simulator. This class implements the basic methods for fitting a model and
    generating a dataset, but does not implement the generation and dump methods.

    Attributes
    ----------
    _data : tsgm.dataset.DatasetProperties
        Properties of the dataset to be used by the simulator.
    _driver : Optional[tsgm.types.Model]
        The model to be used for generating data.
    """
    def __init__(self, data: tsgm.dataset.DatasetProperties, driver: T.Optional[tsgm.types.Model] = None):
        """
        Initialize the Simulator with dataset properties and an optional model.

        Parameters
        ----------
        data : tsgm.dataset.DatasetProperties
            Properties of the dataset to be used.
        driver : Optional[tsgm.types.Model], optional
            The model to be used for generating data, by default None.
        """
        self._data = data
        self._driver = driver

    def fit(self, **kwargs) -> None:
        """
        Fit the model using the dataset properties.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the model's fit method.
        """
        if self._data.y is not None:
            self._driver.fit(self._data.X, self._data.y, **kwargs)
        else:
            self._driver.fit(self._data.X, **kwargs)

    def generate(self, num_samples: int, *args) -> TensorLike:
        """
        Method to generate a dataset. Not implemented in this class.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        *args
            Additional arguments to be passed to the method.

        Returns
        -------
        TensorLike
            The generated dataset.

        Raises
        ------
        NotImplementedError
            This method is not implemented in this class.
        """
        raise NotImplementedError

    def dump(self, path: str, format: str = "csv") -> None:
        """
        Method to save the generated dataset to a file. Not implemented in this class.

        Parameters
        ----------
        path : str
            The file path where the dataset will be saved.
        format : str, optional
            The format in which to save the dataset, by default "csv".

        Raises
        ------
        NotImplementedError
            This method is not implemented in this class.
        """
        raise NotImplementedError

    def clone(self) -> "Simulator":
        """
        Create a deep copy of the simulator.

        Returns
        -------
        Simulator
            A deep copy of the current simulator instance.
        """
        return Simulator(copy.deepcopy(self._data))


class ModelBasedSimulator(Simulator):
    """
    A simulator that is based on a model. This class extends the Simulator class and provides additional
    methods for handling model parameters.

    Methods
    -------
    params() -> T.Dict[str, T.Any]
        Get a dictionary of the simulator's parameters.

    set_params(params: T.Dict[str, T.Any]) -> None
        Set the simulator's parameters from a dictionary.

    generate(num_samples: int, *args) -> None
        Generate a dataset with the specified number of samples.
    """
    def __init__(self, data: tsgm.dataset.DatasetProperties):
        """
        Initialize the ModelBasedSimulator with dataset properties.

        Parameters
        ----------
        data : tsgm.dataset.DatasetProperties
            Properties of the dataset to be used.
        """
        super().__init__(data)

    def params(self) -> T.Dict[str, T.Any]:
        """
        Get a dictionary of the simulator's parameters.

        Returns
        -------
        dict
            A dictionary containing the simulator's parameters.
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

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters to set.
        """
        for param_name, param_value in params.items():
            self.__dict__[param_name] = param_value

    @abc.abstractmethod
    def generate(self, num_samples: int, *args) -> None:
        """
        Abstract method to generate a dataset. Must be implemented by subclasses.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        *args
            Additional arguments to be passed to the method.

        Raises
        ------
        NotImplementedError
            This method is not implemented in this class and must be overridden by subclasses.
        """
        raise NotImplementedError


class NNSimulator(Simulator):
    def clone(self) -> "NNSimulator":
        return NNSimulator(copy.deepcopy(self._data), self._driver.clone())


class SineConstSimulator(ModelBasedSimulator):
    """
    Sine and Constant Function Simulator class that extends the ModelBasedSimulator base class.

    Attributes:
        _scale: TensorFlow probability distribution for scaling factor.
        _const: TensorFlow probability distribution for constant.
        _shift: TensorFlow probability distribution for shift.

    Methods:
        __init__(data, max_scale=10.0, max_const=5.0): Initializes the simulator with dataset properties and optional parameters.
        set_params(max_scale, max_const, *args, **kwargs): Sets the parameters for scale, constant, and shift distributions.
        generate(num_samples, *args) -> tsgm.dataset.Dataset: Generates a dataset based on sine and constant functions.
        clone() -> SineConstSimulator: Creates and returns a deep copy of the current simulator.
    """
    def __init__(self, data: tsgm.dataset.DatasetProperties, max_scale: float = 10.0, max_const: float = 5.0) -> None:
        """
        Initializes the SineConstSimulator with dataset properties and optional maximum scale and constant values.
        Args:
            data (tsgm.dataset.DatasetProperties): Dataset properties for the simulator.
            max_scale (float, optional): Maximum value for the scale parameter. Defaults to 10.0.
            max_const (float, optional): Maximum value for the constant parameter. Defaults to 5.0.
        """
        super().__init__(data)

        self.set_params(max_scale, max_const)

    def set_params(self, max_scale: float, max_const: float, *args, **kwargs):
        """
        Sets the parameters for scale, constant, and shift distributions.

        Args:
            max_scale (float): Maximum value for the scale parameter.
            max_const (float): Maximum value for the constant parameter.
        """
        #  change to pdists usage
        self._scale = distributions.Uniform(0, max_scale)
        self._const = distributions.Uniform(0, max_const)
        self._shift = distributions.Uniform(0, 2)

        super().set_params({"max_scale": max_scale, "max_const": max_const})

    def generate(self, num_samples: int, *args) -> tsgm.dataset.Dataset:
        """
        Generates a dataset based on sine and constant functions.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            tsgm.dataset.Dataset: A dataset containing generated samples.
        """
        result_X, result_y = [], []
        for i in range(num_samples):
            D = self._data.D
            if isinstance(D, int):
                D = (D,) # for PyTorch compatibility
            scales = self._scale.sample(D)
            consts = self._const.sample(D)
            shifts = self._shift.sample(D)
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

        Returns:
            SineConstSimulator: A new instance of SineConstSimulator with copied data and parameters.
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

    Attributes:
        CAT_FEATURES (list): List of categorical feature indices.
        encoders (dict): Dictionary of OneHotEncoders for categorical features.
    Methods:
        __init__(data): Initializes the simulator with dataset properties and sets encoders.
        S(lmbd, t): Calculates the survival curve.
        R(rho, lmbd, t): Calculates the recovery curve parameter.
        set_params(**kwargs): Sets the parameters for the simulator.
        mixture_function(a, x): Calculates the mixture function.
        sample_equipment(num_samples): Samples equipment data and generates the dataset.
        generate(num_samples): Generates the predictive maintenance dataset.
        clone() -> PredictiveMaintenanceSimulator: Creates and returns a deep copy of the current simulator.
    """

    # categorical features
    CAT_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7]

    def __init__(self, data: tsgm.dataset.DatasetProperties) -> None:
        """
        Initializes the PredictiveMaintenanceSimulator with dataset properties and sets encoders for categorical features.

        Args:
            data (tsgm.dataset.DatasetProperties): Dataset properties for the simulator.
        """
        self._data = data
        self.encoders = {d: sklearn.preprocessing.OneHotEncoder() for d in self.CAT_FEATURES}

        for d in self.CAT_FEATURES:
            self.encoders[d].fit([[d], [d + 2], [d + 4], [d + 1], [d + 3], [d + 5], [d + 7]])
        self.set_params()

    def S(self, lmbd, t):
        """
        Calculates the survival curve.

        Args:
            lmbd: Lambda parameter for the exponential distribution.
            t: Time variable.

        Returns:
            float: Survival probability at time t.
        """
        return np.exp(-lmbd * t)

    def R(self, rho, lmbd, t):
        """
        Calculates the recovery curve parameter.

        Args:
            rho: Rho parameter for the recovery function.
            lmbd: Lambda parameter for the exponential distribution.
            t: Time variable.

        Returns:
            float: Recovery curve parameter at time t.
        """
        s_ = self.S(lmbd, t)
        return (1 - s_) - rho

    def set_params(self, **kwargs):
        """
        Sets the parameters for the simulator.

        Args:
            **kwargs: Arbitrary keyword arguments for setting simulator parameters.
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

        Args:
            a: Mixture parameter.
            x: Input variable.

        Returns:
            float: Mixture function value.
        """
        return (a**x - 1) / (a - 1)

    def sample_equipment(self, num_samples):
        """
        Samples equipment data and generates the dataset.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            tuple: A tuple containing the dataset and equipment information.
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

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            tuple: A tuple containing the dataset and equipment information.
        """
        return self.sample_equipment(num_samples)

    def clone(self) -> "PredictiveMaintenanceSimulator":
        """
        Creates a deep copy of the current PredictiveMaintenanceSimulator instance.

        Returns:
            PredictiveMaintenanceSimulator: A new instance of PredictiveMaintenanceSimulator with copied data and parameters.
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

        Args:
            data (tsgm.dataset.DatasetProperties): The dataset properties.
            alpha (float): The maximum prey per capita growth rate. Default is 1.
            beta (float): The effect of the presence of predators on the prey death rate. Default is 1.
            gamma (float): The predator's per capita death rate. Default is 1.
            delta (float): The effect of the presence of prey on the predator's growth rate. Default is 1.
            x0 (float): The initial population density of prey. Default is 1.
            y0 (float): The initial population density of predator. Default is 1.
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

        Args:
            alpha (float): The maximum prey per capita growth rate.
            beta (float): The effect of the presence of predators on the prey death rate.
            gamma (float): The predator's per capita death rate.
            delta (float): The effect of the presence of prey on the predator's growth rate.
            x0 (float): The initial population density of prey.
            y0 (float): The initial population density of predator.
            **kwargs: Arbitrary keyword arguments for setting simulator parameters.
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

        Args:
            num_samples (int): The number of sample points to generate.
            tmax (float): The maximum time value for the simulation. Default is 1.

        Returns:
            np.ndarray: An array containing the population densities of prey and predators over time.
        """
        t = np.linspace(0., tmax, num_samples)
        X0 = [self.x0, self.y0]
        res = integrate.odeint(_lv_derivative, X0, t, args=(self.alpha, self.beta, self.delta, self.gamma))
        return res

    def clone(self) -> "LotkaVolterraSimulator":
        """
        Creates a deep copy of the current LotkaVolterraSimulator instance.

        Returns:
            LotkaVolterraSimulator: A new instance of LotkaVolterraSimulator with copied data and parameters.
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
