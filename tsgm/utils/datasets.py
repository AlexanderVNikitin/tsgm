import os
import typing as T
import glob
import scipy
import collections
import logging

import yfinance as yf
import wfdb

import sklearn
import sklearn.datasets
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io.arff

from pathlib import Path

from tensorflow import keras
from tensorflow.python.types.core import TensorLike

from tsgm.utils import covid19_data_utils
from tsgm.utils import file_utils


logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG)


def gen_sine_dataset(N: int, T: int, D: int, max_value: int = 10) -> npt.NDArray:
    """
    Generates a dataset of sinusoidal waves with random parameters.

    :param N: Number of samples in the dataset.
    :type N: int
    :param T: Length of each time series in the dataset.
    :type T: int
    :param D: Number of dimensions (sinusoids) in each time series.
    :type D: int
    :param max_value: Maximum value for amplitude and shift of the sinusoids. Defaults to 10.
    :type max_value: int, optional

    :return: Generated dataset with shape (N, T, D).
    :rtype: numpy.ndarray
    """
    result = []
    for i in range(N):
        result.append([])
        a = np.random.random() * max_value
        shift = np.random.random() * max_value + 1
        ts = np.arange(0, T, 1)
        for d in range(1, D + 1):
            result[-1].append((a * np.sin((d + 3) * ts / 25.0 + shift)).T)

    return np.transpose(np.array(result), [0, 2, 1])


def gen_sine_const_switch_dataset(
    N: int,
    T: int,
    D: int,
    max_value: int = 10,
    const: int = 0,
    frequency_switch: float = 0.1,
) -> T.Tuple[TensorLike, TensorLike]:
    """
    Generates a dataset with alternating constant and sinusoidal sequences.

    :param N: Number of samples in the dataset.
    :type N: int
    :param T: Length of each sequence in the dataset.
    :type T: int
    :param D: Number of dimensions in each sequence.
    :type D: int
    :param max_value: Maximum value for amplitude and shift of the sinusoids. Defaults to 10.
    :type max_value: int, optional
    :param const: Value indicating whether the sequence is constant or sinusoidal. Defaults to 0.
    :type const: int, optional
    :param frequency_switch: Probability of switching between constant and sinusoidal sequences. Defaults to 0.1.
    :type frequency_switch: float, optional

    :return: Tuple containing input data (X) and target labels (y).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    result_X, result_y = [], []
    cur_y = 0
    scales = np.random.random(D) * max_value
    shifts = np.random.random(D) * max_value
    for i in range(N):
        result_X.append([])
        result_y.append([])

        for t in range(T):
            if np.random.random() < frequency_switch:
                cur_y = (cur_y + 1) % 2

            result_y[-1].append(cur_y)
            if cur_y == 1:
                result_X[-1].append(scales * np.sin(t / 10 + shifts))
            else:
                result_X[-1].append(scales)
    return np.array(result_X), np.array(result_y)


def gen_sine_vs_const_dataset(
    N: int, T: int, D: int, max_value: int = 10, const: int = 0
) -> T.Tuple[TensorLike, TensorLike]:
    """
    Generates a dataset with alternating sinusoidal and constant sequences.

    :param N: Number of samples in the dataset.
    :type N: int
    :param T: Length of each sequence in the dataset.
    :type T: int
    :param D: Number of dimensions in each sequence.
    :type D: int
    :param max_value: Maximum value for amplitude and shift of the sinusoids. Defaults to 10.
    :type max_value: int, optional
    :param const: Maximum value for the constant sequence. Defaults to 0.
    :type const: int, optional

    :return: Tuple containing input data (X) and target labels (y).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    result_X, result_y = [], []
    for i in range(N):
        scales = np.random.random(D) * max_value
        consts = np.random.random(D) * const
        shifts = np.random.random(D) * 2
        alpha = np.random.random()
        if np.random.random() < 0.5:
            times = np.repeat(np.arange(0, T, 1)[:, None], D, axis=1) / 10
            result_X.append(np.sin(alpha * times + shifts) * scales)
            result_y.append(0)
        else:
            result_X.append(np.tile(consts, (T, 1)))
            result_y.append(1)
    return np.array(result_X), np.array(result_y)


class UCRDataManager:
    """
    A manager for UCR collection of time series datasets.
    If you find these datasets useful, please cite:
    @misc{UCRArchive2018,
        title = {The UCR Time Series Classification Archive},
        author = {Dau, Hoang Anh and Keogh, Eamonn and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan
                  and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Yanping and Hu, Bing
                  and Begum, Nurjahan and Bagnall, Anthony and Mueen, Abdullah and Batista, Gustavo, and Hexagon-ML},
        year = {2018},
        month = {October},
        note = {\\url{https://www.cs.ucr.edu/~eamonn/time_series_data_2018/}}
    }
    """

    mirrors = ["https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"]
    resources = [("UCRArchive_2018.zip", 0)]
    key = "someone"
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../data"
    )

    def __init__(self, path: str = default_path, ds: str = "gunpoint") -> None:
        """
        :param path: a relative path to the stored UCR dataset.
        :type path: str
        :param ds: Name of the dataset. The list of names is available at https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ (case sensitive!).
        :type ds: str

        :raises ValueError: When there is no stored UCR archive, or the name of the dataset is incorrect.
        """
        file_utils.download_all_resources(
            self.mirrors[0], path, self.resources, pwd=bytes(self.key, "utf-8")
        )
        path = os.path.join(path, "UCRArchive_2018")

        self.ds = ds.strip().lower()
        self.y_all: T.Optional[T.Collection[T.Hashable]] = None
        path = os.path.join(path, ds)
        train_files = glob.glob(os.path.join(path, "*TRAIN.tsv"))

        if len(train_files) == 0:
            raise ValueError("ds should be listed at UCR website")
        self.train_df = pd.read_csv(
            glob.glob(os.path.join(path, "*TRAIN.tsv"))[0], sep="\t", header=None
        )
        self.test_df = pd.read_csv(
            glob.glob(os.path.join(path, "*TEST.tsv"))[0], sep="\t", header=None
        )

        self.X_train, self.y_train = (
            self.train_df[self.train_df.columns[1:]].to_numpy(),
            self.train_df[self.train_df.columns[0]].to_numpy(),
        )
        self.X_test, self.y_test = (
            self.test_df[self.test_df.columns[1:]].to_numpy(),
            self.test_df[self.test_df.columns[0]].to_numpy(),
        )
        self.y_all = np.concatenate((self.y_train, self.y_test), axis=0)

    def get(self) -> T.Tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
        """
        Returns a tuple containing training and testing data.

        :return: A tuple (X_train, y_train, X_test, y_test).
        :rtype: tuple[TensorLike, TensorLike, TensorLike, TensorLike]
        """
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_classes_distribution(self) -> T.Dict:
        """
        Returns a dictionary with the fraction of occurrences for each class.

        :return: A dictionary containing the fraction of occurrences for each class.
        :rtype: dict[Any, float]
        """
        if self.y_all is not None:
            return {
                k: v / len(self.y_all)
                for k, v in collections.Counter(self.y_all).items()
            }
        else:
            logger.warning("y_all is None, cannot get classes distribution")
            return {}

    def summary(self) -> None:
        """
        Prints a summary of the dataset.
        """
        print("====Summary====")
        print("Name: ", self.ds)
        print("Train Size: ", self.y_train.shape[0])
        print("Test Size: ", self.y_test.shape[0])
        print("Number of classes: ", len(set(self.y_train)))
        print("Distribution of classes: ", self.get_classes_distribution())


def get_mauna_loa() -> T.Tuple[TensorLike, TensorLike]:
    """
    Loads the Mauna Loa CO2 dataset.

    This function loads the Mauna Loa CO2 dataset, which contains measurements of atmospheric CO2 concentrations
    at the Mauna Loa Observatory in Hawaii.

    :return: A tuple containing the input data (X) and target labels (y).
    :rtype: tuple[TensorLike, TensorLike]
    """
    co2 = sklearn.datasets.fetch_openml(data_id=41187, as_frame=True)
    co2_data = co2.frame
    co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
    co2_data = co2_data[["date", "co2"]].set_index("date")
    X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
    y = co2_data["co2"].to_numpy()
    return X, y


def split_dataset_into_objects(
    X: TensorLike, y: TensorLike, step: int = 10
) -> T.Tuple[TensorLike, TensorLike]:
    """
    Splits the dataset into objects of fixed length.

    This function splits the input dataset into objects of fixed length along the first dimension,
    0-padding if necessary.

    :param X: Input data.
    :type X: TensorLike
    :param y: Target labels.
    :type y: TensorLike
    :param step: Length of each object. Defaults to 10.
    :type step: int, optional

    :return: A tuple containing input data objects and corresponding target label objects.
    :rtype: tuple[TensorLike, TensorLike]
    """
    assert X.shape[0] == y.shape[0]

    Xs, ys = [], []
    for start in range(0, X.shape[0], step):
        cur_x, cur_y = X[start : start + step], y[start : start + step]
        Xs.append(np.pad(cur_x, [(0, step - cur_x.shape[0]), (0, 0)]))
        ys.append(np.pad(cur_y, [(0, step - cur_y.shape[0])]))

    return np.array(Xs), np.array(ys)


def load_arff(path: str) -> pd.DataFrame:
    """
    Loads data from an ARFF (Attribute-Relation File Format) file.

    This function reads data from an ARFF file located at the specified path and returns it as a pandas DataFrame.

    :param path: Path to the ARFF file.
    :type path: str

    :return: DataFrame containing the loaded data.
    :rtype: pandas.DataFrame
    """
    data = scipy.io.arff.loadarff(path)
    return pd.DataFrame(data[0])


def get_eeg() -> T.Tuple[TensorLike, TensorLike]:
    """
    Loads the EEG Eye State dataset.

    This function downloads the EEG Eye State dataset from the UCI Machine Learning Repository
    and returns the input features (X) and target labels (y).

    :return: A tuple containing the input features (X) and target labels (y).
    :rtype: tuple[TensorLike, TensorLike]
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG Eye State.arff"
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "EEG Eye State.arff")
    if not os.path.exists(path_to_resource):
        file_utils.download(url, path_to_folder)

    df = load_arff(path_to_resource)
    X = df.drop("eyeDetection", axis=1).to_numpy()
    y = df["eyeDetection"].astype(np.int64).to_numpy()
    return X, y


def get_synchronized_brainwave_dataset() -> T.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the EEG Synchronized Brainwave dataset.

    This function downloads the EEG Synchronized Brainwave dataset from dropbox
    and returns the input features (X) and target labels (y).

    :return: A tuple containing the input features (X) and target labels (y).
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    url = (
        "https://www.dropbox.com/scl/fi/uqah9rthwrt5i2q6evtws/eeg-data.csv.zip?rlkey=z7sautwq74jow2xt9o6q7lcij&st"
        "=hvpvvfez&dl=1"
    )
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "eeg-data.csv.zip")
    path_to_renamed_csv = os.path.join(
        path_to_folder, "synchronized_brainwave_dataset.csv"
    )
    os.makedirs(path_to_folder, exist_ok=True)
    if not os.path.exists(path_to_renamed_csv):
        file_utils.download(url, path_to_folder)
        logger.info("Download completed.")
        file_utils.extract_archive(path_to_resource, path_to_folder)
        logger.info("Extraction completed.")
        original_csv = os.path.join(path_to_folder, "eeg-data.csv")
        if os.path.exists(original_csv):
            os.rename(original_csv, path_to_renamed_csv)
            logger.info(f"File renamed to {path_to_renamed_csv}")
        else:
            logger.warning("The expected CSV file was not found.")
    else:
        logger.info("File exist")
    df = pd.read_csv(path_to_renamed_csv)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


def get_power_consumption() -> npt.NDArray:
    """
    Retrieves the household power consumption dataset.

    This function downloads and loads the household power consumption dataset from the UCI Machine Learning Repository.
    It returns the dataset as a NumPy array.

    :return: Household power consumption dataset.
    :rtype: numpy.ndarray
    """
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, "../../data/")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    file_utils.download_all_resources(
        url, path, resources=[("household_power_consumption.zip", None)]
    )

    df = pd.read_csv(
        os.path.join(path, "household_power_consumption.txt"),
        sep=";",
        parse_dates={"dt": ["Date", "Time"]},
        infer_datetime_format=True,
        low_memory=False,
        na_values=["nan", "?"],
        index_col="dt",
    )
    return df.to_numpy()


def get_stock_data(stock_name: str) -> npt.NDArray:
    """
    Downloads historical stock data for the specified stock ticker.

    This function downloads historical stock data for the specified stock ticker using the Yahoo Finance API.
    It returns the stock data as a NumPy array with an additional axis representing the batch dimension.

    :param stock_name: Ticker symbol of the stock.
    :type stock_name: str

    :return: Historical stock data.
    :rtype: numpy.ndarray
    :raises ValueError: If the provided stock ticker is invalid or no data is available.
    """
    stock_df = yf.download(stock_name)
    if stock_df.empty:
        raise ValueError(f"Cannot download ticker {stock_name}")
    return stock_df.to_numpy()[None, :, :]


def get_energy_data() -> npt.NDArray:
    """
    Retrieves the energy consumption dataset.

    This function downloads and loads the energy consumption dataset from the UCI Machine Learning Repository.
    It returns the dataset as a NumPy array.

    :return: Energy consumption dataset.
    :rtype: numpy.ndarray
    """
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "energydata_complete.csv")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
    if not os.path.exists(path_to_resource):
        file_utils.download(url, path_to_folder)

    return pd.read_csv(path_to_resource).to_numpy()[None, :, 1:]


def get_mnist_data() -> T.Tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    """
    Retrieves the MNIST dataset.

    This function loads the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits,
    and returns the training and testing data along with their corresponding labels.

    :return: A tuple containing the training data, training labels, testing data, and testing labels.
    :rtype: tuple[TensorLike, TensorLike, TensorLike, TensorLike]
    """
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "mnist.npz")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(
        path_to_resource
    )
    x_train = x_train.reshape(-1, 28 * 28, 1)
    x_test = x_test.reshape(-1, 28 * 28, 1)

    return x_train, y_train, x_test, y_test


def _exponential_quadratic(x: npt.NDArray, y: npt.NDArray) -> float:
    """
    This function calculates the exponential quadratic kernel matrix between two sets of points,
    given by matrices `x` and `y`.

    :param x: First set of points.
    :type x: numpy.ndarray
    :param y: Second set of points.
    :type y: numpy.ndarray

    :return: Exponential quadratic kernel matrix.
    :rtype: numpy.ndarray
    """
    return np.exp(-0.5 * scipy.spatial.distance.cdist(x, y))


def get_gp_samples_data(
    num_samples: int, max_time: int, covar_func: T.Callable = _exponential_quadratic
) -> npt.NDArray:
    """
    Generates samples from a Gaussian process.

    This function generates samples from a Gaussian process using the specified covariance function.
    It returns the generated samples as a NumPy array.

    :param num_samples: Number of samples to generate.
    :type num_samples: int
    :param max_time: Maximum time value for the samples.
    :type max_time: int
    :param covar_func: Covariance function to use. Defaults to `_exponential_quadratic`.
    :type covar_func: Callable, optional

    :return: Generated samples from the Gaussian process.
    :rtype: numpy.ndarray
    """

    #  TODO: connect this implementation with `models.gp
    times = np.expand_dims(np.linspace(0, max_time, max_time), 1)
    sigma = covar_func(times, times)

    return np.random.multivariate_normal(
        mean=np.zeros(max_time), cov=sigma, size=num_samples
    )[:, None, :]


def get_physionet2012() -> (
    T.Tuple[TensorLike, TensorLike, TensorLike, TensorLike, TensorLike, TensorLike]
):
    """
    Retrieves the Physionet 2012 dataset.

    This function downloads and retrieves the Physionet 2012 dataset, which consists of physiological data
    and corresponding outcomes. It returns the training, testing, and validation datasets along with their labels.

    :return: A tuple containing the training, testing, and validation datasets along with their labels. (train_X, train_y, test_X, test_y, val_X, val_y)
    :rtype: tuple[TensorLike, TensorLike, TensorLike, TensorLike, TensorLike, TensorLike]
    """
    download_physionet2012()
    train_X = _get_physionet_X_dataframe("physionet2012/set-a")
    train_y = _get_physionet_y_dataframe("physionet2012/Outcomes-a.txt")
    test_X = _get_physionet_X_dataframe("physionet2012/set-b")
    test_y = _get_physionet_y_dataframe("physionet2012/Outcomes-b.txt")
    val_X = _get_physionet_X_dataframe("physionet2012/set-c")
    val_y = _get_physionet_y_dataframe("physionet2012/Outcomes-c.txt")
    return train_X, train_y, test_X, test_y, val_X, val_y


def download_physionet2012() -> None:
    """
    Downloads the Physionet 2012 dataset files from the Physionet website
    and extracts them in local folder 'physionet2012'
    """
    base_url = "https://physionet.org/files/challenge-2012/1.0.0/"
    destination_folder = "physionet2012"
    if (
        os.path.exists(destination_folder)
        and not os.path.isfile(destination_folder)
        and len(os.listdir(destination_folder))
    ):
        logger.info(f"Using downloaded dataset from {destination_folder}")
        return
    X_a = "set-a.tar.gz"
    y_a = "Outcomes-a.txt"

    X_b = "set-b.zip"
    y_b = "Outcomes-b.txt"

    X_c = "set-c.tar.gz"
    y_c = "Outcomes-c.txt"

    all_files = [(X_a, y_a), (X_b, y_b), (X_c, y_c)]
    for X, y in all_files:
        file_utils.download(base_url + X, destination_folder)
        file_utils.download(base_url + y, destination_folder)

    for X, y in all_files:
        file_utils.extract_archive(
            os.path.join(destination_folder, X), destination_folder
        )


def _get_physionet_X_dataframe(dataset_path: str) -> pd.DataFrame:
    """
    Reads txt files from folder 'dataset_path' and returns
    a dataframe (X) with the Physionet dataset.

    Args:
        dataset_path (str): Path to the dataset folder.

    Returns:
        pd.DataFrame: The features (X) dataframe.
    """
    txt_all = list()
    for f in os.listdir(dataset_path):
        with open(os.path.join(dataset_path, f), "r") as fp:
            txt = fp.readlines()

        # add recordid as a column
        recordid = txt[1].rstrip("\n").split(",")[-1]
        txt = [t.rstrip("\n").split(",") + [int(recordid)] for t in txt]
        txt_all.extend(txt[1:])
    df = pd.DataFrame(txt_all, columns=["time", "parameter", "value", "recordid"])
    return df


def _get_physionet_y_dataframe(file_path: str) -> pd.DataFrame:
    """
    Reads txt files from folder 'dataset_path' and returns
    a dataframe (y) with the Physionet data.

    Args:
        dataset_path (str): Path to the dataset folder.

    Returns:
        pd.DataFrame: The target (y) dataframe.
    """
    y = pd.read_csv(file_path)
    y.set_index("RecordID", inplace=True)
    y.index.name = "recordid"
    y.reset_index(inplace=True)
    return y


def get_covid_19() -> T.Tuple[TensorLike, T.Tuple, T.List]:
    """
    Loads Covid-19 dataset with additional graph information
    The dataset is based on data from The New York Times, based on reports from state and local health agencies [1].

    And was adapted to graph case in [2].
    [1] The New York Times. (2021). Coronavirus (Covid-19) Data in the United States. Retrieved [Insert Date Here], from https://github.com/nytimes/covid-19-data.
    [2] Alexander V. Nikitin, St John, Arno Solin, Samuel Kaski Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, PMLR 151:10640-10660, 2022.

    Returns:
    -------
    tuple
        First element is time series data (n_nodes x n_timestamps x n_features). Each timestamp consists of
        the number of deaths, cases, deaths normalized by the population, and cases normalized by the population.
        The second element is the graph tuple (nodes, edges).
        The third element is the order of states.
    """
    base_url = (
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
    )
    destination_folder = "covid19"
    file_utils.download(base_url, destination_folder)
    result, graph = covid19_data_utils.covid_dataset(
        os.path.join(destination_folder, "us-states.csv")
    )

    processed_dataset = []
    for timestamp in result.keys():
        processed_dataset.append([])
        for state in covid19_data_utils.LIST_OF_STATES:
            cur_data = result[timestamp][state]
            processed_dataset[-1].append(
                [
                    cur_data["deaths"],
                    cur_data["cases"],
                    cur_data["deaths_normalized"],
                    cur_data["cases_normalized"],
                ]
            )
    return (
        np.transpose(np.array(processed_dataset), (1, 0, 2)),
        graph,
        covid19_data_utils.LIST_OF_STATES,
    )


def get_arrythmia() -> T.Tuple[TensorLike, TensorLike]:
    """
    Downloads and loads the Arrhythmia dataset from
    https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
    and returns the input features (X) and target labels (y).

    The Arrhythmia dataset contains ECG recordings of patients with arrhythmia.

    :return: A tuple containing the input features (X) and target labels (y).
        X has shape (N, M, D) where
        N is the number of samples,
        M is the signal length (650000),
        D is the number of dimensions (2).
    :rtype: tuple[TensorLike, TensorLike]
    """
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "arrhythmia.data")
    dataset = "mit-bih-arrhythmia-database-1.0.0"
    url = f"https://physionet.org/static/published-projects/mitdb/{dataset}.zip"
    if not os.path.exists(path_to_resource):
        file_utils.download(url, path_to_folder)
        file_utils.extract_archive(
            os.path.join(path_to_folder, f"{dataset}.zip"), path_to_folder
        )

    # load the dataset
    X = []
    y = []
    for i in range(100, 235):
        try:
            record_path = os.path.join(path_to_folder, f"{dataset}/{i}")
            record = wfdb.rdrecord(record_path)
            # equivalent to:
            # wfdb.rdsamp(record_path, sampto=3000))

            # next line does not work
            # annotation = wfdb.rdann(record_path, 'atr',)

            # The signal is an (MxN) 2d numpy array, where M is the signal length.
            X.append(record.p_signal)
            # comments are in record.comments

            # annotation data (e.g., sample numbers and symbols)
            # y.append((annotation.sample, annotation.symbol))
        except Exception as e:
            logger.error(f"Failed to parse {record_path}: {e}")
            pass

    return np.array(X), np.array(y)
