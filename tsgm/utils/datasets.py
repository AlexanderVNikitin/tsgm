import os
import typing
import glob
import scipy
import collections
import logging

import yfinance as yf

import sklearn
import sklearn.datasets
import numpy as np
import pandas as pd
import scipy.io.arff

from tensorflow import keras

from tsgm.utils import file_utils


logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)


def gen_sine_dataset(N, T, D, max_value=10):
    result = []
    for i in range(N):
        result.append([])
        a = np.random.random() * max_value
        shift = np.random.random() * max_value + 1
        ts = np.arange(0, T, 1)
        for d in range(1, D + 1):
            result[-1].append((a * np.sin((d + 3) * ts / 25. + shift)).T)

    return np.transpose(np.array(result), [0, 2, 1])


def gen_sine_const_switch_dataset(N, T, D, max_value=10, const=0, frequency_switch=0.1):
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


def gen_sine_vs_const_dataset(N: int, T: int, D: int, max_value: int = 10, const: int = 0) -> tuple:
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
    """
    mirrors = ["https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"]
    resources = [("UCRArchive_2018.zip", 0)]
    key = "someone"
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")

    def __init__(self, path: str = default_path, ds: str = "gunpoint") -> None:
        """
        :param path: a relative path to the stored UCR dataset.
        :type path: str
        :param ds: Name of the dataset. Should be in (beef | coffee | ecg200 | freezer | gunpoint | insect | mixed_shapes | starlight).
        :type ds: str

        :raises ValueError: When there is no stored UCR archive, or the name of the dataset is incorrect.
        """
        file_utils.download_all_resources(self.mirrors[0], path, self.resources, pwd=bytes(self.key, 'utf-8'))
        path = os.path.join(path, "UCRArchive_2018")

        self.ds = ds.strip().lower()
        self.y_all: typing.Optional[typing.Collection[typing.Hashable]] = None

        if ds == "beef":
            self.regular_train_path = os.path.join(path, "Beef")
            self.small_train_path = os.path.join(path, "Beef")
        elif ds == "coffee":
            self.regular_train_path = os.path.join(path, "Coffee")
            self.small_train_path = os.path.join(path, "Coffee")
        elif ds == "ecg200":
            self.regular_train_path = os.path.join(path, "ECG200")
            self.small_train_path = os.path.join(path, "ECG200")
        elif ds == "electric":
            self.regular_train_path = os.path.join(path, "ElectricDevices")
            self.small_train_path = os.path.join(path, "ElectricDevices")
        elif ds == "freezer":
            self.regular_train_path = os.path.join(path, "FreezerRegularTrain")
            self.small_train_path = os.path.join(path, "FreezerSmallTrain")
        elif ds == "gunpoint":
            self.regular_train_path = os.path.join(path, "GunPoint")
            self.small_train_path = os.path.join(path, "GunPoint")
        elif ds == "insect":
            self.regular_train_path = os.path.join(path, "InsectEPGRegularTrain")
            self.small_train_path = os.path.join(path, path, "InsectEPGSmallTrain")
        elif ds == "mixed_shapes":
            self.regular_train_path = os.path.join(path, path, "MixedShapesRegularTrain")
            self.small_train_path = os.path.join(path, path, "MixedShapesSmallTrain")
        elif ds == "starlight":
            self.regular_train_path = os.path.join(path, path, "StarLightCurves")
            self.small_train_path = os.path.join(path, path, "StarLightCurves")
        elif ds == "wafer":
            self.regular_train_path = os.path.join(path, path, "Wafer")
            self.small_train_path = os.path.join(path, path, "Wafer")
        else:
            raise ValueError("ds should be in (beef | coffee | ecg200 | freezer | gunpoint | insect | mixed_shapes | starlight)")

        self.small_train_df = pd.read_csv(
            glob.glob(os.path.join(self.small_train_path, "*TRAIN.tsv"))[0],
            sep='\t', header=None)
        self.train_df = pd.read_csv(
            glob.glob(os.path.join(self.regular_train_path, "*TRAIN.tsv"))[0],
            sep='\t', header=None)
        self.test_df = pd.read_csv(
            glob.glob(os.path.join(self.regular_train_path, "*TEST.tsv"))[0],
            sep='\t', header=None)

        self.X_train, self.y_train = self.train_df[self.train_df.columns[1:]].to_numpy(), self.train_df[self.train_df.columns[0]].to_numpy()
        self.X_test, self.y_test = self.test_df[self.test_df.columns[1:]].to_numpy(), self.test_df[self.test_df.columns[0]].to_numpy()
        self.y_all = np.concatenate((self.y_train, self.y_test), axis=0)

    def get(self) -> tuple:
        """
        Returns a tuple (X_train, y_train, X_test, y_test).
        """
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_classes_distribution(self) -> dict:
        """
        Returns a dict with fraction for each of classes.
        """
        if self.y_all is not None:
            return {k: v / len(self.y_all) for k, v in collections.Counter(self.y_all).items()}
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


def get_mauna_loa() -> tuple:
    """
    Loads mauna loa dataset.
    """
    co2 = sklearn.datasets.fetch_openml(data_id=41187, as_frame=True)
    co2_data = co2.frame
    co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
    co2_data = co2_data[["date", "co2"]].set_index("date")
    X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
    y = co2_data["co2"].to_numpy()
    return X, y


def split_dataset_into_objects(X, y, step=10):
    assert X.shape[0] == y.shape[0]

    Xs, ys = [], []
    for start in range(0, X.shape[0], step):
        cur_x, cur_y = X[start:start + step], y[start:start + step]
        Xs.append(np.pad(cur_x, [(0, step - cur_x.shape[0]), (0, 0)]))
        ys.append(np.pad(cur_y, step - cur_y.shape[0]))
    return np.array(Xs), np.array(ys)


def load_arff(path: str) -> pd.DataFrame:
    data = scipy.io.arff.loadarff(path)
    return pd.DataFrame(data[0])


def get_eeg() -> tuple:
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


def get_power_consumption() -> np.ndarray:
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, '../../data/')

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    file_utils.download_all_resources(url, path, resources=[("household_power_consumption.zip", None)])

    df = pd.read_csv(
        os.path.join(path, "household_power_consumption.txt"), sep=';', parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
        low_memory=False, na_values=['nan', '?'], index_col='dt')
    return df.to_numpy()


def get_stock_data(stock_name: str) -> np.ndarray:
    stock_df = yf.download(stock_name)
    if stock_df.empty:
        raise ValueError(f"Cannot download ticker {stock_name}")
    return stock_df.to_numpy()[None, :, :]


def get_energy_data() -> np.ndarray:
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "energydata_complete.csv")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
    if not os.path.exists(path_to_resource):
        file_utils.download(url, path_to_folder)

    return pd.read_csv(path_to_resource).to_numpy()[None, :, 1:]


def get_mnist_data() -> tuple:
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "mnist.npz")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path_to_resource)
    x_train = x_train.reshape(-1, 28 * 28, 1)
    x_test = x_test.reshape(-1, 28 * 28, 1)

    return x_train, y_train, x_test, y_test


def _exponential_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    return np.exp(-0.5 * scipy.spatial.distance.cdist(x, y))


def get_gp_samples_data(num_samples: int, max_time: int,
                        covar_func: typing.Callable = _exponential_quadratic) -> np.ndarray:

    #  TODO: connect this implementation with `models.gp
    times = np.expand_dims(np.linspace(0, max_time, max_time), 1)
    sigma = covar_func(times, times)

    return np.random.multivariate_normal(
        mean=np.zeros(max_time), cov=sigma, size=num_samples)[:, None, :]


def get_physionet2012() -> tuple:
    """
    Downloads and retrieves the Physionet 2012 dataset.

    Returns:
        tuple: A tuple containing the training, testing, and validation dataframes as
            (train_X, train_y, test_X, test_y, val_X, val_y)
    """
    download_physionet2012()
    train_X = _get_physionet_X_dataframe("physionet2012/set-a")
    train_y = _get_physionet_y_dataframe("physionet2012/Outcomes-a.txt")
    test_X = _get_physionet_X_dataframe("physionet2012/set-b")
    test_y = _get_physionet_y_dataframe("physionet2012/Outcomes-b.txt")
    val_X = _get_physionet_X_dataframe("physionet2012/set-c")
    val_y = _get_physionet_y_dataframe("physionet2012/Outcomes-c.txt")
    return train_X, train_y, test_X, test_y, val_X, val_y


def download_physionet2012():
    """
    Downloads the Physionet 2012 dataset files from the Physionet website
    and extracts them in local folder 'physionet2012'
    """
    _base_url = "https://physionet.org/files/challenge-2012/1.0.0/"
    _destination_folder = "physionet2012"
    X_a = "set-a.tar.gz"
    y_a = "Outcomes-a.txt"

    X_b = "set-b.zip"
    y_b = "Outcomes-b.txt"

    X_c = "set-c.tar.gz"
    y_c = "Outcomes-c.txt"

    file_utils.download(_base_url + X_a, _destination_folder)
    file_utils.download(_base_url + y_a, _destination_folder)
    file_utils.download(_base_url + X_b, _destination_folder)
    file_utils.download(_base_url + y_b, _destination_folder)
    file_utils.download(_base_url + X_c, _destination_folder)
    file_utils.download(_base_url + y_c, _destination_folder)

    file_utils.extract_archive(_destination_folder + X_a, _destination_folder)
    file_utils.extract_archive(_destination_folder + X_b, _destination_folder)
    file_utils.extract_archive(_destination_folder + X_c, _destination_folder)

    return


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
        with open(os.path.join(dataset_path, f), 'r') as fp:
            txt = fp.readlines()

        # add recordid as a column
        recordid = txt[1].rstrip('\n').split(',')[-1]
        txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
        txt_all.extend(txt[1:])
    df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])
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
    y.set_index('RecordID', inplace=True)
    y.index.name = 'recordid'
    y.reset_index(inplace=True)
    return y
