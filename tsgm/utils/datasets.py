import os
import typing
import glob
import collections
import logging

import sklearn
import sklearn.datasets
import numpy as np
import pandas as pd
import scipy.io.arff

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
            result[-1].append((a * np.sin(d * ts + shift)).T)

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


def _load_arff(path: str) -> pd.DataFrame:
    data = scipy.io.arff.loadarff(path)
    return pd.DataFrame(data[0])


def get_eeg() -> tuple:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG Eye State.arff"
    cur_path = os.path.dirname(__file__)
    path_to_folder = os.path.join(cur_path, "../../data/")
    path_to_resource = os.path.join(path_to_folder, "EEG Eye State.arff")
    if not os.path.exists(path_to_resource):
        file_utils.download(url, path_to_folder)

    df = _load_arff(path_to_resource)
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
