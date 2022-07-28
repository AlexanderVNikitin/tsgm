import os
import glob
import typing
import logging

import sklearn
import sklearn.manifold
import sklearn.datasets
import scipy.io.arff

import seaborn as sns
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

import tsgm.types
import tsgm.dataset


logger = logging.getLogger('utils')
logging.basicConfig(level=logging.DEBUG)

EPS = 1e-18


def visualize_dataset(dataset: tsgm.dataset.Dataset, obj_id: int = 0, path: str = "/tmp/generated_data.pdf") -> None:
    """
    The function visualizes time series dataset with target values.
    It can be handy for regression problems.
    :param dataset: A time series dataset.
    :type dataset: tsgm.dataset.DatasetOrTensor.
    """
    plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

    T = dataset.X.shape[-1]

    sns.lineplot(np.arange(0, T, 1), dataset.X[obj_id, -3], label="Feature #1")
    sns.lineplot(np.arange(0, T, 1), dataset.X[obj_id, -2], label="Feature #2")
    sns.lineplot(np.arange(0, T, 1), dataset.X[obj_id, -1], label="Feature #3")

    plt.xlabel("Time")
    plt.ylabel("Absolute value (measurements)")

    print([int(el) for el in dataset.y[obj_id]])
    plt.ylabel("Target value(y)")
    plt.title("Generated data")

    plt.savefig(path)


def visualize_tsne(X: tsgm.types.Tensor, y: tsgm.types.Tensor, X_gen: tsgm.types.Tensor, y_gen: tsgm.types.Tensor,
                   path: str = "/tmp/tsne_embeddings.pdf"):
    """
    Visualizes TSNE of real and synthetic data.
    """
    tsne = sklearn.manifold.TSNE(n_components=2, learning_rate='auto', init='random')

    X_all = np.concatenate((X, X_gen))
    y_all = np.concatenate((y, y_gen))

    c = np.argmax(y_all, axis=1)
    colors = {0: "class 0", 1: "class 1"}
    c = [colors[el] for el in c]
    point_styles = ["hist"] * X.shape[0] + ["gen"] * X_gen.shape[0]
    X_emb = tsne.fit_transform(np.resize(X_all, (X_all.shape[0], X_all.shape[1] * X_all.shape[2])))

    plt.figure(figsize=(8, 6), dpi=80)
    sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1], hue=c[:], style=point_styles[:], markers={"hist": "<", "gen": "H"}, alpha=0.7)
    plt.legend()
    plt.box(False)
    plt.axis('off')
    plt.savefig(path)


def _graph_convolve(node_value: float, neighbor_values: tsgm.types.Tensor) -> float:
    return (node_value + np.mean(neighbor_values)) / 2


def graph_convolution(values: dict, graph: nx.Graph) -> dict:
    result = dict()
    for v in graph.nodes():
        neighbors = graph.neighbors(v)
        result[v] = _graph_convolve(values[v], [values[n] for n in neighbors])
    return result


class TSGlobalScaler():
    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min + EPS)

    def inverse_transform(self, X):
        X *= (self.max - self.min + EPS)
        X += self.min
        return X

    def fit_transform(self, X):
        self.fit(X)
        scaled_X = self.transform(X)
        return scaled_X


class TSFeatureWiseScaler():

    def __init__(self, feature_range: tuple = (0, 1)):
        assert len(feature_range) == 2

        self._min_v, self._max_v = feature_range

    # X: N x T x D
    def fit(self, X):
        D = X.shape[2]
        self.mins = np.zeros(D)
        self.maxs = np.zeros(D)

        for i in range(D):
            self.mins[i] = np.min(X[:, :, i])
            self.maxs[i] = np.max(X[:, :, i])

        return self

    def transform(self, X):
        return ((X - self.mins) / (self.maxs - self.mins + EPS)) * (self._max_v - self._min_v) + self._min_v

    def inverse_transform(self, X):
        X -= self._min_v
        X /= self._max_v - self._min_v
        X *= (self.maxs - self.mins + EPS)
        X += self.mins
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


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


def visualize_ts(ts: tsgm.types.Tensor, num: int = 5):
    """
    Visualizes time series tensor.
    """
    assert len(ts.shape) == 3

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i].imshow(ts[sample_id].T, aspect='auto')


def visualize_ts_lineplot(ts: tsgm.types.Tensor, ys: tsgm.types.OptTensor = None, num: int = 5):
    assert len(ts.shape) == 3

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))
    if num == 1:
        axs = [axs]

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        feature_id = np.random.randint(ts.shape[2])
        sns.lineplot(x=range(ts.shape[1]), y=ts[sample_id, :, feature_id], ax=axs[i], label=f"feature #{feature_id}")
        if ys is not None:
            if len(ys.shape) == 1:
                print(ys[sample_id])
            elif len(ys.shape) == 2:
                sns.lineplot(x=range(ts.shape[1]), y=ys[sample_id], ax=axs[i].twinx(), color="g", label="Target variable")
            else:
                raise ValueError("ys contains too many dimensions")


def visualize_original_and_reconst_ts(original, reconst, num=5, vmin=0, vmax=1):
    assert original.shape == reconst.shape

    fig, axs = plt.subplots(num, 2, figsize=(14, 10))

    ids = np.random.choice(original.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i, 0].imshow(original[sample_id].T, aspect='auto', vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(reconst[sample_id].T, aspect='auto', vmin=vmin, vmax=vmax)


def reconstruction_loss_by_axis(original, reconstructed, axis=0):
    # axis=0 all (sum of squared diffs)
    # axis=1 features (MSE)
    # axis=2 times (MSE)
    if axis == 0:
        return tf.reduce_sum(tf.math.squared_difference(original, reconstructed))
    else:
        return tf.losses.mean_squared_error(tf.reduce_mean(original, axis=axis), tf.reduce_mean(reconstructed, axis=axis))


class UCRDataManager:
    """
    A manager for UCR collection of time series datasets.
    """
    def __init__(self, path: str, ds: str = "gunpoint") -> None:
        """
        :param path: a relative path to the stored UCR dataset.
        :type path: str
        :param ds: Name of the dataset. Should be in (beef | coffee | ecg200 | freezer | gunpoint | insect | mixed_shapes | starlight).
        :type ds: str

        :raises ValueError: When there is no stored UCR archive, or the name of the dataset is incorrect.
        """

        self.ds = ds.strip().lower()
        self.y_all: typing.Optional[typing.Collection[typing.Hashable]] = None

        cur_path = os.path.abspath(".")
        if ds == "beef":
            self.regular_train_path = os.path.join(cur_path, path, "Beef")
            self.small_train_path = os.path.join(cur_path, path, "Beef")
        elif ds == "coffee":
            self.regular_train_path = os.path.join(cur_path, path, "Coffee")
            self.small_train_path = os.path.join(cur_path, path, "Coffee")
        elif ds == "ecg200":
            self.regular_train_path = os.path.join(cur_path, path, "ECG200")
            self.small_train_path = os.path.join(cur_path, path, "ECG200")
        elif ds == "electric":
            self.regular_train_path = os.path.join(cur_path, path, "ElectricDevices")
            self.small_train_path = os.path.join(cur_path, path, "ElectricDevices")
        elif ds == "freezer":
            self.regular_train_path = os.path.join(cur_path, path, "FreezerRegularTrain")
            self.small_train_path = os.path.join(cur_path, path, "FreezerSmallTrain")
        elif ds == "gunpoint":
            self.regular_train_path = os.path.join(cur_path, path, "GunPoint")
            self.small_train_path = os.path.join(cur_path, path, "GunPoint")
        elif ds == "insect":
            self.regular_train_path = os.path.join(cur_path, path, "InsectEPGRegularTrain")
            self.small_train_path = os.path.join(cur_path, path, "InsectEPGSmallTrain")
        elif ds == "mixed_shapes":
            self.regular_train_path = os.path.join(cur_path, path, "MixedShapesRegularTrain")
            self.small_train_path = os.path.join(cur_path, path, "MixedShapesSmallTrain")
        elif ds == "starlight":
            self.regular_train_path = os.path.join(cur_path, path, "StarLightCurves")
            self.small_train_path = os.path.join(cur_path, path, "StarLightCurves")
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
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, "../data/EEG Eye State.arff")
    df = _load_arff(path)
    X = df.drop("eyeDetection", axis=1).to_numpy()
    y = df["eyeDetection"].astype(np.int64).to_numpy()
    return X, y


def get_power_consumption() -> np.ndarray:
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, '../data/household_power_consumption.txt')
    df = pd.read_csv(
        path, sep=';', parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True,
        low_memory=False, na_values=['nan', '?'], index_col='dt')
    return df.to_numpy()
