import pytest

import numpy as np
import tsgm


def test_TSFeatureWiseScaler():
    ts = np.array([[[0, 2], [1, 0], [1, 2]]])
    scaler = tsgm.utils.TSFeatureWiseScaler()
    scaler.fit(ts)
    assert np.allclose(scaler.transform(ts), np.array([[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]))
    assert np.allclose(scaler.inverse_transform(scaler.transform(ts)), ts)

    scaler1 = tsgm.utils.TSFeatureWiseScaler()
    assert np.allclose(scaler1.fit_transform(ts), np.array([[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]))
    assert np.allclose(scaler1.inverse_transform(scaler1.transform(ts)), ts)

    scaler2 = tsgm.utils.TSFeatureWiseScaler(feature_range=(-2, 2))
    assert np.allclose(scaler2.fit_transform(ts), np.array([[[-2, 2.0], [2.0, -2.0], [2.0, 2.0]]]))
    assert np.allclose(scaler2.inverse_transform(scaler2.fit_transform(ts)), ts)

    ts1 = np.array([[[10, 5], [0, 100], [-1, 2]]])
    scaler3 = tsgm.utils.TSFeatureWiseScaler(feature_range=(-10, 10))
    assert np.allclose(scaler3.fit_transform(ts1), np.array([[[ 10.0,  -9.3877551],
                                                             [-8.18181818, 10.0],
                                                             [-10.0, -10.0]]]))
    assert np.allclose(scaler3.inverse_transform(scaler3.fit_transform(ts1)), ts1)
    assert not np.allclose(scaler.inverse_transform(scaler3.fit_transform(ts1)), ts)


def test_TSGlobalScaler():
    ts = np.array([[[0, 2], [1, 0], [1, 2]]])
    scaler = tsgm.utils.TSGlobalScaler()
    scaler.fit(ts)
    assert np.allclose(scaler.transform(ts), np.array([[[0.0, 1.0], [0.5, 0.0], [0.5, 1.0]]]))

    scaler1 = tsgm.utils.TSGlobalScaler()
    assert np.allclose(scaler1.fit_transform(ts), np.array([[[0.0, 1.0], [0.5, 0.0], [0.5, 1.0]]]))


def test_sine_generator():
    ts = tsgm.utils.gen_sine_dataset(10, 100, 20, max_value=2)
    assert ts.shape == (10, 100, 20)
    assert np.max(ts) <= 2 and np.min(ts) >= -2


def test_reconstruction_loss():
    original = np.array([[[0, 2], [1, 0], [1, 2]]])
    reconstructed = np.array([[[0.1, 1.5], [1.1, 0.1], [1, 2]]])

    # TODO finalize


def test_switch_generator():
    Xs, ys = tsgm.utils.gen_sine_const_switch_dataset(10, 100, 20)

    assert Xs.shape == (10, 100, 20)
    assert ys.shape == (10, 100)


def test_ucr_manager():
    DATASET = "gunpoint"
    ucr_data_manager = tsgm.utils.UCRDataManager(path="./data/UCRArchive_2018/", ds=DATASET)
    assert ucr_data_manager.summary() is None
    X_train, y_train, X_test, y_test = ucr_data_manager.get()
    assert X_train.shape == (50, 150) and X_test.shape == (150, 150)


def test_sine_vs_const_dataset():
    Xs, ys = tsgm.utils.gen_sine_vs_const_dataset(10, 100, 20, max_value=2, const=1)

    assert Xs.shape == (10, 100, 20) and ys.shape == (10,)
    assert np.max(Xs) <= 2 and np.min(Xs) >= - 2


def test_mauna_loa_load():
    X, y = tsgm.utils.get_mauna_loa()

    assert X.shape == (2225, 1)    
    assert y.shape == (2225,)


def test_split_dataset_into_objects():
    X, y = tsgm.utils.get_mauna_loa()
    X, y = tsgm.utils.split_dataset_into_objects(X, y, step=10)
    assert X.shape == (223, 10, 1)
    assert y.shape == (223,)

    X, y = tsgm.utils.get_mauna_loa()
    X, y = tsgm.utils.split_dataset_into_objects(X, y, step=1)
    assert X.shape == (2225, 1, 1)
    assert y.shape == (2225, 1)


def test_get_eeg():
    X, y = tsgm.utils.get_eeg()

    assert X.shape == (14980, 14)
    assert y.shape == (14980,)


def test_get_power_consumption():
    X = tsgm.utils.get_power_consumption()

    assert X.shape == (2075259, 7)
