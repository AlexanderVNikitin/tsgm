import pytest

import functools
import numpy as np
import sklearn.metrics.pairwise

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
    ucr_data_manager = tsgm.utils.UCRDataManager(ds=DATASET)
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
    assert y.shape == (223, 10)

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


def test_get_power_consumption_second_call(mocker):
    X = tsgm.utils.get_power_consumption()
    file_download_mock = mocker.patch('tsgm.utils.download')
    file_download_mock.side_effect = tsgm.utils.download
    X = tsgm.utils.get_power_consumption()
    assert file_download_mock.call_count == 0


def test_get_stock_data():
    X = tsgm.utils.get_stock_data("AAPL")

    assert len(X.shape) == 3


def test_get_energy_data():
    X = tsgm.utils.get_energy_data()

    assert X.shape == (1, 19735, 28)


def test_get_mnist_data():
    X_train, y_train, X_test, y_test = tsgm.utils.get_mnist_data()
    
    assert X_train.shape == (60000, 784, 1)
    assert y_train.shape == (60000,)

    assert X_test.shape == (10000, 784, 1)
    assert y_test.shape == (10000,)


def test_get_gp_data():
    X = tsgm.utils.get_gp_samples_data(num_samples=10, max_time=10)

    assert X.shape == (10, 1, 10)


def test_mmd():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[1, 2, 3], [4, 5, 6]])
    Z = np.array([[1, 1, 1], [1, 1, 1]])
    rbf_kernel = functools.partial(sklearn.metrics.pairwise.rbf_kernel, gamma=1.0)
    assert tsgm.utils.MMD(X, Y, rbf_kernel) == 0
    assert tsgm.utils.MMD(X, Z, rbf_kernel) != 0


def test_mmd_kernel_heuristic():
    X1 = np.random.normal(0, 1, 100)[:, None]
    X2 = np.random.normal(10, 100, 100)[:, None]
    X11 = np.random.normal(0, 1, 100)[:, None]

    kernel_width = tsgm.utils.kernel_median_heuristic(X1, X2)
    assert kernel_width > 1

    kernel_width = tsgm.utils.kernel_median_heuristic(X1, X11)
    assert kernel_width > 0 and kernel_width < 1


def test_mmd_diff_var():
    Kyy = np.array([[1.0, 0.0], [0.0, 1.0]])
    Kzz = np.array([[1.0, 0.0], [0.0, 1.0]])
    Kxy = np.array([[1.0, 0.0], [0.0, 1.0]])
    Kxz = np.array([[1.0, 0.0], [0.0, 1.0]])

    mmd_var = tsgm.utils.mmd_diff_var(Kyy, Kzz, Kxy, Kxz)
    assert mmd_var == 0


def test_mmd_3_test():
    X = np.random.normal(0, 1, 100)[:, None]
    Y = np.random.normal(10, 100, 100)[:, None]
    Z = np.random.normal(0, 1, 100)[:, None]

    #  Use custome kernels with this (TF-sklearn compatibility)
    # sigma_XY = tsgm.utils.kernel_median_heuristic(X, Y);
    # sigma_XZ = tsgm.utils.kernel_median_heuristic(X, Z);
    # sigma = (sigma_XY + sigma_XZ) / 2

    rbf_kernel = functools.partial(sklearn.metrics.pairwise.rbf_kernel, gamma=1.)
    pvalue, tstat, mmd_xy, mmd_xz = tsgm.utils.mmd_3_test(X=X, Y=Y, Z=Z, kernel=rbf_kernel)

    assert pvalue < 1e-10  # the null hypothesis is rejected


def test_get_wafer():
    DATASET = "wafer"
    ucr_data_manager = tsgm.utils.UCRDataManager(ds=DATASET)
    assert ucr_data_manager.summary() is None
    X_train, y_train, X_test, y_test = ucr_data_manager.get()
    assert X_train.shape == (1000, 152)
    assert y_train.shape == (1000,)

    assert X_test.shape == (6164, 152)
    assert y_test.shape == (6164,)
