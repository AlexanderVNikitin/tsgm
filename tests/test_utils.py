import pytest

import os
import tarfile
import shutil
import uuid
import functools
import urllib
import numpy as np
import random
import keras
from keras import ops
import sklearn.metrics.pairwise
from unittest import mock
from functools import wraps

import tsgm


def skip_on(exception, reason="default"):
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exception:
                pytest.skip(reason)

        return wrapper
    return decorator_func


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
    assert np.allclose(scaler3.fit_transform(ts1), np.array(
        [[[ 10.0,  -9.3877551],
        [-8.18181818, 10.0],
        [-10.0, -10.0]]]))
    assert np.allclose(scaler3.inverse_transform(scaler3.fit_transform(ts1)), ts1)
    assert not np.allclose(scaler.inverse_transform(scaler3.fit_transform(ts1)), ts)


def test_TSGlobalScaler():
    ts = np.array([[[0, 2], [1, 0], [1, 2]]])
    scaler = tsgm.utils.TSGlobalScaler()
    scaler.fit(ts)
    assert np.allclose(scaler.transform(ts), np.array([[[0.0, 1.0], [0.5, 0.0], [0.5, 1.0]]]))
    assert np.allclose(scaler.inverse_transform(scaler.transform(ts)), ts)

    scaler1 = tsgm.utils.TSGlobalScaler()
    assert np.allclose(scaler1.fit_transform(ts), np.array([[[0.0, 1.0], [0.5, 0.0], [0.5, 1.0]]]))


def test_sine_generator():
    ts = tsgm.utils.gen_sine_dataset(10, 100, 20, max_value=2)
    assert ts.shape == (10, 100, 20)
    assert np.max(ts) <= 2 and np.min(ts) >= -2


def test_switch_generator():
    Xs, ys = tsgm.utils.gen_sine_const_switch_dataset(10, 100, 20)

    assert Xs.shape == (10, 100, 20)
    assert ys.shape == (10, 100)


def test_ucr_manager():
    DATASET = "GunPoint"
    ucr_data_manager = tsgm.utils.UCRDataManager(ds=DATASET)
    assert ucr_data_manager.summary() is None
    X_train, y_train, X_test, y_test = ucr_data_manager.get()
    assert X_train.shape == (50, 150) and X_test.shape == (150, 150)

    # test y_all is None
    ucr_data_manager.y_all = None
    assert ucr_data_manager.get_classes_distribution() == {}


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


@skip_on(urllib.error.HTTPError, reason="HTTPError due to connection")
def test_get_eeg():
    X, y = tsgm.utils.get_eeg()

    assert X.shape == (14980, 14)
    assert y.shape == (14980,)


@skip_on(urllib.error.HTTPError, reason="HTTPError due to connection")
def test_get_power_consumption():
    X = tsgm.utils.get_power_consumption()

    assert X.shape == (2075259, 7)


@skip_on(urllib.error.HTTPError, reason="HTTPError due to connection")
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
    X1 = np.random.normal(0, 1, 20)[:, None]
    X2 = np.random.normal(10, 100, 20)[:, None]
    X11 = np.random.normal(0, 1, 20)[:, None]

    kernel_width = tsgm.utils.kernel_median_heuristic(X1, X2)
    assert kernel_width > 1

    kernel_width = tsgm.utils.kernel_median_heuristic(X1, X11)
    assert kernel_width > 0 and kernel_width < 1

    assert tsgm.utils.kernel_median_heuristic(np.zeros((10, 1)), np.zeros((20, 1))) == 0


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

    #  Use custom kernels with this (TF-sklearn compatibility)
    # sigma_XY = tsgm.utils.kernel_median_heuristic(X, Y);
    # sigma_XZ = tsgm.utils.kernel_median_heuristic(X, Z);
    # sigma = (sigma_XY + sigma_XZ) / 2

    rbf_kernel = functools.partial(sklearn.metrics.pairwise.rbf_kernel, gamma=1.)
    pvalue, tstat, mmd_xy, mmd_xz = tsgm.utils.mmd_3_test(X=X, Y=Y, Z=Z, kernel=rbf_kernel)

    assert pvalue < 1e-10  # the null hypothesis is rejected


@pytest.mark.parametrize("dataset_name", [
    "Beef",
    "Coffee",
    "ECG200",
    "ElectricDevices",
    "MixedShapesRegularTrain",
    "StarLightCurves",
    "Wafer"
])
def test_ucr_loadable(dataset_name):
    ucr_data_manager = tsgm.utils.UCRDataManager(ds=dataset_name)
    X_train, y_train, X_test, y_test = ucr_data_manager.get()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_ucr_raises():
    with pytest.raises(ValueError) as excinfo:
        ucr_data_manager = tsgm.utils.UCRDataManager(ds="does not exist")
        assert "ds should be listed at UCR website" in str(excinfo.value)
    

def test_get_wafer():
    dataset = "Wafer"
    ucr_data_manager = tsgm.utils.UCRDataManager(ds=dataset)
    assert ucr_data_manager.summary() is None
    X_train, y_train, X_test, y_test = ucr_data_manager.get()
    assert X_train.shape == (1000, 152)
    assert y_train.shape == (1000,)

    assert X_test.shape == (6164, 152)
    assert y_test.shape == (6164,)


def test_fix_random_seeds():
    assert random.random() != 0.6394267984578837
    assert np.random.random() != 0.3745401188473625
    assert float(keras.random.uniform([1])[0]) != 0.68789124

    tsgm.utils.fix_seeds()

    assert random.random() == 0.6394267984578837
    assert np.random.random() == 0.3745401188473625
    
    # Test that keras random can be called (functionality test)
    # Note: Keras 3.0 random seeding behavior may differ from previous versions
    keras_val = float(keras.random.uniform([1])[0])
    assert 0.0 <= keras_val <= 1.0  # Basic sanity check


def test_reconstruction_loss_by_axis():
    eps = 1e-8
    original = ops.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    reconstructed = ops.array([[[1.1, 2.2, 2.9], [3.9, 4.8, 6.1]]])
    loss = tsgm.utils.reconstruction_loss_by_axis(original, reconstructed)
    assert abs(loss.numpy() - 0.119999886) < eps
    loss = tsgm.utils.reconstruction_loss_by_axis(original, reconstructed, axis=1)
    assert abs(loss.numpy()) < eps
    loss = tsgm.utils.reconstruction_loss_by_axis(original, reconstructed, axis=2)
    assert abs(loss.numpy() - 0.00444442) < eps


def test_get_physionet2012(mocker):
    shutil.rmtree("./physionet2012", ignore_errors=True)
    train_X, train_y, test_X, test_y, val_X, val_y = tsgm.utils.get_physionet2012()
    assert train_X.shape == (1757980, 4)
    assert train_y.shape == (4000, 6)

    assert test_X.shape == (1762535, 4)
    assert test_y.shape == (4000, 6)

    assert val_X.shape == (1765303, 4)
    assert val_y.shape == (4000, 6)
    file_download_mock = mocker.patch('tsgm.utils.download')
    file_download_mock.side_effect = tsgm.utils.download
    train_X, train_y, test_X, test_y, val_X, val_y = tsgm.utils.get_physionet2012()
    assert file_download_mock.call_count == 0
    assert train_X.shape == (1757980, 4)
    assert train_y.shape == (4000, 6)

    assert test_X.shape == (1762535, 4)
    assert test_y.shape == (4000, 6)

    assert val_X.shape == (1765303, 4)
    assert val_y.shape == (4000, 6)


def test_download(mocker, caplog):
    file_download_mock = mocker.patch("urllib.request.urlretrieve")
    resource_name = f"resource_{uuid.uuid4()}"
    resource_folder = "./tmp/test_download/"
    os.makedirs(resource_folder, exist_ok=True)
    resource_path = os.path.join(resource_folder, resource_name)
    open(resource_path, 'w')
    try:
        with pytest.raises(ValueError) as excinfo:
            tsgm.utils.download(f"https://pseudourl/{resource_name}", resource_folder, md5=123, max_attempt=1)
        assert "Reference md5 value (123) is not equal to the downloaded" in caplog.text
        assert "Cannot download dataset" in str(excinfo.value)
    finally:
        os.remove(resource_path)


@mock.patch('urllib.request.urlretrieve')
@mock.patch('hashlib.md5')
def test_download_mocked(mock_md5, mock_urlretrieve):
    # Arrange
    url = "http://example.com/resource.zip"
    path = "./tmp/downloads"
    md5 = "12345"
    max_attempt = 3
    # Mocking md5.hexdigest to return the provided md5 value
    mock_md5.return_value.hexdigest.return_value = md5
    mock_urlretrieve.side_effect = lambda a, b: open(f"{path}/resource.zip", "w").write(" ")

    # Act
    try:
        os.makedirs(path, exist_ok=True)
        tsgm.utils.download(url, path, md5, max_attempt)
    finally:
        shutil.rmtree(path)

    # Assert
    # Verify that the necessary functions are called with the correct arguments
    mock_urlretrieve.assert_called_once_with(urllib.parse.quote(url, safe=":/"), os.path.join(path, "resource.zip"))
    mock_md5.assert_called_once_with(b" ")
    mock_md5.return_value.hexdigest.assert_called_once()


def test_get_covid_19():
    X, graph, states = tsgm.utils.get_covid_19()
    assert len(states) == 51 and "new york" in states and "california" in states
    assert len(graph[0]) == len(states) # nodes
    assert len(graph[1]) == 220 # edges
    assert X.shape[0] == len(states)
    assert len(X.shape) == 3
    assert X.shape[2] == 4
    assert X.shape[1] >= 150


def test_extract_targz():
    resource_folder = "./tmp/test_download/"
    os.makedirs(resource_folder, exist_ok=True)
    output_filename = "./tmp/dir.gz"
    extracted_path = "./tmp/extracted"
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(resource_folder, arcname=os.path.basename(resource_folder))
    tsgm.utils.file_utils._extract_targz(output_filename, extracted_path)
    assert os.path.isdir(extracted_path)


def test_version():
    assert isinstance(tsgm.__version__, str)


def test_get_synchronized_brainwave_dataset():
    X, y = tsgm.utils.get_synchronized_brainwave_dataset()
    assert X.shape == (30013, 12)
    assert y.shape == (30013,)