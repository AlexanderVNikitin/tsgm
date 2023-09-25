import pytest
import numpy as np

import tsgm


@pytest.mark.parametrize("ds", [
    np.array([[[1, 2, 3], [3, 4, 5]]]), tsgm.dataset.Dataset(np.array([[[1, 2, 3], [3, 4, 5]]]), y=None)
])
def test_visualize_dataset(ds):
    tsgm.utils.visualize_dataset(ds)


@pytest.mark.parametrize("feature_averaging", [
    True, False
])
def test_visualize_tsne_unlabeled(feature_averaging):
    Xs = np.array([
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]]
    ])
    Xgen = Xs
    ys = np.ones((Xs.shape[0], 1))
    tsgm.utils.visualize_tsne_unlabeled(Xs, Xgen, perplexity=2, feature_averaging=feature_averaging)


@pytest.mark.parametrize("feature_averaging", [
    True, False
])
def test_visualize_tsne(feature_averaging):
    Xs = np.array([
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]]
    ])
    X_gen = Xs
    ys = np.ones((Xs.shape[0], 1))
    y_gen = ys
    tsgm.utils.visualize_tsne(X=Xs, y=ys, X_gen=X_gen, y_gen=y_gen, perplexity=2, feature_averaging=feature_averaging)


def test_visualize_ts():
    Xs = np.array([[[1, 2, 3], [3, 4, 5]]])
    tsgm.utils.visualize_ts(Xs, num=1)


@pytest.mark.parametrize("unite_features", [
    True, False
])
def test_visualize_ts_lineplot(unite_features):
    Xs = np.array([[[1, 2, 3], [3, 4, 5]]])
    tsgm.utils.visualize_ts_lineplot(Xs, num=1, unite_features=unite_features)

    ys = np.array([1, 2])
    tsgm.utils.visualize_ts_lineplot(Xs, ys, num=1, unite_features=unite_features)


def test_visualize_training_loss():
    loss = np.array([[10.0], [9.0], [8.0], [7.0]])
    tsgm.utils.visualize_training_loss(loss)


def test_visualize_original_and_reconst_ts():
    original = np.array([
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]]
    ])
    reconstructed = original
    tsgm.utils.visualize_original_and_reconst_ts(original, reconstructed)

