import pytest
import numpy as np

import tsgm


def test_visualize_dataset():
    Xs = np.array([[[1, 2, 3], [3, 4, 5]]])
    tsgm.utils.visualize_dataset(Xs)


def test_visualize_tsne_unlabeled():
    Xs = np.array([
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]],
        [[1, 2, 3], [3, 4, 5]]
    ])
    Xgen = Xs
    ys = np.ones((Xs.shape[0], 1))
    tsgm.utils.visualize_tsne_unlabeled(Xs, Xgen, perplexity=2)


def test_visualize_tsne():
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
    tsgm.utils.visualize_tsne(X=Xs, y=ys, X_gen=X_gen, y_gen=y_gen, perplexity=2)


def test_visualize_ts():
    Xs = np.array([[[1, 2, 3], [3, 4, 5]]])
    tsgm.utils.visualize_ts(Xs, num=1)


def test_visualize_ts_lineplot():
    Xs = np.array([[[1, 2, 3], [3, 4, 5]]])
    tsgm.utils.visualize_ts_lineplot(Xs, num=1)


def visualize_training_loss():
    loss = np.array([10, 9, 8, 7])
    tsgm.utils.visualize_training_loss(loss)
