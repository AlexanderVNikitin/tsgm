import typing

import numpy as np

import tsgm


def mmd(X: tsgm.types.Tensor, Y: tsgm.types.Tensor, kernel: typing.Callable):
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def kernel_median_heuristic(X1: tsgm.types.Tensor, X2: tsgm.types.Tensor) -> float:
    '''
    Median heuristic (Gretton, 2012) for RBF kernel width.
    '''
    n = X1.shape[0]
    m = X2.shape[0]

    X1_squared = np.tile((X1 * X1).ravel(), (m, 1)).transpose()
    X2_squared = np.tile((X2 * X2).ravel(), (n, 1))

    distances = X1_squared + X2_squared - 2 * np.dot(X1, X2.transpose())
    assert np.min(distances) >= 0

    non_zero_distances = list(filter(lambda x: x != 0, distances.flatten()))
    if non_zero_distances:
        median_distance = np.median(non_zero_distances)
    else:
        median_distance = 0

    return np.sqrt(median_distance / 2)  # 2 * sigma**2
