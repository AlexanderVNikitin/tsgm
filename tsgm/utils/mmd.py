import typing

import tsgm


def mmd(X: tsgm.types.Tensor, Y: tsgm.types.Tensor, kernel: typing.Callable):
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)
    return XX.mean() + YY.mean() - 2 * XY.mean()
