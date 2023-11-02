import typing
import numpy as np
import scipy
import functools
import tensorflow as tf
from statsmodels.tsa.stattools import acf

import tsgm


'''
All statistics should return lists.
'''


def _validate_axis(axis: typing.Optional[int]):
    assert axis == 1 or axis == 2 or axis is None


def _apply_percacf(x):
    return np.percentile(acf(x), .75)


def _apply_power(x):
    return np.power(x, 2).sum() / len(x)


def axis_max_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.array([np.max(ts)]) if axis is None else np.max(np.max(ts, axis=axis), axis=0).flatten()


global_max_s = functools.partial(lambda x: axis_max_s(x, axis=None))


def axis_min_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.array([np.min(ts)]) if axis is None else np.min(np.min(ts, axis=axis), axis=0).flatten()


global_min_s = functools.partial(lambda x: axis_min_s(x, axis=None))


def axis_mean_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.array([np.mean(ts)]) if axis is None else np.mean(np.mean(ts, axis=axis), axis=0).flatten()


def axis_mode_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.array(scipy.stats.mode(ts, axis=None)[0]) if axis is None else scipy.stats.mode(scipy.stats.mode(ts, axis=axis)[0], axis=0)[0].flatten()


def axis_percentile_s(ts: tsgm.types.Tensor, axis: typing.Optional[int], percentile: float) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.percentile(ts, percentile, axis=axis).flatten()


def axis_percautocorr_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.array([_apply_percacf(tf.reshape(ts, [-1]))]) if axis is None else \
        np.apply_along_axis(_apply_percacf, 0, np.apply_along_axis(_apply_percacf, axis, ts))


def axis_power_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)

    return np.array([_apply_power(tf.reshape(ts, [-1]))]) if axis is None else \
        np.apply_along_axis(_apply_power, 0, np.apply_along_axis(_apply_power, axis, ts))
