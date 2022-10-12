import typing
import numpy as np
import scipy
import functools

import tsgm


'''
All statistics should return lists.
'''


def _validate_axis(axis: typing.Optional[int]):
    assert axis == 1 or axis == 2 or axis is None


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
