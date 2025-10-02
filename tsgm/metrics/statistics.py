import typing
import numpy as np
import scipy
import functools
import os
from statsmodels.tsa.stattools import acf

import tsgm


'''
All statistics should return lists.
'''


def _to_numpy(ts: tsgm.types.Tensor) -> np.ndarray:
    """Convert tensor to numpy array based on current backend."""
    if os.environ.get("KERAS_BACKEND") == "torch":
        import torch
        if isinstance(ts, torch.Tensor):
            return ts.detach().cpu().numpy()
    elif os.environ.get("KERAS_BACKEND") == "tensorflow":
        try:
            import tensorflow as tf
            if hasattr(ts, 'numpy'):
                tf.__version__
                return ts.numpy()
        except ImportError:
            pass
    elif os.environ.get("KERAS_BACKEND") == "jax":
        try:
            import jax.numpy as jnp
            jnp.__version__
            if hasattr(ts, '__array__'):
                return np.array(ts)
        except ImportError:
            pass

    # Fallback: assume it's already a numpy array or can be converted
    return np.asarray(ts)


def _validate_axis(axis: typing.Optional[int]) -> int:
    assert axis == 1 or axis == 2 or axis is None


def _apply_percacf(x: tsgm.types.Tensor) -> tsgm.types.Tensor:
    x_np = _to_numpy(x)
    return np.percentile(acf(x_np), .75)


def _apply_power(x: tsgm.types.Tensor) -> tsgm.types.Tensor:
    x_np = _to_numpy(x)
    return np.power(x_np, 2).sum() / len(x_np)


def axis_max_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)

    return np.array([np.max(ts_np)]) if axis is None else np.max(np.max(ts_np, axis=axis), axis=0).flatten()


global_max_s = functools.partial(lambda x: axis_max_s(x, axis=None))


def axis_min_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)

    return np.array([np.min(ts_np)]) if axis is None else np.min(np.min(ts_np, axis=axis), axis=0).flatten()


global_min_s = functools.partial(lambda x: axis_min_s(x, axis=None))


def axis_mean_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)

    return np.array([np.mean(ts_np)]) if axis is None else np.mean(np.mean(ts_np, axis=axis), axis=0).flatten()


def axis_mode_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)

    return np.array(scipy.stats.mode(ts_np, axis=None)[0]) if axis is None else scipy.stats.mode(scipy.stats.mode(ts_np, axis=axis)[0], axis=0)[0].flatten()


def axis_percentile_s(ts: tsgm.types.Tensor, axis: typing.Optional[int], percentile: float) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)

    return np.percentile(ts_np, percentile, axis=axis).flatten()


def axis_percautocorr_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)
    #  According to _apply_percacf function, I think np.reshape would work fine
    return np.array([_apply_percacf(np.reshape(ts_np, [-1]))]) if axis is None else \
        np.apply_along_axis(_apply_percacf, 0, np.apply_along_axis(_apply_percacf, axis, ts_np))


def axis_power_s(ts: tsgm.types.Tensor, axis: typing.Optional[int]) -> tsgm.types.Tensor:
    _validate_axis(axis)
    ts_np = _to_numpy(ts)
    #  According to __apply_power__ function, I think np.reshape would work fine
    return np.array([_apply_power(np.reshape(ts_np, [-1]))]) if axis is None else \
        np.apply_along_axis(_apply_power, 0, np.apply_along_axis(_apply_power, axis, ts_np))
