import pytest

import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tsgm
import tsgm.backend


def test_dataset():
    Xr, yr = tsgm.utils.gen_sine_vs_const_dataset(10, 100, 20, max_value=2, const=1)
    d_real = tsgm.dataset.Dataset(Xr, yr)

    assert d_real.output_dim == 2

    X1, y1 = tsgm.utils.gen_sine_vs_const_dataset(10, 20, 21, max_value=2, const=1)
    d1 = tsgm.dataset.Dataset(X1, y1)

    assert d_real.X.shape == (10, 100, 20)
    assert d_real.y.shape == (10,)

    assert (d_real + d_real).shape == (20, 100, 20)
    assert isinstance(d_real.Xy, tuple)

    assert d_real.Xy_concat.shape == (10, 100, 21)
    assert len(d_real) == len(d_real.X)
    assert len(d_real) == 10

    assert d_real.seq_len == Xr.shape[1]
    assert d_real.feat_dim == Xr.shape[2]

    with pytest.raises(AssertionError):
        d_real + d1

    X1, y1 = tsgm.utils.gen_sine_vs_const_dataset(10, 20, 21, max_value=2, const=1)
    y11 = np.concatenate([y1[:, None], y1[:, None]], axis=1)
    d1 = tsgm.dataset.Dataset(X1, y11)
    assert d1.Xy_concat.shape == (10, 20, 23)


def test_temporally_labeled_ds():
    X = np.ones((10, 100, 2))
    y = np.ones((10, 100))

    ds = tsgm.dataset.Dataset(X, y)

    assert ds.Xy_concat.shape == (10, 100, 3)
    assert np.array_equal(ds.Xy_concat[:, :, :2], X)
    assert np.array_equal(ds.Xy_concat[:, :, 2], y)
