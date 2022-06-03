import pytest

import tsgm


def test_dataset():
    Xr, yr = tsgm.utils.gen_sine_vs_const_dataset(10, 100, 20, max_value=2, const=1)
    d_real = tsgm.dataset.Dataset(Xr, yr)

    assert d_real.X.shape == (10, 100, 20)
    assert d_real.y.shape == (10,)

    assert (d_real + d_real).shape == (20, 100, 20)
    assert isinstance(d_real.Xy, tuple)

    assert d_real.Xy_concat.shape == (10, 100, 21)
    assert len(d_real) == len(d_real.X)
    assert len(d_real) == 10

    assert d_real.seq_len == Xr.shape[1]
    assert d_real.feat_dim == Xr.shape[2]
