import pytest

import numpy as np

import tsgm


def test_sts():
    ts = np.sin(np.arange(0, 10, 0.1))[None, :, None]
    sine_ds = tsgm.dataset.Dataset(ts, y=None)
    sts_model = tsgm.models.sts.STS()
    sts_model.train(sine_ds)

    samples = sts_model.generate(10)
    
    assert samples.shape == (10, 10)
