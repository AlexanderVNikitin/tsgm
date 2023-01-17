import pytest
import numpy as np

import tsgm


def test_base_compose():
    _mock_aug = tsgm.models.augmentations.GaussianNoise()
    compose = tsgm.models.augmentations.BaseCompose([_mock_aug] * 10, p=1, seed=1234)

    assert len(compose) == 10
    assert compose[1] == _mock_aug
    with pytest.raises(NotImplementedError) as e:
        compose()


def test_gaussian_augmentation():
    t = np.arange(0, 10, 0.01)
    xs = np.cos(t)[None, :, None]
    n_gen = 123
    gn_aug = tsgm.models.augmentations.GaussianNoise()
    gn_aug.fit(xs)
    xs_gen = gn_aug.generate(n_gen)
    assert xs_gen.shape == (n_gen, xs.shape[1], xs.shape[2])

    xs_gen1 = gn_aug.generate(n_gen)
    assert not np.array_equal(xs_gen, xs_gen1)
    assert xs_gen1.shape == xs_gen.shape
