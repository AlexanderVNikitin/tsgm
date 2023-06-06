import pytest
import numpy as np

import tsgm


def test_base_compose():
    _mock_aug = tsgm.models.augmentations.GaussianNoise()
    compose = tsgm.models.augmentations.BaseCompose([_mock_aug] * 10,)

    assert len(compose) == 10
    assert compose[1] == _mock_aug
    with pytest.raises(NotImplementedError) as e:
        compose()


def test_gaussian_augmentation_dims():
    t = np.arange(0, 10, 0.01)
    xs = np.cos(t)[None, :, None]
    n_gen = 123
    gn_aug = tsgm.models.augmentations.GaussianNoise()
    xs_gen = gn_aug.generate(X=xs, n_samples=n_gen)
    assert xs_gen.shape == (n_gen, xs.shape[1], xs.shape[2])

    xs_gen1 = gn_aug.generate(X=xs, n_samples=n_gen)
    assert not np.array_equal(xs_gen, xs_gen1)
    assert xs_gen1.shape == xs_gen.shape

    gn_aug = tsgm.models.augmentations.GaussianNoise(per_feature=False)
    xs_gen = gn_aug.generate(X=xs, n_samples=n_gen)
    assert xs_gen.shape == (n_gen, xs.shape[1], xs.shape[2])


def test_shuffle():
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    shfl_aug = tsgm.models.augmentations.Shuffle()
    xs_gen = shfl_aug.generate(X=xs, n_samples=2)
    assert xs_gen.shape == (2, 2, 4)

    xs_gen = shfl_aug.generate(X=xs, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)
    assert all([np.allclose(x[0], x[1]) for x in xs_gen])


def test_magnitude_warping():
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    magn_warp_aug = tsgm.models.augmentations.MagnitudeWarping()
    xs_gen = magn_warp_aug.generate(X=xs, n_samples=2)
    assert xs_gen.shape == (2, 2, 4)

    xs_gen = magn_warp_aug.generate(X=xs, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)


def test_window_warping():
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    magn_warp_aug = tsgm.models.augmentations.WindowWarping()
    xs_gen = magn_warp_aug.generate(X=xs, n_samples=2)
    assert xs_gen.shape == (2, 2, 4)

    xs_gen = magn_warp_aug.generate(X=xs, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)
