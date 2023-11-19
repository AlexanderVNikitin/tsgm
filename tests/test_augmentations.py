import pytest
import numpy as np

import tsgm

@pytest.mark.parametrize("mock_aug", [
    [tsgm.models.augmentations.GaussianNoise()] * 10,
    tsgm.models.augmentations.GaussianNoise()
])
def test_base_compose(mock_aug):
    compose = tsgm.models.augmentations.BaseCompose(mock_aug)

    if not isinstance(mock_aug, list):
        mock_aug = [mock_aug]
    assert len(compose) == len(mock_aug)

    for i in range(len(compose)):
        assert compose[i] == mock_aug[i]
    with pytest.raises(NotImplementedError) as e:
        compose()

@pytest.mark.parametrize("aug_model", [
    tsgm.models.augmentations.GaussianNoise(),
    tsgm.models.augmentations.Shuffle(),
    tsgm.models.augmentations.SliceAndShuffle(),
    tsgm.models.augmentations.MagnitudeWarping(),
    tsgm.models.augmentations.WindowWarping(),
    tsgm.models.augmentations.DTWBarycentricAveraging(),
])
def test_dimensions(aug_model):
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    xs_gen = aug_model.generate(X=xs, n_samples=2)
    assert xs_gen.shape == (2, 2, 4)

    xs_gen = aug_model.generate(X=xs, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)

    ys = np.ones((xs.shape[0], 1))
    xs_gen, ys_gen = aug_model.generate(X=xs, y=ys, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)
    assert ys_gen.shape == (17, 1)


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
    xs_gen = shfl_aug.generate(X=xs, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)
    assert all([np.allclose(x[0], x[1]) for x in xs_gen])


def test_magnitude_warping():
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    magn_warp_aug = tsgm.models.augmentations.MagnitudeWarping()

    ys = np.ones((xs.shape[0], 1))
    xs_gen, ys_gen = magn_warp_aug.generate(X=xs, y=ys, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)
    assert ys_gen.shape == (17, 1)
    assert np.allclose(ys_gen, np.ones((17, 1)))


def test_window_warping():
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    ys = np.ones((xs.shape[0], 1))
    window_warp_aug = tsgm.models.augmentations.WindowWarping()
    xs_gen, ys_gen = window_warp_aug.generate(X=xs, y=ys, n_samples=17)
    assert xs_gen.shape == (17, 2, 4)
    assert ys_gen.shape == (17, 1)
    assert np.allclose(ys_gen, np.ones((17, 1)))


@pytest.mark.parametrize("initial_labels", [[0] * 17, None])
def test_dtw_ba(initial_labels):
    xs = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
    ys = [0, 1]
    dtw_ba_aug = tsgm.models.augmentations.DTWBarycentricAveraging()
    xs_gen, ys_gen = dtw_ba_aug.generate(X=xs, y=ys, n_samples=17, initial_labels=initial_labels)
    assert xs_gen.shape == (17, 2, 4)
    assert ys_gen.shape == (17, 1)

    # check with lists
    xs_gen, ys_gen = dtw_ba_aug.generate(X=list(xs), y=ys, n_samples=17, initial_labels=initial_labels)
    assert xs_gen.shape == (17, 2, 4)
    assert ys_gen.shape == (17, 1)
