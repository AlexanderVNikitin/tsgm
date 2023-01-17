import pytest
import tsgm


def test_base_compose():
	_mock_aug = tsgm.models.augmentations.GaussianNoise()
	compose = tsgm.models.augmentations.BaseCompose([_mock_aug] * 10, p=1, seed=1234)

	assert len(compose) == 10
	assert compose[1] == _mock_aug
	with pytest.raises(NotImplementedError) as e:
		compose()


def test_gaussian_augmentation():
	# TODO
	pass
