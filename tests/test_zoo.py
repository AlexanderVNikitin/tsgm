import pytest
import tensorflow as tf

import tsgm


def test_zoo():
    assert isinstance(tsgm.models.zoo, tsgm.models.architectures.Zoo)
    assert len(tsgm.models.zoo.keys()) == len(tsgm.models.zoo.values())

    assert tsgm.models.zoo.summary() is None

    assert isinstance(tsgm.models.zoo, dict)

    with pytest.raises(TypeError):
        result = tsgm.models.architectures.BaseGANArchitecture()
    with pytest.raises(TypeError):
        result = tsgm.models.architectures.BaseVAEArchitecture()


def test_sampling():
    input_sampling = [0.0, 1.0]
    result = tsgm.models.architectures.Sampling()(input_sampling)
    assert isinstance(result, tf.Tensor)


def test_dict_types():
    for k, v in tsgm.models.zoo.items():
        assert issubclass(v, tsgm.models.architectures.Architecture)
