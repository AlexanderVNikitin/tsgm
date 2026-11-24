"""Regressions for the JAX simulator and distribution backend."""

import os

import numpy as np
import pytest

import tsgm
from tsgm.backend import JAXNormal, JAXUniform


pytestmark = pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND") != "jax", reason="JAX backend regression tests"
)


@pytest.mark.parametrize("distribution,args", [(JAXUniform, (-2., 3.)), (JAXNormal, (0., 1.))])
def test_distribution_advances_reproducible_random_stream(distribution, args):
    first = distribution(*args, seed=17)
    replica = distribution(*args, seed=17)
    draws = [np.asarray(first.sample((16,))) for _ in range(3)]

    for draw in draws:
        np.testing.assert_array_equal(draw, np.asarray(replica.sample((16,))))
        assert np.isfinite(draw).all()
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])


def test_uniform_instances_have_independent_reproducible_streams():
    tsgm.utils.fix_seeds(42)
    first = np.asarray(JAXUniform(0., 1.).sample(16))
    second = np.asarray(JAXUniform(0., 1.).sample(16))
    assert not np.array_equal(first, second)
    assert np.all((first >= 0.) & (first < 1.))

    tsgm.utils.fix_seeds(42)
    np.testing.assert_array_equal(first, np.asarray(JAXUniform(0., 1.).sample(16)))
    np.testing.assert_array_equal(second, np.asarray(JAXUniform(0., 1.).sample(16)))


@pytest.mark.parametrize("distribution,args", [(JAXUniform, (0., 1.)), (JAXNormal, (0., 1.))])
def test_distribution_sample_shapes(distribution, args):
    sampler = distribution(*args, seed=17)
    assert sampler.sample().shape == ()
    assert sampler.sample(3).shape == (3,)
    assert sampler.sample((2, 3)).shape == (2, 3)


def test_sine_simulator_generates_more_than_two_distinct_series():
    tsgm.utils.fix_seeds(42)
    properties = tsgm.dataset.DatasetProperties(N=20, T=10, D=2)
    simulator = tsgm.simulator.SineConstSimulator(properties)
    first = simulator.generate(20).X
    second = simulator.generate(20).X

    assert first.shape == second.shape == (20, 10, 2)
    assert len(np.unique(first, axis=0)) > 2
    assert not np.array_equal(first, second)


def test_abc_can_sample_explicit_and_default_jax_priors():
    samples = tsgm.optimization.abc.prior_samples(
        {"scale": JAXUniform(2., 3., seed=17)}, ["scale", "offset"]
    )
    assert 2. <= float(samples["scale"]) < 3.
    assert np.isfinite(float(samples["offset"]))
