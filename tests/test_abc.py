import pytest
import tsgm
from tsgm.backend import get_distributions

distributions = get_distributions()

import numpy as np
import keras


def test_abc_rejection_sampler_nn_simulator():
    ts = np.array([[[0, 2], [1, 0], [1, 2], [3, 4]]]).astype(np.float32)
    num_samples, seq_len, feature_dim = ts.shape
    latent_dim = 2
    output_dim = 0

    data = tsgm.dataset.Dataset(ts, y=None)
    statistics = [tsgm.metrics.statistics.global_max_s, tsgm.metrics.statistics.global_min_s]
    architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    discriminator, generator = architecture.discriminator, architecture.generator

    gan = tsgm.models.cgan.GAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    simulator = tsgm.simulator.NNSimulator(data=data, driver=gan)
    simulator.fit(epochs=1)

    discrepancy = lambda x, y: np.linalg.norm(x - y)
    sampler = tsgm.optimization.abc.RejectionSampler(
        data=data, simulator=simulator, statistics=statistics, discrepancy=discrepancy, epsilon=0.4)


def test_abc_rejection_sampler_model_based_simulator():
    statistics = [tsgm.metrics.statistics.global_max_s, tsgm.metrics.statistics.global_min_s]
    max_scale = 10
    max_const = 20
    data = tsgm.dataset.DatasetProperties(N=100, D=2, T=100)
    simulator = tsgm.simulator.SineConstSimulator(data=data, max_scale=max_scale, max_const=20)
    priors = {
        "max_scale": distributions.Uniform(9, 11),
        "max_const": distributions.Uniform(19, 21)
    }
    samples_ref = simulator.generate(10)

    discrepancy = lambda x, y: np.linalg.norm(x - y)
    sampler = tsgm.optimization.abc.RejectionSampler(
        data=samples_ref, simulator=simulator,
        statistics=statistics, discrepancy=discrepancy, epsilon=0.5,
        priors=priors
    )

    sampled_params = sampler.sample_parameters(10)
    assert abs(np.mean([p["max_scale"] for p in sampled_params]) - max_scale) < 1
    assert abs(np.mean([p["max_const"] for p in sampled_params]) - max_const) < 1
