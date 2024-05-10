import pytest

import numpy as np
import tensorflow as tf
import functools
import sklearn

import tsgm


def test_statistics():
    eps = 1e-8
    ts = np.array([
        [[0, 2], [11, -11], [1, 2]],
        [[10, 21], [1, -1], [6, 8]]])
    assert tsgm.metrics.statistics.axis_max_s(ts, axis=None) == [21]
    assert tsgm.metrics.statistics.axis_min_s(ts, axis=None) == [-11]

    assert (tsgm.metrics.statistics.axis_max_s(ts, axis=1) == [11, 21]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts, axis=1) == [0, -11]).all()

    assert (tsgm.metrics.statistics.axis_max_s(ts, axis=2) == [21, 11, 8]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts, axis=2) == [0, -11, 1]).all()

    assert (tsgm.metrics.statistics.axis_mode_s(ts, axis=None) == [1]).all()

    assert (tsgm.metrics.statistics.axis_mean_s(ts, axis=None) - np.asarray([4.16666667]) < eps).all()
    assert (tsgm.metrics.statistics.axis_mean_s(ts, axis=1) - np.asarray([4.83333333, 3.5]) < eps).all()
    assert (tsgm.metrics.statistics.axis_mean_s(ts, axis=2) - np.asarray([8.25, 0., 4.25]) < eps).all()

    assert (tsgm.metrics.statistics.axis_percentile_s(ts, axis=None, percentile=50) - np.asarray([2]) < eps).all()

    assert (tsgm.metrics.statistics.axis_percautocorr_s(ts, axis=None) - np.asarray([-0.245016]) < eps).all()
    assert (tsgm.metrics.statistics.axis_percautocorr_s(ts, axis=1) - np.asarray([-0.48875, -0.48875]) < eps).all()

    assert (tsgm.metrics.statistics.axis_power_s(ts, axis=None) - np.asarray([74.5]) < eps).all()
    assert (tsgm.metrics.statistics.axis_power_s(ts, axis=1) - np.asarray([1869.61111111, 15148.72222222]) < eps).all()
    assert (tsgm.metrics.statistics.axis_power_s(ts, axis=2) - np.asarray([36587.13, 7321., 1253.13]) < eps).all()

    # Now, checking with tf.Tensor
    ts_tf = tf.convert_to_tensor(ts)

    assert tsgm.metrics.statistics.axis_max_s(ts_tf, axis=None) == [21]
    assert tsgm.metrics.statistics.axis_min_s(ts_tf, axis=None) == [-11]

    assert (tsgm.metrics.statistics.axis_max_s(ts_tf, axis=1) == [11, 21]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts_tf, axis=1) == [0, -11]).all()

    assert (tsgm.metrics.statistics.axis_max_s(ts_tf, axis=2) == [21, 11, 8]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts_tf, axis=2) == [0, -11, 1]).all()

    assert (tsgm.metrics.statistics.axis_mode_s(ts_tf, axis=None) == [1]).all()
    assert (tsgm.metrics.statistics.axis_mean_s(ts_tf, axis=None) - np.asarray([4.16666667]) < eps).all()
    assert (tsgm.metrics.statistics.axis_mean_s(ts_tf, axis=1) - np.asarray([4.83333333, 3.5]) < eps).all()
    assert (tsgm.metrics.statistics.axis_mean_s(ts, axis=2) - np.asarray([8.25, 0., 4.25]) < eps).all()

    assert (tsgm.metrics.statistics.axis_percentile_s(ts_tf, axis=None, percentile=50) - np.asarray([2]) < eps).all()

    assert (tsgm.metrics.statistics.axis_percautocorr_s(ts_tf, axis=None) - np.asarray([-0.245016]) < eps).all()
    assert (tsgm.metrics.statistics.axis_percautocorr_s(ts_tf, axis=1) - np.asarray([-0.48875, -0.48875]) < eps).all()

    assert (tsgm.metrics.statistics.axis_power_s(ts_tf, axis=None) - np.asarray([74.5]) < eps).all()
    assert (tsgm.metrics.statistics.axis_power_s(ts_tf, axis=1) - np.asarray([1869.61111111, 15148.72222222]) < eps).all()
    assert (tsgm.metrics.statistics.axis_power_s(ts_tf, axis=2) - np.asarray([36587.13, 7321., 1253.13]) < eps).all()


def test_distance_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    sim_ts = ts + 1e-7
    diff_ts = 10 * ts
    y = np.ones((ts.shape[0], 1))

    statistics = [functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1)]

    dist_metric = tsgm.metrics.DistanceMetric(
        statistics=statistics, discrepancy=lambda x, y: np.linalg.norm(x - y)
    )
    assert dist_metric(ts, diff_ts) > dist_metric(ts, sim_ts)
    stat_results = dist_metric.stats(ts)

    assert len(stat_results) == 6
    assert dist_metric._discrepancy(dist_metric.stats(ts), dist_metric.stats(sim_ts)) == dist_metric(ts, sim_ts)
    assert dist_metric(ts, sim_ts) != dist_metric(ts, diff_ts)
    assert dist_metric(ts, ts) == 0
    assert dist_metric(diff_ts, ts) == dist_metric(ts, diff_ts)

    # with labels
    ds = tsgm.dataset.Dataset(ts, y)
    ds_diff = tsgm.dataset.Dataset(diff_ts, y)
    ds_sim = tsgm.dataset.Dataset(sim_ts, y)
    assert dist_metric(ts, diff_ts) != 0
    assert dist_metric(ds, ds) == 0
    assert dist_metric(ds_sim, ds) < dist_metric(ds_diff, ds)
    assert dist_metric(ds, ds_diff) == dist_metric(ds_diff, ds)


class MockEvaluator:
    def evaluate(self, D: tsgm.dataset.Dataset, Dtest: tsgm.dataset.Dataset) -> float:
        return 0.42


def test_consistency_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D1 = tsgm.dataset.Dataset(ts, y=None)

    diff_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D2 = tsgm.dataset.Dataset(diff_ts, y=None)

    test_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D_test = tsgm.dataset.Dataset(test_ts, y=None)

    n_models = 5
    consistency_metric = tsgm.metrics.ConsistencyMetric(
        evaluators = [MockEvaluator() for _ in range(n_models)]
    )
    model_results = consistency_metric._apply_models(D1, D_test)
    assert len(model_results) == n_models
    assert consistency_metric(D1, D2, D_test) == 1.0


def test_downstream_performance_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D1 = tsgm.dataset.Dataset(ts, y=None)

    diff_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D2 = tsgm.dataset.Dataset(diff_ts, y=None)

    test_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D_test = tsgm.dataset.Dataset(diff_ts, y=None)

    downstream_perf_metric = tsgm.metrics.DownstreamPerformanceMetric(
        evaluator=MockEvaluator()
    )
    assert downstream_perf_metric(D1, D2, D_test) == downstream_perf_metric(D2, D1, D_test)
    assert downstream_perf_metric(D1, D2, D_test) == 0

    assert downstream_perf_metric(D1, D2, D_test) == downstream_perf_metric(ts, diff_ts, test_ts)
    assert downstream_perf_metric(D1, D2, D_test) == downstream_perf_metric(D1, diff_ts, D_test)
    assert downstream_perf_metric(D1, D2, D_test) == downstream_perf_metric(ts, D2, D_test)
    mean, std = downstream_perf_metric(D1, D2, D_test, return_std=True)
    assert mean == 0 and std == 0


class FlattenTSOneClassSVM:
    def __init__(self, clf):
        self._clf = clf

    def fit(self, X):
        X_fl = X.reshape(X.shape[0], -1)
        self._clf.fit(X_fl)

    def predict(self, X):
        X_fl = X.reshape(X.shape[0], -1)
        return self._clf.predict(X_fl)


def test_privacy_inference_attack_metric():
    tr_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D_tr = tsgm.dataset.Dataset(tr_ts, y=None)

    test_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D_test = tsgm.dataset.Dataset(test_ts, y=None)

    sim_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D_sim = tsgm.dataset.Dataset(sim_ts, y=None)

    attacker = FlattenTSOneClassSVM(sklearn.svm.OneClassSVM())
    privacy_metric = tsgm.metrics.PrivacyMembershipInferenceMetric(
        attacker=attacker
    )

    assert isinstance(privacy_metric(D_tr, D_test, D_sim), float)


def test_mmd_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]]).astype(np.float32)
    D1 = tsgm.dataset.Dataset(ts, y=None)

    diff_ts = np.array([[[12, 13], [10, 10], [-1, -2]], [[-1, 32], [2, 1], [10, 8]]]).astype(np.float32)
    D2 = tsgm.dataset.Dataset(diff_ts, y=None)

    mmd_metric = tsgm.metrics.MMDMetric()
    assert mmd_metric(ts, diff_ts) == 1.0
    assert mmd_metric(ts, ts) == 0 and mmd_metric(diff_ts, diff_ts) == 0

    assert mmd_metric(D1, D2) == mmd_metric(ts, diff_ts)
    assert mmd_metric(D1, D1) == 0 and mmd_metric(D2, D2) == 0


def test_discriminative_metric():
    ts = np.sin(np.arange(10)[:, None, None] + np.arange(6)[None, :, None])  # sin_sequence, [10, 6, 3]
    D1 = tsgm.dataset.Dataset(ts, y=None)

    diff_ts = np.sin(np.arange(10)[:, None, None] + np.arange(6)[None, :, None]) + 1000  # sin_sequence, [10, 6, 3]
    D2 = tsgm.dataset.Dataset(diff_ts, y=None)

    model = tsgm.models.zoo["clf_cl_n"](seq_len=ts.shape[1], feat_dim=ts.shape[2], output_dim=2).model
    model.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    )
    discr_metric = tsgm.metrics.DiscriminativeMetric()
    # should be easy to be classified 
    assert discr_metric(d_hist=D1, d_syn=D2, model=model, test_size=0.2, random_seed=42, n_epochs=5) == 1.0
    assert discr_metric(d_hist=D1, d_syn=D2, model=model, metric=sklearn.metrics.precision_score, test_size=0.2, random_seed=42, n_epochs=5) == 1.0


def test_entropy_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]]).astype(np.float32)
    D1 = tsgm.dataset.Dataset(ts, y=None)
    spec_entropy_metric = tsgm.metrics.EntropyMetric()
    assert spec_entropy_metric(D1) == 2.6402430161833763


def test_demographic_parity():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]]).astype(np.float32)
    y = np.array([0, 1, 1])
    groups = np.array([0, 1 ,2])
    D = tsgm.dataset.Dataset(ts, y)

    synth_ts = ts
    synth_y = np.array([0, 1, 1])
    synth_groups = np.array([1, 2, 3])
    D_synth = tsgm.dataset.Dataset(synth_ts, synth_y)
    demographic_parity_metric = tsgm.metrics.DemographicParityMetric()
    result =  demographic_parity_metric(D, groups, D_synth, synth_groups)

    assert result == {
        0: np.inf,
        1: 1.0,
        2: 0,
        3: -np.inf
    }
