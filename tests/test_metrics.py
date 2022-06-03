import pytest

import numpy as np
import tensorflow as tf
import functools
import sklearn

import tsgm


def test_statistics():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    assert tsgm.metrics.statistics.axis_max_s(ts, axis=None) == [21]
    assert tsgm.metrics.statistics.axis_min_s(ts, axis=None) == [-11]

    assert (tsgm.metrics.statistics.axis_max_s(ts, axis=1) == [11,  2, 10, 21]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts, axis=1) == [0, -11, 1, -1]).all()

    assert (tsgm.metrics.statistics.axis_max_s(ts, axis=2) == [2, 11, 2, 21, 1, 8]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts, axis=2) == [0, -11, 1, 10, -1, 6]).all()

    assert (tsgm.metrics.statistics.axis_mode_s(ts, axis=None) == [1]).all()

    # Now, checking with tf.Tensor
    ts_tf = tf.convert_to_tensor(ts)

    assert (tsgm.metrics.statistics.axis_max_s(ts_tf, axis=None) == [21]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts_tf, axis=None) == [-11]).all()

    assert (tsgm.metrics.statistics.axis_max_s(ts, axis=1) == [11,  2, 10, 21]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts, axis=1) == [0, -11, 1, -1]).all()

    assert (tsgm.metrics.statistics.axis_max_s(ts, axis=2) == [2, 11, 2, 21, 1, 8]).all()
    assert (tsgm.metrics.statistics.axis_min_s(ts, axis=2) == [0, -11, 1, 10, -1, 6]).all()

    assert (tsgm.metrics.statistics.axis_mode_s(ts, axis=None) == [1]).all()


def test_similarity_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    diff_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    sim_ts = ts + 1e-7
    statistics = [functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1)]

    sim_metric = tsgm.metrics.SimilarityMetric(
        statistics=statistics, discrepancy=lambda x, y: np.linalg.norm(x - y)
    )
    assert sim_metric(ts, diff_ts) < sim_metric(ts, sim_ts)
    stat_results = sim_metric.stats(ts)
    
    assert len(stat_results) == 10
    assert sim_metric._discrepancy(sim_metric.stats(ts), sim_metric.stats(diff_ts)) == sim_metric(ts, diff_ts)
    assert sim_metric(ts, diff_ts) == sim_metric(diff_ts, ts)


class MockEvaluator:
    def evaluate(self, D: tsgm.dataset.Dataset) -> float:
        return 0.42


def test_consistency_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D1 = tsgm.dataset.Dataset(ts, y=None)

    diff_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D2 = tsgm.dataset.Dataset(diff_ts, y=None)

    n_models = 5
    consistency_metric = tsgm.metrics.ConsistencyMetric(
        evaluators = [MockEvaluator() for _ in range(n_models)]
    )
    model_results = consistency_metric._apply_models(D1)
    assert len(model_results) == n_models

    assert consistency_metric(D1, D2) == 1.0


def test_downstream_performance_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D1 = tsgm.dataset.Dataset(ts, y=None)

    diff_ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]])
    D2 = tsgm.dataset.Dataset(diff_ts, y=None)

    downstream_perf_metric = tsgm.metrics.DownstreamPerformanceMetric(
        evaluator=MockEvaluator()
    )
    assert downstream_perf_metric(D1, D2) == downstream_perf_metric(D2, D1)
    assert downstream_perf_metric(D1, D2) == 0


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
