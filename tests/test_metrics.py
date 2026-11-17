import logging
import pytest

import numpy as np
import keras
from keras import ops
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
    ts_tf = ops.convert_to_tensor(ts)

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


class MockAttacker:
    """Flags a fixed set of samples as members (+1) and the rest as non-members (-1)."""
    def __init__(self, member_idx):
        self._member_idx = member_idx

    def fit(self, X):
        pass

    def predict(self, X):
        labels = -np.ones(X.shape[0], dtype=int)
        labels[self._member_idx] = 1
        return labels


def test_privacy_inference_attack_metric_is_one_minus_precision():
    n = 10
    D_tr = tsgm.dataset.Dataset(np.zeros((n, 3, 2)), y=None)
    D_test = tsgm.dataset.Dataset(np.ones((n, 3, 2)), y=None)
    D_syn = tsgm.dataset.Dataset(np.zeros((n, 3, 2)), y=None)

    # Two samples are flagged as members, both come from the training set: precision = 1.0, recall = 0.2
    cautious_attacker = MockAttacker(member_idx=[0, 1])
    privacy_metric = tsgm.metrics.PrivacyMembershipInferenceMetric(attacker=cautious_attacker)
    assert privacy_metric(D_tr, D_syn, D_test) == pytest.approx(0.0)

    # Every sample is flagged as a member: precision = 0.5, recall = 1.0
    greedy_attacker = MockAttacker(member_idx=np.arange(2 * n))
    privacy_metric = tsgm.metrics.PrivacyMembershipInferenceMetric(attacker=greedy_attacker)
    assert privacy_metric(D_tr, D_syn, D_test) == pytest.approx(0.5)

    # A custom metric is called with the scikit-learn convention, metric(y_true, y_pred)
    privacy_metric = tsgm.metrics.PrivacyMembershipInferenceMetric(
        attacker=cautious_attacker, metric=sklearn.metrics.recall_score
    )
    assert privacy_metric(D_tr, D_syn, D_test) == pytest.approx(0.8)


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
        keras.optimizers.Adam(),
        keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    )
    discr_metric = tsgm.metrics.DiscriminativeMetric()
    # should be easy to be classified - allow for slight variance across backends
    score1 = discr_metric(d_hist=D1, d_syn=D2, model=model, test_size=0.2, random_seed=42, n_epochs=5)
    assert score1 >= 0.5  # Should be better than random chance
    score2 = discr_metric(d_hist=D1, d_syn=D2, model=model, metric=sklearn.metrics.precision_score, test_size=0.2, random_seed=42, n_epochs=5)
    assert score2 >= 0.5  # Should be better than random chance


class MockDiscriminator:
    """Tells apart all-ones (real, label 1) and all-zeros (synthetic, label 0) samples, returning outputs of a given shape."""
    def __init__(self, output, invert=False):
        self._output = output
        self._invert = invert

    def fit(self, X, y, epochs=1):
        pass

    def predict(self, X):
        p_real = (X.mean(axis=(1, 2)) > 0.5).astype(np.float32)
        if self._invert:
            p_real = 1 - p_real
        if self._output == "sigmoid_flat":  # shape (n,)
            return p_real
        elif self._output == "sigmoid_column":  # shape (n, 1), as returned by a Dense(1, activation="sigmoid") head
            return p_real[:, None]
        else:  # shape (n, 2), as returned by a softmax head
            return np.stack([1 - p_real, p_real], axis=1)


@pytest.mark.parametrize("output", ["sigmoid_flat", "sigmoid_column", "softmax"])
def test_discriminative_metric_output_shapes(output):
    real, syn = np.ones((20, 6, 2)), np.zeros((20, 6, 2))
    discr_metric = tsgm.metrics.DiscriminativeMetric()

    score = discr_metric(d_hist=real, d_syn=syn, model=MockDiscriminator(output), test_size=0.5, n_epochs=1, random_seed=42)
    assert score == 1.0

    score = discr_metric(d_hist=real, d_syn=syn, model=MockDiscriminator(output, invert=True), test_size=0.5, n_epochs=1, random_seed=42)
    assert score == 0.0


def test_discriminative_metric_keras_sigmoid_output():
    seq_len, feat_dim = 6, 2
    real, syn = np.ones((20, seq_len, feat_dim), dtype=np.float32), np.zeros((20, seq_len, feat_dim), dtype=np.float32)

    model = keras.Sequential([
        keras.Input((seq_len, feat_dim)),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    # sigmoid(12 - 6) ~ 1 for real samples and sigmoid(-6) ~ 0 for synthetic ones
    model.layers[-1].set_weights([np.ones((seq_len * feat_dim, 1), dtype=np.float32), np.full((1,), -6.0, dtype=np.float32)])
    # zero learning rate keeps the hand-set weights, so that the test is deterministic
    model.compile(keras.optimizers.SGD(learning_rate=0.0), keras.losses.BinaryCrossentropy())

    discr_metric = tsgm.metrics.DiscriminativeMetric()
    score = discr_metric(d_hist=real, d_syn=syn, model=model, test_size=0.5, n_epochs=1, random_seed=42)
    assert score == 1.0


def test_entropy_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]]).astype(np.float32)
    D1 = tsgm.dataset.Dataset(ts, y=None)
    spec_entropy_metric = tsgm.metrics.EntropyMetric()
    assert abs(spec_entropy_metric(D1) - 2.6402430161833763) < 1e-6


def test_shannon_entropy_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]]).astype(np.float32)
    y = np.array([1] * ts.shape[0])
    D1 = tsgm.dataset.Dataset(ts, y=y)
    sdi_metric = tsgm.metrics.ShannonEntropyMetric()
    assert sdi_metric(D1) == 0
    y = np.array([1, 2])
    D2 = tsgm.dataset.Dataset(ts, y=y)
    assert sdi_metric(D2) > 0


def test_pairwise_distance_metric():
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[0, 2], [11, -11], [1, 2]]]).astype(np.float32)
    D1 = tsgm.dataset.Dataset(ts, y=None)
    pd_metric = tsgm.metrics.PairwiseDistanceMetric()
    assert np.mean(pd_metric(D1)) == 0
    ts = np.array([[[0, 2], [11, -11], [1, 2]], [[10, 21], [1, -1], [6, 8]]]).astype(np.float32)
    D2 = tsgm.dataset.Dataset(ts, y=None)
    assert np.mean(pd_metric(D2)) > 0


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


def test_predictive_parity():
    metric = tsgm.metrics.PredictiveParityMetric()
    y_pred_hist = np.array([0, 1, 1, 0, 0, 1])
    y_true_hist = np.array([0, 1, 0, 0, 0, 1])
    groups_hist = np.array([0, 0, 0, 1, 1, 1])
    y_true_synth = np.array([0, 0, 1, 0, 0, 1])
    y_pred_synth = np.array([1, 0, 0, 0, 0, 1])
    groups_synth = np.array([0, 0, 0, 1, 1, 1])
    result = metric(y_true_hist, y_pred_hist, groups_hist, y_true_synth, y_pred_synth, groups_synth)
    assert result[0] > result[1] and result[1] == 0


def test_demographic_parity_list_groups():
    ts = np.zeros((6, 3, 2), dtype=np.float32)
    D_hist = tsgm.dataset.Dataset(ts, y=np.array([0, 0, 0, 1, 1, 1]))
    D_synth = tsgm.dataset.Dataset(ts, y=np.array([1, 1, 1, 1, 1, 1]))
    groups_hist = [0, 1, 0, 1, 1, 0]
    groups_synth = [0, 0, 1, 1, 0, 1]

    demographic_parity_metric = tsgm.metrics.DemographicParityMetric()
    result = demographic_parity_metric(D_hist, groups_hist, D_synth, groups_synth)

    # group 0: historical labels are [0, 0, 1], synthetic labels are [1, 1, 1]
    # group 1: historical labels are [0, 1, 1], synthetic labels are [1, 1, 1]
    assert result == {0: pytest.approx(2 / 3), 1: pytest.approx(1 / 3)}
    assert result == demographic_parity_metric(D_hist, np.array(groups_hist), D_synth, np.array(groups_synth))


def test_predictive_parity_list_inputs():
    # the example from the docstring of PredictiveParityMetric
    y_true_hist = [0, 1, 0, 1, 1, 0]
    y_pred_hist = [0, 1, 0, 0, 1, 1]
    groups_hist = [0, 1, 0, 1, 1, 0]
    y_true_synth = [1, 0, 1, 0, 0, 1]
    y_pred_synth = [1, 0, 1, 1, 0, 0]
    groups_synth = [1, 1, 0, 0, 0, 1]

    metric = tsgm.metrics.PredictiveParityMetric()
    result = metric(y_true_hist, y_pred_hist, groups_hist, y_true_synth, y_pred_synth, groups_synth)

    # group 0: precision is 0 on historical data and 0.5 on synthetic data
    # group 1: precision is 1 on both
    assert result == {0: pytest.approx(-0.5), 1: pytest.approx(0.0)}
    assert result == metric(
        np.array(y_true_hist), np.array(y_pred_hist), np.array(groups_hist),
        np.array(y_true_synth), np.array(y_pred_synth), np.array(groups_synth))


def test_predictive_parity_groups_warning(caplog):
    metric = tsgm.metrics.PredictiveParityMetric()
    y_true, y_pred = np.array([0, 1, 0, 0, 0, 1]), np.array([0, 1, 1, 0, 0, 1])
    groups = np.array([0, 0, 0, 1, 1, 1])
    warning = "not entirely identical"

    with caplog.at_level(logging.WARNING):
        metric(y_true, y_pred, groups, y_true, y_pred, groups)
    assert warning not in caplog.text

    other_groups = np.array([0, 0, 0, 2, 2, 2])
    with caplog.at_level(logging.WARNING):
        result = metric(y_true, y_pred, groups, y_true, y_pred, other_groups)
    assert warning in caplog.text
    assert result[1] == np.inf and result[2] == -np.inf
