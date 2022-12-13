import argparse
import pickle
import copy
import functools
import numpy as np
import sklearn.metrics
import tensorflow as tf

import tsgm


N_EPOCHS_DEFAULT = 10
BATCH_SIZE_DEFAULT = 256
LATENT_DIM_DEFAULT = 16
N_EPOCHS_EVALUATOR = 100
N_LAYERS_EVALUATOR = 1
N_EVALUATORS = 2


class PredictNextEvaluator:
    def __init__(self, hidden_dim, output_dim, n_layers, epochs=100, return_mean=False):
        self.model = tsgm.models.zoo["recurrent"](
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            network_type="GRU",
            name="AR_model",
        ).build(activation="tanh", return_sequences=False)
        self._zero_model = copy.deepcopy(self.model)
        self.model.compile(optimizer="adam", loss="mse")
        self.epochs = epochs
        self._return_mean = return_mean

    def reset(self):
        self.model = copy.deepcopy(self._zero_model)
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, X, y, epochs=100) -> None:
        self.model.fit(X, y, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, X_test):
        results = []
        for i in range(5):
            self.train(X[:, 0 : -1, :], X[:, -1, :], epochs=self.epochs)
            X_predicted = self.predict(X_test[:, 0 : -1, :])
            self.reset()
            results.append(sklearn.metrics.mean_squared_error(X_test[:, -1, :], X_predicted))
        if self._return_mean:
            return np.mean(results)
        else:
            return np.array(results)


class FlattenTSOneClassSVM:
    def __init__(self, clf):
        self._clf = clf

    def fit(self, X):
        X_fl = X.reshape(X.shape[0], -1)
        self._clf.fit(X_fl)

    def predict(self, X):
        X_fl = X.reshape(X.shape[0], -1)
        return self._clf.predict(X_fl)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Experiments with data generations')

    parser.add_argument("--generated-data", type=str, help='Path to the pickled data you want to model (.pkl)',
                        default=None, required=True)
    parser.add_argument("--source-data", type=str, help="Source data that was used to produce generated-data (.pkl)",
                        default=None, required=True)
    parser.add_argument("--holdout-data", type=str, help="Holdout test data (.pkl)",
                        default=None, required=True)

    parser.add_argument("--batch-size", type=int, help="Batch size", default=None)
    parser.add_argument("--latent-dim", type=int, help="Latent dimensionality", default=LATENT_DIM_DEFAULT)
    parser.add_argument("--n-epochs-evaluator", type=int, help="Number of epochs of the evaluator", default=N_EPOCHS_EVALUATOR)
    parser.add_argument("--n-evaluators", type=int, help="Number of epochs of the evaluator", default=N_EPOCHS_EVALUATOR)
    return parser.parse_args()


def _get_scale(generated_data) -> tuple:
    if np.min(generated_data) < 0:
        return (-1, 1)
    else:
        return (0, 1)


def load_data(data_path: str) -> tsgm.dataset.DatasetOrTensor:
    try:
        return pickle.load(open(data_path, "rb"))
    except pickle.PickleError:
        raise ValueError(f"Unsupported filetype in {data_path}")


def evaluate_similarity_metric(
        X_source: tsgm.dataset.DatasetOrTensor, X_syn: tsgm.dataset.DatasetOrTensor,
        X_holdout: tsgm.dataset.DatasetOrTensor) -> None:
    statistics = [functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_max_s, axis=2),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=2),

                  # mean
                  functools.partial(tsgm.metrics.statistics.axis_mean_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_mean_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_mean_s, axis=2),

                  # mode
                  functools.partial(tsgm.metrics.statistics.axis_mode_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_mode_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_mode_s, axis=2)]
    sim_metric = tsgm.metrics.SimilarityMetric(
        statistics=statistics, discrepancy=lambda x, y: np.linalg.norm(x - y)
    )
    print(f"Similarity Metric: {sim_metric(X_source, X_syn)}")


def evaluate_downstream_performance_metric(X_source, X_syn, X_holdout):
    downstream_perf_metric = tsgm.metrics.DownstreamPerformanceMetric(
        evaluator=PredictNextEvaluator(hidden_dim=8, output_dim=X_source.shape[2], n_layers=N_LAYERS_EVALUATOR, epochs=N_EPOCHS_EVALUATOR)
    )
    print(f"Downstream Performance Metric: {downstream_perf_metric(X_source, X_syn[:100], X_holdout)}")


def evaluate_consistency_metric(X_source, X_syn, X_holdout):
    consistency_metric = tsgm.metrics.ConsistencyMetric(
        evaluators=[PredictNextEvaluator(hidden_dim=8, output_dim=X_source.shape[2], n_layers=n_layers + 1, epochs=N_EPOCHS_EVALUATOR, return_mean=True) for n_layers in range(N_EVALUATORS)]
    )
    print(f"Consistency Metric: {consistency_metric(X_source, X_syn, X_holdout)}")


def evaluate_privacy_metric(X_source, X_syn, X_holdout):
    attacker = FlattenTSOneClassSVM(sklearn.svm.OneClassSVM())
    privacy_metric = tsgm.metrics.PrivacyMembershipInferenceMetric(
        attacker=attacker
    )
    print("Privacy Metric: ", privacy_metric(
        tsgm.dataset.Dataset(X_source, y=None),
        tsgm.dataset.Dataset(X_syn, y=None),
        tsgm.dataset.Dataset(X_holdout, y=None)))


def evaluate_metrics(X_source, X_syn, X_holdout):
    evaluate_similarity_metric(X_source, X_syn, X_holdout)
    evaluate_downstream_performance_metric(X_source, X_syn, X_holdout)
    evaluate_consistency_metric(X_source, X_syn, X_holdout)
    evaluate_privacy_metric(X_source, X_syn, X_holdout)


def main():
    tsgm.utils.fix_seeds()
    args = parse_arguments()
    source_data = load_data(args.source_data)
    X_syn = load_data(args.generated_data)
    holdout_data = load_data(args.holdout_data)

    scale = _get_scale(X_syn)
    scaler = tsgm.utils.TSFeatureWiseScaler(scale)
    X_train = scaler.fit_transform(source_data).astype(np.float32)
    X_holdout = scaler.transform(holdout_data).astype(np.float32)
    if isinstance(X_syn, tf.Tensor):
        X_syn = X_syn.numpy()
    evaluate_metrics(X_train, X_syn, X_holdout)


if __name__ == "__main__":
    main()
