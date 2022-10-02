import argparse
import pickle
import copy
import functools
import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras

from io import BytesIO

from keras import layers

import tsgm


N_EPOCHS_DEFAULT = 10
BATCH_SIZE_DEFAULT = 256
LATENT_DIM_DEFAULT = 16


def _gen_dataset(X, batch_size=BATCH_SIZE_DEFAULT):
    scaler = tsgm.utils.TSFeatureWiseScaler((-1, 1))
    X_train = scaler.fit_transform(X).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset, X_train


def parse_arguments():
    parser = argparse.ArgumentParser(description='Experiments with data generations')

    parser.add_argument('--source_data', type=str, help='Path to the pickled data you want to model',
                        default="source_data")
    parser.add_argument('--source_data_label', type=str, help="Path to the pickled source data labels", 
                        default="")

    parser.add_argument('--dest_data', type=str, help='Destination path for the pickled data',
                        default="generated_data")
    parser.add_argument('--n_epochs', type=int, help='Destination path for the pickled data',
                        default=N_EPOCHS_DEFAULT)
    parser.add_argument('--batch_size', type=int, help='Batch size for training',
                        default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--dump_model_path", type=str, help="Path where the serialized model is saved",
                        default="./saved_model.pkl")
    parser.add_argument("--architecture_type", type=str, help="[GAN|TimeGAN]", default="GAN")

    return parser.parse_args()


class PredictNextEvaluator:
    def __init__(self, hidden_dim, output_dim, n_layers, epochs=100):
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

    def reset(self):
        self.model = self._zero_model
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, X, y, epochs=100) -> None:
        self.model.fit(X, y, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X):
        results = []
        ln_interval = 5
        for split in range(X.shape[1] - ln_interval):
            self.train(X[:, split : split + ln_interval, :], X[:, split + ln_interval, :], epochs=self.epochs)
            X_predicted = self.predict(X)
            self.reset()
            results.append(sklearn.metrics.mean_squared_error(X[:, split + ln_interval, :], X_predicted))
        return np.array(results)


def evaluate_metrics(X_source, X_syn):
    statistics = [functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None),
                  functools.partial(tsgm.metrics.statistics.axis_max_s, axis=1),
                  functools.partial(tsgm.metrics.statistics.axis_min_s, axis=1)]

    sim_metric = tsgm.metrics.SimilarityMetric(
        statistics=statistics, discrepancy=lambda x, y: np.linalg.norm(x - y)
    )
    print(f"Similarity Metric: {sim_metric(X_source, X_syn)}")

    downstream_perf_metric = tsgm.metrics.DownstreamPerformanceMetric(
        evaluator=PredictNextEvaluator(hidden_dim=2, output_dim=X_source.shape[2], n_layers=1, epochs=100)
    )
    print(f"Downstream Performance Metric: {downstream_perf_metric(X_source, X_syn)}")


if __name__ == "__main__":
    args = parse_arguments()

    X_source = pickle.load(open(args.source_data, "rb"))
    if args.source_data_label != "":
        y_sourse = pickle.load(open(args.source_data, "rb"))
    assert len(X_source.shape) == 3
    n, seq_len, feature_dim = X_source.shape
    print(f"n={n}, seq_len={seq_len} feature_dim={feature_dim}")

    dataset, X_preprocessed = _gen_dataset(X_source, batch_size=args.batch_size)

    architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=LATENT_DIM_DEFAULT, output_dim=0)
    if args.architecture_type.lower() == "gan":
        discriminator, generator = architecture.discriminator, architecture.generator

        gan = tsgm.models.cgan.GAN(
            discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM_DEFAULT
        )
        gan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss_fn=keras.losses.BinaryCrossentropy(),
        )
        gan.fit(dataset, epochs=args.n_epochs)
    elif args.architecture.lower() == "timegan":
        gan = tsgm.models.timeGAN.TimeGAN(seq_len=X_preprocessed.shape[1],
            n_features=X_preprocessed.shape[2], module='lstm', epochs=args.n_epochs)
        gan.compile()
        gan.fit(X_preprocessed, epochs=args.n_epochs)
    else:
        raise ValueError(f"Unknown architecture type: {args.architecture_type}")
    X_syn = gan.generate(n)
    pickle.dump(X_syn, open(args.dest_data, "wb"))
    print(f"The generated dataset is saved to {args.dest_data}")

    evaluate_metrics(X_preprocessed, X_syn)

    # TODO: save the model
