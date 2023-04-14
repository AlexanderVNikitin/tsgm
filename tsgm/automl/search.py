import itertools
import optuna
import typing
import tqdm

import urllib

import numpy as np
from packaging import version

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist

import logging

import tsgm

from tsgm.models.architectures.zoo import ConvnArchitecture


logger = logging.getLogger("automl")
logger.setLevel(logging.DEBUG)


class ModelSelection:
    """
    """
    def __init__(
        self,
        # model: tsgm.models.timeGAN.TimeGAN | tsgm.models.cvae | tsgm.models.cgan = None,
        n_trials: int = 100,
        optimize_towards: str = "maximize",
        search_space: typing.Optional[optuna.trial.Trial] = None,
        **model_kwargs
    ):
        """
        :param model:
        :param n_trials: int, how many points to sample in the parameters space
        :param optimize_towards: str, "maximize" or "minimize"
        :param search_space:
        :param **model_kwargs: keywords parametrs to the model
        """
        assert n_trials > 0, "n_trials needs to be greater than 0"
        assert optimize_towards in ["maximize", "minimize"]

        self.n_trials = n_trials
        self.direction = optimize_towards
        # self.model = model
        self.model = ConvnArchitecture
        self.model_args = model_kwargs
        self.search_space = search_space
        # if self.search_space is None:
        #     logger.info("Looking for a config file for the parameters space...")
        #     raise NotImplementedError

        self.study = optuna.create_study(direction=self.direction)
        return

    def learn(self, model, optimizer, dataset, mode="eval"):
        # objective
        accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

        for batch, (images, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(images, training=(mode == "train"))
                loss_value = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                )
                if mode == "eval":
                    accuracy(
                        tf.argmax(logits, axis=1, output_type=tf.int64), tf.cast(labels, tf.int64)
                    )
                else:
                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(zip(grads, model.variables))

        if mode == "eval":
            return accuracy

    def objective(self, trial):
        # Get MNIST data.
        train_ds, valid_ds = get_mnist()

        # Build model and optimizer.
        n_layers = trial.suggest_int("n_layers", 1, 10)
        num_hidden = trial.suggest_int("n_units", 4, 128, log=True)
        model = self.model(n_conv_blocks=n_layers, **self.model_args).model
        optimizer = create_optimizer(trial)

        # Training and validating cycle.
        EPOCHS = 2
        with tf.device("/cpu:0"):
            for _ in range(EPOCHS):
                self.learn(model, optimizer, train_ds, "train")

            accuracy = self.learn(model, optimizer, valid_ds, "eval")

        # Return last validation accuracy.
        return accuracy.result()

    def start(self):
        """
        Run the optimization
        """
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return


# TODO: Remove below once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128


def get_mnist():
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_valid = x_valid.astype("float32") / 255

    y_train = y_train.astype("int32")
    y_valid = y_valid.astype("int32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(60000).batch(BATCHSIZE).take(N_TRAIN_EXAMPLES)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.shuffle(10000).batch(BATCHSIZE).take(N_VALID_EXAMPLES)
    return train_ds, valid_ds


def create_optimizer(trial):
    # optimize the choice of optimizers as well as their parameters
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        # kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


if __name__ == "__main__":
    ModelSelection(**dict(seq_len=28, feat_dim=28, output_dim=10)).start()
