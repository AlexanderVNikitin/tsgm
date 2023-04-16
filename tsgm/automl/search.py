import optuna
import typing
from tensorflow.python.types.core import TensorLike
import tensorflow as tf

import tqdm

import logging

import tsgm

from tsgm.metrics.metrics import SimilarityMetric
from tsgm.models.architectures.zoo import ConvnArchitecture

# TODO: remove those 2
import urllib
from tensorflow.keras.datasets import mnist


logger = logging.getLogger("automl")
logger.setLevel(logging.DEBUG)


class ModelSelection:
    """ """

    def __init__(
        self,
        model: tsgm.models.timeGAN.TimeGAN
        | tsgm.models.cvae.BetaVAE
        | tsgm.models.cvae.cBetaVAE
        | tsgm.models.cgan.GAN = None,
        n_trials: int = 100,
        optimize_towards: str = "maximize",
        search_space: typing.Optional[
            typing.Dict[str, typing.Tuple[str, typing.Dict[str, typing.Any]]]
        ] = None,
        **model_kwargs,
    ):
        """
        :param model:
        :param n_trials: int, how many points to sample in the parameters space
        :param optimize_towards: str, "maximize" or "minimize"
        :param search_space:
        :param **model_kwargs: keywords parameters to the model
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

        # data
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        return

    def _get_dataset(self) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        :return: (train dataset, validation dataset)
        """
        N_TRAIN_EXAMPLES = 3000
        N_VALID_EXAMPLES = 1000
        BATCHSIZE = 128
        if self.y_train is not None and self.y_val is not None:
            train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
            valid_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        else:
            train_ds = tf.data.Dataset.from_tensor_slices((self.X_train))
            valid_ds = tf.data.Dataset.from_tensor_slices((self.X_val))

        train_ds = train_ds.shuffle(60000).batch(BATCHSIZE).take(N_TRAIN_EXAMPLES)
        valid_ds = valid_ds.shuffle(10000).batch(BATCHSIZE).take(N_VALID_EXAMPLES)
        return train_ds, valid_ds

    def _set_search_space_params(self, trial):
        for name, (type_to_search, search_args) in self.search_space.items():
            if type_to_search == "int":
                setattr(self, name, trial.suggest_int(name=name, **search_args))
            elif type_to_search == "float":
                setattr(self, name, trial.suggest_float(name=name, **search_args))
            else:
                pass
        return

    def learn(self, model, optimizer, dataset, mode="eval"):
        # objective
        accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)
        # accuracy = tf.metrics.KLDivergence(name='kullback_leibler_divergence', dtype=tf.float32)
        for batch, (images, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(images, training=(mode == "train"))
                loss_value = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels
                    )
                )
                if mode == "eval":
                    accuracy(
                        tf.argmax(logits, axis=1, output_type=tf.int64),
                        tf.cast(labels, tf.int64),
                    )
                else:
                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(zip(grads, model.variables))

        if mode == "eval":
            return accuracy

    def objective(self, trial):
        # Get data
        train_ds, valid_ds = self._get_dataset()

        # Build model and optimizer
        self._set_search_space_params(trial)
        n_layers = getattr(self, "n_layers")
        num_hidden = getattr(self, "num_hidden")
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

    def start(
        self,
        X_train: TensorLike,
        X_val: TensorLike,
        y_train: typing.Optional[TensorLike] = None,
        y_val: typing.Optional[TensorLike] = None,
    ):
        """
        Run the optimization
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return


# TODO: Remove below once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


def get_mnist():
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_valid = x_valid.astype("float32") / 255

    y_train = y_train.astype("int32")
    y_valid = y_valid.astype("int32")

    return x_train, y_train, x_valid, y_valid


def create_optimizer(trial):
    # optimize the choice of optimizers as well as their parameters
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float(
            "rmsprop_momentum", 1e-5, 1e-1, log=True
        )
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float(
            "adam_learning_rate", 1e-5, 1e-1, log=True
        )
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float(
            "sgd_opt_momentum", 1e-5, 1e-1, log=True
        )

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


if __name__ == "__main__":
    search_space = {
        "n_layers": ("int", dict(low=1, high=10)),
        "num_hidden": ("int", dict(low=4, high=128, log=True)),
    }
    X_train, y_train, X_val, y_val = get_mnist()
    ModelSelection(
        search_space=search_space, **dict(seq_len=28, feat_dim=28, output_dim=10)
    ).start(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
