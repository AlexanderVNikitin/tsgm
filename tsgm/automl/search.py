import optuna
import typing
from tensorflow.python.types.core import TensorLike
import tensorflow as tf

import tqdm

import logging

import tsgm

from tsgm.metrics.metrics import SimilarityMetric
from tsgm.models.timeGAN import TimeGAN
# from tsgm.models.architectures.zoo import ConvnArchitecture

# TODO: remove those when everything works
import numpy as np
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
        self.model = model
        # self.model = ConvnArchitecture
        self.model_args = model_kwargs
        self.search_space = search_space
        # if self.search_space is None:
        #     logger.info("Looking for a config file for the parameters space...")
        #     raise NotImplementedError

        self.study = optuna.create_study(direction=self.direction)

        # optimize for
        self.metric_to_optimize = None

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

    def _create_optimizer(self, trial):
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

    def learn(self, model, optimizer, dataset, mode="eval"):
        # objective
        objective_to_optimize = self.metric_to_optimize
        for batch, (images, labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(images, training=(mode == "train"))
                loss_value = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels
                    )
                )
                if mode == "eval":
                    objective_to_optimize(
                        tf.argmax(logits, axis=1, output_type=tf.int64),
                        tf.cast(labels, tf.int64),
                    )
                else:
                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(zip(grads, model.variables))

        if mode == "eval":
            return objective_to_optimize

    def objective(self, trial):
        # Get data
        # train_ds, valid_ds = self._get_dataset()
        train_ds, valid_ds = self.X_train, self.X_val

        # Build model and optimizer
        self._set_search_space_params(trial)
        n_layers = getattr(self, "n_layers")
        num_hidden = getattr(self, "num_hidden")

        # model = self.model(n_conv_blocks=n_layers, **self.model_args).model
        model = self.model(n_layers=n_layers)
        optimizer = self._create_optimizer(trial)
        model.compile(optimizer)

        # Training and validating cycle.
        EPOCHS = 2
        # with tf.device("/cpu:0"):
        #     for _ in range(EPOCHS):
        #         self.learn(model, optimizer, train_ds, "train")

        #    objective_to_optimize = self.learn(model, optimizer, valid_ds, "eval")
        model.fit(data=train_ds, epochs=EPOCHS)
        _y = model.generate(n_samples=10)
        objective_to_optimize = self.metric_to_optimize(_y, valid_ds)
        # Return last validation score
        # return objective_to_optimize.result()
        return objective_to_optimize

    def start(
        self,
        metric_to_optimize: typing.Callable,
        X_train: TensorLike,
        X_val: TensorLike,
        y_train: typing.Optional[TensorLike] = None,
        y_val: typing.Optional[TensorLike] = None,
    ):
        """
        Run the optimization
        """
        self.metric_to_optimize = metric_to_optimize
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


def MinMaxScaler(data):
  """Min Max normalizer.

  Args:
    - data: original data

  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.

  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

  Returns:
    - data: generated data
  """
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)
      phase = np.random.uniform(0, 0.1)

      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
      temp.append(temp_data)

    # Align row/column
    temp = np.transpose(np.asarray(temp))
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)

  return data


if __name__ == "__main__":
    search_space = {
        "n_layers": ("int", dict(low=1, high=10)),
        "num_hidden": ("int", dict(low=4, high=128, log=True)),
    }
    # X_train, y_train, X_val, y_val = get_mnist()
    X = sine_data_generation(2000, 24, 6)
    ModelSelection(
        model=TimeGAN, search_space=search_space, **dict(seq_len=28, feat_dim=28, output_dim=10)
    ).start(
        metric_to_optimize= SimilarityMetric, # tf.metrics.Accuracy("accuracy", dtype=tf.float32),
        X_train=X,
        X_val=X,
        # y_train=y_train,
        # y_val=y_val,
    )
