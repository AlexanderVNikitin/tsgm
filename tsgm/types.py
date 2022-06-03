import typing

import tensorflow as tf
import numpy as np


Tensor = typing.Union[tf.Tensor, np.ndarray]
OptTensor = typing.Optional[Tensor]

Model = typing.Any  # TODO -- restrict
