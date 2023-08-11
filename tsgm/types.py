import typing

import tensorflow as tf
import numpy.typing as npt


Tensor = typing.Union[tf.Tensor, npt.NDArray]
OptTensor = typing.Optional[Tensor]

Model = typing.Any  # TODO -- restrict
