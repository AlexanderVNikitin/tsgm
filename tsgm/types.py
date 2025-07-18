import typing
import numpy.typing as npt
from tsgm.backend import get_backend

backend = get_backend()

#  more flexible Tensor type
Tensor = typing.Union[backend.Tensor, npt.NDArray]

OptTensor = typing.Optional[Tensor]

Model = typing.Any  # TODO -- restrict
