import typing
import numpy.typing as npt
import os
from tsgm.backend import get_backend

backend = get_backend()

# More flexible Tensor type that supports JAX arrays
if os.environ["KERAS_BACKEND"] == "jax":
    import jax.numpy as jnp
    Tensor = typing.Union[jnp.ndarray, npt.NDArray]
elif hasattr(backend, 'Tensor'):
    Tensor = typing.Union[backend.Tensor, npt.NDArray]
else:
    # Fallback for backends without explicit Tensor type
    Tensor = npt.NDArray

OptTensor = typing.Optional[Tensor]

Model = typing.Any  # TODO -- restrict
