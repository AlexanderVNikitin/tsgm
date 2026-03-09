import typing
import numpy.typing as npt

from tsgm.backend import _backend_name

# More flexible Tensor type that supports JAX arrays
if _backend_name == "jax":
    from tsgm.backend import get_backend
    backend = get_backend()
    import jax.numpy as jnp
    Tensor = typing.Union[jnp.ndarray, npt.NDArray]
elif _backend_name is not None:
    from tsgm.backend import get_backend
    backend = get_backend()
    if hasattr(backend, 'Tensor'):
        Tensor = typing.Union[backend.Tensor, npt.NDArray]
    else:
        Tensor = npt.NDArray
else:
    # No backend available (e.g. doc builds)
    Tensor = npt.NDArray

OptTensor = typing.Optional[Tensor]

Model = typing.Any  # TODO -- restrict
