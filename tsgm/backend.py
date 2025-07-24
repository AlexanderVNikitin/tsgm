import os

# Set the Keras backend before importing Keras anywhere
# This must be done before any Keras imports


def _set_keras_backend():
    """Set the Keras backend based on available libraries."""
    # Check if JAX is properly installed
    try:
        import jax
        import jax.numpy as jnp
        if hasattr(jax, 'random') and hasattr(jnp, 'array'):
            os.environ["KERAS_BACKEND"] = "jax"
            return "jax"
    except (ImportError, AttributeError):
        pass

    # Check if TensorFlow is properly installed
    try:
        import tensorflow as tf_test
        if hasattr(tf_test, 'data') and hasattr(tf_test, 'compat'):
            os.environ["KERAS_BACKEND"] = "tensorflow"
            return "tensorflow"
    except (ImportError, AttributeError):
        pass

    # Try PyTorch as fallback
    try:
        import torch  # noqa: F401
        os.environ["KERAS_BACKEND"] = "torch"
        return "torch"
    except ImportError:
        pass

    raise ImportError("No backend found. Please install jax, tensorflow, or torch.")


# Set backend before any other imports
_backend_name = _set_keras_backend()

# Global variables to store backend modules
tf = None
torch = None
jax = None
jax_numpy = None
tensorflow_probability = None
Keras_Dataset = None

# Try to import JAX first
try:
    import jax as jax_module
    import jax.numpy as jnp_module
    jax = jax_module
    jax_numpy = jnp_module
    # Check if JAX is properly installed by accessing core modules
    if hasattr(jax, 'random') and hasattr(jnp_module, 'array'):
        _has_jax = True
    else:
        jax = None
        jax_numpy = None
        _has_jax = False
except (ImportError, AttributeError):
    jax = None
    jax_numpy = None
    _has_jax = False

# Try to import TensorFlow
try:
    import tensorflow as tf_module
    tf = tf_module
    # Check if TensorFlow is properly installed by accessing a core module
    if hasattr(tf, 'data') and hasattr(tf, 'compat'):
        if not _has_jax:
            Keras_Dataset = tf.data.Dataset
        _has_tensorflow = True
    else:
        tf = None
        _has_tensorflow = False
except (ImportError, AttributeError):
    tf = None
    _has_tensorflow = False

# Try to import PyTorch
try:
    import torch as torch_module
    torch = torch_module
    import torch.utils
    import torch.utils.data
    if not _has_jax and not _has_tensorflow:
        Keras_Dataset = torch.utils.data.DataLoader
    _has_torch = True
except ImportError:
    _has_torch = False

# Try to import TensorFlow Probability
try:
    import tensorflow_probability as tfp
    tensorflow_probability = tfp
    _has_tfp = True
except (ImportError, AttributeError):
    tensorflow_probability = None
    _has_tfp = False

# If no backend is available, raise an error
if not _has_jax and not _has_tensorflow and not _has_torch:
    raise ImportError("No backend found. Please install jax, tensorflow, or torch.")


def get_backend():
    """Get the current backend module."""
    if os.environ["KERAS_BACKEND"] == "jax":
        if jax is None:
            raise ImportError("JAX backend requested but not available.")
        return jax
    elif os.environ["KERAS_BACKEND"] == "tensorflow":
        if tf is None:
            raise ImportError("TensorFlow backend requested but not available.")
        return tf
    elif os.environ["KERAS_BACKEND"] == "torch":
        if torch is None:
            raise ImportError("PyTorch backend requested but not available.")
        return torch
    else:
        raise ValueError("No backend found. Please install jax, tensorflow, or torch.")


def get_distributions():
    """Get the distributions module for the current backend."""
    if os.environ["KERAS_BACKEND"] == "jax":
        try:
            import jax.scipy.stats as jax_distributions
            return jax_distributions
        except ImportError:
            raise ImportError("JAX distributions not available. Install with: pip install jax")
    elif os.environ["KERAS_BACKEND"] == "tensorflow":
        if tensorflow_probability is None:
            raise ImportError("TensorFlow Probability not available. Install with: pip install tensorflow-probability")
        return tensorflow_probability.distributions
    elif os.environ["KERAS_BACKEND"] == "torch":
        if torch is None:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        return torch.distributions
    else:
        raise ValueError("No backend found. Please install jax, tensorflow, or torch.")

# tf.function decorator for tensorflow backend, jax.jit for jax backend, or no op decorator for torch backend


def tf_function_decorator(func):
    """Decorator that applies tf.function for TensorFlow, jax.jit for JAX, or no-op for PyTorch backend."""
    if os.environ["KERAS_BACKEND"] == "tensorflow" and tf is not None:
        return tf.function(func)
    elif os.environ["KERAS_BACKEND"] == "jax" and jax is not None:
        return jax.jit(func)
    else:
        # no op decorator
        return func
