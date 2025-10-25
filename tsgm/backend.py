import os

# Enable MPS fallback for unsupported operations early (like linalg_qr in LSTM)
# This must be set before importing PyTorch for the first time
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# For TimeGAN compatibility, check if we should disable MPS entirely
if os.environ.get("DISABLE_MPS_FOR_TIMEGAN", "0") == "1":
    # Completely disable MPS to avoid broadcasting and device issues
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Set the Keras backend before importing Keras anywhere
# This must be done before any Keras imports


def _is_backend_available(backend_name):
    """Check if a specific backend is available."""
    if backend_name == "jax":
        try:
            import jax
            import jax.numpy as jnp
            return hasattr(jax, 'random') and hasattr(jnp, 'array')
        except (ImportError, AttributeError):
            return False
    elif backend_name == "tensorflow":
        try:
            import tensorflow as tf_test
            return hasattr(tf_test, 'data') and hasattr(tf_test, 'compat')
        except (ImportError, AttributeError):
            return False
    elif backend_name == "torch":
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False
    return False


def _set_keras_backend():
    """Set the Keras backend based on available libraries."""
    # Check if KERAS_BACKEND is already set and respect it
    current_backend = os.environ.get("KERAS_BACKEND", "").lower()

    if current_backend in ["jax", "tensorflow", "torch"]:
        if _is_backend_available(current_backend):
            return current_backend
        else:
            print(f"Warning: Requested backend '{current_backend}' is not available. Auto-detecting...")

    # Auto-detect backend priority: JAX > TensorFlow > PyTorch
    for backend in ["jax", "tensorflow", "torch"]:
        if _is_backend_available(backend):
            os.environ["KERAS_BACKEND"] = backend
            return backend

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

    # Set default tensor type to float32 for MPS compatibility
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)

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

# Set Keras_Dataset based on the current backend
if os.environ["KERAS_BACKEND"] == "tensorflow" and _has_tensorflow:
    Keras_Dataset = tf.data.Dataset
elif os.environ["KERAS_BACKEND"] == "torch" and _has_torch:
    Keras_Dataset = torch.utils.data.DataLoader
else:
    # For JAX backend or when no backend is available, keep Keras_Dataset as None
    # since JAX doesn't have a direct equivalent
    Keras_Dataset = None


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


class JAXUniform:
    """JAX wrapper for uniform distribution to match TensorFlow Probability API."""
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, shape=()):
        import jax
        # Generate a new key each time (this is not ideal but works for testing)
        key = jax.random.PRNGKey(hash((self.low, self.high)) % (2**32))
        return jax.random.uniform(key, shape=shape, minval=self.low, maxval=self.high)


class JAXDistributionsWrapper:
    """Wrapper to provide TensorFlow Probability-like API for JAX."""
    def __init__(self):
        pass

    @property
    def Uniform(self):
        return JAXUniform


def get_distributions():
    """Get the distributions module for the current backend."""
    if os.environ["KERAS_BACKEND"] == "jax":
        try:
            import jax.scipy.stats as jax_distributions  # noqa: F401
            return JAXDistributionsWrapper()
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
