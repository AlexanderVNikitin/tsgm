import os

# Set the Keras backend before importing Keras anywhere
# This must be done before any Keras imports
def _set_keras_backend():
    """Set the Keras backend based on available libraries."""
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
        import torch as torch_test
        os.environ["KERAS_BACKEND"] = "torch"
        return "torch"
    except ImportError:
        pass
    
    raise ImportError("No backend found. Please install tensorflow or torch.")

# Set backend before any other imports
_backend_name = _set_keras_backend()

# Global variables to store backend modules
tf = None
torch = None
tensorflow_probability = None
Keras_Dataset = None

# Try to import TensorFlow first
try:
    import tensorflow as tf_module
    tf = tf_module
    # Check if TensorFlow is properly installed by accessing a core module
    if hasattr(tf, 'data') and hasattr(tf, 'compat'):
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
    if not _has_tensorflow:
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

# If neither backend is available, raise an error
if not _has_tensorflow and not _has_torch:
    raise ImportError("No backend found. Please install tensorflow or torch.")

def get_backend():
    """Get the current backend module."""
    if os.environ["KERAS_BACKEND"] == "tensorflow":
        if tf is None:
            raise ImportError("TensorFlow backend requested but not available.")
        return tf
    elif os.environ["KERAS_BACKEND"] == "torch":
        if torch is None:
            raise ImportError("PyTorch backend requested but not available.")
        return torch
    else:
        raise ValueError("No backend found. Please install tensorflow or torch.")

def get_distributions():
    """Get the distributions module for the current backend."""
    if os.environ["KERAS_BACKEND"] == "tensorflow":
        if tensorflow_probability is None:
            raise ImportError("TensorFlow Probability not available. Install with: pip install tensorflow-probability")
        return tensorflow_probability.distributions
    elif os.environ["KERAS_BACKEND"] == "torch":
        if torch is None:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        return torch.distributions
    else:
        raise ValueError("No backend found. Please install tensorflow or torch.")

# tf.function decorator for tensorflow backend or no op decorator for torch backend
def tf_function_decorator(func):
    """Decorator that applies tf.function only for TensorFlow backend."""
    if os.environ["KERAS_BACKEND"] == "tensorflow" and tf is not None:
        return tf.function(func)
    else:
        # no op decorator
        return func
    