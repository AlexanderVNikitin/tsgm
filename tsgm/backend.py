import os

import torch.utils
import torch.utils.data

try:
    import tensorflow as tf
    os.environ["KERAS_BACKEND"] = "tensorflow"
    Keras_Dataset = tf.data.Dataset
except ImportError:
    try:
        import torch
        os.environ["KERAS_BACKEND"] = "torch"
        Keras_Dataset = torch.utils.data.DataLoader
    except ImportError:
        raise ImportError("No backend found. Please install tensorflow or torch .")

def get_backend():
    if os.environ["KERAS_BACKEND"] == "tensorflow":
        return tf
    elif os.environ["KERAS_BACKEND"] == "torch":
        return torch
    else:
        raise ValueError("No backend found. Please install tensorflow or torch.")


#  I am not sure if this is correct to import distributions here
def get_distributions():
    if os.environ["KERAS_BACKEND"] == "tensorflow":
        return tensorflow_probability.distributions
    elif os.environ["KERAS_BACKEND"] == "torch":
        return torch.distributions
    else:
        raise ValueError("No backend found. Please install tensorflow or torch.")
    

# tf.function decorator for tensorflow backend or no op decorator for torch backend
if os.environ["KERAS_BACKEND"] == "tensorflow":
    import tensorflow as tf
    def tf_function_decorator(func):
        return tf.function(func)
else:
    # no op decorator
    def tf_function_decorator(func):
        return func
    