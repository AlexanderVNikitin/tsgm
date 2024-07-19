import os

try:
    import tensorflow as tf
    os.environ["KERAS_BACKEND"] = "tensorflow"
except ImportError:
    try:
        import torch
        os.environ["KERAS_BACKEND"] = "torch"
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