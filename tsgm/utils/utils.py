#  more flexible Tensor, supports both TensorFlow and PyTorch
from tsgm.types import Tensor
import keras
from keras import ops


def reconstruction_loss_by_axis(original: Tensor, reconstructed: Tensor, axis: int = 0) -> Tensor:
    """
    Calculate the reconstruction loss based on a specified axis.

    This function computes the reconstruction loss between the original data and
    the reconstructed data along a specified axis. The loss can be computed in
    two ways depending on the chosen axis:

    - When `axis` is 0, it computes the loss as the sum of squared differences
      between the original and reconstructed data for all elements.
    - When `axis` is 1 or 2, it computes the mean squared error (MSE) between the
      mean values along the chosen axis for the original and reconstructed data.

    Parameters:
    ----------
    original : tf.Tensor
        The original data tensor.

    reconstructed : tf.Tensor
        The reconstructed data tensor, typically produced by an autoencoder.

    axis : int, optional (default=0)
        The axis along which to compute the reconstruction loss:
        - 0: All elements (sum of squared differences).
        - 1: Along features (MSE).
        - 2: Along time steps (MSE).

    Returns:
    -------
    tf.Tensor
        The computed reconstruction loss as a TensorFlow tensor.

    Example:
    --------
    >>> original = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> reconstructed = tf.constant([[1.1, 2.2, 2.9], [3.9, 4.8, 6.1]])
    >>> loss = reconstruction_loss_by_axis(original, reconstructed, axis=0)
    >>> print(loss.numpy())

    Notes:
    ------
    - This function is commonly used in the context of autoencoders and other
      reconstruction-based models to assess the quality of the reconstruction.
    - The choice of `axis` determines how the loss is calculated, and it should
      align with the data's structure.
    """

    # axis=0 all (sum of squared diffs)
    # axis=1 features (MSE)
    # axis=2 times (MSE)
    if axis == 0:
        return ops.sum(ops.square(original - reconstructed))
    else:
        return keras.losses.mean_squared_error(ops.mean(original, axis=axis), ops.mean(reconstructed, axis=axis))


def fix_seeds(seed_value: int = 42) -> None:
    """
    Fix random number generator seeds for reproducibility.

    Parameters:
    ----------
    seed_value : int, optional (default=42)
        The seed value to use for fixing the random number generator seeds.
        This value is used to initialize the random number generators.

    Returns:
    -------
    None
        This function does not return a value; it modifies the random number generators
        in-place to fix their seeds.
    """
    # Set the seed using keras.utils.set_random_seed. This will set:
    # 1) `numpy` seed
    # 2) backend random seed
    # 3) `python` random seed
    keras.utils.set_random_seed(seed_value)
