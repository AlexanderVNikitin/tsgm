# Follows the implementation from TensorFlow:
# https://github.com/keras-team/keras/blob/v2.15.0/keras/layers/locally_connected/locally_connected1d.py
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import Layer
from keras.layers import InputSpec

import itertools
import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend


def int_shape(x):
    """Returns shape of tensor/variable as a tuple of int/None entries.

    Args:
        x: Tensor or variable.

    Returns:
        A tuple of integers (or None entries).

    Examples:

    >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
    >>> tf.keras.backend.int_shape(input)
    (2, 4, 5)
    >>> val = np.array([[1, 2], [3, 4]])
    >>> kvar = tf.keras.backend.variable(value=val)
    >>> tf.keras.backend.int_shape(kvar)
    (2, 2)

    """
    try:
        shape = x.shape
        if not isinstance(shape, tuple):
            shape = tuple(shape.as_list())
        return shape
    except ValueError:
        return None


def normalize_tuple(value, n, name, allow_zero=False):
    """Transforms non-negative/positive integer/integers into an integer tuple.

    Args:
      value: The value to validate and convert. Could an int, or any iterable of
        ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
      allow_zero: A ValueError will be raised if zero is received
        and this param is False. Defaults to `False`.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof or a
      negative value is
        passed.
    """
    error_msg = (
        f"The `{name}` argument must be a tuple of {n} "
        f"integers. Received: {value}"
    )

    if isinstance(value, int):
        value_tuple = (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(error_msg)
        if len(value_tuple) != n:
            raise ValueError(error_msg)
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                error_msg += (
                    f"including element {single_value} of "
                    f"type {type(single_value)}"
                )
                raise ValueError(error_msg)

    if allow_zero:
        unqualified_values = {v for v in value_tuple if v < 0}
        req_msg = ">= 0"
    else:
        unqualified_values = {v for v in value_tuple if v <= 0}
        req_msg = "> 0"

    if unqualified_values:
        error_msg += (
            f" including {unqualified_values}"
            f" that does not satisfy the requirement `{req_msg}`."
        )
        raise ValueError(error_msg)

    return value_tuple


def local_conv(
    inputs, kernel, kernel_size, strides, output_shape, data_format=None
):
    """Apply N-D convolution with un-shared weights.

    Args:
        inputs: (N+2)-D tensor with shape
            (batch_size, channels_in, d_in1, ..., d_inN)
            if data_format='channels_first', or
            (batch_size, d_in1, ..., d_inN, channels_in)
            if data_format='channels_last'.
        kernel: the unshared weight for N-D convolution,
            with shape (output_items, feature_dim, channels_out), where
            feature_dim = np.prod(kernel_size) * channels_in,
            output_items = np.prod(output_shape).
        kernel_size: a tuple of N integers, specifying the
            spatial dimensions of the N-D convolution window.
        strides: a tuple of N integers, specifying the strides
            of the convolution along the spatial dimensions.
        output_shape: a tuple of (d_out1, ..., d_outN) specifying the spatial
            dimensionality of the output.
        data_format: string, "channels_first" or "channels_last".

    Returns:
        An (N+2)-D tensor with shape:
        (batch_size, channels_out) + output_shape
        if data_format='channels_first', or:
        (batch_size,) + output_shape + (channels_out,)
        if data_format='channels_last'.

    Raises:
        ValueError: if `data_format` is neither
        `channels_last` nor `channels_first`.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: " + str(data_format))

    kernel_shape = int_shape(kernel)
    feature_dim = kernel_shape[1]
    channels_out = kernel_shape[-1]
    ndims = len(output_shape)
    spatial_dimensions = list(range(ndims))

    xs = []
    output_axes_ticks = [range(axis_max) for axis_max in output_shape]
    for position in itertools.product(*output_axes_ticks):
        slices = [slice(None)]

        if data_format == "channels_first":
            slices.append(slice(None))

        slices.extend(
            slice(
                position[d] * strides[d],
                position[d] * strides[d] + kernel_size[d],
            )
            for d in spatial_dimensions
        )

        if data_format == "channels_last":
            slices.append(slice(None))

        xs.append(backend.reshape(inputs[slices], (1, -1, feature_dim)))

    x_aggregate = backend.concatenate(xs, axis=0)
    output = backend.batch_dot(x_aggregate, kernel)
    output = backend.reshape(output, output_shape + (-1, channels_out))

    if data_format == "channels_first":
        permutation = [ndims, ndims + 1] + spatial_dimensions
    else:
        permutation = [ndims] + spatial_dimensions + [ndims + 1]

    return backend.permute_dimensions(output, permutation)


def conv_kernel_idxs(
    input_shape,
    kernel_shape,
    strides,
    padding,
    filters_in,
    filters_out,
    data_format,
):
    """Yields output-input tuples of indices in a CNN layer.

    The generator iterates over all `(output_idx, input_idx)` tuples, where
    `output_idx` is an integer index in a flattened tensor representing a single
    output image of a convolutional layer that is connected (via the layer
    weights) to the respective single input image at `input_idx`

    Example:

      >>> input_shape = (2, 2)
      >>> kernel_shape = (2, 1)
      >>> strides = (1, 1)
      >>> padding = "valid"
      >>> filters_in = 1
      >>> filters_out = 1
      >>> data_format = "channels_last"
      >>> list(conv_kernel_idxs(input_shape, kernel_shape, strides, padding,
      ...                       filters_in, filters_out, data_format))
      [(0, 0), (0, 2), (1, 1), (1, 3)]

    Args:
      input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
        input.
      kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
        receptive field.
      strides: tuple of size N, strides along each spatial dimension.
      padding: type of padding, string `"same"` or `"valid"`.
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      filters_in: `int`, number if filters in the input to the layer.
      filters_out: `int', number if filters in the output of the layer.
      data_format: string, "channels_first" or "channels_last".

    Yields:
      The next tuple `(output_idx, input_idx)`, where `output_idx` is an integer
      index in a flattened tensor representing a single output image of a
      convolutional layer that is connected (via the layer weights) to the
      respective single input image at `input_idx`.

    Raises:
        ValueError: if `data_format` is neither `"channels_last"` nor
          `"channels_first"`, or if number of strides, input, and kernel number
          of dimensions do not match.

        NotImplementedError: if `padding` is neither `"same"` nor `"valid"`.
    """
    if padding not in ("same", "valid"):
        raise NotImplementedError(
            f"Padding type {padding} not supported. "
            'Only "valid" and "same" are implemented.'
        )

    in_dims = len(input_shape)
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,) * in_dims
    if isinstance(strides, int):
        strides = (strides,) * in_dims

    kernel_dims = len(kernel_shape)
    stride_dims = len(strides)
    if kernel_dims != in_dims or stride_dims != in_dims:
        raise ValueError(
            "Number of strides, input and kernel dimensions must all "
            f"match. Received: stride_dims={stride_dims}, "
            f"in_dims={in_dims}, kernel_dims={kernel_dims}"
        )

    output_shape = conv_output_shape(
        input_shape, kernel_shape, strides, padding
    )
    output_axes_ticks = [range(dim) for dim in output_shape]

    if data_format == "channels_first":
        concat_idxs = (
            lambda spatial_idx, filter_idx: (filter_idx,) + spatial_idx
        )
    elif data_format == "channels_last":
        concat_idxs = lambda spatial_idx, filter_idx: spatial_idx + (
            filter_idx,
        )
    else:
        raise ValueError(
            f"Data format `{data_format}` not recognized."
            '`data_format` must be "channels_first" or "channels_last".'
        )

    for output_position in itertools.product(*output_axes_ticks):
        input_axes_ticks = conv_connected_inputs(
            input_shape, kernel_shape, output_position, strides, padding
        )
        for input_position in itertools.product(*input_axes_ticks):
            for f_in in range(filters_in):
                for f_out in range(filters_out):
                    out_idx = np.ravel_multi_index(
                        multi_index=concat_idxs(output_position, f_out),
                        dims=concat_idxs(output_shape, filters_out),
                    )
                    in_idx = np.ravel_multi_index(
                        multi_index=concat_idxs(input_position, f_in),
                        dims=concat_idxs(input_shape, filters_in),
                    )
                    yield (out_idx, in_idx)


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Args:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.

    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {"same", "valid", "full", "causal"}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ["same", "causal"]:
        output_length = input_length
    elif padding == "valid":
        output_length = input_length - dilated_filter_size + 1
    elif padding == "full":
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_connected_inputs(
    input_shape, kernel_shape, output_position, strides, padding
):
    """Return locations of the input connected to an output position.

    Assume a convolution with given parameters is applied to an input having N
    spatial dimensions with `input_shape = (d_in1, ..., d_inN)`. This method
    returns N ranges specifying the input region that was convolved with the
    kernel to produce the output at position
    `output_position = (p_out1, ..., p_outN)`.

    Example:

      >>> input_shape = (4, 4)
      >>> kernel_shape = (2, 1)
      >>> output_position = (1, 1)
      >>> strides = (1, 1)
      >>> padding = "valid"
      >>> conv_connected_inputs(input_shape, kernel_shape, output_position,
      ...                       strides, padding)
      [range(1, 3), range(1, 2)]

    Args:
      input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
        input.
      kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
        receptive field.
      output_position: tuple of size N: `(p_out1, ..., p_outN)`, a single
        position in the output of the convolution.
      strides: tuple of size N, strides along each spatial dimension.
      padding: type of padding, string `"same"` or `"valid"`.
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.

    Returns:
      N ranges `[[p_in_left1, ..., p_in_right1], ...,
                [p_in_leftN, ..., p_in_rightN]]` specifying the region in the
      input connected to output_position.
    """
    ranges = []

    ndims = len(input_shape)
    for d in range(ndims):
        left_shift = int(kernel_shape[d] / 2)
        right_shift = kernel_shape[d] - left_shift

        center = output_position[d] * strides[d]

        if padding == "valid":
            center += left_shift

        start = max(0, center - left_shift)
        end = min(input_shape[d], center + right_shift)

        ranges.append(range(start, end))

    return ranges


def conv_kernel_mask(input_shape, kernel_shape, strides, padding):
    """Compute a mask representing the connectivity of a convolution operation.

    Assume a convolution with given parameters is applied to an input having N
    spatial dimensions with `input_shape = (d_in1, ..., d_inN)` to produce an
    output with shape `(d_out1, ..., d_outN)`. This method returns a boolean
    array of shape `(d_in1, ..., d_inN, d_out1, ..., d_outN)` with `True`
    entries indicating pairs of input and output locations that are connected by
    a weight.

    Example:

      >>> input_shape = (4,)
      >>> kernel_shape = (2,)
      >>> strides = (1,)
      >>> padding = "valid"
      >>> conv_kernel_mask(input_shape, kernel_shape, strides, padding)
      array([[ True, False, False],
             [ True,  True, False],
             [False,  True,  True],
             [False, False,  True]])

      where rows and columns correspond to inputs and outputs respectively.


    Args:
      input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
        input.
      kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
        receptive field.
      strides: tuple of size N, strides along each spatial dimension.
      padding: type of padding, string `"same"` or `"valid"`.
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.

    Returns:
      A boolean 2N-D `np.ndarray` of shape
      `(d_in1, ..., d_inN, d_out1, ..., d_outN)`, where `(d_out1, ..., d_outN)`
      is the spatial shape of the output. `True` entries in the mask represent
      pairs of input-output locations that are connected by a weight.

    Raises:
      ValueError: if `input_shape`, `kernel_shape` and `strides` don't have the
          same number of dimensions.
      NotImplementedError: if `padding` is not in {`"same"`, `"valid"`}.
    """
    if padding not in {"same", "valid"}:
        raise NotImplementedError(
            f"Padding type {padding} not supported. "
            'Only "valid" and "same" are implemented.'
        )

    in_dims = len(input_shape)
    if isinstance(kernel_shape, int):
        kernel_shape = (kernel_shape,) * in_dims
    if isinstance(strides, int):
        strides = (strides,) * in_dims

    kernel_dims = len(kernel_shape)
    stride_dims = len(strides)
    if kernel_dims != in_dims or stride_dims != in_dims:
        raise ValueError(
            "Number of strides, input and kernel dimensions must all "
            f"match. Received: stride_dims={stride_dims}, "
            f"in_dims={in_dims}, kernel_dims={kernel_dims}"
        )

    output_shape = conv_output_shape(
        input_shape, kernel_shape, strides, padding
    )

    mask_shape = input_shape + output_shape
    mask = np.zeros(mask_shape, bool)

    output_axes_ticks = [range(dim) for dim in output_shape]
    for output_position in itertools.product(*output_axes_ticks):
        input_axes_ticks = conv_connected_inputs(
            input_shape, kernel_shape, output_position, strides, padding
        )
        for input_position in itertools.product(*input_axes_ticks):
            mask[input_position + output_position] = True

    return mask


def normalize_data_format(value):
    if value is None:
        value = backend.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            f'"channels_first", "channels_last". Received: {value}'
        )
    return data_format


def normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {"valid", "same", "causal"}:
        raise ValueError(
            "The `padding` argument must be a list/tuple or one of "
            '"valid", "same" (or "causal", only for `Conv1D). '
            f"Received: {padding}"
        )
    return padding



def get_locallyconnected_mask(
    input_shape, kernel_shape, strides, padding, data_format
):
    """Return a mask representing connectivity of a locally-connected operation.

    This method returns a masking numpy array of 0s and 1s (of type
    `np.float32`) that, when element-wise multiplied with a fully-connected
    weight tensor, masks out the weights between disconnected input-output pairs
    and thus implements local connectivity through a sparse fully-connected
    weight tensor.

    Assume an unshared convolution with given parameters is applied to an input
    having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
    to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
    by layer parameters such as `strides`).

    This method returns a mask which can be broadcast-multiplied (element-wise)
    with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer
    between (N+1)-D activations (N spatial + 1 channel dimensions for input and
    output) to make it perform an unshared convolution with given
    `kernel_shape`, `strides`, `padding` and `data_format`.

    Args:
      input_shape: tuple of size N: `(d_in1, ..., d_inN)` spatial shape of the
        input.
      kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
        receptive field.
      strides: tuple of size N, strides along each spatial dimension.
      padding: type of padding, string `"same"` or `"valid"`.
      data_format: a string, `"channels_first"` or `"channels_last"`.

    Returns:
      a `np.float32`-type `np.ndarray` of shape
      `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
      if `data_format == `"channels_first"`, or
      `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
      if `data_format == "channels_last"`.

    Raises:
      ValueError: if `data_format` is neither `"channels_first"` nor
                  `"channels_last"`.
    """
    mask = conv_kernel_mask(
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        strides=strides,
        padding=padding,
    )

    ndims = int(mask.ndim / 2)

    if data_format == "channels_first":
        mask = np.expand_dims(mask, 0)
        mask = np.expand_dims(mask, -ndims - 1)

    elif data_format == "channels_last":
        mask = np.expand_dims(mask, ndims)
        mask = np.expand_dims(mask, -1)

    else:
        raise ValueError("Unrecognized data_format: " + str(data_format))

    return mask


def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
    """Apply N-D convolution with un-shared weights using a single matmul call.

    This method outputs `inputs . (kernel * kernel_mask)`
    (with `.` standing for matrix-multiply and `*` for element-wise multiply)
    and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
    hence perform the same operation as a convolution with un-shared
    (the remaining entries in `kernel`) weights. It also does the necessary
    reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.

    Args:
        inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
          d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
        kernel: the unshared weights for N-D convolution,
            an (N+2)-D tensor of shape: `(d_in1, ..., d_inN, channels_in,
            d_out2, ..., d_outN, channels_out)` or `(channels_in, d_in1, ...,
            d_inN, channels_out, d_out2, ..., d_outN)`, with the ordering of
            channels and spatial dimensions matching that of the input. Each
            entry is the weight between a particular input and output location,
            similarly to a fully-connected weight matrix.
        kernel_mask: a float 0/1 mask tensor of shape: `(d_in1, ..., d_inN, 1,
          d_out2, ..., d_outN, 1)` or `(1, d_in1, ..., d_inN, 1, d_out2, ...,
          d_outN)`, with the ordering of singleton and spatial dimensions
          matching that of the input. Mask represents the connectivity pattern
          of the layer and is precomputed elsewhere based on layer parameters:
          stride, padding, and the receptive field shape.
        output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
          d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
          spatial dimensions matching that of the input.

    Returns:
        Output (N+2)-D tensor with shape `output_shape`.
    """
    inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))

    kernel = kernel_mask * kernel
    kernel = make_2d(kernel, split_dim=backend.ndim(kernel) // 2)

    output_flat = tf.matmul(inputs_flat, kernel, b_is_sparse=True)
    output = backend.reshape(
        output_flat,
        [
            backend.shape(output_flat)[0],
        ]
        + output_shape.as_list()[1:],
    )
    return output


def local_conv_sparse_matmul(
    inputs, kernel, kernel_idxs, kernel_shape, output_shape
):
    """Apply N-D convolution with unshared weights using a single sparse matmul.

    This method outputs `inputs . tf.sparse.SparseTensor(indices=kernel_idxs,
    values=kernel, dense_shape=kernel_shape)`, with `.` standing for
    matrix-multiply. It also reshapes `inputs` to 2-D and `output` to (N+2)-D.

    Args:
        inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
          d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
        kernel: a 1-D tensor with shape `(len(kernel_idxs),)` containing all the
          weights of the layer.
        kernel_idxs:  a list of integer tuples representing indices in a sparse
          matrix performing the un-shared convolution as a matrix-multiply.
        kernel_shape: a tuple `(input_size, output_size)`, where `input_size =
          channels_in * d_in1 * ... * d_inN` and `output_size = channels_out *
          d_out1 * ... * d_outN`.
        output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
          d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
          spatial dimensions matching that of the input.

    Returns:
        Output (N+2)-D dense tensor with shape `output_shape`.
    """
    inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))
    output_flat = tf.sparse.sparse_dense_matmul(
        sp_a=tf.SparseTensor(kernel_idxs, kernel, kernel_shape),
        b=inputs_flat,
        adjoint_b=True,
    )
    output_flat_transpose = backend.transpose(output_flat)

    output_reshaped = backend.reshape(
        output_flat_transpose,
        [
            backend.shape(output_flat_transpose)[0],
        ]
        + output_shape.as_list()[1:],
    )
    return output_reshaped


def make_2d(tensor, split_dim):
    """Reshapes an N-dimensional tensor into a 2D tensor.

    Dimensions before (excluding) and after (including) `split_dim` are grouped
    together.

    Args:
      tensor: a tensor of shape `(d0, ..., d(N-1))`.
      split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).

    Returns:
      Tensor of shape
      `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
    """
    shape = tf.shape(tensor)
    in_dims = shape[:split_dim]
    out_dims = shape[split_dim:]

    in_size = tf.reduce_prod(in_dims)
    out_size = tf.reduce_prod(out_dims)

    return tf.reshape(tensor, (in_size, out_size))


class LocallyConnected1D(Layer):
    """Locally-connected layer for 1D inputs.

    The `LocallyConnected1D` layer works similarly to
    the `Conv1D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each different patch
    of the input.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Example:
    ```python
        # apply a unshared weight convolution 1d of length 3 to a sequence with
        # 10 timesteps, with 64 output filters
        model = Sequential()
        model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
        # now model.output_shape == (None, 8, 64)
        # add a new conv1d on top
        model.add(LocallyConnected1D(32, 3))
        # now model.output_shape == (None, 6, 32)
    ```

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the
          number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer, specifying
          the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer, specifying the
          stride length of the convolution.
        padding: Currently only supports `"valid"` (case-insensitive). `"same"`
          may be supported in the future. `"valid"` means no padding.
        data_format: A string, one of `channels_last` (default) or
          `channels_first`. The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape `(batch, length,
          channels)` while `channels_first` corresponds to inputs with shape
          `(batch, channels, length)`. When unspecified, uses
          `image_data_format` value found in your Keras config file at
          `~/.keras/keras.json` (if exists) else 'channels_last'.
          Defaults to 'channels_last'.
        activation: Activation function to use. If you don't specify anything,
          no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
          matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the
          layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
        implementation: implementation mode, either `1`, `2`, or `3`. `1` loops
          over input spatial locations to perform the forward pass. It is
          memory-efficient but performs a lot of (small) ops.  `2` stores layer
          weights in a dense but sparsely-populated 2D matrix and implements the
          forward pass as a single matrix-multiply. It uses a lot of RAM but
          performs few (large) ops.  `3` stores layer weights in a sparse tensor
          and implements the forward pass as a single sparse matrix-multiply.
            How to choose:
            `1`: large, dense models,
            `2`: small models,
            `3`: large, sparse models,  where "large" stands for large
              input/output activations (i.e. many `filters`, `input_filters`,
              large `input_size`, `output_size`), and "sparse" stands for few
              connections between inputs and outputs, i.e. small ratio
              `filters * input_filters * kernel_size / (input_size * strides)`,
              where inputs to and outputs of the layer are assumed to have
              shapes `(input_size, input_filters)`, `(output_size, filters)`
              respectively.  It is recommended to benchmark each in the setting
              of interest to pick the most efficient one (in terms of speed and
              memory usage). Correct choice of implementation can lead to
              dramatic speed improvements (e.g. 50X), potentially at the expense
              of RAM.  Also, only `padding="valid"` is supported by
              `implementation=1`.
    Input shape:
        3D tensor with shape: `(batch_size, steps, input_dim)`
    Output shape:
        3D tensor with shape: `(batch_size, new_steps, filters)` `steps` value
          might have changed due to padding or strides.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        implementation=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = normalize_tuple(
            kernel_size, 1, "kernel_size"
        )
        self.strides = normalize_tuple(
            strides, 1, "strides", allow_zero=True
        )
        self.padding = normalize_padding(padding)
        if self.padding != "valid" and implementation == 1:
            raise ValueError(
                "Invalid border mode for LocallyConnected1D "
                '(only "valid" is supported if implementation is 1): ' + padding
            )
        self.data_format = normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.implementation = implementation
        self.input_spec = InputSpec(ndim=3)

    @property
    def _use_input_spec_as_call_signature(self):
        return False

    def build(self, input_shape):
        if self.data_format == "channels_first":
            input_dim, input_length = input_shape[1], input_shape[2]
        else:
            input_dim, input_length = input_shape[2], input_shape[1]

        if input_dim is None:
            raise ValueError(
                "Axis 2 of input should be fully-defined. Found shape:",
                input_shape,
            )
        self.output_length = conv_output_length(
            input_length, self.kernel_size[0], self.padding, self.strides[0]
        )

        if self.output_length <= 0:
            raise ValueError(
                "One of the dimensions in the output is <= 0 "
                f"due to downsampling in {self.name}. Consider "
                "increasing the input size. "
                f"Received input shape {input_shape} which would produce "
                "output shape with a zero or negative value in a "
                "dimension."
            )

        if self.implementation == 1:
            self.kernel_shape = (
                self.output_length,
                self.kernel_size[0] * input_dim,
                self.filters,
            )

            self.kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        elif self.implementation == 2:
            if self.data_format == "channels_first":
                self.kernel_shape = (
                    input_dim,
                    input_length,
                    self.filters,
                    self.output_length,
                )
            else:
                self.kernel_shape = (
                    input_length,
                    input_dim,
                    self.output_length,
                    self.filters,
                )

            self.kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

            self.kernel_mask = (
                get_locallyconnected_mask(
                    input_shape=(input_length,),
                    kernel_shape=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                )
            )

        elif self.implementation == 3:
            self.kernel_shape = (
                self.output_length * self.filters,
                input_length * input_dim,
            )

            self.kernel_idxs = sorted(
                conv_kernel_idxs(
                    input_shape=(input_length,),
                    kernel_shape=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    filters_in=input_dim,
                    filters_out=self.filters,
                    data_format=self.data_format,
                )
            )

            self.kernel = self.add_weight(
                shape=(len(self.kernel_idxs),),
                initializer=self.kernel_initializer,
                name="kernel",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        else:
            raise ValueError(
                "Unrecognized implementation mode: %d." % self.implementation
            )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_length, self.filters),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        if self.data_format == "channels_first":
            self.input_spec = InputSpec(ndim=3, axes={1: input_dim})
        else:
            self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            input_length = input_shape[2]
        else:
            input_length = input_shape[1]

        length = conv_output_length(
            input_length, self.kernel_size[0], self.padding, self.strides[0]
        )

        if self.data_format == "channels_first":
            return (input_shape[0], self.filters, length)
        elif self.data_format == "channels_last":
            return (input_shape[0], length, self.filters)

    def call(self, inputs):
        if self.implementation == 1:
            output = local_conv(
                inputs,
                self.kernel,
                self.kernel_size,
                self.strides,
                (self.output_length,),
                self.data_format,
            )

        elif self.implementation == 2:
            output = local_conv_matmul(
                inputs,
                self.kernel,
                self.kernel_mask,
                self.compute_output_shape(inputs.shape),
            )

        elif self.implementation == 3:
            output = local_conv_sparse_matmul(
                inputs,
                self.kernel,
                self.kernel_idxs,
                self.kernel_shape,
                self.compute_output_shape(inputs.shape),
            )

        else:
            raise ValueError(
                "Unrecognized implementation mode: %d." % self.implementation
            )

        if self.use_bias:
            output = backend.bias_add(
                output, self.bias, data_format=self.data_format
            )

        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "implementation": self.implementation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
