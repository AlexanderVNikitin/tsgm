import abc
import math
import tsgm
import typing as T
#  import keras3.0 here, the LocallyConnected1D is not supported in keras3.0, I replace it with Conv1D
import keras
from keras import layers
#  replace tf with ops in keras3.0
from keras import ops

from prettytable import PrettyTable


class Sampling(keras.layers.Layer):
    """
    Custom Keras layer for sampling from a latent space.

    This layer samples from a latent space using the reparameterization trick during training.
    It takes as input the mean and log variance of the latent distribution and generates
    samples by adding random noise scaled by the standard deviation to the mean.
    """
    def call(self, inputs: T.Tuple[tsgm.types.Tensor, tsgm.types.Tensor]) -> tsgm.types.Tensor:
        """
        Generates samples from a latent space.

        :param inputs: Tuple containing mean and log variance tensors of the latent distribution.
        :type inputs: tuple[tsgm.types.Tensor, tsgm.types.Tensor]

        :returns: Sampled latent vector.
        :rtype: tsgm.types.Tensor
        """
        z_mean, z_log_var = inputs
        #  random noise for keras3.0
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        #  ops for keras3.0
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class Architecture(abc.ABC):
    @abc.abstractproperty
    def arch_type(self):
        raise NotImplementedError


class BaseGANArchitecture(Architecture):
    """
    Base class for defining architectures of Generative Adversarial Networks (GANs).
    """
    @property
    def discriminator(self) -> keras.models.Model:
        """
        Property for accessing the discriminator model.

        :returns: The discriminator model.
        :rtype: keras.models.Model
        :raises NotImplementedError: If the discriminator model is not found.
        """
        if hasattr(self, "_discriminator"):
            return self._discriminator
        else:
            raise NotImplementedError

    @property
    def generator(self) -> keras.models.Model:
        """
        Property for accessing the generator model.

        :returns: The generator model.
        :rtype: keras.models.Model
        :raises NotImplementedError: If the generator model is not implemented.
        """
        if hasattr(self, "_generator"):
            return self._generator
        else:
            raise NotImplementedError

    def get(self) -> T.Dict:
        """
        Retrieves both discriminator and generator models as a dictionary.

        :return: A dictionary containing discriminator and generator models.
        :rtype: dict
        :raises NotImplementedError: If either discriminator or generator models are not implemented.
        """
        if hasattr(self, "_discriminator") and hasattr(self, "_generator"):
            return {"discriminator": self._discriminator, "generator": self._generator}
        else:
            raise NotImplementedError


class BaseVAEArchitecture(Architecture):
    """
    Base class for defining architectures of Variational Autoencoders (VAEs).
    """
    @property
    def encoder(self) -> keras.models.Model:
        """
        Property for accessing the encoder model.

        :return: The encoder model.
        :rtype: keras.models.Model
        :raises NotImplementedError: If the encoder model is not implemented.
        """
        if hasattr(self, "_encoder"):
            return self._encoder
        else:
            raise NotImplementedError

    @property
    def decoder(self) -> keras.models.Model:
        """
        Property for accessing the decoder model.

        :return: The decoder model.
        :rtype: keras.models.Model
        :raises NotImplementedError: If the decoder model is not implemented.
        """
        if hasattr(self, "_decoder"):
            return self._decoder
        else:
            raise NotImplementedError

    def get(self) -> T.Dict:
        """
        Retrieves both encoder and decoder models as a dictionary.

        :return: A dictionary containing encoder and decoder models.
        :rtype: dict
        :raises NotImplementedError: If either encoder or decoder models are not implemented.
        """
        if hasattr(self, "_encoder") and hasattr(self, "_decoder"):
            return {"encoder": self._encoder, "decoder": self._decoder}
        else:
            raise NotImplementedError


class VAE_CONV5Architecture(BaseVAEArchitecture):
    """
    This class defines the architecture for a Variational Autoencoder (VAE) with Convolutional Layers.

    Parameters:
        seq_len (int): Length of input sequence.
        feat_dim (int): Dimensionality of input features.
        latent_dim (int): Dimensionality of latent space.
    """
    arch_type = "vae:unconditional"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int) -> None:
        """
        Initializes the VAE_CONV5Architecture.

        :parameter seq_len: Length of input sequences.
        :type seq_len: int
        :parameter feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :parameter latent_dim: Dimensionality of latent space.
        :type latent_dim: int
        """
        super().__init__()
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(shape=(self._seq_len, self._feat_dim))
        x = layers.Conv1D(64, 10, activation="relu", strides=1, padding="same")(
            encoder_inputs
        )
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 2, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 2, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 2, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 4, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(self._latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def _build_decoder(self) -> keras.models.Model:
        latent_inputs = keras.Input(shape=(self._latent_dim,))
        x = layers.Dense(64, activation="relu")(latent_inputs)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)

        dense_shape = self._encoder.layers[-6].output_shape[1] * self._seq_len

        x = layers.Dense(dense_shape, activation="relu")(x)

        x = layers.Reshape((self._seq_len, dense_shape // self._seq_len))(x)
        x = layers.Conv1DTranspose(64, 2, activation="relu", strides=1, padding="same")(
            x
        )
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, activation="relu", strides=1, padding="same")(
            x
        )
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, activation="relu", strides=1, padding="same")(
            x
        )
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, activation="relu", strides=1, padding="same")(
            x
        )
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(
            64, 10, activation="relu", strides=1, padding="same"
        )(x)
        x = layers.Dropout(rate=0.2)(x)
        decoder_outputs = layers.Conv1DTranspose(
            self._feat_dim, 3, activation="sigmoid", padding="same"
        )(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder


class cVAE_CONV5Architecture(BaseVAEArchitecture):
    arch_type = "vae:conditional"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, output_dim: int = 2) -> None:
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim

        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self) -> keras.models.Model:
        encoder_inputs = keras.Input(
            shape=(self._seq_len, self._feat_dim + self._output_dim)
        )

        x = layers.Conv1D(64, 10, activation="relu", strides=1, padding="same")(
            encoder_inputs
        )
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 2, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 2, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 2, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(64, 4, activation="relu", strides=1, padding="same")(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(self._latent_dim * self._seq_len, name="z_mean")(x)
        z_log_var = layers.Dense(self._latent_dim * self._seq_len, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def _build_decoder(self) -> keras.models.Model:
        inputs = keras.Input(
            shape=(
                self._seq_len,
                self._latent_dim + self._output_dim,
            )
        )
        x = layers.Conv1DTranspose(64, 2, strides=2, padding="same")(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)

        pool_and_stride = round((x.shape[1] + 1) / (self._seq_len + 1))
        x = layers.AveragePooling1D(pool_size=pool_and_stride, strides=pool_and_stride)(
            x
        )
        d_output = layers.Conv1D(self._feat_dim, 1, activation="sigmoid")(x)

        decoder = keras.Model(inputs, d_output, name="decoder")
        return decoder


class cGAN_Conv4Architecture(BaseGANArchitecture):
    """
    Architecture for Conditional Generative Adversarial Network (cGAN) with Convolutional Layers.
    """
    arch_type = "gan:conditional"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, output_dim: int) -> None:
        """
        Initializes the cGAN_Conv4Architecture.

        :parameter seq_len: Length of input sequence.
        :type seq_len: int
        :parameter feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :parameter latent_dim: Dimensionality of latent space.
        :type latent_dim: int
        :parameter output_dim: Dimensionality of output.
        :type output_dim: int
        """
        super().__init__()
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self.generator_in_channels = latent_dim + output_dim
        self.discriminator_in_channels = feat_dim + output_dim

        self._discriminator = self._build_discriminator()
        self._generator = self._build_generator()

    def _build_discriminator(self) -> keras.models.Model:
        d_input = keras.Input((self._seq_len, self.discriminator_in_channels))
        x = layers.Conv1D(64, 3, strides=2, padding="same")(d_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.GlobalAvgPool1D()(x)
        d_output = layers.Dense(1, activation="sigmoid")(x)
        discriminator = keras.Model(d_input, d_output, name="discriminator")
        return discriminator

    def _build_generator(self) -> keras.models.Model:
        g_input = keras.Input((self.generator_in_channels,))
        x = layers.Dense(8 * 8 * self._seq_len)(g_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((self._seq_len, 64))(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1D(1, 8, padding="same")(x)
        x = layers.LSTM(256, return_sequences=True)(x)

        pool_and_stride = math.ceil((x.shape[1] + 1) / (self._seq_len + 1))

        x = layers.AveragePooling1D(pool_size=pool_and_stride, strides=pool_and_stride)(
            x
        )
        g_output = layers.Conv1D(self._feat_dim, 1, activation="tanh")(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator


class tcGAN_Conv4Architecture(BaseGANArchitecture):
    """
    Architecture for Temporal Conditional Generative Adversarial Network (tcGAN) with Convolutional Layers.
    """
    arch_type = "gan:t-conditional"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, output_dim: int) -> None:
        """
        Initializes the tcGAN_Conv4Architecture.

        :parameter seq_len: Length of input sequence.
        :type seq_len: int
        :parameter feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :parameter latent_dim: Dimensionality of latent space.
        :type latent_dim: int
        :parameter output_dim: Dimensionality of output.
        :type output_dim: int
        """
        super().__init__()
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim

        self.generator_in_channels = latent_dim + output_dim
        self.discriminator_in_channels = feat_dim + output_dim

        self._discriminator = self._build_discriminator()
        self._generator = self._build_generator()

    def _build_discriminator(self) -> keras.models.Model:
        d_input = keras.Input((self._seq_len, self.discriminator_in_channels))
        x = layers.Conv1D(64, 3, strides=2, padding="same")(d_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.GlobalAvgPool1D()(x)
        d_output = layers.Dense(1, activation="sigmoid")(x)
        discriminator = keras.Model(d_input, d_output, name="discriminator")
        return discriminator

    def _build_generator(self) -> keras.models.Model:
        g_input = keras.Input((self._seq_len, self.generator_in_channels))
        x = layers.Conv1DTranspose(64, 2, strides=2, padding="same")(g_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1DTranspose(64, 2, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)

        pool_and_stride = round((x.shape[1] + 1) / (self._seq_len + 1))
        x = layers.AveragePooling1D(pool_size=pool_and_stride, strides=pool_and_stride)(
            x
        )
        g_output = layers.Conv1D(self._feat_dim, 1, activation="tanh")(x)

        generator = keras.Model(g_input, g_output, name="generator")
        return generator


class cGAN_LSTMConv3Architecture(BaseGANArchitecture):
    """
    Architecture for Conditional Generative Adversarial Network (cGAN) with LSTM and Convolutional Layers.
    """
    arch_type = "gan:conditional"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, output_dim: int) -> None:
        """
        Initializes the cGAN_LSTMConv3Architecture.

        :parameter seq_len: Length of input sequence.
        :type seq_len: int
        :parameter feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :parameter latent_dim: Dimensionality of latent space.
        :type latent_dim: int
        :parameter output_dim: Dimensionality of output.
        :type output_dim: int
        """
        super().__init__()
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim

        self.generator_in_channels = latent_dim + output_dim
        self.discriminator_in_channels = feat_dim + output_dim

        self._discriminator = self._build_discriminator()
        self._generator = self._build_generator()

    def _build_discriminator(self) -> keras.models.Model:
        d_input = keras.Input((self._seq_len, self.discriminator_in_channels))
        x = layers.LSTM(64, return_sequences=True)(d_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Conv1D(128, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.GlobalAvgPool1D()(x)
        d_output = layers.Dense(1, activation="sigmoid")(x)
        discriminator = keras.Model(d_input, d_output, name="discriminator")
        return discriminator

    def _build_generator(self) -> keras.models.Model:
        g_input = keras.Input((self.generator_in_channels,))
        x = layers.Dense(8 * 8 * self._seq_len)(g_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((self._seq_len, 64))(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1DTranspose(128, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv1D(1, 8, padding="same")(x)
        x = layers.LSTM(256, return_sequences=True)(x)

        pool_and_stride = round((x.shape[1] + 1) / (self._seq_len + 1))

        x = layers.AveragePooling1D(pool_size=pool_and_stride, strides=pool_and_stride)(x)
        g_output = layers.Conv1D(self._feat_dim, 1, activation="tanh")(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator


class BaseClassificationArchitecture(Architecture):
    """
    Base class for classification architectures.

    :param seq_len: Length of input sequences.
    :type seq_len: int
    :param feat_dim: Dimensionality of input features.
    :type feat_dim: int
    :param output_dim: Dimensionality of the output.
    :type output_dim: int
    """

    arch_type = "downstream:classification"

    def __init__(self, seq_len: int, feat_dim: int, output_dim: int) -> None:
        """
        Initializes the base classification architecture.

        :param seq_len: Length of input sequences.
        :type seq_len: int
        :param feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :param output_dim: Dimensionality of the output.
        :type output_dim: int
        """
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._output_dim = output_dim
        self._model = self._build_model()

    @property
    def model(self) -> keras.models.Model:
        """
        Property to access the underlying Keras model.

        :returns: The Keras model.
        :rtype: keras.models.Model
        """
        return self._model

    def get(self) -> T.Dict:
        """
        Returns a dictionary containing the model.

        :returns: A dictionary containing the model.
        :rtype: dict
        """
        return {"model": self.model}

    def _build_model(self) -> None:
        raise NotImplementedError


class ConvnArchitecture(BaseClassificationArchitecture):
    """
    Convolutional neural network architecture for classification.
    Inherits from BaseClassificationArchitecture.
    """
    def __init__(
        self, seq_len: int, feat_dim: int, output_dim: int, n_conv_blocks: int = 1
    ) -> None:
        """
        Initializes the convolutional neural network architecture.

        :param seq_len: Length of input sequences.
        :type seq_len: int
        :param feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :param output_dim: Dimensionality of the output.
        :type output_dim: int
        :param n_conv_blocks: Number of convolutional blocks to use (default is 1).
        :type n_conv_blocks: int, optional
        """
        self._n_conv_blocks = n_conv_blocks
        super().__init__(seq_len, feat_dim, output_dim)

    def _build_model(self) -> keras.models.Model:
        m_input = keras.Input((self._seq_len, self._feat_dim))
        x = m_input
        for _ in range(self._n_conv_blocks):
            x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        m_output = layers.Dense(self._output_dim, activation="softmax")(x)
        return keras.Model(m_input, m_output, name="classification_model")


class ConvnLSTMnArchitecture(BaseClassificationArchitecture):
    def __init__(
        self, seq_len: int, feat_dim: int, output_dim: int, n_conv_lstm_blocks: int = 1
    ) -> None:
        self._n_conv_lstm_blocks = n_conv_lstm_blocks
        super().__init__(seq_len, feat_dim, output_dim)

    def _build_model(self) -> keras.models.Model:
        m_input = keras.Input((self._seq_len, self._feat_dim))
        x = m_input
        for _ in range(self._n_conv_lstm_blocks):
            x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            x = layers.LSTM(128, activation="relu", return_sequences=True)(x)
            x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        m_output = layers.Dense(self._output_dim, activation="softmax")(x)
        return keras.Model(m_input, m_output, name="classification_model")


class BlockClfArchitecture(BaseClassificationArchitecture):
    """
    Architecture for classification using a sequence of blocks.

    Inherits from BaseClassificationArchitecture.
    """

    arch_type = "downstream:classification"

    def __init__(self, seq_len: int, feat_dim: int, output_dim: int, blocks: list) -> None:
        """
        Initializes the BlockClfArchitecture.

        :param seq_len: Length of input sequences.
        :type seq_len: int
        :param feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :param output_dim: Dimensionality of the output.
        :type output_dim: int
        :param blocks: List of blocks used in the architecture.
        :type blocks: list
        """
        self._blocks = blocks
        super().__init__(seq_len, feat_dim, output_dim)

    def _build_model(self) -> keras.Model:
        m_input = keras.Input((self._seq_len, self._feat_dim))
        x = m_input
        for block in self._blocks:
            x = block(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        m_output = layers.Dense(self._output_dim, activation="softmax")(x)
        return keras.Model(m_input, m_output, name="classification_model")


class BasicRecurrentArchitecture(Architecture):
    """
    Base class for recurrent neural network architectures.

    Inherits from Architecture.
    """

    arch_type = "rnn_architecture"

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        network_type: str,
        name: str = "Sequential",
    ) -> None:
        """
        :param hidden_dim: int, the number of units (e.g. 24)
        :param output_dim: int, the number of output units (e.g. 1)
        :param n_layers: int, the number of layers (e.g. 3)
        :param network_type: str, one of 'gru', 'lstm', or 'lstmLN'
        :param name: str, model name
            Default: "Sequential"
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.network_type = network_type.lower()
        assert self.network_type in ["gru", "lstm"]

        self._name = name

    def _rnn_cell(self) -> keras.layers.Layer:
        """
        Basic RNN Cell
        :return cell: keras.layers.Layer
        """
        cell = None
        # GRU
        if self.network_type == "gru":
            cell = keras.layers.GRUCell(self.hidden_dim, activation="tanh")
        # LSTM
        elif self.network_type == "lstm":
            cell = keras.layers.LSTMCell(self.hidden_dim, activation="tanh")
        return cell

    def _make_network(self, model: keras.models.Model, activation: str, return_sequences: bool) -> keras.models.Model:
        _cells = keras.layers.StackedRNNCells(
            [self._rnn_cell() for _ in range(self.n_layers)],
            name=f"{self.network_type}_x{self.n_layers}",
        )
        model.add(keras.layers.RNN(_cells, return_sequences=return_sequences))
        model.add(
            keras.layers.Dense(units=self.output_dim, activation=activation, name="OUT")
        )
        return model

    def build(self, activation: str = "sigmoid", return_sequences: bool = True) -> keras.models.Model:
        """
        Builds the recurrent neural network model.

        :param activation: Activation function for the output layer (default is 'sigmoid').
        :type activation: str
        :param return_sequences: Whether to return the full sequence of outputs (default is True).
        :type return_sequences: bool
        :return: The built Keras model.
        :rtype: keras.models.Model
        """
        model = keras.models.Sequential(name=f"{self._name}")
        model = self._make_network(model, activation=activation, return_sequences=return_sequences)
        return model


class TransformerClfArchitecture(BaseClassificationArchitecture):
    """
    Base class for transformer architectures.

    Inherits from BaseClassificationArchitecture.
    """

    arch_type = "downstream:classification"

    def __init__(self, seq_len: int, feat_dim: int, num_heads: int = 2, ff_dim: int = 64, n_blocks: int = 1, dropout_rate=0.5, output_dim: int = 2) -> None:
        """
        Initializes the TransformerClfArchitecture.

        :param seq_len: Length of input sequences.
        :type seq_len: int
        :param feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :param num_heads: Number of attention heads (default is 2).
        :type num_heads: int
        :param ff_dim: Feed forward dimension in the attention block (default is 64).
        :type ff_dim: int
        :param output_dim: Dimensionality of the output.
        :type output_dim: int
        :param dropout_rate: Dropout probability (default is 0.5).
        :type dropout_rate: float, optional
        :param n_blocks: Number of transformer blocks (default is 1).
        :type n_blocks: int, optional
        :param output_dim: Number of classes (default is 2).
        :type output_dim: int, optional
        """

        self._num_heads = num_heads
        self._ff_dim = ff_dim
        self._n_blocks = n_blocks
        self._dropout_rate = dropout_rate

        super().__init__(seq_len, feat_dim, output_dim)

    def transformer_block(self, inputs):
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=inputs.shape[-1]
        )(inputs, inputs)
        attention_output = layers.Dropout(self._dropout_rate)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

        # Feed-Forward Network
        ff_output = layers.Dense(self._ff_dim, activation="relu")(attention_output)
        ff_output = layers.Dense(inputs.shape[-1])(ff_output)
        ff_output = layers.Dropout(self._dropout_rate)(ff_output)
        ff_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

        return ff_output

    def _build_model(self) -> keras.Model:
        inputs = layers.Input(shape=(self._seq_len, self._feat_dim))

        x = inputs
        for _ in range(self._n_blocks):
            x = self.transformer_block(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self._dropout_rate)(x)
        outputs = layers.Dense(self._output_dim, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        return model


class cGAN_LSTMnArchitecture(BaseGANArchitecture):
    """
    Conditional Generative Adversarial Network (cGAN) with LSTM-based architecture.

    Inherits from BaseGANArchitecture.
    """

    arch_type = "gan:conditional"

    def __init__(self, seq_len: int, feat_dim: int, latent_dim: int, output_dim: int, n_blocks: int = 1, output_activation: str = "tanh") -> None:
        """
        Initializes the cGAN_LSTMnArchitecture.

        :param seq_len: Length of input sequences.
        :type seq_len: int
        :param feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :param latent_dim: Dimensionality of the latent space.
        :type latent_dim: int
        :param output_dim: Dimensionality of the output.
        :type output_dim: int
        :param n_blocks: Number of LSTM blocks in the architecture (default is 1).
        :type n_blocks: int, optional
        :param output_activation: Activation function for the output layer (default is "tanh").
        :type output_activation: str, optional
        """
        super().__init__()
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._n_blocks = n_blocks
        self._output_activation = output_activation

        self.generator_in_channels = latent_dim + output_dim
        self.discriminator_in_channels = feat_dim + output_dim

        self._discriminator = self._build_discriminator()
        self._generator = self._build_generator(output_activation=output_activation)

    def _build_discriminator(self) -> keras.Model:
        d_input = keras.Input((self._seq_len, self.discriminator_in_channels))
        x = d_input
        for i in range(self._n_blocks - 1):
            x = layers.LSTM(64, return_sequences=True)(x)
            x = layers.Dropout(rate=0.2)(x)

        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(rate=0.2)(x)

        x = layers.GlobalAvgPool1D()(x)
        d_output = layers.Dense(1, activation="sigmoid")(x)
        discriminator = keras.Model(d_input, d_output, name="discriminator")
        return discriminator

    def _build_generator(self, output_activation: str) -> keras.Model:
        g_input = keras.Input((self.generator_in_channels,))

        x = layers.Dense(8 * 8 * self._seq_len)(g_input)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((self._seq_len, 64))(x)

        for i in range(self._n_blocks - 1):
            x = layers.LSTM(64, return_sequences=True)(x)
            x = layers.Dropout(rate=0.2)(x)
        x = layers.LSTM(256, return_sequences=True)(x)

        pool_and_stride = round((x.shape[1] + 1) / (self._seq_len + 1))

        x = layers.AveragePooling1D(pool_size=pool_and_stride, strides=pool_and_stride)(x)
        g_output = layers.Conv1D(self._feat_dim, 1, activation=output_activation)(x)
        generator = keras.Model(g_input, g_output, name="generator")
        return generator


class WaveGANArchitecture(BaseGANArchitecture):
    """
    WaveGAN architecture, from https://arxiv.org/abs/1802.04208

    Inherits from BaseGANArchitecture.
    """
    arch_type = "gan:raw"

    def __init__(self, seq_len: int, feat_dim: int = 64, latent_dim: int = 32, output_dim: int = 1, kernel_size: int = 32, phase_rad: int = 2, use_batchnorm: bool = False):
        """
        Initializes the WaveGANArchitecture.

        :param seq_len: Length of input sequences.
        :type seq_len: int
        :param feat_dim: Dimensionality of input features.
        :type feat_dim: int
        :param latent_dim: Dimensionality of the latent space.
        :type latent_dim: int
        :param output_dim: Dimensionality of the output.
        :type output_dim: int
        :param kernel_size: Sizes of convolutions
        :type kernel_size: int, optional
        :param phase_rad: Phase shuffle radius for wavegan (default is 2)
        :type phase_rad: int, optional
        :param use_batchnorm: Whether to use batchnorm (default is False)
        :type use_batchnorm: bool, optional
        """
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.phase_rad = phase_rad
        self.output_dim = output_dim
        self.use_batchnorm = use_batchnorm

        self._discriminator = self._build_discriminator()
        self._generator = self._build_generator()

    def _apply_phaseshuffle(self, x, rad):
        '''
        Based on
        https://github.com/chrisdonahue/wavegan/
        '''
        if rad <= 0 or x.shape[1] <= 1:
            return x

        b, x_len, nch = x.shape
        #  for keras 3.0
        phase = keras.random.randint([], minval=-rad, maxval=rad + 1, dtype="int32")
        pad_l, pad_r = ops.maximum(phase, 0), ops.maximum(-phase, 0)
        phase_start = pad_r
        x = ops.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode="reflect")

        x = x[:, phase_start:phase_start + x_len]
        x = ops.reshape(x, [b, x_len, nch])

        return x

    def _conv_transpose_block(self, inputs, channels, strides=4):
        x = layers.Conv1DTranspose(channels, self.kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x) if self.use_batchnorm else x
        x = layers.LeakyReLU()(x)
        return x

    def _build_generator(self):
        inputs = layers.Input((self.latent_dim,))
        x = layers.Dense(16 * 1024, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x) if self.use_batchnorm else x
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((16, 1024))(x)

        for conv_size in [512, 256, 128, 64]:
            x = self._conv_transpose_block(x, conv_size)

        x = layers.Conv1DTranspose(1, self.kernel_size, strides=4, padding='same', use_bias=False, activation='tanh')(x)
        pool_and_stride = math.ceil((x.shape[1] + 1) / (self.seq_len + 1))
        x = layers.AveragePooling1D(pool_size=pool_and_stride, strides=pool_and_stride)(x)
        return keras.Model(inputs, x)

    def _build_discriminator(self):
        inputs = layers.Input((self.seq_len, self.feat_dim))
        for conv_size in [64, 128, 256, 512]:
            x = layers.Conv1D(conv_size, self.kernel_size, strides=4, padding='same')(inputs)
            x = layers.BatchNormalization()(x) if self.use_batchnorm else x
            x = layers.LeakyReLU()(x)
            x = self._apply_phaseshuffle(x, self.phase_rad)

        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)

        return keras.Model(inputs, x)


class TimeEmbedding(layers.Layer):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs: tsgm.types.Tensor) -> tsgm.types.Tensor:
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

        return emb


class BaseDenoisingArchitecture(Architecture):
    """
    Base class for denoising architectures in DDPM (Denoising Diffusion Probabilistic Models, `tsgm.models.ddpm`).

    Attributes:
        arch_type: A string indicating the type of architecture, set to "ddpm:denoising".
        _seq_len: The length of the input sequences.
        _feat_dim: The dimensionality of the input features.
        _n_filters: The number of filters used in the convolutional layers.
        _n_conv_layers: The number of convolutional layers in the model.
        _model: The Keras model instance built using the `_build_model` method.
    """

    arch_type = "ddpm:denoising"

    def __init__(self, seq_len: int, feat_dim: int, n_filters: int = 64, n_conv_layers: int = 3, **kwargs) -> None:
        """
        Initializes the BaseDenoisingArchitecture with the specified parameters.

        Args:
            seq_len (int): The length of the input sequences.
            feat_dim (int): The dimensionality of the input features.
            n_filters (int, optional): The number of filters for convolutional layers. Default is 64.
            n_conv_layers (int, optional): The number of convolutional layers. Default is 3.
            **kwargs: Additional keyword arguments to be passed to the parent class `Architecture`.
        """
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._n_filters = n_filters
        self._n_conv_layers = n_conv_layers
        self._model = self._build_model()

    @property
    def model(self) -> keras.models.Model:
        """
        Provides access to the Keras model instance.

        Returns:
            keras.models.Model: The Keras model instance built by `_build_model`.
        """
        return self._model

    def get(self) -> T.Dict:
        """
        Returns a dictionary containing the model.

        :returns: A dictionary containing the model.
        :rtype: dict
        """
        return {"model": self.model}

    def _build_model(self) -> None:
        """
        Abstract method for building the Keras model.
        Subclasses must implement this method to define the specific architecture of the model.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError


class DDPMConvDenoiser(BaseDenoisingArchitecture):
    """
    A convolutional denoising model for DDPM.

    This class defines a convolutional neural network architecture used as a denoiser in DDPM.
    It predicts the noise added to the input samples during the diffusion process.

    Attributes:
        arch_type: A string indicating the architecture type, set to "ddpm:denoiser".
    """
    arch_type = "ddpm:denoiser"

    def __init__(self, **kwargs):
        """
        Initializes the DDPMConvDenoiser model with additional parameters.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """
        super().__init__(**kwargs)

    def _build_model(self) -> keras.Model:
        """
        Constructs and returns the Keras model for the DDPM denoiser.

        The model consists of:
            - A 1D convolutional layer to process input features.
            - An additional input layer for time embedding to incorporate timestep information.
            - `n_conv_layers` convolutional layers to process the combined features and time embeddings.
            - A final convolutional layer to output the predicted noise.

        Returns:
            keras.Model: The Keras model instance for the DDPM denoiser.
        """
        inputs = keras.Input(shape=(self._seq_len, self._feat_dim))

        # Input for the additional float parameter
        time_input = keras.Input(shape=(1,))

        temb = TimeEmbedding(dim=self._seq_len)(time_input)
        temb = keras.layers.Reshape((temb.shape[-1], 1))(temb)

        x = layers.Concatenate()([inputs, temb])

        for l_id in range(self._n_conv_layers):
            x = layers.Conv1D(self._n_filters, 3, padding="same", activation="relu")(x)

        outputs = layers.Conv1D(self._feat_dim, 3, padding="same")(x)
        return keras.Model(inputs=[inputs, time_input], outputs=outputs)


class Zoo(dict):
    """
    A collection of architectures represented. It behaves like supports Python `dict` API.
    """
    def __init__(self, *arg, **kwargs) -> None:
        """
        Initializes the Zoo.
        """
        super(Zoo, self).__init__(*arg, **kwargs)

    def summary(self) -> None:
        """
        Prints a summary of architectures in the Zoo.
        """
        summary_table = PrettyTable()
        summary_table.field_names = ["id", "type"]
        for k, v in self.items():
            summary_table.add_row([k, v.arch_type])
        print(summary_table)


zoo = Zoo(
    {
        # Generative models
        "vae_conv5": VAE_CONV5Architecture,
        "cvae_conv5": cVAE_CONV5Architecture,
        "cgan_base_c4_l1": cGAN_Conv4Architecture,
        "t-cgan_c4": tcGAN_Conv4Architecture,
        "cgan_lstm_n": cGAN_LSTMnArchitecture,
        "cgan_lstm_3": cGAN_LSTMConv3Architecture,
        "wavegan": WaveGANArchitecture,
        "ddpm_denoiser": DDPMConvDenoiser,

        # Downstream models
        "clf_cn": ConvnArchitecture,
        "clf_cl_n": ConvnLSTMnArchitecture,
        "clf_block": BlockClfArchitecture,
        "recurrent": BasicRecurrentArchitecture,
        "clf_transformer": TransformerClfArchitecture
    }
)
