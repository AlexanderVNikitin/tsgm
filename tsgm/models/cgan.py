import typing as T
import keras
from keras import ops
try:
    import tensorflow_privacy as tf_privacy
    __tf_privacy_available = True
except (ModuleNotFoundError, ImportError):
    __tf_privacy_available = False

import logging

import tsgm

import os
from tsgm.backend import get_backend


logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


def _is_dp_optimizer(optimizer: keras.optimizers.Optimizer) -> bool:
    return __tf_privacy_available \
        and (isinstance(optimizer, tf_privacy.DPKerasAdagradOptimizer)
             or isinstance(optimizer, tf_privacy.DPKerasAdamOptimizer)
             or isinstance(optimizer, tf_privacy.DPKerasSGDOptimizer))


def _discriminator_loss(discriminator: keras.Model, loss_fn: T.Callable,
                        fake_data: tsgm.types.Tensor, real_data: tsgm.types.Tensor) -> tsgm.types.Tensor:
    """
    Discriminator loss for a batch of generated samples (label 1) and a batch of real samples (label 0) of the same size.

    The discriminator scores the two batches in separate forward passes. Layers that depend on the composition
    of a batch (e.g., BatchNormalization) would otherwise normalize generated and real samples together in the
    discriminator step, but generated samples alone in the generator step.

    The two losses are combined after they are computed, and the predictions are never concatenated:
    for a discriminator that ends with a sigmoid, Keras with the TensorFlow backend computes the crossentropy
    from the logits (which keeps the gradients of a saturated discriminator) only if the loss receives
    the output of the discriminator itself.
    """
    fake_predictions = discriminator(fake_data, training=True)
    real_predictions = discriminator(real_data, training=True)
    fake_loss = loss_fn(ops.ones_like(fake_predictions), fake_predictions)
    real_loss = loss_fn(ops.zeros_like(real_predictions), real_predictions)
    if len(fake_loss.shape) > 0:
        # per-sample losses (a loss without reduction, e.g., for DP optimizers)
        return ops.concatenate([fake_loss, real_loss], axis=0)
    if getattr(loss_fn, "reduction", None) == "sum":
        return fake_loss + real_loss
    return 0.5 * (fake_loss + real_loss)


class _GANOptimizers(keras.optimizers.Optimizer):
    """Expose both optimizers' state to Keras' single-optimizer JAX trainer."""

    def __init__(self, discriminator_optimizer, generator_optimizer, discriminator_variables, generator_variables):
        super().__init__(learning_rate=0.0, name="gan_optimizers")
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.discriminator_variables = list(discriminator_variables)
        self.generator_variables = list(generator_variables)
        discriminator_optimizer.build(self.discriminator_variables)
        generator_optimizer.build(self.generator_variables)
        self.built = True

    @property
    def variables(self):
        return self.discriminator_optimizer.variables + self.generator_optimizer.variables

    @property
    def iterations(self):
        return self.discriminator_optimizer.iterations

    def finalize_variable_values(self, variables):
        self.discriminator_optimizer.finalize_variable_values(self.discriminator_variables)
        self.generator_optimizer.finalize_variable_values(self.generator_variables)


class _JAXGANTraining:
    """Stateless alternating updates shared by unconditional and conditional GANs."""

    def _jax_gradient_penalty(self, jax, real_data, fake_data):
        alpha = keras.random.normal(
            (real_data.shape[0], 1, 1), seed=self.seed_generator
        )
        interpolated = real_data + alpha * (fake_data - real_data)
        trainable_values = [v.value for v in self.trainable_variables]
        non_trainable_values = [v.value for v in self.non_trainable_variables]

        def prediction_sum(inputs):
            # The inner differentiation must own its state updates, so that
            # batch-normalization values do not escape as JAX tracers.
            mapping = list(zip(self.trainable_variables, trainable_values)) + list(
                zip(self.non_trainable_variables, non_trainable_values)
            )
            with keras.StatelessScope(state_mapping=mapping) as scope:
                prediction = self.discriminator(inputs, training=True)
            updated_values = [scope.get_current_value(v) for v in self.non_trainable_variables]
            return ops.sum(prediction), updated_values

        (_, updated_values), gradients = jax.value_and_grad(prediction_sum, has_aux=True)(interpolated)
        for variable, value in zip(self.non_trainable_variables, updated_values):
            variable.assign(value)
        norm = ops.sqrt(ops.sum(ops.square(gradients), axis=(1, 2)) + keras.backend.epsilon())
        return ops.mean(ops.square(norm - 1.0))

    def train_step_jax(self, jax, state, data):
        trainable_values, non_trainable_values, optimizer_values, metrics_values = state
        real_ts, labels, repeated_labels = self._prepare_jax_data(data)
        variable_indices = {id(v): i for i, v in enumerate(self.trainable_variables)}
        d_indices = [variable_indices[id(v)] for v in self.discriminator.trainable_variables]
        g_indices = [variable_indices[id(v)] for v in self.generator.trainable_variables]
        d_optimizer_size = len(self.d_optimizer.variables)

        def discriminator_inputs(samples):
            if repeated_labels is None:
                return samples
            return ops.concatenate([samples, repeated_labels], axis=-1)

        def loss_fn(active_values, all_values, non_trainable, indices, discriminator_step):
            all_values = list(all_values)
            for index, value in zip(indices, active_values):
                all_values[index] = value
            mapping = list(zip(self.trainable_variables, all_values)) + list(
                zip(self.non_trainable_variables, non_trainable)
            )
            with keras.StatelessScope(state_mapping=mapping) as scope:
                noise = self._get_random_vector_labels(real_ts.shape[0], labels)
                fake_data = discriminator_inputs(self.generator(noise, training=True))
                if discriminator_step:
                    real_data = discriminator_inputs(real_ts)
                    if getattr(self, "use_wgan", False):
                        fake_logits = self.discriminator(fake_data, training=True)
                        real_logits = self.discriminator(real_data, training=True)
                        loss = self.wgan_discriminator_loss(real_logits, fake_logits)
                        loss += self.gp_weight * self._jax_gradient_penalty(jax, real_data, fake_data)
                    else:
                        loss = _discriminator_loss(self.discriminator, self.loss_fn, fake_data, real_data)
                else:
                    predictions = self.discriminator(fake_data, training=True)
                    if getattr(self, "use_wgan", False):
                        loss = self.wgan_generator_loss(predictions)
                    else:
                        loss = self.loss_fn(ops.zeros_like(predictions), predictions)
                # JAX gradients require a scalar, including for unreduced losses.
                loss = ops.mean(loss)
            non_trainable = [scope.get_current_value(v) for v in self.non_trainable_variables]
            return loss, non_trainable

        def d_loss_fn(values, all_values, non_trainable):
            return loss_fn(values, all_values, non_trainable, d_indices, True)

        (d_loss, non_trainable_values), d_gradients = jax.value_and_grad(d_loss_fn, has_aux=True)(
            [trainable_values[i] for i in d_indices], trainable_values, non_trainable_values
        )
        d_values, d_optimizer_values = self.d_optimizer.stateless_apply(
            optimizer_values[:d_optimizer_size], d_gradients, [trainable_values[i] for i in d_indices]
        )
        trainable_values = list(trainable_values)
        for index, value in zip(d_indices, d_values):
            trainable_values[index] = value

        def g_loss_fn(values, all_values, non_trainable):
            return loss_fn(values, all_values, non_trainable, g_indices, False)

        (g_loss, non_trainable_values), g_gradients = jax.value_and_grad(g_loss_fn, has_aux=True)(
            [trainable_values[i] for i in g_indices], trainable_values, non_trainable_values
        )
        g_values, g_optimizer_values = self.g_optimizer.stateless_apply(
            optimizer_values[d_optimizer_size:], g_gradients, [trainable_values[i] for i in g_indices]
        )
        for index, value in zip(g_indices, g_values):
            trainable_values[index] = value

        with keras.StatelessScope(state_mapping=list(zip(self.metrics_variables, metrics_values))) as scope:
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            logs = {"g_loss": self.gen_loss_tracker.result(), "d_loss": self.disc_loss_tracker.result()}
        metrics_values = [scope.get_current_value(v) for v in self.metrics_variables]
        return logs, (
            trainable_values, non_trainable_values,
            d_optimizer_values + g_optimizer_values, metrics_values,
        )


class GAN(_JAXGANTraining, keras.Model):
    """
    GAN implementation for unlabeled time series.
    """
    def __init__(self, discriminator: keras.Model, generator: keras.Model, latent_dim: int, use_wgan: bool = False) -> None:
        """
        :param discriminator: A discriminator model which takes a time series as input and check
            whether the sample is real or fake.
        :type discriminator: keras.Model
        :param generator: Takes as input a random noise vector of `latent_dim` length and returns
            a simulated time-series.
        :type generator: keras.Model
        :param latent_dim: The size of the noise vector.
        :type latent_dim: int
        :param use_wgan: Use Wasserstein GAN with gradient penalty.
        :type use_wgan: bool
        """
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self._seq_len = self.generator.output_shape[1]
        self.use_wgan = use_wgan
        self.gp_weight = 10.0
        self.seed_generator = keras.random.SeedGenerator() if keras.backend.backend() == "jax" else None

        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def call(self, inputs):
        """
        Forward pass for the GAN model.
        This method is required for Keras 3 compatibility with PyTorch backend.
        """
        # For GAN training, we don't typically call the model directly.
        # This is just a placeholder to satisfy Keras 3 requirements.
        # Return the inputs as-is since this is mainly used for building the model.
        return inputs

    def wgan_discriminator_loss(self, real_sample, fake_sample):
        real_loss = ops.mean(real_sample)
        fake_loss = ops.mean(fake_sample)
        return fake_loss - real_loss

    # Define the loss functions to be used for generator
    def wgan_generator_loss(self, fake_sample):
        return -ops.mean(fake_sample)

    def gradient_penalty_tf(self, tf, interpolated):
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated sample.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated sample.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        return grads

    def gradient_penalty_torch(self, torch, interpolated):
        # Create a new tensor that requires grad instead of modifying existing one
        interpolated = interpolated.detach().requires_grad_(True)
        pred = self.discriminator(interpolated, training=True)
        grads = torch.autograd.grad(outputs=pred, inputs=interpolated,
                                    grad_outputs=ops.ones_like(pred),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        return grads

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        # get the interpolated samples
        alpha_shape = [batch_size, 1, 1]
        # Create alpha on the same device as real_samples
        alpha = ops.ones_like(real_samples[:, :1, :1]) * keras.random.normal(alpha_shape, 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            grads = self.gradient_penalty_tf(backend, interpolated)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            grads = self.gradient_penalty_torch(backend, interpolated)
        # 3. Calcuate the norm of the gradients
        norm = ops.sqrt(ops.sum(ops.square(grads), axis=[1, 2]))
        gp = ops.mean((norm - 1.0) ** 2)
        return gp

    @property
    def metrics(self) -> T.List:
        """
        :returns: A list of metrics trackers (e.g., generator's loss and discriminator's loss).
        """
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer: keras.optimizers.Optimizer, g_optimizer: keras.optimizers.Optimizer,
                loss_fn: keras.losses.Loss) -> None:
        """
        Compiles the generator and discriminator models.

        :param d_optimizer: An optimizer for the GAN's discriminator.
        :type d_optimizer: keras.optimizers.Optimizer
        :param g_optimizer: An optimizer for the GAN's generator.
        :type g_optimizer: keras.optimizers.Optimizer
        :param loss_fn: Loss function.
        :type loss_fn: keras.losses.Loss
        """
        if keras.backend.backend() == "jax":
            super(GAN, self).compile(optimizer=_GANOptimizers(
                d_optimizer, g_optimizer, self.discriminator.trainable_variables, self.generator.trainable_variables
            ))
        else:
            super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

        generator_dp = _is_dp_optimizer(d_optimizer)
        discriminator_dp = _is_dp_optimizer(g_optimizer)

        if generator_dp != discriminator_dp:
            logger.warning(f"One of the optimizers is DP and another one is not. generator_dp={generator_dp}, discriminator_dp={discriminator_dp}")

        self.dp = generator_dp and discriminator_dp

    def _get_random_vector_labels(self, batch_size: int, labels=None) -> tsgm.types.Tensor:
        return keras.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)

    def _prepare_jax_data(self, data):
        real_data, _, _ = keras.utils.unpack_x_y_sample_weight(data)
        return real_data, None, None

    def train_step_tf(self, tf, data: tsgm.types.Tensor) -> T.Dict[str, float]:
        real_data = data
        batch_size = ops.shape(real_data)[0]
        # Generate ts
        random_vector = self._get_random_vector_labels(batch_size)
        fake_data = self.generator(random_vector, training=True)

        with tf.GradientTape() as tape:
            if self.use_wgan:
                fake_logits = self.discriminator(fake_data, training=True)
                # Get the logits for the real samples
                real_logits = self.discriminator(real_data, training=True)

                # Calculate the discriminator loss using the fake and real sample logits
                d_cost = self.wgan_discriminator_loss(real_logits, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight
            else:
                d_loss = _discriminator_loss(self.discriminator, self.loss_fn, fake_data, real_data)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_vector = self._get_random_vector_labels(batch_size=batch_size)

        # Pretend that all samples are real
        misleading_labels = ops.zeros((batch_size, 1))

        # Train generator (with updating the discriminator)
        with tf.GradientTape() as tape:
            fake_data = self.generator(random_vector, training=True)
            predictions = self.discriminator(fake_data, training=True)
            if self.use_wgan:
                # uses logits
                g_loss = self.wgan_generator_loss(predictions)
            else:
                g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def train_step_torch(self, torch, data: tsgm.types.Tensor) -> T.Dict[str, float]:
        # Handle PyTorch DataLoader format - extract tensor from list
        if isinstance(data, (list, tuple)) and len(data) == 1:
            real_data = data[0]
        else:
            real_data = data

        # Ensure real_data is on the same device as model parameters (MPS)
        if hasattr(real_data, 'device'):
            # Get device from first model parameter
            model_device = next(self.generator.parameters()).device
            if real_data.device != model_device:
                real_data = real_data.to(model_device)

        batch_size = ops.shape(real_data)[0]
        # Generate ts
        random_vector = self._get_random_vector_labels(batch_size)
        fake_data = self.generator(random_vector, training=True)

        if self.use_wgan:
            fake_logits = self.discriminator(fake_data, training=True)
            # Get the logits for the real samples
            real_logits = self.discriminator(real_data, training=True)

            # Calculate the discriminator loss using the fake and real sample logits
            d_cost = self.wgan_discriminator_loss(real_logits, fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(batch_size, real_data, fake_data)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight
        else:
            d_loss = _discriminator_loss(self.discriminator, self.loss_fn, fake_data, real_data)
        self.discriminator.zero_grad()
        d_loss.backward()

        d_trainable_weights = [v for v in self.discriminator.trainable_weights]
        d_gradients = [v.value.grad for v in d_trainable_weights]

        with torch.no_grad():
            # Keras 3 expects (gradient, variable) pairs
            grads_and_vars = list(zip(d_gradients, d_trainable_weights))
            self.d_optimizer.apply_gradients(grads_and_vars)
        random_vector = self._get_random_vector_labels(batch_size=batch_size)
        misleading_labels = ops.zeros((batch_size, 1))
        fake_data = self.generator(random_vector, training=True)
        predictions = self.discriminator(fake_data, training=True)
        if self.use_wgan:
            # uses logits
            g_loss = self.wgan_generator_loss(predictions)
        else:
            g_loss = self.loss_fn(misleading_labels, predictions)

        self.generator.zero_grad()
        g_loss.backward()

        g_trainable_weights = [v for v in self.generator.trainable_weights]
        g_gradients = [v.value.grad for v in g_trainable_weights]

        with torch.no_grad():
            # Keras 3 expects (gradient, variable) pairs
            grads_and_vars = list(zip(g_gradients, g_trainable_weights))
            self.g_optimizer.apply_gradients(grads_and_vars)

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def train_step(self, *args) -> T.Any:
        """
        Performs a training step using a batch of data, stored in data.

        :param data: A batch of data in a format batch_size x seq_len x feat_dim
        :type data: tsgm.types.Tensor

        :returns: A dictionary with generator (key "g_loss") and discriminator (key "d_loss") losses
        :rtype: T.Dict[str, float]
        """
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.train_step_tf(backend, *args)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.train_step_torch(backend, *args)
        elif os.environ.get("KERAS_BACKEND") == "jax":
            return self.train_step_jax(backend, *args)
        raise ValueError(f"Unsupported backend: {os.environ.get('KERAS_BACKEND')}")

    def generate(self, num: int) -> tsgm.types.Tensor:
        """
        Generates new data from the model.

        :param num: the number of samples to be generated.
        :type num: int

        :returns: Generated samples
        :rtype: tsgm.types.Tensor
        """
        random_vector_labels = self._get_random_vector_labels(batch_size=num)
        return self.generator(random_vector_labels)

    def clone(self) -> "GAN":
        """
        Clones GAN object

        :returns: The exact copy of the object
        :rtype: "GAN"
        """
        copy_model = GAN(self.discriminator, self.generator, latent_dim=self.latent_dim)
        copy_model = copy_model.set_weights(self.get_weights())
        return copy_model


class ConditionalGAN(_JAXGANTraining, keras.Model):
    """
    Conditional GAN implementation for labeled and temporally labeled time series.
    """
    def __init__(self, discriminator: keras.Model, generator: keras.Model, latent_dim: int, temporal=False, use_wgan=False) -> None:
        """
        :param discriminator: A discriminator model which takes a time series as input and check
            whether the sample is real or fake.
        :type discriminator: keras.Model
        :param generator: Takes as input a random noise vector of `latent_dim` length and return
            a simulated time-series.
        :type generator: keras.Model
        :param latent_dim: The size of the noise vector.
        :type latent_dim: int
        :param temporal: Indicates whether the time series temporally labeled or not.
        :type temporal: bool
        :param use_wgan: Use Wasserstein GAN with gradient penalty. Default is False.
        :type use_wgan: bool
        """
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self._seq_len = self.generator.output_shape[1]

        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self._temporal = temporal
        self.seed_generator = keras.random.SeedGenerator() if keras.backend.backend() == "jax" else None

    def call(self, inputs):
        """
        Forward pass for the ConditionalGAN model.
        This method is required for Keras 3 compatibility with PyTorch backend.
        """
        # For Conditional GAN training, we don't typically call the model directly.
        # This is just a placeholder to satisfy Keras 3 requirements.
        # Return the inputs as-is since this is mainly used for building the model.
        return inputs

    @property
    def metrics(self) -> T.List:
        """
        :returns: A list of metrics trackers (e.g., generator's loss and discriminator's loss).
        :rtype: T.List
        """
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer: keras.optimizers.Optimizer, g_optimizer: keras.optimizers.Optimizer, loss_fn: T.Callable) -> None:
        """
        Compiles the generator and discriminator models.

        :param d_optimizer: An optimizer for the GAN's discriminator.
        :type d_optimizer: keras.optimizers.Optimizer
        :param g_optimizer: An optimizer for the GAN's generator.
        :type g_optimizer: keras.optimizers.Optimizer
        :param loss_fn: Loss function.
        :type loss_fn: keras.losses.Loss
        """
        # TODO: move `.compile logic to a base GAN class
        if keras.backend.backend() == "jax":
            super(ConditionalGAN, self).compile(optimizer=_GANOptimizers(
                d_optimizer, g_optimizer, self.discriminator.trainable_variables, self.generator.trainable_variables
            ))
        else:
            super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

        generator_dp = _is_dp_optimizer(d_optimizer)
        discriminator_dp = _is_dp_optimizer(g_optimizer)

        if generator_dp != discriminator_dp:
            logger.warning(f"One of the optimizers is DP and another one is not. generator_dp={generator_dp}, discriminator_dp={discriminator_dp}")

        self.dp = generator_dp and discriminator_dp

    def _get_random_vector_labels(self, batch_size: int, labels: tsgm.types.Tensor) -> None:
        if self._temporal:
            random_latent_vectors = keras.random.normal(
                shape=(batch_size, self._seq_len, self.latent_dim), seed=self.seed_generator
            )
            random_vector_labels = ops.concatenate(
                [random_latent_vectors, labels[:, :, None]], axis=2
            )
        else:
            random_latent_vectors = keras.random.normal(
                shape=(batch_size, self.latent_dim), seed=self.seed_generator
            )
            random_vector_labels = ops.concatenate(
                [random_latent_vectors, labels], axis=1
            )
        return random_vector_labels

    def _prepare_jax_data(self, data):
        real_ts, labels, _ = keras.utils.unpack_x_y_sample_weight(data)
        output_dim = self._get_output_shape(labels)
        if self._temporal:
            repeated_labels = labels
        else:
            repeated_labels = ops.repeat(labels[:, :, None], repeats=self._seq_len)
        repeated_labels = ops.reshape(repeated_labels, (-1, self._seq_len, output_dim))
        return real_ts, labels, repeated_labels

    def _get_output_shape(self, labels: tsgm.types.Tensor) -> int:
        if self._temporal:
            if len(labels.shape) == 2:
                return 1
            else:
                return labels.shape[2]
        else:
            return labels.shape[1]

    def train_step_tf(self, tf, data: T.Tuple) -> T.Dict[str, float]:
        real_ts = data[0]
        labels = data[1]
        output_dim = self._get_output_shape(labels)
        batch_size = ops.shape(real_ts)[0]
        if not self._temporal:
            rep_labels = labels[:, :, None]
            rep_labels = ops.repeat(
                rep_labels, repeats=[self._seq_len]
            )
        else:
            rep_labels = labels

        rep_labels = ops.reshape(
            rep_labels, (-1, self._seq_len, output_dim)
        )

        # Generate ts
        random_vector_labels = self._get_random_vector_labels(batch_size=batch_size, labels=labels)
        generated_ts = self.generator(random_vector_labels, training=True)

        fake_data = ops.concatenate([generated_ts, rep_labels], -1)
        real_data = ops.concatenate([real_ts, rep_labels], -1)

        with tf.GradientTape() as tape:
            d_loss = _discriminator_loss(self.discriminator, self.loss_fn, fake_data, real_data)

        if self.dp:
            # For DP optimizers from `tensorflow.privacy`
            self.d_optimizer.minimize(d_loss, self.discriminator.trainable_weights, tape=tape)
        else:
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        random_vector_labels = self._get_random_vector_labels(batch_size=batch_size, labels=labels)

        # Pretend that all samples are real
        misleading_labels = ops.zeros((batch_size, 1))

        # Train generator (with updating the discriminator)
        with tf.GradientTape() as tape:
            fake_samples = self.generator(random_vector_labels, training=True)
            fake_data = ops.concatenate([fake_samples, rep_labels], -1)
            predictions = self.discriminator(fake_data, training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)

        if self.dp:
            # For DP optimizers from `tensorflow.privacy`
            self.g_optimizer.minimize(g_loss, self.generator.trainable_weights, tape=tape)
        else:
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def train_step_torch(self, torch, data: T.Tuple) -> T.Dict[str, float]:
        # Handle PyTorch DataLoader format
        if isinstance(data, (list, tuple)) and len(data) == 2:
            real_ts, labels = data
        else:
            # Fallback for single input
            real_ts, labels = data, None
        output_dim = self._get_output_shape(labels)
        batch_size = ops.shape(real_ts)[0]
        if not self._temporal:
            rep_labels = labels[:, :, None]
            rep_labels = ops.repeat(
                rep_labels, repeats=[self._seq_len]
            )
        else:
            rep_labels = labels

        rep_labels = ops.reshape(
            rep_labels, (-1, self._seq_len, output_dim)
        )

        # Generate ts
        random_vector_labels = self._get_random_vector_labels(batch_size=batch_size, labels=labels)
        generated_ts = self.generator(random_vector_labels, training=True)

        fake_data = ops.concatenate([generated_ts, rep_labels], -1)
        real_data = ops.concatenate([real_ts, rep_labels], -1)

        d_loss = _discriminator_loss(self.discriminator, self.loss_fn, fake_data, real_data)
        self.discriminator.zero_grad()
        d_loss.backward()

        d_trainable_weights = [v for v in self.discriminator.trainable_weights]
        d_gradients = [v.value.grad for v in d_trainable_weights]

        with torch.no_grad():
            # Keras 3 expects (gradient, variable) pairs
            grads_and_vars = list(zip(d_gradients, d_trainable_weights))
            self.d_optimizer.apply_gradients(grads_and_vars)
        random_vector_labels = self._get_random_vector_labels(batch_size=batch_size, labels=labels)

        # Pretend that all samples are real
        misleading_labels = ops.zeros((batch_size, 1))

        # Train generator (with updating the discriminator)
        fake_samples = self.generator(random_vector_labels, training=True)
        fake_data = ops.concatenate([fake_samples, rep_labels], -1)
        predictions = self.discriminator(fake_data, training=True)
        g_loss = self.loss_fn(misleading_labels, predictions)

        self.generator.zero_grad()
        g_loss.backward()

        g_trainable_weights = [v for v in self.generator.trainable_weights]
        g_gradients = [v.value.grad for v in g_trainable_weights]

        with torch.no_grad():
            # Keras 3 expects (gradient, variable) pairs
            grads_and_vars = list(zip(g_gradients, g_trainable_weights))
            self.g_optimizer.apply_gradients(grads_and_vars)

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

    def train_step(self, *args) -> T.Any:
        """
        Performs a training step using a batch of data, stored in data.

        :param data: A batch of data in a format batch_size x seq_len x feat_dim
        :type data: tsgm.types.Tensor

        :returns: A dictionary with generator (key "g_loss") and discriminator (key "d_loss") losses
        :rtype: T.Dict[str, float]
        """
        backend = get_backend()
        if os.environ.get("KERAS_BACKEND") == "tensorflow":
            return self.train_step_tf(backend, *args)
        elif os.environ.get("KERAS_BACKEND") == "torch":
            return self.train_step_torch(backend, *args)
        elif os.environ.get("KERAS_BACKEND") == "jax":
            return self.train_step_jax(backend, *args)
        else:
            raise ValueError(f"Unsupported backend: {os.environ.get('KERAS_BACKEND')}")

    def generate(self, labels: tsgm.types.Tensor) -> tsgm.types.Tensor:
        """
        Generates new data from the model.

        :param labels: The labels for which to generate samples.
        :type labels: tsgm.types.Tensor

        :returns: Generated samples.
        :rtype: tsgm.types.Tensor
        """
        batch_size = labels.shape[0]

        random_vector_labels = self._get_random_vector_labels(
            batch_size=batch_size, labels=labels)
        return self.generator(random_vector_labels)
