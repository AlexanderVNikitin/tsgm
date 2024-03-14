Introduction
=======================

`Time Series Generative Modeling (TSGM) <https://github.com/AlexanderVNikitin/tsgm>`_ is a generative modeling framework for synthetic time series data. It builds on open-source libraries and implements various methods, such as GANs, VAEs, or ABC, for synthetic time series simulation. Moreover, *TSGM* provides many approaches for evaluating synthetic time series data. It is built on top of `TensorFlow <https://www.tensorflow.org/>`_.


Key Features
------------

TSGM offers a wide range of features to support the generation and evaluation of synthetic time series data. Some of its key features include:

- **Generative Models:** TSGM integrates popular generative modeling techniques such as GANs, VAEs, and ABC to simulate realistic time series data. These models enable the generation of diverse and representative synthetic time series.

- **Augmentations:** TSGM allows users to easily augment their time series data.

- **Evaluation Approaches:** TSGM provides multiple approaches for evaluating the quality of synthetic time series data. These evaluation methods help assess the fidelity of the generated data by comparing it to real-world time series, enabling researchers to measure the accuracy and statistical properties of the synthetic data.

- **Built on Keras:** TSGM is built on top of the `Keras <https://www.keras.io/>`_ deep learning framework. It offers efficient computation and enables seamless integration with other TensorFlow-based models and libraries, allowing users to leverage its extensive ecosystem for further customization and experimentation.


Augmentations
=============================
TSGM implements multiple augmentation approaches including window warping, shuffling, slicing, and dynamic time barycentric average. See specific methods in `tsgm.models.augmentations`. A minimalistic example of time series augmentation:

.. code-block:: python

	import tsgm
	X = tsgm.utils.gen_sine_dataset(100, 64, 2, max_value=20)
	aug_model = tsgm.models.augmentations.GaussianNoise(variance=0.2)
	samples = aug_model.generate(X=X, n_samples=10)

More examples are available in `the augmentation tutorial. <https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/augmentations.ipynb>`_

Generators
=============================
A central concept of TSGM is `Generator`. The generator can be trained on historical data using method `.fit` in a standard for Keras models way.

The training of data-driven simulators can be done via likelihood optimization, adversarial training procedures, or variational methods. Some of the implemented data-driven simulators include:

- `tsgm.models.sts.STS` - Structural Time Series model for time sires generation,\\
- `tsgm.models.cgan.GAN` - standard GAN model adapted for time-series simulation,\\
- `tsgm.models.cgan.ConditionalGAN` - conditional GAN model for labeled and temporally labeled time-series simulation,\\
- `tsgm.models.cvae.BetaVAE` - beta-VAE model adapted for time-series simulation,\\
- `tsgm.models.cvae.cBetaVAE` - conditional beta-VAE model for labeled and temporally labeled time-series simulation,\\
- `tsgm.models.cvae.TimeGAN` - extended GAN-based model for time series generation.

A minimalistic example of synthetic data generation with VAEs:

.. code-block:: python

	import tsgm
	from tensorflow import keras

	n, n_ts, n_features  = 1000, 24, 5
	data = tsgm.utils.gen_sine_dataset(n, n_ts, n_features)
	scaler = tsgm.utils.TSFeatureWiseScaler()        
	scaled_data = scaler.fit_transform(data)
	architecture = tsgm.models.zoo["vae_conv5"](n_ts, n_features, 10)
	encoder, decoder = architecture.encoder, architecture.decoder
	vae = tsgm.models.cvae.BetaVAE(encoder, decoder)
	vae.compile(optimizer=keras.optimizers.Adam())

	vae.fit(scaled_data, epochs=1, batch_size=64)
	vae.generate(10)

Dataset
=============================
In TSGM, time series datasets are often stored in one of two ways: wrapped in a `tss.dataset.Dataset` object, or as a Tensor with shape `n_samples x n_timestamps x n_features`.

Class `tsgm.dataset.DatasetProperties` implements generic placeholder for data when they are unavailable.

`tsgm.utils` has a plenty of datasets, see :ref:`datasets-label`. For instance,

.. code-block:: python

    import tsgm

    ucr_data_manager = tsgm.utils.UCRDataManager(ds="gunpoint")
    assert ucr_data_manager.summary() is None
    X_train, y_train, X_test, y_test = ucr_data_manager.get()


Architectures Zoo
=============================
Architectures Zoo is a storage object of NN architectures that can be utilized by the framework users.
It provides architectures for GANs, VAEs, and downstream task models. It also provides additional information on the implemented architectures via `tsgm.models.zoo.summary()`. `tsgm.models.zoo` object support API of Python dictionary. In particular the users can add their custom models to it.

For example, the models from zoo can be used as follows:

.. code-block:: python

    import tsgm

    model_type = tsgm.models.architectures.zoo["cgan_lstm_n"]
    arch = model_type(
        seq_len=seq_len, feat_dim=feat_dim,
        latent_dim=latent_dim, output_dim=output_dim)
    arch_dict = arch.get()
    # arch will store `.generator` and `.discriminator` fields for cGAN


Metrics
=============================
In `tsgm.metrics`, we implemented several metrics for evaluation of generated time series. Essentially, these metrics are subdivided into five types:

- data similarity / distance,
- predictive consistency,
- fairness,
- privacy,
- downstream effectiveness,
- visual similarity.

See the following code for an example of using metrics:

.. code-block:: python

	import tsgm
	import functools
	import numpy as np

	Xr, yr = tsgm.utils.gen_sine_vs_const_dataset(10, 100, 20, max_value=2, const=1)  # real data
	Xs, ys = Xr + 1e-5, yr  # synthetic data

	d_real = tsgm.dataset.Dataset(Xr, yr)
	d_syn = tsgm.dataset.Dataset(Xs, ys)

	statistics = [
		functools.partial(tsgm.metrics.statistics.axis_max_s, axis=None),
        functools.partial(tsgm.metrics.statistics.axis_min_s, axis=None)]
	sim_metric = tsgm.metrics.DistanceMetric(
    	statistics=statistics, discrepancy=lambda x, y: np.linalg.norm(x - y)
	)
	sim_metric = tsgm.metrics.DistanceMetric(
    	statistics=statistics, discrepancy=discrepancy_func
	)
	sim_metric(d_real, d_syn)


Implementations and examples of these methods are described in `tutorials/metrics.ipynb`.


Citing
=======================
If you find the *TSGM* useful, please consider citing our paper:

.. code-block:: latex

	@article{nikitin2023tsgm,
	  title={TSGM: A Flexible Framework for Generative Modeling of Synthetic Time Series},
	  author={Nikitin, Alexander and Iannucci, Letizia and Kaski, Samuel},
	  journal={arXiv preprint arXiv:2305.11567},
	  year={2023}
	}


