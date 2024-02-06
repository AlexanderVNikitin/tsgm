:github_url: https://github.com/

Time Series Simulator (TSGM) Official Documentation
========================================

Time Series Generative Modeling (TSGM) is a Python framework for time series data generation. It include data-driven and model-based approaches to synthetic time-series generation. It uses both generative


The package is built on top of `Tensorflow <https://www.tensorflow.org/>`_ that allows training the models on CPUs, GPUs, or TPUs.

Quick start:

.. code-block:: bash

    pip install tsgm


.. code-block:: python

    import tsgm

    # ... Define hyperparameters ...
    # dataset is a tensor of shape n_samples x seq_len x feature_dim

    # Zoo contains several prebuilt architectures: we choose a conditional GAN architecture
    architecture = tsgm.models.architectures.zoo["cgan_base_c4_l1"](
        seq_len=seq_len, feat_dim=feature_dim,
        latent_dim=latent_dim, output_dim=0)
    discriminator, generator = architecture.discriminator, architecture.generator

    # Initialize GAN object with selected discriminator and generator
    gan = tsgm.models.cgan.GAN(
        discriminator=discriminator, generator=generator, latent_dim=latent_dim
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    gan.fit(dataset, epochs=1)

    # Generate 10 synthetic samples
    result = gan.generate(10)

For more examples, see `our tutorials <https://github.com/AlexanderVNikitin/tsgm/tree/main/tutorials>`_.

If you find this repo useful, please consider citing our paper:

.. code-block:: latex

	@article{nikitin2023tsgm,
	    title={TSGM: A Flexible Framework for Generative Modeling of Synthetic Time Series},
	    author={Nikitin, Alexander and Iannucci, Letizia and Kaski, Samuel},
	    journal={arXiv preprint arXiv:2305.11567},
	    year={2023}
	}

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Guides

   guides/installation
   guides/introduction
   guides/datasets
   guides/resources

.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Package Reference

   modules/root
   modules/optimization
   modules/utils
   modules/metrics
