Introduction
=======================

`Time Series Generative Modeling (TSGM) <https://github.com/AlexanderVNikitin/tsgm>`_ is a generative modeling framework for synthetic time series data. It builds on open-source libraries and implements various methods, such as GANs, VAEs, or ABC, for synthetic time series simulation. Moreover, *TSGM* provides many approaches for evaluating synthetic time series data. It is built on top of `TensorFlow <https://www.tensorflow.org/>`_.


Key Features
------------

TSGM offers a wide range of features to support the generation and evaluation of synthetic time series data. Some of its key features include:

- **Generative Models:** TSGM integrates popular generative modeling techniques such as GANs, VAEs, and ABC to simulate realistic time series data. These models enable the generation of diverse and representative synthetic time series.

- **Augmentations:** TSGM allows users to easily augment their time series data.

- **Evaluation Approaches:** TSGM provides multiple approaches for evaluating the quality of synthetic time series data. These evaluation methods help assess the fidelity of the generated data by comparing it to real-world time series, enabling researchers to measure the accuracy and statistical properties of the synthetic data.

- **Built on TensorFlow:** TSGM is built on top of the `TensorFlow <https://www.tensorflow.org/>`_ deep learning framework. TensorFlow offers efficient computation and enables seamless integration with other TensorFlow-based models and libraries, allowing users to leverage its extensive ecosystem for further customization and experimentation.

Next, we explain the main concepts by considering several examples.

Generators
=============================
A central concept of TSGM is `Generator`. The generator can be trained on historical data using method `.fit` in a standard for Keras models way.

The training of data-driven simulators can be done via likelihood optimization, adversarial training procedures, or variational methods. Some of the implemented data-driven simulators include:

- `tss.models.cgan.GAN` - standard GAN model adapted for time-series simulation,\\
- `tss.models.cgan.ConditionalGAN` - conditional GAN model for labeled and temporally labeled time-series simulation,\\
- `tss.models.cvae.BetaVAE` - beta-VAE model adapted for time-series simulation,\\
- `tss.models.cvae.cBetaVAE` - conditional beta-VAE model for labeled and temporally labeled time-series simulation.


Combined generators
--------------------------
The combined simulators generate data based on an underlying model, but allow to inference parameters using, for example, approximate Bayesian computation.

Example, of the use of a combined simulator is provided in `https://github.com/AlexanderVNikitin/tsgm/blob/main/tests/test_abc.py`.


Dataset
=============================
In TSGM, time series datasets are often stored in one of two ways: wrapped in a `tss.dataset.Dataset` object, or as a Tensor with shape `n_samples x n_timestamps x n_features`.

Class `tsgm.dataset.DatasetProperties` implements generic placeholder for data when they are unavailable.

`tsgm.utils` has a plenty of datasets, see :ref:`datasets-label`.


Architectures Zoo
=============================
Architectures Zoo is a storage object of NN architectures that can be utilized by the framework users. It provides architectures for GANs, VAEs, and downstream task models. It also provides additional information on the implemented architectures via `zoo.summary()`.


Metrics
=============================
In `tsgm.metrics`, we implemented several metrics for evaluation of generated time series. Essentially, these metrics are subdivided into five types:

- data similarity,
- predictive consistency,
- privacy,
- downstream effectiveness,
- visual similarity.

Implementations and examples of these methods are described in `tutorials/metrics.ipynb`.


Citing
=======================
If you find the *Time Series Generator Modeling framework* useful, please consider citing our paper:

.. code-block:: latex

    >@article{
        nikitin2023tsgm,
        author = {Alexander Nikitin and Samuel Kaski},
        title = {TSGM --- A Flexible Framework for Synthetic Time Series Generative Modeling},
        year = {2022},
    }

