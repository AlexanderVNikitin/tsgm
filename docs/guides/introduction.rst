Introduction
=======================

`Time Series Generative Modeling (TSGM) <https://github.com/TODO>`_ is a generative modeling framework for synthetic time series data. It builds on open-source libraries and implements various methods, such as GANs, VAEs, or ABC, for synthetic time series simulation. Moreover, *TSGM* provides many approaches for evaluating synthetic time series data. It is built on top of `TensorFlow <https://www.tensorflow.org/>`.


Citing
=======================
If you find the *Time Series Generator Modeling framework* useful, please consider citing our paper:

.. code-block:: latex

    >@article{
        nikitin2022gen,
        author = {Alexander Nikitin and Samuel Kaski},
        title = {TSGM --- A Flexible Framework for Synthetic Time Series Generative Modeling},
        year = {2022},
    }

Next, we explain the main concepts by considering several examples.

Generators
=============================
A central concept of TSGM is `Generator`. We classify generators into three types:

- `simulator-based` - if you **do not have any real data**, but you **can program the generative process**,
- `data-driven` - if you have **collected real data**, and you **do not wish to model generative process**,
- `combined` - if you have **collected real data**, and you **can program generative process** without feasible inference procedure for all the parameters.

NB, all `combined` simulators become `model-based` when data are unavailable.

The generators implement a method `.generate()`, that returns simulated time series. Next, we consider some examples of generators.

Simulator-based generators
--------------------------

TSGM allows model-based simulators to be coded. An example of a model-based simulator is provided in `tss.simulators.SineConstSimulator`.

Data-driven generators
--------------------------

Data-driven simulators often use generative approaches for synthetic data generation. The training of data-driven simulators can be done via likelihood optimization, adversarial training procedures, or variational methods. Some of the implemented data-driven simulators include:

- `tss.models.cgan.GAN` - standard GAN model adapted for time-series simulation,\\
- `tss.models.cgan.ConditionalGAN` - conditional GAN model for labeled and temporally labeled time-series simulation,\\
- `tss.models.cvae.BetaVAE` - beta-VAE model adapted for time-series simulation,\\
- `tss.models.cvae.cBetaVAE` - conditional beta-VAE model for labeled and temporally labeled time-series simulation.


Combined generators
--------------------------
The combined simulators generate data based on an underlying model, but allow to inference parameters using, for example, approximate Bayesian computation.

Example, of the use of a combined simulator is provided in `github.com/TODO`.


Dataset
=============================
In TSGM, time series datasets are often stored in one of two ways: wrapped in a `tss.dataset.Dataset` object, or as a tensor with shape `n_samples x n_timestamps x n_features`.

Class `tsgm.dataset.DatasetProperties` implements generic placeholder for data when they are unavailable.


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
