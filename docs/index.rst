:github_url: https://github.com/

Time Series Simulator (TSGM) Official Documentation
========================================

Time Series Generative Modeling (TSGM) is a Python framework for time series data generation. It include data-driven and model-based approaches to synthetic time-series generation. It uses both generative


The package is built on top of `Tensorflow <https://www.tensorflow.org/>`_ that allows training the models on CPUs, or GPUs. 

.. code-block:: latex

    >@article{
        nikitin2022gen,
        author = {Alexander Nikitin and Samuel Kaski},
        title = {{TSGM} --- A Flexible Framework for Synthetic Time Series Generative Modeling},
        year = {2022},
    }

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Guides

   guides/installation
   guides/introduction
   guides/resources

.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Package Reference

   modules/root
   modules/optimization
   modules/utils
   modules/metrics
