Datasets
============

The package provides easy access to many time series datasets.

.. list-table:: List of Datasets
   :widths: 25 25 50
   :header-rows: 1

   * - Dataset Name
     - API
     - Description
   * - UCR Dataset
     - tsgm.utils.UCRDataManager
     - https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
   * - Mauna Loa
     - tsgm.utils.get_mauna_loa()
     - https://gml.noaa.gov/ccgg/trends/data.html
   * - EEG & Eye state
     - tsgm.utils.get_eeg()
     - https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
   * - Power consumption dataset
     - tsgm.utils.get_power_consumption()
     - https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
   * - Stock data
     - tsgm.utils.get_stock_data(ticker_name)
     - Gets historical stock data from YFinance
   * - Energy Data (UCI)
     - tsgm.utils.get_energy_data
     - https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
   * - MNIST as time series
     - tsgm.utils.get_mnist_data
     - https://en.wikipedia.org/wiki/MNIST_database
   * - Samples from GPs
     - tsgm.utils.get_gp_samples_data
     - https://en.wikipedia.org/wiki/Gaussian_process
