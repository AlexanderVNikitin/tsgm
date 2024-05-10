.. _augmentations-label:

Augmentations
============

[Recommended] A more in-depth tutorial on augmentations for time series data `is available in our repo. <https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/augmentations.ipynb>`_

TSGM provides a wide variety of augmentation techniques beyond generative models.
For the following demonstrations, we first need to generate a toy dataset:

.. code-block:: python

	import tsgm
	X = tsgm.utils.gen_sine_dataset(100, 64, 2, max_value=20)

Jittering
------------
In tsgm, Gaussian noise augmentation can be applied as follows:

.. code-block:: python

  aug_model = tsgm.models.augmentations.GaussianNoise()
  samples = aug_model.generate(X=X, n_samples=10, variance=0.2)

The idea behind Gaussian noise augmentation is that adding a small amount of jittering to time series probably will not change it significantly but will increase the amount of such noisy samples in our dataset.

Shuffle Features
------------
Another approach to time series augmentation is simply shuffle the features. This approach is suitable only for particular multivariate time series, where they are invariant to all or particular permutations of features. For instance, it can be applied to time series where each feature represents same independent measurements from various sensors.

.. code-block:: python

  aug_model = tsgm.models.augmentations.Shuffle()
  samples = aug_model.generate(X=X, n_samples=3)

Slice and shuffle
------------
Slice and shuffle augmentation [3] cuts a time series into slices and shuffles those pieces. This augmentation can be performed for time series that exhibit some form of invariance over time. For instance, imagine a time series measured from wearable devices for several days. The good strategy for this case is to slice time series by days and, by shuffling those days, get additional samples. 

.. code-block:: python

  aug_model = tsgm.models.augmentations.SliceAndShuffle()
  samples = aug_model.generate(X=X, n_samples=10, n_segments=3)

Magnitude Warping
------------
Magnitude warping [3] changes the magnitude of each sample in a time series dataset by multiplication of the original time series with a cubic spline curve. This process scales the magnitude of time series, which can be beneficial in many cases, such as our synthetic example with sines n_knots number of knots at random magnitudes distributed as N(1, σ^2) where σ is set by a parameter sigma in function .generate.

.. code-block:: python

  aug_model = tsgm.models.augmentations.MagnitudeWarping()
  samples = aug_model.generate(X=X, n_samples=10, sigma=1)  



Window Warping
------------
In this technique [4], the selected windows in time series data are either speeding up or down. Then, the whole resulting time series is scaled back to the original size in order to keep the timesteps at the original length. See an example of such augmentation below:

.. code-block:: python

  aug_model = tsgm.models.augmentations.WindowWarping()
  samples = aug_model.generate(X=X, n_samples=10, scales=(0.5,), window_ratio=0.5) 


Dynamic Time Warping Barycentric Average (DTWBA)
------------
Dynamic Time Warping Barycentric Average (DTWBA)[2] is an augmentation method that is based on Dynamic Time Warping (DTW). DTW is a method of measuring similarity between time series. The idea is to "sync" those time series, as it is demonstrated in the following picture.

DTWBA goes like this:

  1. The algorithm picks one time series to initialize the DTW_BA result. 
  2. This time series can either be given explicitly or can be chosen randomly from the dataset
  3. For each of the N time series, the algorithm computes DTW distance and the path (the path is the mapping that minimizes the distance)
  4. After computing all DTW distances, the algorithm updates the DTWBA result by doing the average with respect to all the paths found above
  5. The algorithm repeats steps (2) and (3) until the DTWBA result converges

.. code-block:: python

  aug_model = tsgm.models.augmentations.DTWBarycentricAveraging()
  initial_timeseries = random.sample(range(X.shape[0]), 10)
  initial_timeseries = X[initial_timeseries]
  samples = aug_model.generate(X=X, n_samples=10, initial_timeseries=initial_timeseries ) 


References
------------
[1] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization for spoken word recognition”. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43-49 (1978).

[2] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693

[3] Um TT, Pfister FM, Pichler D, Endo S, Lang M, Hirche S, Fietzek U, Kulic´ D (2017) Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks. In: Proceedings of the 19th ACM international conference on multimodal interaction, pp. 216–220

[4] Rashid, K.M. and Louis, J., 2019. Window-warping: a time series data augmentation of IMU data for construction equipment activity identification. In ISARC. Proceedings of the international symposium on automation and robotics in construction (Vol. 36, pp. 651-657). IAARC Publications.
