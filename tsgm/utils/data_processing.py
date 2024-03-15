import numpy as np
from tensorflow.python.types.core import TensorLike
import typing as T


EPS = 1e-18


class TSGlobalScaler():
    """
    Scales time series data globally.

    Attributes:
    -----------
    min : float
        Minimum value encountered in the data.
    max : float
        Maximum value encountered in the data.
    """
    def fit(self, X: TensorLike) -> "TSGlobalScaler":
        """
        Fits the scaler to the data.

        :parameter X: Input data.
        :type X: TensorLike

        :returns: The fitted scaler object.
        :rtype: TSGlobalScaler
        """
        self.min = np.min(X)
        self.max = np.max(X)
        return self

    def transform(self, X: TensorLike) -> TensorLike:
        """
        Transforms the data.

        :parameter X: Input data.
        :type X: TensorLike

        :returns: Scaled X.
        :rtype: TensorLike
        """
        return (X - self.min) / (self.max - self.min + EPS)

    def inverse_transform(self, X: TensorLike) -> TensorLike:
        """
        Inverse-transforms the data.

        :parameter X: Input data.
        :type X: TensorLike

        :returns: Original data.
        :rtype: TensorLike
        """
        X *= (self.max - self.min + EPS)
        X += self.min
        return X

    def fit_transform(self, X: TensorLike) -> TensorLike:
        """
        Fits the scaler to the data and transforms it.

        :parameter X: Input data
        :type X: TensorLike

        :returns: Scaled input data X
        :rtype: TensorLike
        """
        self.fit(X)
        scaled_X = self.transform(X)
        return scaled_X


class TSFeatureWiseScaler():
    """
    Scales time series data feature-wise.

    Parameters:
    -----------
    feature_range : tuple(float, float), optional
        Tuple representing the minimum and maximum feature values (default is (0, 1)).

    Attributes:
    -----------
    _min_v : float
        Minimum feature value.
    _max_v : float
        Maximum feature value.
    """
    def __init__(self, feature_range: T.Tuple[float, float] = (0, 1)) -> None:
        """
        Initializes a new instance of the TSFeatureWiseScaler class.

        :parameter feature_range: Tuple representing the minimum and maximum feature values, defaults to (0, 1)
        :type tuple(float, float), optional:
        """
        assert len(feature_range) == 2

        self._min_v, self._max_v = feature_range

    # X: N x T x D
    def fit(self, X: TensorLike) -> "TSFeatureWiseScaler":
        """
        Fits the scaler to the data.

        :parameter X: Input data.
        :type X: TensorLike

        :returns: The fitted scaler object.
        :rtype: TSGlobalScaler
        """
        D = X.shape[2]
        self.mins = np.zeros(D)
        self.maxs = np.zeros(D)

        for i in range(D):
            self.mins[i] = np.min(X[:, :, i])
            self.maxs[i] = np.max(X[:, :, i])

        return self

    def transform(self, X: TensorLike) -> TensorLike:
        """
        Transforms the data.

        :parameter X: Input data.
        :type X: TensorLike

        :returns: Scaled X.
        :rtype: TensorLike
        """
        return ((X - self.mins) / (self.maxs - self.mins + EPS)) * (self._max_v - self._min_v) + self._min_v

    def inverse_transform(self, X: TensorLike) -> TensorLike:
        """
        Inverse-transforms the data.

        :parameter X: Input data.
        :type X: TensorLike

        :returns: Original data.
        :rtype: TensorLike
        """
        X -= self._min_v
        X /= self._max_v - self._min_v
        X *= (self.maxs - self.mins + EPS)
        X += self.mins
        return X

    def fit_transform(self, X: TensorLike) -> TensorLike:
        """
        Fits the scaler to the data and transforms it.

        :parameter X: Input data
        :type X: TensorLike

        :returns: Scaled input data X
        :rtype: TensorLike
        """
        self.fit(X)
        return self.transform(X)
