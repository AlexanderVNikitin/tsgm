import numpy as np
from tensorflow.python.types.core import TensorLike
import typing as T


EPS = 1e-18


class TSGlobalScaler():
    def fit(self, X: TensorLike) -> "TSGlobalScaler":
        self.min = np.min(X)
        self.max = np.max(X)
        return self

    def transform(self, X: TensorLike) -> TensorLike:
        return (X - self.min) / (self.max - self.min + EPS)

    def inverse_transform(self, X: TensorLike) -> TensorLike:
        X *= (self.max - self.min + EPS)
        X += self.min
        return X

    def fit_transform(self, X: TensorLike) -> TensorLike:
        self.fit(X)
        scaled_X = self.transform(X)
        return scaled_X


class TSFeatureWiseScaler():
    def __init__(self, feature_range: T.Tuple[float, float] = (0, 1)) -> None:
        assert len(feature_range) == 2

        self._min_v, self._max_v = feature_range

    # X: N x T x D
    def fit(self, X: TensorLike) -> "TSFeatureWiseScaler":
        D = X.shape[2]
        self.mins = np.zeros(D)
        self.maxs = np.zeros(D)

        for i in range(D):
            self.mins[i] = np.min(X[:, :, i])
            self.maxs[i] = np.max(X[:, :, i])

        return self

    def transform(self, X: TensorLike) -> TensorLike:
        return ((X - self.mins) / (self.maxs - self.mins + EPS)) * (self._max_v - self._min_v) + self._min_v

    def inverse_transform(self, X: TensorLike) -> TensorLike:
        X -= self._min_v
        X /= self._max_v - self._min_v
        X *= (self.maxs - self.mins + EPS)
        X += self.mins
        return X

    def fit_transform(self, X: TensorLike) -> TensorLike:
        self.fit(X)
        return self.transform(X)
