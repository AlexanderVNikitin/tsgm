import typing
import logging
import numpy as np

import tsgm.types


logger = logging.getLogger('dataset')
logger.setLevel(logging.DEBUG)


class DatasetProperties:
    """
    Stores the properties of a dataset. Along with dimensions it can store properties of the covariates.
    """
    def __init__(self, N: int, D: int, T: int, variables=None):
        """
        :param N: The number of samples.
        :type N: int
        :param D: The number of dimensions.
        :type data: int
        :param T: The number of timestemps.
        :type statistics: list
        :param variables: The properties of each covariate.
        :type variables: list
        """
        self.N = N
        self.D = D
        self.T = T
        self._variables = variables
        assert variables is None or self.D == len(variables)


class Dataset(DatasetProperties):
    """
    Wrapper for time-series datasets. Additional information is stored in `metadata` field.
    """
    def __init__(self, x: tsgm.types.Tensor, y: tsgm.types.Tensor, metadata: typing.Optional[typing.Dict] = None):
        """
        :param x: The matrix of time series with dimensions NxDxT
        :type x: tsgm.types.Tensor
        :param y: The lables of a time series.
        :type y: tsgm.types.Tensor
        :param metadata: Additional info for the dataset.
        :type statistics: typing.Optional[typing.Dict]
        """
        self._x = x
        self._y = y
        assert self._y is None or self._x.shape[0] == self._y.shape[0]

        self._metadata = metadata or {}
        self._graph = self._metadata.get("graph")
        super().__init__(N=self._x.shape[0], D=self._x.shape[1], T=self._x.shape[2])

    @property
    def X(self) -> tsgm.types.Tensor:
        """
        Returns the time series tensor in format: n_samples x seq_len x feat_dim.
        """
        return self._x

    @property
    def y(self) -> tsgm.types.Tensor:
        """
        Returns labels tensor.
        """
        return self._y

    @property
    def Xy(self) -> tuple:
        """
        Returns a tuple of a time series tensor and labels tensor.
        """
        return self._x, self._y

    @property
    def Xy_concat(self) -> tsgm.types.Tensor:
        """
        Returns a concatenated time series and labels in a tensor.
        Output shape is n_sample x seq_len x feat_dim + y_dim
        """
        if self._y is None:
            return self._x
        elif len(self._y.shape) == 1:
            return np.concatenate((self._x, np.repeat(self._y[:, None, None], self._x.shape[1], axis=1)), axis=2)
        elif len(self._y.shape) == 2:
            if self._y.shape[1] == 1:
                return np.concatenate((self._x, np.repeat(self._y[:, :, None], self._x.shape[1], axis=1)), axis=2)
            else:
                return np.concatenate((self._x, self._y[:, :, None]), axis=2)
        else:
            raise ValueError("X & y are not compatible for Xy_concat operation")

    def _compatible(self, other_ds: "Dataset") -> bool:
        if self.X.shape[1:] == other_ds.X.shape[1:]:
            return self.y is None and other_ds.y is None or self.y.shape[1:] == other_ds.y.shape[1:]
        else:
            return False

    def _merge_meta(self, other_meta: dict) -> dict:
        return {**self._metadata, **other_meta}

    def __add__(self, other_ds: "Dataset") -> "Dataset":
        """
        Returns a concatenated time series and labels in a tensor.
        Output shape is n_sample x seq_len x feat_dim + y_dim
        """
        assert self._compatible(other_ds)
        return Dataset(
            np.concatenate((self.X, other_ds.X), axis=0),
            np.concatenate((self.y, other_ds.y), axis=0) if self.y is not None else None,
            self._merge_meta(other_ds._metadata)
        )

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the time series in the dataset.
        """
        return self.X.shape

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def seq_len(self) -> int:
        """
        Returns the length of sequences in the dataset.
        """
        return self.X.shape[1]

    @property
    def feat_dim(self) -> int:
        """
        Returns the size of feature dimension in the time series.
        """
        return self.X.shape[2]

    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes in the dataset.
        """
        num_classes = len(set(self.y))
        if num_classes > len(self.y) * 0.5:
            logger.warning("either the number of classes if huge or it is not a classification dataset")
        return len(set(self.y))


DatasetOrTensor = typing.Union[Dataset, tsgm.types.Tensor]
