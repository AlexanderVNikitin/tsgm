import math
import numpy as np
import random
import scipy.interpolate
from dtaidistance import dtw
from typing import List, Dict, Any, Optional
from tensorflow.python.types.core import TensorLike

import logging

logger = logging.getLogger("augmentations")
logger.setLevel(logging.DEBUG)


class BaseAugmenter:
    def __init__(
        self,
        per_feature: bool,
    ):
        self.per_channel = per_feature

    def _get_seeds(self, total_num: int, n_seeds: int) -> TensorLike:
        seeds_idx = np.random.choice(range(total_num), size=n_seeds, replace=True)
        return seeds_idx

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        raise NotImplementedError


class BaseCompose:
    def __init__(
        self,
        augmentations: List[BaseAugmenter],
    ):
        if isinstance(augmentations, (BaseCompose, BaseAugmenter)):
            augmentations = [augmentations]

        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.augmentations)

    def __call__(self, *args, **data) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> BaseAugmenter:
        return self.augmentations[item]


class GaussianNoise(BaseAugmenter):
    """Apply noise to the input time series.
    Args:
        variance ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_feature (bool): if set to True, noise will be sampled for each feature independently.
            Otherwise, the noise will be sampled once for all features. Default: True
    """

    def __init__(
        self,
        mean: float = 0,
        variance: float = 1.0,
        per_feature: bool = True,
    ):
        super(GaussianNoise, self).__init__(per_feature)
        self.variance = variance
        self.mean = mean

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        seeds_idx = self._get_seeds(total_num=X.shape[0], n_seeds=n_samples)

        sigma = self.variance**0.5
        has_labels = y is not None
        if self.per_channel:
            gauss = np.random.normal(self.mean, sigma, (n_samples, X.shape[1], X.shape[2]))
        else:
            gauss = np.random.normal(self.mean, sigma, (n_samples, X.shape[1]))
            gauss = np.expand_dims(gauss, -1)
        synthetic_X = X[seeds_idx] + gauss
        if has_labels:
            synthetic_y = y[seeds_idx]
            return np.array(synthetic_X), np.array(synthetic_y)
        else:
            return np.array(synthetic_X)


class SliceAndShuffle(BaseAugmenter):
    """Slice the time series in k pieces and create a new time series by shuffling.
    Args:
        n_segments (int): the number of slices
        per_feature (bool): if set to True, each time series is sliced independently.
            Otherwise, all features are sliced in the same way. Default: True
    """

    def __init__(
        self,
        n_segments: int,
        per_feature: bool = True,
    ):
        super(SliceAndShuffle, self).__init__(per_feature)
        self.n_segments = n_segments

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        assert 0 < self.n_segments < X.shape[1]

        seeds_idx = self._get_seeds(n_samples=X.shape[0], n_seeds=n_samples)

        synthetic_data = []
        has_labels = y is not None
        if has_labels:
            new_labels = []
        for i in seeds_idx:
            sequence = X[i]
            if self.per_channel:
                raise NotImplementedError(
                    "SliceAndShuffle separately by feature is not supported yet."
                )
            else:
                # Randomly pick n_segments-1 points where to slice
                idxs = np.random.randint(0, sequence.shape[0], size=self.n_segments - 1)
                slices = []
                start_idx = 0
                for j in sorted(idxs):
                    s = sequence[start_idx:j]
                    start_idx = j
                    slices.append(s)
                slices.append(sequence[start_idx:])
                np.random.shuffle(slices)
            synthetic_data.append(sequence)
            if has_labels:
                new_labels.append(self._targets[i])
        if has_labels:
            return np.array(synthetic_data), np.array(new_labels)
        else:
            return np.array(synthetic_data)


class Shuffle(BaseAugmenter):
    """Shuffles time series features.
    Shuffling is beneficial when each feature corresponds to interchangeable sensors.
    """

    def __init__(self):
        super(Shuffle, self).__init__(per_feature=False)

    def _n_repeats(self, n: int, total_num: int) -> int:
        return math.ceil(n / total_num)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1) -> TensorLike:
        seeds_idx = self._get_seeds(X.shape[0], n_samples)
        n_features = X.shape[2]

        n_repeats = self._n_repeats(n_samples, total_num=len(X))
        shuffle_ids = [np.random.choice(np.arange(n_features), n_features, replace=False) for _ in range(n_repeats)]

        synthetic_data = []
        has_labels = y is not None
        if has_labels:
            new_labels = []
        for num, i in enumerate(seeds_idx):
            sequence = X[i]
            id_repeat = self._n_repeats(num, total_num=len(X))
            synthetic_data.append(sequence[:, shuffle_ids[id_repeat]])
            if has_labels:
                new_labels.append(y[i])
        if has_labels:
            return np.array(synthetic_data), np.array(new_labels)
        else:
            return np.array(synthetic_data)


class MagnitudeWarping(BaseAugmenter):
    """
    Magnitude warping changes the magnitude of each
    sample by convolving the data window with a smooth curve varying around one
    https://dl.acm.org/doi/pdf/10.1145/3136755.3136817
    """
    def __init__(self):
        super(MagnitudeWarping, self).__init__(per_feature=False)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, n_samples: int = 1, sigma: float = 0.2, knot: int = 4) -> TensorLike:
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        orig_steps = np.arange(n_timesteps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(n_samples, knot + 2, n_features))
        warp_steps = (np.ones(
            (n_features, 1)) * (np.linspace(0, n_timesteps - 1., num=knot + 2))).T
        result = np.zeros((n_samples, n_timesteps, n_features))
        has_labels = y is not None

        if has_labels:
            result_y = np.zeros((n_samples, 1))

        for i in range(n_samples):
            random_sample_id = random.randint(0, n_data - 1)
            warper = np.array([scipy.interpolate.CubicSpline(
                warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(n_features)]).T
            result[i] = X[random_sample_id] * warper
            if has_labels:
                result_y[i] = y[random_sample_id]
        if has_labels:
            return result, result_y
        else:
            return result


class WindowWarping(BaseAugmenter):
    """
    https://halshs.archives-ouvertes.fr/halshs-01357973/document
    """
    def __init__(self):
        super(WindowWarping, self).__init__(per_feature=False)

    def generate(self, X: TensorLike, y: Optional[TensorLike] = None, window_ratio=0.2, scales=[0.25, 1.0], n_samples=1):
        n_data = X.shape[0]
        n_timesteps = X.shape[1]
        n_features = X.shape[2]

        scales_per_sample = np.random.choice(scales, n_samples)
        warp_size = max(np.round(window_ratio * n_timesteps).astype(np.int64), 1)
        window_starts = np.random.randint(
            low=0, high=n_timesteps - warp_size,
            size=(n_samples))
        window_ends = window_starts + warp_size

        result = np.zeros((n_samples, n_timesteps, n_features))
        result_y = np.zeros((n_samples, 1))
        has_labels = y is not None
        for i in range(n_samples):
            for dim in range(n_features):
                random_sample_id = random.randint(0, n_data - 1)
                random_sample = X[random_sample_id]
                start_seg = random_sample[:window_starts[i], dim]
                warp_ts_size = max(round(warp_size * scales_per_sample[i]), 1)
                window_seg = np.interp(
                    x=np.linspace(0, warp_size - 1, num=warp_ts_size),
                    xp=np.arange(warp_size),
                    fp=random_sample[window_starts[i] : window_ends[i], dim])
                end_seg = random_sample[window_ends[i]:, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                result[i, :, dim] = np.interp(
                    np.arange(n_timesteps),
                    np.linspace(0, n_timesteps - 1., num=warped.size), warped).T
                if has_labels:
                    result_y[i] = y[random_sample_id]

        if has_labels:
            return result, result_y
        else:
            return result


class DTWBarycentricAveraging(BaseAugmenter):
    """
    DTW Barycenter Averaging (DBA) [1] method estimated through
        Expectation-Maximization algorithm [2] as in https://github.com/tslearn-team/tslearn/
    ----------
    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    .. [2] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
       for Averaging in Dynamic Time Warping Spaces.
       Pattern Recognition, 74, 340-358.
    """

    def __init__(self):
        super(DTWBarycentricAveraging, self).__init__(per_feature=False)
        self.data = None

    def generate(
        self,
        X: TensorLike,
        y: Optional[TensorLike] = None,
        n_samples: int = 1,
        sample_size:Optional[int] = None,
        output_size=None,
        seed_timeseries=None,
        max_iter=30,
        tol=1e-5,
        weights=None,
        metric_params=None,
        verbose=False,
        n_init=1,
    ) -> TensorLike:
        """
        Parameters
        ----------
        output_size : int or None (default: None)
            Size of the timeseries to generate.
            If None, the size is equal to the data provided at fit
            unless seed_timeseries is provided.
        sample_size : int or None (default: None)
            The number of timeseries to draw from the dataset before computing DTW_BA.
            If None, use the entire set.
        seed_timeseries : array or None (default: None)
            Initial timesteries to start from for the optimization process.
        max_iter : int (default: 30)
            Maximum number of iterations for Expectation-Maximization optimization.
        tol : float (default: 1e-5)
            Tolerance to use for early stopping: if the decrease in cost is lower
            than this value, the Expectation-Maximization stops.
        weights: None or array
            Weights of each timeseries. Must be the same size as len(X).
            If None, uniform weights are used.
        metric_params: dict or None (default: None)
            DTW constraint parameters to be used.
            If None, no constraint is used for DTW computations.
        verbose : boolean (default: False)
            Whether to log information about the cost at each iteration or not.
        n_init : int (default: 1)
            Number of different initializations to be tried (useful only is
            seed_timeseries is set to None, otherwise all trials will reach the
            same performance)
        Returns
        -------
        np.array of shape (n_samples, output_size, d)
            or (n_samples, original_size, d) if output_size is None
            or (n_samples, seed_timeseries_size, d) if seed_timeseries is not None
        """
        self.data = X
        _samples = []
        for _ in range(n_samples):
            _samples.append(
                self._one_dtwba(
                    sample_size=sample_size,
                    output_size=output_size,
                    seed_timeseries=seed_timeseries,
                    max_iter=max_iter,
                    tol=tol,
                    weights=weights,
                    metric_params=metric_params,
                    verbose=verbose,
                    n_init=n_init,
                )
            )
        return np.array(_samples)

    def _one_dtwba(
        self,
        sample_size: int,
        output_size=None,
        seed_timeseries=None,
        max_iter=30,
        tol=1e-5,
        weights=None,
        metric_params=None,
        verbose=False,
        n_init=1,
    ) -> TensorLike:
        best_cost = np.inf
        best_barycenter = None
        for i in range(n_init):
            ## TODO: draw sample_size timeseries at random
            _sample_from_original = self.data
            if verbose:
                logger.info(f"Initialization {i + 1}")
            curr_avg, curr_loss = self._dtwba_iteration(
                X=_sample_from_original,  # array-like, shape=(sample_size, sz, d)
                output_size=output_size,
                seed_timeseries=seed_timeseries,
                max_iter=max_iter,
                tol=tol,
                weights=weights,
                metric_params=metric_params,
                verbose=verbose,
            )
            if loss < best_cost:
                best_cost = curr_loss
                best_barycenter = curr_avg
        return best_barycenter

    def _dtwba_iteration(
        self,
        X: TensorLike,
        output_size=None,
        seed_timeseries=None,
        max_iter=30,
        tol=1e-5,
        weights=None,
        metric_params=None,
        verbose=False,
    ):
        X_ = X  # (sample_size, sz, d)
        if output_size is None:
            output_size = X_.shape[1]
        weights = self._set_weights(weights, X_.shape[0])
        if seed_timeseries is None:
            barycenter = self._init_avg(X_, output_size)
        else:
            output_size = seed_timeseries.shape[0]
            barycenter = seed_timeseries
        cost_prev, cost = np.inf, np.inf
        for it in range(max_iter):
            list_p_k, cost = self._mm_assignment(X_, barycenter, weights, metric_params)
            diag_sum_v_k, list_w_k = self._mm_valence_warping(
                list_p_k, output_size, weights
            )
            if verbose:
                logger.info(f"[DTW_BA] epoch {it + 1}, cost: {round(cost, 3)}")
            barycenter = self._mm_update_barycenter(X_, diag_sum_v_k, list_w_k)
            if abs(cost_prev - cost) < tol:
                break
            elif cost_prev < cost:
                logger.warning("DTW_BA loss is increasing. Stopping optimization.")
                break
            else:
                cost_prev = cost
        return barycenter, cost


    @staticmethod
    def _mm_assignment(X, barycenter, weights, metric_params=None):
        """Computes item assignement based on DTW alignments and return cost.
        Parameters
        ----------
        X : np.array of shape (n, sz, d)
            Time-series to be averaged
        barycenter : np.array of shape (barycenter_size, d)
            Barycenter as computed at the current step of the algorithm.
        weights: array
            Weights of each X[i]. Must be the same size as len(X).
        metric_params: dict or None (default: None)
            DTW constraint parameters to be used.
            See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
            a list of accepted parameters
            If None, no constraint is used for DTW computations.
        Returns
        -------
        list of index pairs
            Warping paths
        float
            Current alignment cost
        """
        if metric_params is None:
            metric_params = {}
        n = X.shape[0]
        cost = 0.
        list_p_k = []
        for i in range(n):
            dist_i, paths = dtw.warping_paths(barycenter, X[i], **metric_params)
            path = dtw.best_path(paths)
            cost += dist_i ** 2 * weights[i]
            list_p_k.append(path)
        cost /= weights.sum()
        return list_p_k, cost

    @staticmethod
    def _mm_valence_warping(list_p_k, barycenter_size, weights):
        """Compute Valence and Warping matrices from paths.
        Valence matrices are denoted :math:`V^{(k)}` and Warping matrices are
        :math:`W^{(k)}` in [1]_.
        This function returns the sum of :math:`V^{(k)}` diagonals (as a vector)
        and a list of :math:`W^{(k)}` matrices.
        Parameters
        ----------
        list_p_k : list of index pairs
            Warping paths
        barycenter_size : int
            Size of the barycenter to generate.
        weights: array
            Weights of each X[i]. Must be the same size as len(X).
        Returns
        -------
        np.array of shape (barycenter_size, )
            sum of weighted :math:`V^{(k)}` diagonals (as a vector)
        list of np.array of shape (barycenter_size, sz_k)
            list of weighted :math:`W^{(k)}` matrices
        References
        ----------
        .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
           for Averaging in Dynamic Time Warping Spaces.
           Pattern Recognition, 74, 340-358.
        """
        list_v_k, list_w_k = _subgradient_valence_warping(
            list_p_k=list_p_k,
            barycenter_size=barycenter_size,
            weights=weights)
        diag_sum_v_k = np.zeros(list_v_k[0].shape)
        for v_k in list_v_k:
            diag_sum_v_k += v_k
        return diag_sum_v_k, list_w_k

    @staticmethod
    def _mm_update_barycenter(X, diag_sum_v_k, list_w_k):
        """Update barycenters using the formula from Algorithm 2 in [1]_.
        Parameters
        ----------
        X : np.array of shape (n, sz, d)
            Time-series to be averaged
        diag_sum_v_k : np.array of shape (barycenter_size, )
            sum of weighted :math:`V^{(k)}` diagonals (as a vector)
        list_w_k : list of np.array of shape (barycenter_size, sz_k)
            list of weighted :math:`W^{(k)}` matrices
        Returns
        -------
        np.array of shape (barycenter_size, d)
            Updated barycenter
        References
        ----------
        .. [1] D. Schultz and B. Jain. Nonsmooth Analysis and Subgradient Methods
           for Averaging in Dynamic Time Warping Spaces.
           Pattern Recognition, 74, 340-358.
        """
        d = X.shape[2]
        barycenter_size = diag_sum_v_k.shape[0]
        sum_w_x = np.zeros((barycenter_size, d))
        for k, (w_k, x_k) in enumerate(zip(list_w_k, X)):
            sum_w_x += w_k.dot(x_k[:ts_size(x_k)])
        barycenter = np.diag(1. / diag_sum_v_k).dot(sum_w_x)
        return barycenter

    @staticmethod
    def _set_weights(w, n):
        """Return w if it is a valid weight vector of size n, and a vector of n 1s
        otherwise.
        """
        if w is None or len(w) != n:
            w = np.ones((n,))
        return w

    @staticmethod
    def _init_avg(X, barycenter_size):
        if X.shape[1] == barycenter_size:
            return np.nanmean(X, axis=0)
        else:
            X_avg = np.nanmean(X, axis=0)
            xnew = np.linspace(0, 1, barycenter_size)
            f = scipy.interpolate(np.linspace(0, 1, X_avg.shape[0]), X_avg,
                         kind="linear", axis=0)
            return f(xnew)
