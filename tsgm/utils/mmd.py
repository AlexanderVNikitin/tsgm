import typing as T
import logging
import scipy

import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.types.core import TensorLike

import tsgm


logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)


EXP_QUAD_KERNEL = tfp.math.psd_kernels.ExponentiatedQuadratic(feature_ndims=2)


def exp_quad_kernel(x: TensorLike, y: TensorLike):
    return EXP_QUAD_KERNEL.matrix(x, y)


def MMD(X: tsgm.types.Tensor, Y: tsgm.types.Tensor, kernel: T.Callable = exp_quad_kernel) -> TensorLike:
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


def kernel_median_heuristic(X1: tsgm.types.Tensor, X2: tsgm.types.Tensor) -> float:
    '''
    Median heuristic (Gretton, 2012) for RBF kernel width.
    '''
    n = X1.shape[0]
    m = X2.shape[0]

    if n * m >= 10 ** 8:
        logger.warning("n * m >= 10^8, consider subsampling for kernel median heuristic")

    X1_squared = tf.transpose(tf.tile((X1 * X1).ravel()[None, :], (m, 1)))
    X2_squared = tf.tile((X2 * X2).ravel()[None, :], (n, 1))

    distances = X1_squared + X2_squared - 2 * tf.tensordot(X1, tf.transpose(X2), axes=1)
    assert np.min(distances) >= 0

    non_zero_distances = list(filter(lambda x: x != 0, tf.reshape(distances, [-1])))
    if non_zero_distances:
        median_distance = np.median(non_zero_distances)
    else:
        median_distance = 0

    return tf.sqrt(median_distance / 2)  # 2 * sigma**2


def mmd_diff_var(Kyy: tsgm.types.Tensor, Kzz: tsgm.types.Tensor, Kxy: tsgm.types.Tensor, Kxz: tsgm.types.Tensor) -> float:
    '''
    Computes the variance of the difference statistic MMD_{XY} - MMD_{XZ}
    See http://arxiv.org/pdf/1511.04581.pdf Appendix A for more details.
    '''
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    Kyy_nd = Kyy - tf.linalg.diag(tf.linalg.diag_part(Kyy))  # Kyy - diag[Kyy]
    Kzz_nd = Kzz - tf.linalg.diag(tf.linalg.diag_part(Kzz))  # Kzz - diag[Kzz]

    # Approximations from Eq. 31
    u_yy = tf.math.reduce_sum(Kyy_nd) / (n * (n - 1))
    u_zz = tf.math.reduce_sum(Kzz_nd) / (r * (r - 1))
    u_xy = tf.math.reduce_sum(Kxy) / (m * n)
    u_xz = tf.math.reduce_sum(Kxz) / (m * r)

    Kyy_nd_T = tf.transpose(Kyy_nd)
    Kxy_T = tf.transpose(Kxy)
    Kzz_nd_T = tf.transpose(Kzz_nd)
    Kxz_T = tf.transpose(Kxz)

    # zeta_1 computation, Eq. 30 & 31 in the paper
    term1 = (1 / (n * (n - 1) ** 2)) * tf.math.reduce_sum(Kyy_nd_T @ Kyy_nd) - u_yy ** 2
    term2 = (1 / (n ** 2 * m)) * tf.math.reduce_sum(Kxy_T @ Kxy) - u_xy ** 2
    term3 = (1 / (m ** 2 * n)) * tf.math.reduce_sum(Kxy @ Kxy_T) - u_xy ** 2
    term4 = (1 / (r * (r - 1) ** 2)) * tf.math.reduce_sum(Kzz_nd_T @ Kzz_nd) - u_zz ** 2
    term5 = (1 / (r * m ** 2)) * tf.math.reduce_sum(Kxz @ Kxz_T) - u_xz ** 2
    term6 = (1 / (m * r ** 2)) * tf.math.reduce_sum(Kxz_T @ Kxz) - u_xz ** 2

    term7 = (1 / (m * n * (n - 1))) * tf.math.reduce_sum(Kyy_nd @ Kxy_T) - u_yy * u_xy
    term8 = (1 / (n * m * r)) * tf.math.reduce_sum(Kxy_T @ Kxz) - u_xz * u_xy
    term9 = (1 / (m * r * (r - 1))) * tf.math.reduce_sum(Kzz_nd @ Kxz_T) - u_zz * u_xz

    zeta1 = (term1 + term2 + term3 + term4 + term5 + term6 - 2 * (term7 + term8 + term9))
    zeta2 = (1 / (m * (m - 1))) * tf.math.reduce_sum((Kyy_nd - Kzz_nd - Kxy_T - Kxy + Kxz + Kxz_T) ** 2) - \
        (u_yy - 2 * u_xy - (u_zz - 2 * u_xz)) ** 2

    var_z1 = (4 * (m - 2) / (m * (m - 1))) * zeta1  # Eq (13)
    var_z2 = (2 / (m * (m - 1))) * zeta2  # Eq (13)

    return var_z1 + var_z2


def mmd_3_test(
        X: tsgm.types.Tensor, Y: tsgm.types.Tensor,
        Z: tsgm.types.Tensor, kernel: T.Callable) -> T.Tuple[float, float, float, float]:
    '''
    Relative MMD test --- returns a test statistic for whether Y is closer to X or than Z.
    See http://arxiv.org/pdf/1511.04581.pdf
    '''

    Kxx = kernel(X, X)
    Kyy = kernel(Y, Y)
    Kzz = kernel(Z, Z)
    Kxy = kernel(X, Y)
    Kxz = kernel(X, Z)

    Kxx_nd = Kxx - tf.linalg.diag(tf.linalg.diag_part(Kxx))
    Kyy_nd = Kyy - tf.linalg.diag(tf.linalg.diag_part(Kyy))
    Kzz_nd = Kzz - tf.linalg.diag(tf.linalg.diag_part(Kzz))

    m = Kxy.shape[0]
    n = Kyy.shape[0]
    r = Kzz.shape[0]

    u_xx = tf.math.reduce_sum(Kxx_nd) * (1 / (m * (m - 1)))
    u_yy = tf.math.reduce_sum(Kyy_nd) * (1 / (n * (n - 1)))
    u_zz = tf.math.reduce_sum(Kzz_nd) * (1 / (r * (r - 1)))
    u_xy = tf.math.reduce_sum(Kxy) / (m * n)
    u_xz = tf.math.reduce_sum(Kxz) / (m * r)

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)  # test stat
    diff_var = mmd_diff_var(Kyy, Kzz, Kxy, Kxz)
    sqrt_diff_var = math.sqrt(diff_var)

    pvalue = scipy.stats.norm.cdf(-t / sqrt_diff_var)
    tstat = t / sqrt_diff_var

    mmd_xy = u_xx + u_yy - 2 * u_xy
    mmd_xz = u_xx + u_zz - 2 * u_xz
    return pvalue, tstat, mmd_xy, mmd_xz
