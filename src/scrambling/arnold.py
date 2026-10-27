"""Arnold 猫映射置乱算法实现（正变换 / 逆变换 / 周期计算）。

标准 Arnold 映射矩阵 A = [[1, 1], [1, 2]]，对 N×N 方阵像素坐标做线性变换：

    x' = (x + y)  mod N
    y' = (x + 2y) mod N

迭代 k 次即得到 k 次置乱。该映射是双射（可逆），且具有周期性 T(N)：
A^T ≡ I (mod N)，即迭代满一个周期后图像完全还原。

实测要点：T(224) = 24，因此：
  * (0, 12] 是当前选取的比较范围，识别难度不保证单调变化；
  * 迭代 24 次完全还原；迭代 50 次等效 2 次、迭代 100 次等效 4 次。

只置换像素空间位置、不改变灰度值；RGB 三通道使用同一套坐标映射。
"""

from functools import lru_cache

import numpy as np

# 正变换矩阵 A 与逆变换矩阵 A^{-1}（det(A)=1，逆矩阵为整数矩阵）
FORWARD = np.array([[1, 1], [1, 2]], dtype=np.int64)
INVERSE = np.array([[2, -1], [-1, 1]], dtype=np.int64)


def _apply_coords(X, Y, matrix, iterations, N):
    """对坐标网格 (X, Y) 迭代应用矩阵 matrix 共 iterations 次（模 N）。"""
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    for _ in range(int(iterations)):
        Xn = (a * X + b * Y) % N
        Yn = (c * X + d * Y) % N
        X, Y = Xn, Yn
    return X, Y


@lru_cache(maxsize=None)
def get_permutation(N, iterations, inverse=False):
    """计算 N×N 网格经 iterations 次 Arnold 映射后的坐标置换（带缓存）。

    返回 (Xp, Yp)，含义：原坐标为 (Y, X)（Y=行、X=列）的像素被移动到新坐标
    (Yp, Xp)，即 scrambled[Yp, Xp] = original[Y, X]。
    """
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    X = X.astype(np.int64)
    Y = Y.astype(np.int64)
    M = INVERSE if inverse else FORWARD
    Xp, Yp = _apply_coords(X, Y, M, iterations, N)
    return Xp, Yp


def scramble(image, iterations, inverse=False):
    """对单张方阵图像做 Arnold 置乱 / 还原。

    参数
    ----
    image : np.ndarray, 形状 (N, N) 或 (N, N, C)
    iterations : int, 迭代次数
    inverse : bool, 为 True 时执行逆变换（用于还原）

    返回
    ----
    与输入同形状、同 dtype 的新数组。
    """
    arr = np.asarray(image)
    single_channel = arr.ndim == 2
    if single_channel:
        arr = arr[..., None]

    H, W, C = arr.shape
    if H != W:
        raise ValueError(f"Arnold 变换要求方阵图像，当前尺寸为 {H}x{W}")
    N = H

    Xp, Yp = get_permutation(N, int(iterations), inverse)

    out = np.empty_like(arr)
    for ch in range(C):
        out[Yp, Xp, ch] = arr[:, :, ch]

    return out[..., 0] if single_channel else out


def period(N, max_iter=10 ** 6):
    """计算 Arnold 映射在 N×N 网格上的周期 T(N)。"""
    I = np.eye(2, dtype=np.int64)
    M = np.eye(2, dtype=np.int64)
    for k in range(1, max_iter + 1):
        M = (M @ FORWARD) % N
        if np.array_equal(M, I):
            return k
    raise RuntimeError(f"在 {max_iter} 次迭代内未找到 N={N} 的周期")


def effective_iterations(iterations, N):
    """返回给定迭代次数在一个周期内的有效迭代次数（iterations % T(N)）。"""
    return int(iterations) % period(N)
