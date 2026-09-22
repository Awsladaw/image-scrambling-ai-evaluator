"""Arnold 变换单元自检：周期、正/逆变换往返一致、有效迭代次数。"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scrambling.arnold import effective_iterations, period, scramble  # noqa: E402

N = 224
print("T(224) =", period(N))
assert period(N) == 24

# 用一个可区分的测试图验证：置乱后应变化、逆变换后应还原
img = np.arange(N * N, dtype=np.uint8).reshape(N, N)
sc = scramble(img, 6)
assert not np.array_equal(img, sc), "置乱后不应等于原图"
restored = scramble(sc, 6, inverse=True)
assert np.array_equal(img, restored), "逆变换应精确还原"

# RGB 三通道
rgb = np.stack([img, img // 2, img % 251], axis=-1)
assert scramble(scramble(rgb, 8), 8, inverse=True).shape == rgb.shape

# 有效迭代次数
print("有效迭代：", {k: effective_iterations(k, N) for k in [5, 10, 20, 50, 100]})
assert effective_iterations(50, N) == 2
assert effective_iterations(100, N) == 4
assert effective_iterations(24, N) == 0

print("Arnold 自检全部通过 (OK)")
