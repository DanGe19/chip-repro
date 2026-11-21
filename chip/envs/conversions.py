"""GMR / 数据集 ↔ IsaacLab 之间的格式转换。纯 numpy，无 IsaacLab 依赖。

两个必须注意的差异：
1. 四元数顺序：GMR 和 scipy 用 (x, y, z, w)，IsaacLab 用 (w, x, y, z)。
2. G1 关节顺序：GMR 按 xml 里的顺序（左腿 6、右腿 6、腰 3、左臂 7、右臂 7），
   IsaacLab 从 USD 读出来的顺序通常是按树的广度优先交错的（left_hip_pitch, right_hip_pitch, waist_yaw, ...）。
   不要假设任何顺序，运行时拿 env 的 joint_names 按名字建映射。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 四元数
# ---------------------------------------------------------------------------

def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """(…, 4) xyzw → 绕 Z 的偏航角 (…,)。用于把观测对齐到机器人朝向。"""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_z(yaw: np.ndarray) -> np.ndarray:
    """(…,) → (…, 3, 3) 绕 Z 轴旋转矩阵。"""
    c, s = np.cos(yaw), np.sin(yaw)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.stack([
        np.stack([c, -s, zero], -1),
        np.stack([s, c, zero], -1),
        np.stack([zero, zero, one], -1),
    ], -2)


# ---------------------------------------------------------------------------
# 关节顺序
# ---------------------------------------------------------------------------

class JointReorder:
    """在两套关节命名顺序之间重排 (…, D) 数组。

    用法：
        r = JointReorder(src_names=G1_DOF_NAMES, dst_names=env.joint_names)
        dof_isaac = r(dof_gmr)         # src → dst
        dof_gmr   = r.inverse(dof_isaac)
    """

    def __init__(self, src_names, dst_names, fill_missing: float | None = None):
        """fill_missing=None 时两边名字必须完全一致；给一个数值时，dst 里 src 没有的关节
        （比如 IsaacLab G1 资产上多出来的 14 个 Dex3 手指关节）用该值填充。src 多出来的关节始终报错。"""
        src_names, dst_names = list(src_names), list(dst_names)
        missing = [n for n in dst_names if n not in src_names]
        extra = [n for n in src_names if n not in dst_names]
        if extra or (missing and fill_missing is None):
            raise ValueError(f"关节名对不上。dst 里 src 没有的: {missing}；src 里 dst 没有的: {extra}")
        self.src_names, self.dst_names = src_names, dst_names
        self.missing_names = missing
        self.fill_missing = fill_missing
        # dst[i] = src[fwd[i]]；缺的关节 fwd[i] = -1
        self._fwd = np.array([src_names.index(n) if n in src_names else -1 for n in dst_names])
        self._inv = np.array([dst_names.index(n) for n in src_names])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if not self.missing_names:
            return x[..., self._fwd]
        out = np.full(x.shape[:-1] + (len(self.dst_names),), self.fill_missing, dtype=x.dtype)
        has = self._fwd >= 0
        out[..., has] = x[..., self._fwd[has]]
        return out

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[..., self._inv]

    @property
    def forward_index(self) -> np.ndarray:
        return self._fwd.copy()
