"""场景和物体的几何编码：高度图、BPS、身体-物体距离。纯 numpy + trimesh，无 IsaacLab 依赖。

论文 3.4 和附录 C 的定义：
- 局部高度图：16×16，覆盖 1.6 m × 1.6 m，以机器人为中心、按机器人朝向对齐，记录场景相对机器人的高度。
- 物体 BPS：512 维。Basis Point Set（Prokudin 2019）：固定一组基点，记录每个基点到物体表面的最近距离。
- 身体-物体交互几何：96 维。论文没写具体定义。我们的假设：32 个 body × 3，每个 body 到物体表面最近点的向量。

性能设计：
- 场景高度图不能每一步都对网格做光线投射（太慢）。做法是每个场景预计算一张稠密的全局高度场
  （SceneHeightField），运行时只在局部 16×16 网格上做双线性采样。
- 物体 BPS 在物体坐标系下只算一次（物体形状不变），运行时把基点按物体位姿变换即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .conversions import rot_z

# ---------------------------------------------------------------------------
# 场景高度场
# ---------------------------------------------------------------------------

@dataclass
class SceneHeightField:
    """稠密的全局高度场。heights[i, j] 是点 (x_min + i*res, y_min + j*res) 处的最高表面高度。"""

    heights: np.ndarray        # (Nx, Ny)
    x_min: float
    y_min: float
    res: float

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, res: float = 0.05, pad: float = 2.0,
                  floor_z: float = 0.0, ray_start_z: float | None = None) -> "SceneHeightField":
        """对网格自上而下做光线投射，得到每个格点的最高表面高度。网格外的格点取 floor_z。

        这一步每个场景只跑一次，结果可以用 save/load 缓存。
        """
        lo, hi = mesh.bounds
        x_min, y_min = lo[0] - pad, lo[1] - pad
        x_max, y_max = hi[0] + pad, hi[1] + pad
        nx = int(np.ceil((x_max - x_min) / res)) + 1
        ny = int(np.ceil((y_max - y_min) / res)) + 1
        xs = x_min + np.arange(nx) * res
        ys = y_min + np.arange(ny) * res
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        z0 = (hi[2] + 1.0) if ray_start_z is None else ray_start_z
        origins = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)], -1)
        dirs = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))

        heights = np.full(gx.size, floor_z, dtype=np.float32)
        locs, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(idx_ray):
            # multiple_hits=False 给的是第一次命中，即最高表面
            heights[idx_ray] = np.maximum(heights[idx_ray], locs[:, 2].astype(np.float32))
        return cls(heights.reshape(nx, ny), float(x_min), float(y_min), float(res))

    def sample(self, xy: np.ndarray) -> np.ndarray:
        """(…, 2) 世界坐标 → (…,) 高度，双线性插值，越界夹到边缘。"""
        xy = np.asarray(xy, dtype=np.float64)
        fx = (xy[..., 0] - self.x_min) / self.res
        fy = (xy[..., 1] - self.y_min) / self.res
        nx, ny = self.heights.shape
        fx = np.clip(fx, 0, nx - 1 - 1e-6)
        fy = np.clip(fy, 0, ny - 1 - 1e-6)
        i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - i0, fy - j0
        h = self.heights
        return ((1 - tx) * (1 - ty) * h[i0, j0] + tx * (1 - ty) * h[i0 + 1, j0]
                + (1 - tx) * ty * h[i0, j0 + 1] + tx * ty * h[i0 + 1, j0 + 1])

    def local_heightmap(self, base_pos: np.ndarray, yaw: np.ndarray,
                        size: int = 16, extent_m: float = 1.6) -> np.ndarray:
        """论文的局部高度图。

        Args:
            base_pos: (N, 3) 机器人根位置（世界坐标）
            yaw:      (N,)   机器人偏航角
        Returns:
            (N, size, size)，值 = 场景高度 − 机器人根高度（"relative to the humanoid"）。
            第 0 维沿机器人前方（+x_body），第 1 维沿左方（+y_body）。
        """
        base_pos = np.asarray(base_pos, dtype=np.float64).reshape(-1, 3)
        yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
        half = extent_m / 2.0
        g = np.linspace(-half, half, size)
        gx, gy = np.meshgrid(g, g, indexing="ij")                       # (size, size)
        local = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], -1)   # (S², 3)
        R = rot_z(yaw)                                                  # (N, 3, 3)
        world = np.einsum("nij,pj->npi", R, local) + base_pos[:, None, :]   # (N, S², 3)
        h = self.sample(world[..., :2])                                 # (N, S²)
        return (h - base_pos[:, None, 2]).reshape(-1, size, size).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, heights=self.heights, x_min=self.x_min, y_min=self.y_min, res=self.res)

    @classmethod
    def load(cls, path: str | Path) -> "SceneHeightField":
        d = np.load(path)
        return cls(d["heights"], float(d["x_min"]), float(d["y_min"]), float(d["res"]))


# ---------------------------------------------------------------------------
# BPS 物体编码
# ---------------------------------------------------------------------------

def make_basis_points(n: int = 512, radius: float = 1.0, seed: int = 0) -> np.ndarray:
    """在半径 radius 的球内均匀采样 n 个固定基点。seed 固定，所有物体、所有训练共用同一组。"""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    r = rng.uniform(0, 1, size=(n, 1)) ** (1.0 / 3.0)
    return (pts * r * radius).astype(np.float32)


@dataclass
class ObjectBPS:
    """一个物体的 BPS 编码器。物体坐标系下预采样表面点，运行时把基点变换到物体系再算距离。"""

    surface_points: np.ndarray     # (M, 3) 物体坐标系
    basis: np.ndarray              # (K, 3) 世界/机器人坐标系下的基点模板

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, basis: np.ndarray, n_surface: int = 4096,
                  seed: int = 0) -> "ObjectBPS":
        pts, _ = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
        return cls(np.asarray(pts, dtype=np.float32), np.asarray(basis, dtype=np.float32))

    def encode_object_frame(self) -> np.ndarray:
        """基点就放在物体坐标系原点处时的编码。(K,) 每个基点到表面的最近距离。"""
        return _min_dist(self.basis, self.surface_points)

    def encode(self, obj_pose: np.ndarray, anchor_pos: np.ndarray, anchor_yaw: float,
               scale: float = 1.0) -> np.ndarray:
        """基点锚在机器人（anchor）处、随机器人朝向旋转，物体按 obj_pose 摆放时的编码。(K,)

        等价做法：把基点从机器人系变到世界系，再变到物体系，对预采样的表面点求最近距离。
        """
        R_a = rot_z(np.asarray(anchor_yaw))                           # (3, 3)
        basis_world = self.basis @ R_a.T + np.asarray(anchor_pos)[None, :]
        R_o, t_o = obj_pose[:3, :3], obj_pose[:3, 3]
        basis_obj = (basis_world - t_o[None, :]) @ R_o              # R_oᵀ (p − t)
        return _min_dist(basis_obj / scale, self.surface_points) * scale


def _min_dist(query: np.ndarray, points: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """(Q,3) 到 (M,3) 点集的最近距离，分块避免 Q×M 矩阵过大。"""
    out = np.empty(query.shape[0], dtype=np.float32)
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk]
        d2 = ((q[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        out[s:s + chunk] = np.sqrt(d2.min(1))
    return out


# ---------------------------------------------------------------------------
# 身体-物体交互几何
# ---------------------------------------------------------------------------

def body_to_object_vectors(body_pos: np.ndarray, mesh: trimesh.Trimesh, obj_pose: np.ndarray,
                           scale: float = 1.0) -> np.ndarray:
    """每个 body 到物体表面最近点的向量（世界系）。(B, 3)，展平后即论文的 96 维（B=32）。

    假设：论文的 "96-D body–object interaction geometry" = 32 body × 3 维向量。
    """
    body_pos = np.asarray(body_pos, dtype=np.float64).reshape(-1, 3)
    R_o, t_o = obj_pose[:3, :3], obj_pose[:3, 3]
    p_obj = (body_pos - t_o[None, :]) @ R_o / scale
    closest, _, _ = trimesh.proximity.closest_point(mesh, p_obj)
    closest_world = (np.asarray(closest) * scale) @ R_o.T + t_o[None, :]
    return (closest_world - body_pos).astype(np.float32)


def load_mesh(path: str | Path, yup_to_zup: bool = False) -> trimesh.Trimesh:
    """读 OBJ。TRUMANS 的场景/物体网格是 Y 朝上，传 yup_to_zup=True 转成 Z 朝上。"""
    m = trimesh.load(path, force="mesh")
    if yup_to_zup:
        from chip.data.trumans import YUP_TO_ZUP_4x4
        m.apply_transform(YUP_TO_ZUP_4x4.astype(np.float64))
    return m
