# chip.retarget：SMPL-X 人体动作 → Unitree G1

## 为什么要重定向

数据集里是人的动作（SMPL-X 骨架），机器人是 G1（29 个关节，比例和人不一样）。
重定向就是解一个优化问题：让 G1 的关键身体部位（手、脚、头、骨盆）尽量贴着人的对应位置，
同时关节角不超限。做这件事的工具是 **GMR**（General Motion Retargeting）。

```
MotionClip ──export_smplx──► GMR 输入 .npz ──GMR──► .pkl ──robot_motion──► RobotMotion .npz
（人体 SMPL-X）                                                              （G1 关节角 + 原上下文）
```

场景、物体、文本不参与重定向，原样搬到 RobotMotion 里。

## 文件

| 文件 | 作用 |
|---|---|
| `robot_motion.py` | `RobotMotion` 格式、G1 的 29 个关节顺序、读 GMR 输出 |
| `export_smplx.py` | MotionClip → GMR 要的 npz 布局 |
| `run_gmr.py` | 命令行：导出 → 子进程跑 GMR 批处理 → 收集结果 → 索引 |

## 装 GMR（在 GPU 机器上，和 chip-repro 分开的环境）

GMR 依赖 mujoco、mink 等，别和 IsaacLab 装在一起。

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
# 按它 README 的 conda + pip 方式装，或者 uv sync
# SMPL-X 模型文件（SMPLX_NEUTRAL/MALE/FEMALE.pkl）要从 smpl-x.is.tue.mpg.de 注册下载，
# 放到 GMR/assets/body_models/smplx/
export GMR_ROOT=$PWD
```

GMR 原生支持 OMOMO 和 AMASS 格式，G1 是它支持最好的机器人。

## 用法

```bash
# 先在 chip 环境里把数据转成 MotionClip（见 chip/data/README.md）
# 然后：
python -m chip.retarget.run_gmr \
    --clips data/clips --out data/robot_clips --work /tmp/gmr_work \
    --gmr-root $GMR_ROOT --python /path/to/gmr-env/bin/python --num-cpus 16

# 先拿 5 条试：
python -m chip.retarget.run_gmr --clips data/clips --out /tmp/rc --limit 5 --gmr-root $GMR_ROOT --python ...
```

输出 `data/robot_clips/index.csv`，里面 `in_frames` 和 `num_frames` 不一致的条目说明 GMR 丢了帧，要看一眼。

## 验收

- 用 GMR 自带的 `scripts/vis_robot_motion.py` 回放几条 `/tmp/gmr_work/gmr_out/*.pkl`，肉眼看：脚不滑、不穿地、搬东西时手在物体附近。
- 留意 `root_rot` 是 **(x, y, z, w)** 顺序。IsaacLab 用的是 (w, x, y, z)，到 envs 模块里要换。
- 关节顺序是 GMR 的 `g1_mocap_29dof.xml` 顺序（`G1_DOF_NAMES`）。IsaacLab 的 G1 顺序不同，envs 里按名字对齐。

## 和论文的差异

论文：无接触的走路用 GMR，有接触的交互动作用 **OmniRetargeting**（Yang et al. 2025）。
我们第一版全部用 GMR 先跑通。接触动作（搬箱子、坐椅子）用 GMR 可能出现手穿进物体、
接触点不准的问题。如果 Stage I 的物体专家训不起来，再接 OmniRetargeting。
到时候只需要在 `run_gmr.py` 旁边加一个 `run_omni.py`，输出同样的 `RobotMotion` 即可。
