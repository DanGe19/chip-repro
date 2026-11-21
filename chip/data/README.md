# chip.data：数据读取与统一格式

## 设计

两个数据集格式完全不同，这里用"适配器"把它们翻译成同一个中间格式 `MotionClip`：

```
OMOMO  pickle ──omomo.py───┐
                           ├──► MotionClip ──save()──► data/clips/*.npz + index.csv
TRUMANS 大数组 ──trumans.py─┘
```

后面所有阶段只读 `MotionClip`，不碰原始数据。

## 文件

| 文件 | 作用 |
|---|---|
| `schema.py` | `MotionClip` / `ObjectTrack` 定义，`.npz` 存取，形状校验 |
| `omomo.py` | 读 OMOMO pickle，只保留论文用的 4 个物体 |
| `trumans.py` | 读 TRUMANS 拼接数组，按 `seg_name` 切片段，Y-up → Z-up，默认只保留 idle 子集 |
| `build_index.py` | 命令行入口：两个数据集一起转，输出 `.npz` 和索引表 |

## 用法

```bash
# 全量转换（RunPod 路径）
python -m chip.data.build_index \
    --omomo   /data/datasets/omomo/data \
    --trumans /data/datasets/trumans --trumans-max-hours 8.71 \
    --out     /data/clips

# 先试 20 条看看对不对
python -m chip.data.build_index --omomo /data/omomo/data --out data/clips_debug --limit 20
```

转完看 `data/clips/index_stats.txt`，和论文表 4 对一下条数和时长：
TRUMANS idle 应约 1486 条 / 8.71 h，OMOMO 应约 1214 条 / 2.07 h。
对不上说明筛选规则不一样，调整 `trumans.is_idle_segment` 或 `omomo.PAPER_OBJECTS`。

## 拿到真实数据后要做的检查

两个加载器都有 `sanity_check(clip)`，返回可疑点列表：

```python
from chip.data.omomo import iter_omomo_clips, sanity_check
for clip in iter_omomo_clips("/data/omomo/data"):
    issues = sanity_check(clip)
    if issues:
        print(clip.clip_id, issues)
```

## 真实数据实测（2026-11-21，RunPod）

| 项目 | 结果 |
|---|---|
| TRUMANS | 3,792,068 帧，1824 个录制片段，`betas.npy` 为逐帧 (N,10)，192 个场景网格，561 个物体网格 |
| `object_list.npy` | 条目形如 `movable_chair_seat_01(0.87,1.23,1.17)`，括号是各轴缩放，网格在 `Object_all/Object_mesh/<基名>.obj` |
| `action_label.npy` | (N,10) 多热标签：lie down, squat, mouse, keyboard, laptop, phone, book, bottle, pen, vase |
| `Actions/*.txt` | 每行 `起始帧\t结束帧\t描述`，561 个片段有标注 |
| OMOMO | 4 个物体筛出 1323 条 / 2.32 h（论文 1214 / 2.07 h，接近） |

### TRUMANS "idle 子集"对不上论文（待作者回复）

论文：1486 条 / 8.71 h。我们试过的规则（条数 / 时长）：

| 规则 | 结果 |
|---|---|
| 整段无物体 | 303 / 4.81 h |
| 逐帧 action_label 全 0，连续 ≥3 s | ~4600 / 29 h |
| 上条 + 站立（骨盆−脚 > 0.7 m） | ~5100 / 27 h |
| 上条 + 移动（速度 > 0.2 m/s） | ~9700 / 10.5 h（全是碎片） |
| Actions 标注区间之间的空档 | ~1300 / 2.8 h |

**当前采用**：`mode="idle_runs"` = action_label 全 0 且站立，连续 ≥ 90 帧；可用 `max_hours=8.71` 抽样对齐论文规模。
写论文时要注明这一差异。

仍待确认：
1. **OMOMO 坐标系**：代码假设 Z 朝上。跑 `sanity_check` 看骨盆高度。

## 约定

- 坐标系统一 **Z 轴朝上**（IsaacLab 习惯）。
- 人体用 SMPL-X 参数，轴角表示，21 个身体关节。
- 帧率保留原始的 30 fps，不在这一层重采样。
- 场景网格不在这里加载，只记路径；`chip.envs` 加载时要用 `trumans.YUP_TO_ZUP_4x4` 转一下。
