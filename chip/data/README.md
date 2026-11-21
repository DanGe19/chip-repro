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
# 全量转换
python -m chip.data.build_index \
    --omomo   /data/omomo/data \
    --trumans /data/trumans/Data_release \
    --out     data/clips

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

重点确认：
1. **OMOMO 坐标系**：代码假设 Z 朝上。如果 `sanity_check` 报骨盆高度不对，说明猜错了，要加转换。
2. **TRUMANS idle 规则**：论文没写。当前规则是"整段无物体"。看条数对不对。
3. **TRUMANS Action/ 文本格式**：没见过真实文件，`_load_action_texts` 是按最简单的情况写的。
4. **物体网格路径**：`Object_all/Object_mesh/<name>.obj` 是按 README 推的，确认一下实际文件名。

## 约定

- 坐标系统一 **Z 轴朝上**（IsaacLab 习惯）。
- 人体用 SMPL-X 参数，轴角表示，21 个身体关节。
- 帧率保留原始的 30 fps，不在这一层重采样。
- 场景网格不在这里加载，只记路径；`chip.envs` 加载时要用 `trumans.YUP_TO_ZUP_4x4` 转一下。
