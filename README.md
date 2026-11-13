# CHIP 复现：Context-conditioned Humanoid Interaction Prior

复现论文 *Unifying Physics-Based Humanoid Interaction with a Context-Conditioned Interaction Prior*
（Li, Chen, Wong — SIGGRAPH Asia 2026, arXiv:2609.06591）。

目标：在 IsaacLab 中用 Unitree G1 跑通论文的三阶段流水线，
复现表 1 的三个任务（场景导航、物体搬运、组合任务），
并为后续在技能空间上接采样式规划器（MPPI / CEM）打基础。

## 三阶段流水线

```
异构动作数据 (TRUMANS + OMOMO + 走路 + 搬箱子)
        │  重定向 SMPL-X → G1
        ▼
Stage I   动作模仿专家 (PPO, 每类动作一个专家)
        │  用专家跑 rollout 当教师
        ▼
Stage II  VQ-VAE 离散技能 token (K=1024)
          + Transformer 上下文先验 p(token | 本体历史, 场景/物体/目标)
        │  冻结先验、解码器、codebook
        ▼
Stage III PPO 后训练 + CFG 引导的 KL 正则 → 下游任务策略
```

## 目录

| 目录 | 内容 |
|---|---|
| `docs/` | 名词解释、复现清单（附录超参数）、GPU 机器环境安装 |
| `configs/` | 每个阶段的超参数 yaml，来自论文附录表 5–9 |
| `chip/data/` | TRUMANS / OMOMO 读取、筛选、统一格式 |
| `chip/retarget/` | GMR（走路）/ OmniRetargeting（接触动作）封装 |
| `chip/envs/` | IsaacLab 环境：G1、场景高度图、BPS 物体编码 |
| `chip/stage1_imitation/` | 跟踪奖励、专家训练 |
| `chip/stage2_prior/` | VQ-VAE、模态 tokenizer、Transformer 先验 |
| `chip/stage3_posttrain/` | PPO + CFG-KL、三个下游任务 |
| `chip/eval/` | 成功率、距离、碰撞分数、摔倒率 |
| `scripts/` | 各阶段入口脚本 |
| `tests/` | 不需要 GPU 的单元测试 |

## 进度

- [x] 仓库骨架、复现清单
- [ ] GPU 机器装好 IsaacLab，G1 能跑
- [ ] 申请并下载 TRUMANS、OMOMO
- [ ] 数据预处理 + 重定向
- [ ] Stage I：走路专家 → 场景专家 → 物体专家
- [ ] Stage II：VQ-VAE → 先验
- [ ] Stage III：导航 → 搬运 → 组合
- [ ] 评测对照表 1

## 快速开始

```bash
# 开发环境（不含 IsaacLab，只装纯 Python 依赖，可在任意机器跑测试）
pip install -e ".[dev]"
pytest tests/

# IsaacLab 环境见 docs/setup_gpu_machine.md
```

## 参考

- 论文 PDF：`docs/paper/`（自己放，不提交到 git）
- 项目主页：https://jiann-li.github.io/chip-project/
- TRUMANS：https://github.com/jnnan/trumans_utils
- OMOMO：https://lijiaman.github.io/projects/omomo
- IsaacLab：https://isaac-sim.github.io/IsaacLab/
