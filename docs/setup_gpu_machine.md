# GPU 机器环境安装

这一步要在 lab 的 GPU 机器上做。IsaacLab 和 Isaac Sim 版本更新很快，
**以官方文档为准**：https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
下面只是流程和注意事项。

## 硬件要求

- NVIDIA GPU，显存 ≥ 8 GB（4096 个环境并行训练建议 ≥ 24 GB）
- Linux（Ubuntu 22.04 最稳）
- 驱动版本按 Isaac Sim 当前要求来，装之前先看官方表

先确认：
```bash
nvidia-smi          # 能看到 GPU 和驱动版本
lsb_release -a      # 系统版本
df -h ~             # Isaac Sim 本体 + 缓存要几十 GB，看一下剩余空间
```

## 安装流程（概要）

1. **装 conda / miniconda**，新建一个环境（Python 版本按 IsaacLab 文档要求）。
2. **装 Isaac Sim**。官方现在推荐 pip 安装，比以前的 Omniverse Launcher 简单。
3. **clone IsaacLab**，运行它的 `./isaaclab.sh --install`。它会装 torch、rsl_rl 等。
4. **验证**：跑一个官方示例，比如
   ```bash
   ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
   ```
   能弹出窗口（或 headless 不报错）就算过。
5. **验证 G1**：IsaacLab 自带 Unitree G1 的配置，跑一下它的 locomotion 示例：
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
   ```
   能开始训练、reward 在涨，环境就 OK 了。
6. **装本仓库**：
   ```bash
   cd chip-repro
   pip install -e ".[dev,train]"
   pytest tests/
   ```

## 常见坑

- **服务器没显示器**：所有命令加 `--headless`。要看画面用 `--video` 录视频，或者用 Isaac Sim 的 livestream。
- **torch 版本冲突**：不要自己 `pip install torch`，让 IsaacLab 的脚本装。它要和 Isaac Sim 自带的 CUDA 匹配。
- **显存不够**：先把 `num_envs` 从 4096 降到 1024 跑通，再往上加。
- **第一次启动很慢**：Isaac Sim 第一次要编译 shader、下载资源，等几分钟是正常的。

## 重定向工具（后面再装）

- GMR：https://github.com/YanjieZe/GMR
- OmniRetargeting：按论文引用 [Yang et al. 2025] 找官方仓库

这两个等数据到了再装。

## 记录

把你的实际版本填在这里，方便以后复现：

| 项目 | 版本 |
|---|---|
| 系统 | |
| GPU / 驱动 | |
| Isaac Sim | |
| IsaacLab | |
| Python | |
| torch | |
