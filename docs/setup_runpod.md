# 在 RunPod 上租 GPU 跑 IsaacLab

适用于没有实验室 GPU、自己租卡的情况。目标：用 NVIDIA 官方的 IsaacLab 容器镜像，
在一张 RTX 4090 上跑通 G1 示例，然后开始我们的流水线。

预算参考（2026-11 行情）：4090 竞价约 $0.34/h，稳定约 $0.69/h。整个复现约 250–350 GPU 小时。

## 总体策略

| 阶段 | 用什么 | 为什么 |
|---|---|---|
| 第一次搭环境（约 2–3 h） | Secure Cloud，按需（不可中断） | 装到一半被踢下线最浪费时间 |
| 之后的训练 | 竞价（interruptible） | 便宜一半；代码已稳定，靠 checkpoint 续跑 |

三条保命原则，缺一不可：
1. **代码只放 GitHub**。Pod 上永远是 `git clone` 下来的副本，改了就 push。
2. **数据可重下**。TRUMANS / OMOMO 的下载脚本放在仓库里，丢了 30 分钟能重下。
3. **checkpoint 定期备份出去**。训练脚本每 N 分钟存一次，另有一个后台任务把它同步到 Pod 外面（见第 6 步）。

做到这三条，竞价实例被中断只损失几分钟进度。

---

## 第 1 步：注册两个账号

### RunPod
1. https://www.runpod.io 注册，充值。先充 $25 够搭环境和前几次实验。
2. Settings → SSH Public Keys，把你 Mac 的公钥贴进去（`cat ~/.ssh/id_ed25519.pub`，没有就 `ssh-keygen -t ed25519` 生成）。

### NVIDIA NGC（拉 IsaacLab 镜像要用）
1. https://ngc.nvidia.com 注册（可以用 NVIDIA 开发者账号）。
2. 右上角头像 → Setup → Generate API Key，复制保存。**这个 key 只显示一次。**
3. 回到 RunPod：Settings → Container Registry Auth → Add：
   - Name：`ngc`
   - Username：`$oauthtoken`（就是这个字面量，带美元符号）
   - Password：刚才的 NGC API key

---

## 第 2 步：在 RunPod 建一个模板（Template）

Pods → Templates → New Template：

| 字段 | 填什么 |
|---|---|
| Template Name | `isaaclab-chip` |
| Container Image | `nvcr.io/nvidia/isaac-lab:2.3.0` |
| Container Registry Credentials | 选 `ngc` |
| Container Disk | 60 GB（镜像本身约 20 GB，shader 缓存要空间） |
| Volume Disk | 150 GB |
| Volume Mount Path | `/workspace` |
| Expose TCP Ports | `22` |
| Docker Command | `bash -c "sleep infinity"` |
| Environment Variables | `ACCEPT_EULA=Y`，`PRIVACY_CONSENT=Y` |

说明：
- 版本：2.3.0 是经过多人验证的稳定版，自带 G1 配置。3.0.0 刚发布，API 有变动，先不用。
- `sleep infinity` 让容器启动后挂着不退出，我们 SSH 进去手动操作。
- 两个环境变量是接受 Isaac Sim 的许可，不设会在启动时卡住。

---

## 第 3 步：启动第一个 Pod（Secure Cloud，按需）

Pods → Deploy：
1. 选 **Secure Cloud**（不是 Community）。
2. GPU 选 **RTX 4090**（24 GB）。
3. Template 选 `isaaclab-chip`。
4. Pricing 选 **On-Demand**（第一次不要选 Spot）。
5. Deploy。等状态变成 Running（第一次拉镜像要 5–10 分钟）。

连接：Pod 卡片 → Connect → 复制 SSH 命令，形如
```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

---

## 第 4 步：进去之后先验证

```bash
nvidia-smi                        # 能看到 4090
cd /workspace/isaaclab            # 镜像里 IsaacLab 在这里
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```
最后一条不报错、正常退出，说明 Isaac Sim 能起来。第一次会编译 shader，等 2–5 分钟是正常的。

再跑 G1 示例，确认物理和 RL 都正常：
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 1024 --max_iterations 50
```
能看到 reward 在打印、几分钟跑完，环境就 OK 了。

---

## 第 5 步：装我们的仓库

```bash
cd /workspace
git clone https://github.com/DanGe19/chip-repro.git
cd chip-repro
# 用 IsaacLab 自带的 python，不要另建 conda 环境
/workspace/isaaclab/isaaclab.sh -p -m pip install -e ".[dev,train]"
/workspace/isaaclab/isaaclab.sh -p -m pytest tests/ -q
```
`29 passed` 就对了。

把这些常用别名加进 `~/.bashrc`，以后少敲字：
```bash
cat >> ~/.bashrc <<'EOF'
alias ilab='/workspace/isaaclab/isaaclab.sh -p'
export CHIP_ROOT=/workspace/chip-repro
export CHIP_DATA=/workspace/data
EOF
source ~/.bashrc
```

---

## 第 6 步：数据和 checkpoint 的持久化

### 数据下载（放在 `/workspace/data`，在 Volume Disk 上）
```bash
mkdir -p /workspace/data && cd /workspace/data
pip install gdown
# TRUMANS：从 Drive 文件夹下载（需要先在浏览器里用申请时的 Google 账号打开过一次）
gdown --folder "https://drive.google.com/drive/folders/1CPw_cQcQAP_wbMuQ89rREQmMtEMd-2Hz" -O trumans
# OMOMO：链接见官方仓库 README
```
只下需要的：TRUMANS 的 `Scene_mesh/`、`Object_all/`、根目录的 `.npy`；`video_render/`、`Recordings_blend/` 不要。

### checkpoint 备份
训练脚本会把 checkpoint 写到 `/workspace/runs/`。再开一个后台同步，每 10 分钟推一次到外部存储。
最省事的外部存储是 **Backblaze B2**（$6/TB/月，10 GB 以下免费）或 Cloudflare R2：
```bash
pip install rclone  # 或 apt install rclone
rclone config       # 配一个 remote，名字叫 backup
nohup bash -c 'while true; do rclone sync /workspace/runs backup:chip-runs --exclude "*.tmp"; sleep 600; done' > /workspace/sync.log 2>&1 &
```
被中断后在新 Pod 上 `rclone sync backup:chip-runs /workspace/runs` 拉回来，接着训。

---

## 第 7 步：切到竞价实例

环境跑通、第 6 步做好之后：
1. 停掉按需 Pod（Stop，不要 Terminate，Volume Disk 还在）。
2. Deploy 一个新 Pod，同样的 Template，Pricing 选 **Spot**。可以选 Community Cloud 更便宜。
3. 进去 `git pull`、`rclone sync` 拉回 checkpoint、继续。

被中断时 Pod 会变成 Stopped。等几分钟点 Start 试试，那台机器有空闲 GPU 就能恢复，恢复不了就按第 2 条重新部署一个。

---

## 省钱习惯

- 不用的时候 **Stop**。按秒计费，停了只收磁盘钱（Volume 约 $0.10/GB/月）。
- 调代码、写文档在 Mac 上做，只在跑的时候开 Pod。
- 训练命令用 `nohup ... &` 或 `tmux`，SSH 断了训练不断。
- 开始长训练前先用 `--num_envs 256 --max_iterations 20` 跑一遍，确认不报错再放全量。

## 常见坑

| 现象 | 原因 / 解法 |
|---|---|
| 启动卡在 EULA | 没设 `ACCEPT_EULA=Y` |
| 拉镜像 `unauthorized` | NGC 凭据没配对，用户名必须是 `$oauthtoken` |
| `CUDA out of memory` | `--num_envs` 减半 |
| 第一次运行很慢 | 编译 shader，等 |
| 训练完有残留进程 | `pkill -9 -f python` 再启动 |
| Pod 恢复后数据没了 | 数据放在了 Container Disk 而不是 `/workspace` |
| GMR 报 `Sizes of tensors must match ... Expected size N but got size 1` | IsaacLab 自带的 `smplx` 是 PyPI 旧版 0.1.28，多帧姿态时不会把默认 expression 扩展到每帧。换成 GitHub 版：`isaaclab.sh -p -m pip install --no-deps --force-reinstall git+https://github.com/vchoutas/smplx` |
| GMR 报 `einsum(): subscript l has size 26 ... previously seen size 20` | 模型要 16 个 betas，数据只有 10 个（TRUMANS）。`run_gmr.py` 现在默认 `--num-betas 16` 补零 |
| `/data/envs/gmr/bin/python` 里 `No module named numpy` | 那个 venv 是空壳。GMR 实际装在 IsaacLab 的 python 里，`run_gmr.py --python /workspace/isaaclab/_isaac_sim/python.sh` |

## GMR 在 Pod 上的实际布局（2026-11-21）

- GMR 源码：`/data/GMR`（`pip install -e .` 装进了 IsaacLab 的 python，不是独立 venv）
- SMPL-X 模型：`/data/GMR/assets/body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz`
- 重定向命令：

```bash
cd /data/chip-repro && /workspace/isaaclab/isaaclab.sh -p -m chip.retarget.run_gmr \
  --clips /data/clips --out /data/robot_clips --work /data/gmr_work \
  --gmr-root /data/GMR --python /workspace/isaaclab/_isaac_sim/python.sh --num-cpus 7
```

注意：`nproc` 显示 96 核是宿主机的，`cat /sys/fs/cgroup/cpu.max` 才是 Pod 的真实配额（A40 档约 7.65 核）。进程数超过配额只会更慢、更吃内存。后台任务用 `setsid nohup ... < /dev/null &`，SSH 掉线也不受影响。Isaac Sim 的 RTX 渲染（`--enable_cameras`）在 RunPod 容器里会段错误：容器没有 Vulkan/图形驱动能力（`NVIDIA_DRIVER_CAPABILITIES` 为空），物理仿真和训练不受影响。
