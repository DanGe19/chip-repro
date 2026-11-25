#!/usr/bin/env bash
# 新 RunPod Pod（IsaacLab 2.3.0 镜像 + 网络盘挂 /data）的一键初始化。
# 容器盘在 Stop/Terminate 后会丢，所以装在容器里的东西每次都要重来；网络盘 /data 上的不用。
#
# 用法（在 Pod 里）：
#   bash /data/chip-repro/scripts/pod_bootstrap.sh
# 第一次（/data/chip-repro 还不存在时）：
#   git clone https://github.com/DanGe19/chip-repro.git /data/chip-repro && bash /data/chip-repro/scripts/pod_bootstrap.sh
set -euo pipefail

PY=/workspace/isaaclab/isaaclab.sh
log() { echo -e "\n\033[1;34m==> $*\033[0m"; }

log "1/5 sshd（为了 scp/sftp 直连）"
if ! command -v sshd >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null
fi
mkdir -p /run/sshd ~/.ssh && chmod 700 ~/.ssh
# RunPod 把账户里的公钥放在 $PUBLIC_KEY
if [ -n "${PUBLIC_KEY:-}" ]; then echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys; sort -u -o ~/.ssh/authorized_keys ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; fi
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
pgrep -x sshd >/dev/null || /usr/sbin/sshd
echo "sshd 运行中: $(pgrep -xc sshd) 个进程"

log "2/5 git 身份 + 拉最新代码"
git config --global user.name "Jingyang Chen"; git config --global user.email "jychen227@gmail.com"
git config --global --add safe.directory /data/chip-repro
git -C /data/chip-repro pull -q || true
git -C /data/chip-repro log --oneline -1

log "3/5 GitHub 版 smplx（IsaacLab 自带的 0.1.28 太旧，多帧会报 expression 尺寸错误）"
$PY -p -m pip install -q --no-deps --force-reinstall "git+https://github.com/vchoutas/smplx" 2>&1 | grep -v WARNING || true

log "4/5 GMR（装进 IsaacLab 的 python；源码在 /data/GMR 不会丢）"
if ! $PY -p -c "import general_motion_retargeting" 2>/dev/null; then
  $PY -p -m pip install -q -e /data/GMR 2>&1 | grep -v WARNING || true
fi
$PY -p -m pip install -q quadprog imageio imageio-ffmpeg 2>&1 | grep -v WARNING || true
$PY -p -c "import mujoco, smplx, general_motion_retargeting; print('mujoco', mujoco.__version__, 'OK')"

log "5/5 仓库自检"
cd /data/chip-repro && $PY -p -m pytest -q tests 2>&1 | tail -1

echo
echo "完成。常用路径："
echo "  代码 /data/chip-repro   数据 /data/datasets   重定向结果 /data/robot_clips   GMR /data/GMR"
echo "  CPU 配额: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo '?')   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
