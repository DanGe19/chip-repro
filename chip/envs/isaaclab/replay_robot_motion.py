"""在 IsaacLab 里"运动学回放"一条 RobotMotion，并离屏录像。

不训练、不做控制：每一帧直接把根部位姿和 29 个关节角写进仿真，让 G1 像木偶一样
按数据动，用一个跟随相机录下来。目的有三：

1. 验证 GMR 输出 → IsaacLab 的关节顺序映射（``JointReorder``）是对的；
2. 验证根部坐标系/四元数约定一致（站着的人不应该躺下）；
3. 看看脚是否贴地、是否穿模，和 MuJoCo 那边的体检互相印证。

用法（Pod 上）::

    cd /data/chip-repro && /workspace/isaaclab/isaaclab.sh -p -m chip.envs.isaaclab.replay_robot_motion \
        --motion /data/robot_clips_test/omomo/sub11_trashcan_000.npz \
        --out /data/inspect/sub11_trashcan_000_isaac.mp4 --headless --enable_cameras

``--headless --enable_cameras`` 是 IsaacLab 的参数：无窗口但允许相机渲染。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic replay of a RobotMotion in IsaacLab")
parser.add_argument("--motion", required=True, type=Path)
parser.add_argument("--out", type=Path, default=None, help="输出 mp4；不给则只打印检查信息")
parser.add_argument("--max-frames", type=int, default=600)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--no-root-fix", action="store_true", help="不对 USD 资产的根部朝向做补偿（调试用）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 下面才能 import isaaclab 的其它模块 ----
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_mul  # noqa: E402

from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # noqa: E402  # isort: skip

from chip.envs.conversions import JointReorder, quat_xyzw_to_wxyz  # noqa: E402
from chip.retarget.robot_motion import RobotMotion  # noqa: E402

FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def main():
    rm = RobotMotion.load(args_cli.motion)
    T = min(rm.root_pos.shape[0], args_cli.max_frames)
    print(f"[replay] {rm.clip_id}: {rm.root_pos.shape[0]} 帧 @ {rm.fps:g} fps，回放前 {T} 帧")

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / rm.fps, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.7))

    # 场景：地面 + 灯 + G1 + 相机
    AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()).spawn.func(
        "/World/ground", sim_utils.GroundPlaneCfg()
    )
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    robot_cfg = G1_29DOF_CFG.replace(prim_path="/World/Robot")
    # 运动学回放：让根部不受物理影响更稳（我们每帧都覆盖状态，所以也可以不固定）
    robot = Articulation(robot_cfg)
    cam = None
    if args_cli.out is not None:
        cam_cfg = CameraCfg(
            prim_path="/World/Cam",
            update_period=0.0,
            height=args_cli.height,
            width=args_cli.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
        )
        cam = Camera(cam_cfg)

    sim.reset()
    print("[replay] IsaacLab 关节顺序:", robot.joint_names)
    print("[replay] 刚体名:", robot.body_names)
    # IsaacLab 的 G1 资产可能带 Dex3 手（多 14 个手指关节），数据里没有的关节填 0
    reorder = JointReorder(list(rm.dof_names), list(robot.joint_names), fill_missing=0.0)
    if reorder.missing_names:
        print(f"[replay] 资产多出 {len(reorder.missing_names)} 个关节（填 0）:", reorder.missing_names)
    device = sim.device

    # 根部朝向补偿：G1_29DOF_CFG 的 init rot 是绕 x 转 +90°，说明该 USD 资产本体是 Y 朝上。
    # 我们的数据是 Z 朝上，所以要把这个旋转乘进去。先假设需要；视频若躺下则用 --no-root-fix 对比。
    q_fix = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]], device=device)  # wxyz, +90° about x
    use_fix = not args_cli.no_root_fix

    root_pos = torch.as_tensor(rm.root_pos[:T], dtype=torch.float32, device=device)
    root_quat = torch.as_tensor(quat_xyzw_to_wxyz(rm.root_rot[:T]), dtype=torch.float32, device=device)
    dof = torch.as_tensor(reorder(rm.dof_pos[:T]), dtype=torch.float32, device=device)
    zeros_vel = torch.zeros_like(dof)

    foot_ids = [robot.body_names.index(n) for n in FOOT_BODIES if n in robot.body_names]
    foot_z = []
    frames = []
    for t in range(T):
        q = quat_mul(root_quat[t : t + 1], q_fix) if use_fix else root_quat[t : t + 1]
        pose = torch.cat([root_pos[t : t + 1], q], dim=-1)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        robot.write_joint_state_to_sim(dof[t : t + 1], zeros_vel[t : t + 1])
        robot.write_data_to_sim()
        if cam is not None:
            eye = root_pos[t].clone()
            eye[0] += 2.2
            eye[1] += 2.2
            eye[2] = 1.4
            tgt = root_pos[t].clone()
            tgt[2] = 0.7
            cam.set_world_poses_from_view(eye[None], tgt[None])
        sim.step(render=cam is not None)
        robot.update(sim.get_physics_dt())
        if foot_ids:
            foot_z.append(robot.data.body_pos_w[0, foot_ids, 2].cpu().numpy())
        if cam is not None:
            cam.update(sim.get_physics_dt())
            frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

    if foot_z:
        fz = np.stack(foot_z)
        print(f"[replay] IsaacLab 中 ankle_roll 原点最低高度: 左 {fz[:, 0].min():.3f} 右 {fz[:, 1].min():.3f} (MuJoCo 侧约 0.035 为脚底)")
    print(f"[replay] 骨盆高度均值 {rm.root_pos[:T, 2].mean():.3f} m")

    if cam is not None and frames:
        import imageio

        args_cli.out.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args_cli.out, fps=int(round(rm.fps)), codec="libx264", quality=7) as w:
            for f in frames:
                w.append_data(f)
        print("[replay] 已写出", args_cli.out)
        try:
            from PIL import Image

            idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
            sheet = Image.new("RGB", (4 * args_cli.width, 2 * args_cli.height), "white")
            for k, i in enumerate(idx):
                sheet.paste(Image.fromarray(frames[i]), ((k % 4) * args_cli.width, (k // 4) * args_cli.height))
            p = args_cli.out.with_name(args_cli.out.stem + "_sheet.png")
            sheet.save(p)
            print("[replay] 已写出", p)
        except Exception as e:  # noqa: BLE001
            print("拼图失败:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
    simulation_app.close()
