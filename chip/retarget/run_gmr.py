"""用 GMR 把 MotionClip 批量重定向到 Unitree G1。

流程：
    data/clips/**/*.npz (MotionClip)
        │ export_smplx.export_clips_for_gmr
        ▼
    <work>/gmr_src/<name>.npz           GMR 输入
        │ 调用 GMR 自己的 scripts/smplx_to_robot_dataset.py（子进程）
        ▼
    <work>/gmr_out/<name>.pkl           GMR 输出
        │ robot_motion_from_gmr，拼回场景/物体/文本
        ▼
    data/robot_clips/<source>/<name>.npz (RobotMotion) + index.csv

用法：
    export GMR_ROOT=/path/to/GMR       # clone 并按它的 README 装好，SMPL-X 模型放 assets/body_models
    python -m chip.retarget.run_gmr --clips data/clips --out data/robot_clips --work /tmp/gmr_work

GMR 的脚本要在它自己的环境里跑（它依赖 mujoco、mink 等）。用 --python 指定那个环境的解释器，
或者 --uv 让它用 `uv run`。

论文用 GMR 处理无接触的走路动作、用 OmniRetargeting 处理有接触的交互动作。
我们第一版先全部用 GMR，跑通流水线；OmniRetargeting 等拿到它的接口再接（见 README）。
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from chip.data.schema import MotionClip

from .export_smplx import export_clips_for_gmr
from .robot_motion import load_gmr_pickle, robot_motion_from_gmr

log = logging.getLogger(__name__)


def find_clip_files(clips_dir: Path, sources: list[str] | None) -> list[Path]:
    files = sorted(p for p in clips_dir.rglob("*.npz"))
    if sources:
        files = [p for p in files if p.parent.name in sources]
    return files


def run_gmr_batch(gmr_root: Path, src: Path, tgt: Path, robot: str, num_cpus: int,
                  python: str | None, use_uv: bool, override: bool) -> None:
    script = gmr_root / "scripts" / "smplx_to_robot_dataset.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 GMR 脚本: {script}，检查 GMR_ROOT")
    if use_uv:
        cmd = ["uv", "run", str(script)]
    else:
        cmd = [python or sys.executable, str(script)]
    cmd += ["--robot", robot, "--src_folder", str(src), "--tgt_folder", str(tgt), "--num_cpus", str(num_cpus)]
    if override:
        cmd.append("--override")
    log.info("运行 GMR: %s", " ".join(cmd))
    # 在 GMR 根目录下跑，它用相对路径找 assets/
    subprocess.run(cmd, cwd=str(gmr_root), check=True)


def collect_outputs(manifest: list[dict], gmr_out: Path, out_dir: Path) -> list[dict]:
    rows = []
    n_missing = 0
    for m in manifest:
        pkl = gmr_out / f"{m['name']}.pkl"
        if not pkl.exists():
            n_missing += 1
            log.warning("GMR 没有输出 %s（可能重定向失败）", pkl.name)
            continue
        clip = MotionClip.load(m["clip_path"])
        gmr = load_gmr_pickle(pkl)
        rm = robot_motion_from_gmr(
            gmr,
            clip_meta={
                "clip_id": clip.clip_id, "source": clip.source,
                "scene_name": clip.scene_name, "scene_mesh_path": clip.scene_mesh_path,
                "text": clip.text,
            },
            objects=clip.objects,
            bad_frames=clip.bad_frames,
            # GMR 用 1.66 + 0.1*betas[0] 估身高（utils/smpl.py），这里必须用同一个公式
            human_transl=clip.transl,
            human_height=1.66 + 0.1 * float(np.asarray(clip.betas).ravel()[0]),
        )
        rel = Path(clip.clip_id + ".npz")
        rm.save(out_dir / rel)
        rows.append({
            "clip_id": rm.clip_id, "source": rm.source, "num_frames": rm.num_frames,
            "fps": rm.fps, "in_frames": m["num_frames"],
            "has_scene": rm.scene_mesh_path is not None,
            "objects": "|".join(o.name for o in rm.objects),
            "path": str(rel),
        })
    log.info("收集完成：%d 条成功，%d 条缺失", len(rows), n_missing)
    return rows


def write_index(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True, help="MotionClip .npz 目录（build_index 的输出）")
    ap.add_argument("--out", type=Path, required=True, help="RobotMotion 输出目录")
    ap.add_argument("--work", type=Path, default=Path("/tmp/gmr_work"), help="GMR 中间文件目录")
    ap.add_argument("--gmr-root", type=Path, default=os.environ.get("GMR_ROOT"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--sources", nargs="*", help="只处理这些来源，例如 omomo trumans")
    ap.add_argument("--num-betas", type=int, default=None, help="强制 betas 长度（GMR 对 OMOMO 用 16）")
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--python", help="GMR 环境的 python 解释器")
    ap.add_argument("--uv", action="store_true", help="用 `uv run` 跑 GMR 脚本")
    ap.add_argument("--override", action="store_true", help="覆盖已有的 GMR 输出")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    ap.add_argument("--skip-gmr", action="store_true", help="不跑 GMR，只收集已有输出（调试用）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.gmr_root is None and not args.skip_gmr:
        ap.error("请设置 GMR_ROOT 环境变量或传 --gmr-root")

    files = find_clip_files(args.clips, args.sources)
    if args.limit:
        files = files[: args.limit]
    log.info("找到 %d 条 MotionClip", len(files))

    src = args.work / "gmr_src"
    tgt = args.work / "gmr_out"
    manifest = export_clips_for_gmr(files, src, num_betas=args.num_betas)

    if not args.skip_gmr:
        run_gmr_batch(Path(args.gmr_root), src, tgt, args.robot, args.num_cpus,
                      args.python, args.uv, args.override)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = collect_outputs(manifest, tgt, args.out)
    write_index(rows, args.out)


if __name__ == "__main__":
    main()
