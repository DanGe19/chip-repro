"""把原始数据集转成统一的 MotionClip .npz，并生成索引表。

用法：
    python -m chip.data.build_index --omomo /path/to/omomo/data --trumans /path/to/Data_release \
        --out data/clips

输出：
    data/clips/omomo/<seq_name>.npz
    data/clips/trumans/<seg_name>.npz
    data/clips/index.csv       每条一行：clip_id, source, num_frames, duration_s, has_scene, objects, ...
    data/clips/index_stats.txt 按来源汇总的条数和时长，直接对照论文表 4
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

from .schema import MotionClip

log = logging.getLogger(__name__)


def write_clips(clips, out_dir: Path, rows: list[dict]) -> None:
    for clip in clips:
        rel = clip.clip_id.replace("/", "/", 1)       # "omomo/xxx" → 子目录 omomo/
        path = out_dir / f"{rel}.npz"
        clip.save(path)
        row = clip.summary()
        row["path"] = str(path.relative_to(out_dir))
        row["objects"] = "|".join(row["objects"])
        rows.append(row)


def write_index(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        log.warning("没有任何片段被写入")
        return
    keys = list(rows[0].keys())
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    stats = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
    for r in rows:
        stats[r["source"]]["clips"] += 1
        stats[r["source"]]["seconds"] += float(r["duration_s"])
    lines = ["来源        片段数     时长(h)", "-" * 34]
    for src, s in sorted(stats.items()):
        lines.append(f"{src:<10} {s['clips']:>7} {s['seconds'] / 3600:>10.2f}")
    total_c = sum(s["clips"] for s in stats.values())
    total_h = sum(s["seconds"] for s in stats.values()) / 3600
    lines += ["-" * 34, f"{'合计':<10} {total_c:>7} {total_h:>10.2f}",
              "", "论文表 4 参考：TRUMANS idle 1486 条 / 8.71 h；OMOMO 1214 条 / 2.07 h"]
    (out_dir / "index_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n" + "\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omomo", type=Path, help="OMOMO 的 data/ 目录")
    ap.add_argument("--omomo-split", default="train", choices=["train", "test"])
    ap.add_argument("--trumans", type=Path, help="TRUMANS 的 Data_release/ 目录")
    ap.add_argument("--trumans-all", action="store_true", help="不做 idle 筛选，保留全部 TRUMANS 片段")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="每个数据集最多处理几条（调试用）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def limited(it):
        if args.limit is None:
            yield from it
        else:
            for i, x in enumerate(it):
                if i >= args.limit:
                    break
                yield x

    if args.omomo:
        from .omomo import iter_omomo_clips
        write_clips(limited(iter_omomo_clips(args.omomo, split=args.omomo_split)), args.out, rows)

    if args.trumans:
        from .trumans import is_idle_segment, iter_trumans_clips
        flt = None if args.trumans_all else is_idle_segment
        write_clips(limited(iter_trumans_clips(args.trumans, subset_filter=flt)), args.out, rows)

    write_index(rows, args.out)


if __name__ == "__main__":
    main()
