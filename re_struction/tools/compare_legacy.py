"""T32 (07 §6): 新旧系统 A/B 对比脚本 (compare_legacy.py)。

读取 `benchmarks/ground_truth.yaml` 与 `benchmarks/baseline_legacy.json`，
运行/复用 Pass A 缓存与 Pass B 运动学引擎，生成 `benchmarks/report.md` 并在终端输出
新旧 A/B 关键指标与 `07 §6` 验收线核对。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

RE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = RE_ROOT / "backend"
for p in (str(RE_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.eval_benchmark import evaluate_real_videos  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("compare_legacy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="新系统 wormlab vs 旧系统 baseline_legacy A/B 对比评测")
    parser.add_argument("--videos", type=str, default=None, help="逗号分隔视频 ID（默认全部）")
    parser.add_argument("--cache-dir", type=str, default=str(RE_ROOT / "data" / "cache"))
    parser.add_argument("--rerun-pass-a", action="store_true", help="重跑全部 Pass A（应用 labels 中的 head_xy 并测速 FPS）")
    parser.add_argument("--out", type=str, default=str(RE_ROOT / "benchmarks" / "report.md"))
    args = parser.parse_args(argv)

    vid_filter = {v.strip() for v in args.videos.split(",") if v.strip()} if args.videos else None
    summary = evaluate_real_videos(
        video_filter=vid_filter,
        cache_root=Path(args.cache_dir),
        rerun_pass_a=args.rerun_pass_a,
        out_path=Path(args.out),
    )

    print("\n=== A/B Comparison Summary ===")
    print(f"Crawl Pass Rate : {summary['crawl_pass_rate']:.1f}% (target >= 80.0%)")
    print(f"Swim Pass Rate  : {summary['swim_pass_rate']:.1f}% (target >= 75.0% w/o detector ceiling)")
    print(f"Overall MAE     : {summary['new_mae']:.2f} vs legacy {summary['old_mae']:.2f}")
    print(f"gt=0 Max / Mean : {summary['zero_max']} (<=1) / {summary['zero_mean']:.2f} (<=0.5)")
    print(f"Report written  : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
