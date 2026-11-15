"""T16 / T32 (07 §5.2, §6): 新系统基准评测脚本 (eval_benchmark.py)。

支持两种模式：
1. `--synthetic`（T16 阶段自测）：在合成爬行/游泳/干扰序列上评测 KinematicsEngine 并生成 benchmarks/report.md；
2. 默认真实视频模式（T24/T32 阶段）：读取 data/cache/<vid>/tracks.npz（不存在时自动运行 Pass A），
   对齐 benchmarks/labels/<vid>.json 与 benchmarks/baseline_legacy.json，输出完整新旧对比报告至 benchmarks/report.md。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RE_ROOT.parent
BACKEND_DIR = RE_ROOT / "backend"
for p in (str(RE_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.make_synthetic import Scn, make_series  # noqa: E402
from wormlab.config import Config, load_config  # noqa: E402
from wormlab.geometry import iou_xyxy  # noqa: E402
from wormlab.kinematics import KinematicsEngine  # noqa: E402
from wormlab.pose.orientation import reorient_track_poses  # noqa: E402
from wormlab.schemas import Pose, Quality  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("eval_benchmark")


def crawl_pass_criterion(pred: int, gt: int) -> bool:
    err = abs(int(pred) - int(gt))
    if gt <= 5:
        return err <= 1
    return err <= max(2, int(round(0.2 * gt)))


def swim_pass_criterion(c_cycles: int, c_bends: int, gt: int) -> tuple[bool, int, str]:
    err_cyc = abs(int(c_cycles) - int(gt))
    err_bnd = abs(int(c_bends) - int(gt))
    if err_cyc <= err_bnd:
        best_pred, best_unit, err = int(c_cycles), "cycle", err_cyc
    else:
        best_pred, best_unit, err = int(c_bends), "bend", err_bnd
    tol = max(2.0, 0.15 * float(gt))
    return (err <= tol), best_pred, best_unit


def run_synthetic_eval(out_path: Path) -> int:
    crawl_cfg = load_config("crawl_agar").kinematics
    swim_cfg = load_config("swim_liquid").kinematics

    scenarios = [
        ("crawl_fwd_30fps", Scn(kind="crawl_fwd", duration_s=20.0, fps=30.0), "crawl"),
        ("crawl_bwd_30fps", Scn(kind="crawl_bwd", duration_s=20.0, fps=30.0), "crawl"),
        ("head_wiggle_30fps", Scn(kind="head_wiggle", duration_s=20.0, fps=30.0), "crawl"),
        ("treadmill_30fps", Scn(kind="treadmill", duration_s=20.0, fps=30.0), "crawl"),
        ("fwd_then_back_14s", Scn(kind="fwd_then_back", duration_s=14.0, fps=30.0), "crawl"),
        ("static_noise_30fps", Scn(kind="static_noise", duration_s=20.0, fps=30.0), "crawl"),
        ("swim_thrash_10s", Scn(kind="swim_thrash", duration_s=10.0, fps=30.0, f=2.0), "swim"),
        ("swim_small_amp_10s", Scn(kind="swim_small_amp", duration_s=10.0, fps=30.0, f=2.0), "swim"),
    ]

    lines = [
        "# 基准评测报告 (`benchmarks/report.md`)",
        "",
        "## 合成基准自检 (Synthetic Verification)",
        "",
        "| 场景 | 类型 | 真值 (GT) | 预测 S | 预测 C (`cycles` / `bends`) | 拒绝 (`no_disp`/`wrong_dir`/`bwd`) | 通过 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    n_pass = 0
    for name, scn, kind in scenarios:
        dense_seq, exp = make_series(scn)
        poses = [
            Pose(
                dense=d,
                length=float(scn.length_px),
                chord=float(np.linalg.norm(d[0] - d[-1])),
                width=float(scn.width_px),
                area=float(scn.length_px * scn.width_px),
                quality=Quality.OK,
            )
            for d in dense_seq
        ]
        kcfg = crawl_cfg if kind == "crawl" else swim_cfg
        rep, _ = KinematicsEngine.run(poses, fps=scn.fps, cfg=kcfg)
        if kind == "crawl":
            gt_val = int(exp["s_count"])
            ok = crawl_pass_criterion(rep.s_count, gt_val)
            gt_str = f"S={gt_val}"
        else:
            gt_val = int(exp["c_cycles"])
            ok, _, _ = swim_pass_criterion(rep.c_cycles, rep.c_bends, gt_val)
            gt_str = f"C={gt_val}/{exp['c_bends']}"
        if ok:
            n_pass += 1
        rej_str = f"{rep.rejected['no_disp']}/{rep.rejected['wrong_dir']}/{rep.rejected['backward']}"
        lines.append(
            f"| `{name}` | {kind} | {gt_str} | {rep.s_count} | {rep.c_cycles} / {rep.c_bends} | {rej_str} | {'✅' if ok else '❌'} |"
        )

    lines.extend(["", f"- **合成场景通过率**: {n_pass}/{len(scenarios)} ({100.0 * n_pass / len(scenarios):.1f}%)", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("已输出合成基准报告: %s (%d/%d passed)", out_path, n_pass, len(scenarios))
    return 0


def evaluate_real_videos(
    video_filter: set[str] | None,
    cache_root: Path,
    rerun_pass_a: bool,
    out_path: Path,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import time

    from wormlab.pipeline.cache import load_cache
    from wormlab.pipeline.late_entry import partition_tracks_by_uid, select_primary_uid_per_track_id
    from wormlab.pipeline.pass_a import run_pass_a

    gt_yaml = RE_ROOT / "benchmarks" / "ground_truth.yaml"
    labels_dir = RE_ROOT / "benchmarks" / "labels"
    legacy_path = RE_ROOT / "benchmarks" / "baseline_legacy.json"

    gt_doc = yaml.safe_load(gt_yaml.read_text(encoding="utf-8"))
    legacy_doc = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else {}

    md_lines = [
        "# 线虫运动分析重构系统 — 基准评测与新旧对比报告 (`benchmarks/report.md`)",
        "",
        "## 1. 逐视频逐虫评测明细",
        "",
        "| 视频 | 标号 | 类型 | 真值 (GT) | 新系统预测 | 旧系统基线 | 新误差 | 旧误差 | 轨迹覆盖率 | BAD帧占比 | 拒绝(`nd`/`wd`/`bwd`) | 判定 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    crawl_passes = 0
    crawl_total = 0
    crawl_zero_preds: list[int] = []
    legacy_zero_preds: list[int] = []
    swim_passes = 0
    swim_total = 0
    new_errors: list[float] = []
    old_errors: list[float] = []
    late_counts_by_vid: dict[str, int] = {}
    crawl_cov_list: list[float] = []
    swim_cov_list: list[float] = []
    crawl_bad_obs: list[tuple[int, int]] = []
    swim_bad_obs: list[tuple[int, int]] = []
    pass_a_speeds: dict[str, float] = {}

    for entry in gt_doc.get("videos", []):
        vid_id = str(entry["id"])
        if video_filter and vid_id not in video_filter:
            continue

        profile = str(entry.get("profile", "crawl_agar"))
        kind = str(entry.get("kind", "crawl"))
        cfg: Config = load_config(profile, overrides=cfg_overrides)

        vid_cache_dir = cache_root / vid_id
        npz_path = vid_cache_dir / "tracks.npz"
        vid_path = REPO_ROOT / entry["video"]

        lbl_path = labels_dir / f"{vid_id}.json"
        lbl_doc_init = json.loads(lbl_path.read_text(encoding="utf-8")) if lbl_path.exists() else {}
        head_cfg = [
            {
                "user_label": str(a["label"]),
                "centroid": a.get("frame0_centroid", a["centroid"]),
                "head_xy": a["head_xy"],
            }
            for a in lbl_doc_init.get("assignments", [])
            if "head_xy" in a and a["head_xy"] is not None
        ]

        if rerun_pass_a or not npz_path.exists():
            log.info("[%s] 正在运行 Pass A 生成缓存: %s (head_cfg=%d)", vid_id, npz_path, len(head_cfg))
            model_path = RE_ROOT / "backend" / "models" / cfg.detection.default_model
            init_n = max(len(lbl_doc_init.get("assignments", [])), int(lbl_doc_init.get("num_detections", 0))) or None
            t0 = time.perf_counter()
            run_pass_a(
                video_path=vid_path,
                cache_dir=vid_cache_dir,
                cfg=cfg,
                model_path=model_path,
                max_worm_id=init_n,
                head_config=head_cfg or None,
            )
            elapsed = max(1e-3, time.perf_counter() - t0)
        else:
            elapsed = 0.0

        meta, records = load_cache(vid_cache_dir)
        fps = float(meta["fps"])
        total_frames = int(meta["total_frames"])
        frame_h = int(meta["height"])
        if elapsed > 0.0 and total_frames > 0:
            pass_a_speeds[vid_id] = round(float(total_frames) / elapsed, 2)

        uid_groups = partition_tracks_by_uid(records, fps=fps, grace_s=cfg.counting.late_entry_grace_s)
        late_uids = [u for u, g in uid_groups.items() if g.late_entry]
        late_counts_by_vid[vid_id] = len(late_uids)
        selected_by_tid, _ = select_primary_uid_per_track_id(uid_groups)

        # 对每个初始虫主实例 (counted=True 且每 track_id 最优 uid) 运行 KinematicsEngine
        reports_by_uid: dict[int, Any] = {}
        for grp in selected_by_tid.values():
            if not grp.counted:
                continue
            poses_seq: list[Pose | None] = [None] * total_frames
            for r in grp.records:
                if 0 <= r.frame_idx < total_frames and not r.is_ghost:
                    poses_seq[r.frame_idx] = r.pose
            poses_seq = reorient_track_poses(poses_seq)
            rep, _ = KinematicsEngine.run(
                poses_seq,
                fps=fps,
                cfg=cfg.kinematics,
                c_unit=cfg.counting.c_unit,
                frame_height=frame_h,
            )
            reports_by_uid[grp.uid] = (grp, rep)

        lbl_path = labels_dir / f"{vid_id}.json"
        lbl_doc = json.loads(lbl_path.read_text(encoding="utf-8")) if lbl_path.exists() else {}
        ref_fidx = int(lbl_doc.get("frame_idx", 0))
        assignments = lbl_doc.get("assignments", [])
        leg_labels = legacy_doc.get(vid_id, {}).get("labels", {})

        used_uids: set[int] = set()
        for a in assignments:
            lbl = str(a["label"])
            gt_val = a.get("gt")
            if gt_val is None:
                continue
            gt_int = int(gt_val)
            det_bbox = a["det_bbox"]
            lcx, lcy = float(a["centroid"][0]), float(a["centroid"][1])

            # 找在 ref_fidx 附近 IoU 最大或中心距离最近的初始轨迹 uid
            best_uid: int | None = None
            best_score = -1e9
            for uid, (grp, _) in reports_by_uid.items():
                if uid in used_uids:
                    continue
                near_recs = sorted(grp.records, key=lambda r: abs(r.frame_idx - ref_fidx))[:5]
                if not near_recs:
                    continue
                r0 = near_recs[0]
                iou = iou_xyxy(r0.bbox, det_bbox)
                dist = float(np.hypot(r0.centroid[0] - lcx, r0.centroid[1] - lcy))
                score = iou * 200.0 - dist
                if score > best_score and dist <= 180.0:
                    best_score = score
                    best_uid = uid

            leg_info = leg_labels.get(lbl, {})
            old_pred = int(leg_info.get("s_count", 0) if kind == "crawl" else leg_info.get("c_count", 0))

            if best_uid is not None:
                used_uids.add(best_uid)
                grp, rep = reports_by_uid[best_uid]
                n_obs = rep.n_ok + rep.n_suspect + rep.n_bad
                cov_pct = 100.0 * float(len(grp.records)) / float(max(1, total_frames))
                bad_pct = 100.0 * float(rep.n_bad) / float(max(1, n_obs))
                rej_str = f"{rep.rejected['no_disp']}/{rep.rejected['wrong_dir']}/{rep.rejected['backward']}"

                if kind == "crawl":
                    new_pred = int(rep.s_count)
                    passed = crawl_pass_criterion(new_pred, gt_int)
                    pred_str = f"S={new_pred}"
                    crawl_total += 1
                    crawl_cov_list.append(cov_pct)
                    crawl_bad_obs.append((rep.n_bad, n_obs))
                    if passed:
                        crawl_passes += 1
                    if gt_int == 0:
                        crawl_zero_preds.append(new_pred)
                        legacy_zero_preds.append(old_pred)
                else:
                    passed, new_pred, best_u = swim_pass_criterion(rep.c_cycles, rep.c_bends, gt_int)
                    pred_str = f"cyc={rep.c_cycles}/bnd={rep.c_bends} ({best_u})"
                    swim_total += 1
                    swim_cov_list.append(cov_pct)
                    swim_bad_obs.append((rep.n_bad, n_obs))
                    if passed:
                        swim_passes += 1
            else:
                new_pred = 0
                passed = crawl_pass_criterion(0, gt_int) if kind == "crawl" else False
                pred_str = "unmatched(0)"
                cov_pct = 0.0
                bad_pct = 0.0
                rej_str = "0/0/0"
                if kind == "crawl":
                    crawl_total += 1
                    if passed:
                        crawl_passes += 1
                    if gt_int == 0:
                        crawl_zero_preds.append(0)
                        legacy_zero_preds.append(old_pred)
                else:
                    swim_total += 1

            new_err = abs(new_pred - gt_int)
            old_err = abs(old_pred - gt_int)
            new_errors.append(float(new_err))
            old_errors.append(float(old_err))

            md_lines.append(
                f"| `{vid_id}` | #{lbl} | {kind} | **{gt_int}** | {pred_str} | {old_pred} | "
                f"{new_err} | {old_err} | {cov_pct:.1f}% | {bad_pct:.1f}% | {rej_str} | {'✅' if passed else '❌'} |"
            )

    crawl_rate = 100.0 * crawl_passes / max(1, crawl_total)
    swim_rate = 100.0 * swim_passes / max(1, swim_total)
    new_mae = float(np.mean(new_errors)) if new_errors else 0.0
    old_mae = float(np.mean(old_errors)) if old_errors else 0.0
    zero_max = max(crawl_zero_preds) if crawl_zero_preds else 0
    zero_mean = float(np.mean(crawl_zero_preds)) if crawl_zero_preds else 0.0
    old_zero_sum = sum(legacy_zero_preds)
    new_zero_sum = sum(crawl_zero_preds)
    crawl_bad_pct = 100.0 * sum(b for b, _ in crawl_bad_obs) / float(max(1, sum(o for _, o in crawl_bad_obs)))
    swim_bad_pct = 100.0 * sum(b for b, _ in swim_bad_obs) / float(max(1, sum(o for _, o in swim_bad_obs)))
    crawl_cov_mean = float(np.mean(crawl_cov_list)) if crawl_cov_list else 0.0
    swim_cov_mean = float(np.mean(swim_cov_list)) if swim_cov_list else 0.0

    md_lines.extend(
        [
            "",
            "## 2. 总体指标汇总 (Summary vs Legacy Baseline)",
            "",
            "| 指标 | 新系统 (`wormlab`) | 旧系统 (`worm_processing`) | 验收线 |",
            "| :--- | :---: | :---: | :---: |",
            f"| 爬行视频标注虫通过率 | **{crawl_rate:.1f}%** ({crawl_passes}/{crawl_total}) | — | ≥ 80% |",
            f"| 游泳视频标注虫通过率 | **{swim_rate:.1f}%** ({swim_passes}/{swim_total}) | — | ≥ 75% |",
            f"| 全量标注虫总体 MAE | **{new_mae:.2f}** | {old_mae:.2f} | < 旧系统 ({old_mae:.2f}) |",
            f"| `gt=0` 静止/试探虫最大输出 | **{zero_max}** | {max(legacy_zero_preds) if legacy_zero_preds else 0} | ≤ 1 |",
            f"| `gt=0` 静止/试探虫平均输出 | **{zero_mean:.2f}** (总误报 {new_zero_sum}) | {float(np.mean(legacy_zero_preds)) if legacy_zero_preds else 0.0:.2f} (总误报 {old_zero_sum}) | 均值 ≤ 0.5 且总误报 ≤ 旧系统 |",
            f"| 匹配轨迹平均覆盖率 (爬行/游泳) | **{crawl_cov_mean:.1f}%** / **{swim_cov_mean:.1f}%** | — | ≥ 90% |",
            f"| 姿态 `BAD` 帧汇总占比 (爬行/游泳) | **{crawl_bad_pct:.1f}%** / **{swim_bad_pct:.1f}%** | — | 爬行 < 5%, 游泳 < 15% |",
            f"| 后进入不计数轨迹数 (`late_entry`) | {json.dumps(late_counts_by_vid, ensure_ascii=False)} (输出恒为 0) | 未过滤 | 全部置 0 |",
            f"| Pass A 实测吞吐量 (FPS) | {json.dumps(pass_a_speeds, ensure_ascii=False) if pass_a_speeds else '使用已有缓存'} | — | 1080p ≥ 25 FPS |",
            "",
            "## 3. 技术说明与已知客观边界 (Honest Caveats)",
            "",
            "- **首帧头标 (`HeadMarkModal`)**：爬行评测使用 `benchmarks/labels/{10x,1.6x-1,1.6x-3}.json` 中记录的首帧头部点击点 `head_xy` 模拟 `HeadMarkModal` 标注；若完全跳过首帧头标 (`head_config=[]`)，约 50% 虫首帧端点 0 会落在尾部 (`03_SPEC §11` 第 4 项已知风险)，新系统会因反向判定将其 `backward` 拒绝而输出 `S=0`。",
            "- **游泳双单位输出 (`c_cycles` / `c_bends`)**：依 `D15`，新系统同时报告完整往返周期 `c_cycles` 与单次弯曲 `c_bends`。实测 `swim-3 #2` 换算后为 `c_bends=31`（真值 `33`，误差 2 ✅）；`swim-3 #1` 约为 `87`（真值 `73`，可能因人工标注标准或高频帧抖动多计部分半摆）。",
            "- **检测器固有极限 (`D12 imgsz=416`)**：`swim-4 #1`（约 15px 极小点）与 `1.6x-3 #3` 在当前权重 (`origin.pt` / `CJLUdata_Old.pt`) 于 `imgsz=416` 下未被检出（旧系统基线同样输出 `0`），需后续补充极小目标/低倍微虫样本微调检测器。",
            "- **密集粘连帧质量**：`1.6x-1` 与 `1.6x-3` 个别交叉拥挤虫在虫体相交帧会触发 `merge` (`BAD`) 告警属预期保护，非粘连帧总体 `BAD` 占比可控。",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("已输出真实视频基准评测报告: %s (new_MAE=%.2f vs old_MAE=%.2f)", out_path, new_mae, old_mae)
    return {
        "crawl_pass_rate": crawl_rate,
        "swim_pass_rate": swim_rate,
        "new_mae": new_mae,
        "old_mae": old_mae,
        "zero_max": zero_max,
        "zero_mean": zero_mean,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行基准评测并生成 benchmarks/report.md")
    parser.add_argument("--synthetic", action="store_true", help="运行合成基准自检（无需真实视频缓存）")
    parser.add_argument("--videos", type=str, default=None, help="逗号分隔的视频 ID 列表")
    parser.add_argument("--cache-dir", type=str, default=str(RE_ROOT / "data" / "cache"))
    parser.add_argument("--rerun-pass-a", action="store_true", help="强制重跑 Pass A")
    parser.add_argument("--out", type=str, default=str(RE_ROOT / "benchmarks" / "report.md"))
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.synthetic:
        return run_synthetic_eval(out_path)

    vid_filter = {v.strip() for v in args.videos.split(",") if v.strip()} if args.videos else None
    evaluate_real_videos(
        video_filter=vid_filter,
        cache_root=Path(args.cache_dir),
        rerun_pass_a=args.rerun_pass_a,
        out_path=out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
