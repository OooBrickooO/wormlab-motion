"""T16 / T32 (07 §7 + 08_SPEC Workstream D): 运动学超参数网格搜索与留一视频交叉验证工具 (tune.py)。

08_SPEC S7 修复：旧版 `_eval_fast` 与 `KinematicsEngine` 严重脱节（缺 s_run.k_body、freeze、
slowdown_hold_s、settle_partial、deep_coil_R、角动量门限）。现对每组候选超参**直接调用
`KinematicsEngine.run_postprocessed(...)`**（与 PassB 生产管线完全同一口径），不再维护失真的
快速副本。

支持:
- `--dataset synth`: 内置 4 条合成序列快速搜索 (默认向后兼容)
- `--dataset real --profile crawl_agar`: 读取 `data/cache/{10x,1.6x-1,1.6x-3}` 缓存与
  `benchmarks/labels/` 真值，全网格 + 留一视频交叉验证 (LOVO CV)
- `--dataset real --profile swim_liquid`: 读取 `data/cache/{Swim-3,Swim-4}` 缓存，
  网格搜索 swing.bend_high / shape_aspect_prom / shape_motion_min / min_flip_interval_s
输出 `benchmarks/tuned_<profile>.yaml`（不自动修改 default.yaml）。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = RE_ROOT / "backend"
for p in (str(RE_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.make_synthetic import Scn, make_series  # noqa: E402
from wormlab.config import KinCfg, load_config  # noqa: E402
from wormlab.kinematics import KinematicsEngine, TrackReport  # noqa: E402
from wormlab.pipeline.cache import load_cache  # noqa: E402
from wormlab.pipeline.late_entry import partition_tracks_by_uid, select_primary_uid_per_track_id  # noqa: E402
from wormlab.pose.orientation import reorient_track_poses, reorient_track_poses_coil_aware  # noqa: E402
from wormlab.schemas import Pose, Quality  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("tune")

REAL_CRAWL_VIDEOS = ["10x", "1.6x-1", "1.6x-3"]
REAL_SWIM_VIDEOS = ["Swim-3", "Swim-4"]


@dataclass
class _TrackData:
    """单条轨迹的原始姿态序列 + 真值（评测时直接喂给 run_postprocessed）。"""

    video_id: str
    label: str
    gt: int
    fps: float
    poses: list[Pose | None]


def _iou_xyxy(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _build_synth_crawl_dataset() -> list[_TrackData]:
    dataset: list[_TrackData] = []
    for kind, gt in [("crawl_fwd", 10), ("crawl_bwd", 0), ("head_wiggle", 0), ("treadmill", 0)]:
        d_seq, _ = make_series(Scn(kind=kind, duration_s=20.0, fps=30.0))
        poses = [
            Pose(
                dense=d,
                length=120.0,
                chord=float(np.linalg.norm(d[0] - d[-1])),
                width=7.0,
                area=840.0,
                quality=Quality.OK,
            )
            for d in d_seq
        ]
        dataset.append(_TrackData(video_id="synth", label=kind, gt=gt, fps=30.0, poses=poses))
    return dataset


def _align_head_flip(poses_seq: list[Pose | None]) -> list[Pose | None]:
    """依据观测特征的整体前进方向，将首帧头尾对齐（模拟首帧人工头标效果）。"""
    cfg = load_config("crawl_agar").kinematics
    from wormlab.kinematics.features import compute_features

    obs_feats = [
        compute_features(p.dense, cfg)
        for p in poses_seq
        if p is not None and p.dense is not None and p.quality != Quality.BAD
    ]
    obs_feats = [f for f in obs_feats if f is not None]
    if len(obs_feats) < 2:
        return poses_seq
    dC = obs_feats[-1].center - obs_feats[0].center
    u_m = np.mean([f.u for f in obs_feats[: min(150, len(obs_feats))]], axis=0)
    if float(np.dot(dC, u_m)) >= 0.0:
        return poses_seq
    return [
        None
        if p is None or p.dense is None
        else replace(p, dense=p.dense[::-1].copy())
        for p in poses_seq
    ]


def _load_real_tracks(profile: str, videos: list[str], align_head: bool) -> list[_TrackData]:
    """从 data/cache/<vid>/tracks.npz + benchmarks/labels/<vid>.json 构建评测轨迹集。"""
    cfg0 = load_config(profile)
    coil_aware = bool(getattr(cfg0.pose, "coil_aware_reorient", False) and cfg0.kinematics.s_run.enabled)
    tracks: list[_TrackData] = []
    for vid in videos:
        cdir = RE_ROOT / "data" / "cache" / vid
        if not (cdir / "tracks.npz").exists():
            raise FileNotFoundError(
                f"缺少缓存 {cdir / 'tracks.npz'}，请先运行 eval_benchmark.py --rerun-pass-a"
            )
        meta, recs = load_cache(cdir)
        fps = float(meta["fps"])
        T = int(meta["total_frames"])
        sel, _ = select_primary_uid_per_track_id(
            partition_tracks_by_uid(recs, fps=fps, grace_s=cfg0.counting.late_entry_grace_s)
        )
        lbl = json.loads((RE_ROOT / "benchmarks" / "labels" / f"{vid}.json").read_text(encoding="utf-8"))
        ref = int(lbl.get("frame_idx", 0))
        used: set[int] = set()
        for a in lbl["assignments"]:
            best, bs = None, -1e9
            for tid, grp in sel.items():
                if tid in used or not grp.counted or not grp.records:
                    continue
                r0 = min(grp.records, key=lambda r: abs(r.frame_idx - ref))
                d = float(np.hypot(r0.centroid[0] - a["centroid"][0], r0.centroid[1] - a["centroid"][1]))
                s = _iou_xyxy(r0.bbox, a["det_bbox"]) * 200.0 - d
                if d <= 180.0 and s > bs:
                    best, bs = tid, s
            if best is None:
                continue
            used.add(best)
            seq: list[Pose | None] = [None] * T
            for r in sel[best].records:
                if not r.is_ghost and 0 <= r.frame_idx < T:
                    seq[r.frame_idx] = r.pose
            if align_head:
                seq = _align_head_flip(seq)
            if coil_aware:
                seq = reorient_track_poses_coil_aware(
                    seq,
                    cfg=cfg0.kinematics,
                    fps=fps,
                    preserve_first_head=align_head,
                )
            else:
                seq = reorient_track_poses(seq)
            tracks.append(_TrackData(video_id=vid, label=str(a["label"]), gt=int(a["gt"]), fps=fps, poses=seq))
    return tracks


def _eval_postprocessed(track: _TrackData, kcfg: KinCfg, c_unit: str = "cycle") -> TrackReport:
    """直接调用生产管线同款 run_postprocessed（含桥接 / 自适应 k / 免骨架双轨 / 尾数结算）。"""
    rep, _pfs = KinematicsEngine.run_postprocessed(
        track.poses,
        fps=track.fps,
        cfg=kcfg,
        c_unit=c_unit,
    )
    return rep


def _crawl_pass(pred: int, gt: int) -> bool:
    return abs(pred - gt) <= 1 if gt <= 5 else abs(pred - gt) <= max(2, 0.2 * float(gt))


def _crawl_pred(rep: TrackReport) -> int:
    return int(rep.s_count)


def _swim_pred(rep: TrackReport) -> tuple[int, int]:
    return int(rep.c_cycles), int(rep.c_bends)


def _swim_pass(c_cycles: int, c_bends: int, gt: int) -> bool:
    err = min(abs(c_cycles - gt), abs(c_bends - gt))
    return err <= max(2.0, 0.15 * float(gt))


def _crawl_loss(preds: list[int], tracks: list[_TrackData], idxs: list[int]) -> tuple[float, int, float]:
    mae_sum = 0.0
    rel_sum = 0.0
    zero_pen = 0.0
    passes = 0
    for i in idxs:
        p, g = preds[i], tracks[i].gt
        mae_sum += abs(p - g)
        rel_sum += abs(p - g) / float(max(1, g))
        if g == 0 and p > 1:
            zero_pen += 5.0 * (p - 1)
        if _crawl_pass(p, g):
            passes += 1
    return rel_sum + zero_pen, passes, mae_sum / float(max(1, len(idxs)))


def _swim_loss(
    preds: list[tuple[int, int]],
    tracks: list[_TrackData],
    idxs: list[int],
) -> tuple[float, int, float]:
    mae_sum = 0.0
    rel_sum = 0.0
    zero_pen = 0.0
    passes = 0
    for i in idxs:
        cyc, bnd = preds[i]
        g = tracks[i].gt
        err = min(abs(cyc - g), abs(bnd - g))
        mae_sum += err
        rel_sum += err / float(max(1, g))
        if g == 0 and min(cyc, bnd) > 1:
            zero_pen += 5.0 * (min(cyc, bnd) - 1)
        if _swim_pass(cyc, bnd, g):
            passes += 1
    return rel_sum + zero_pen, passes, mae_sum / float(max(1, len(idxs)))


def _grid_eval_crawl(
    tracks: list[_TrackData],
    grid: dict[str, list[Any]],
) -> list[tuple[dict[str, Any], list[int]]]:
    keys = list(grid)
    base_kcfg = load_config("crawl_agar").kinematics
    candidates: list[tuple[dict[str, Any], list[int]]] = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        ov: dict[str, Any] = {}
        for k, v in zip(keys, vals):
            a, b = k.split(".")
            ov.setdefault(a, {})[b] = v
        kcfg = replace(
            base_kcfg,
            wave=replace(base_kcfg.wave, **ov.get("wave", {})),
            s_run=replace(base_kcfg.s_run, **ov.get("s_run", {})),
        )
        preds = [_crawl_pred(_eval_postprocessed(t, kcfg)) for t in tracks]
        candidates.append((ov, preds))
    return candidates


def tune_real_crawl(out_yaml: Path, align_head: bool = True) -> dict[str, Any]:
    """真实爬行视频调参：与生产引擎完全同口径的 run_postprocessed 网格搜索 + LOVO CV。"""
    tracks = _load_real_tracks("crawl_agar", REAL_CRAWL_VIDEOS, align_head=align_head)
    log.info("已加载 %d 条真实爬行轨迹 (align_head=%s)", len(tracks), align_head)

    grid = {
        "wave.r_min": [0.12, 0.14, 0.18],
        "wave.coherence_min": [0.50, 0.55, 0.60],
        "wave.omega_min_rad_s": [0.30, 0.40],
        "s_run.gap_hold_s": [0.8, 0.9],
        "s_run.min_disp_ratio": [0.04, 0.05, 0.08],
        "s_run.phase_bridge_s": [0.35, 0.45],
    }
    candidates = _grid_eval_crawl(tracks, grid)

    all_idx = list(range(len(tracks)))
    best_ov, best_preds = min(candidates, key=lambda c: _crawl_loss(c[1], tracks, all_idx)[0])
    _, best_pass, best_mae = _crawl_loss(best_preds, tracks, all_idx)

    cv_pass = 0
    cv_n = 0
    lovo_details: dict[str, str] = {}
    for hold in REAL_CRAWL_VIDEOS:
        tr = [i for i, t in enumerate(tracks) if t.video_id != hold]
        te = [i for i, t in enumerate(tracks) if t.video_id == hold]
        ov_tr, _ = min(candidates, key=lambda c: _crawl_loss(c[1], tracks, tr)[0])
        # 用训练集最优参数重新评测留出集
        kcfg = replace(
            load_config("crawl_agar").kinematics,
            wave=replace(load_config("crawl_agar").kinematics.wave, **ov_tr.get("wave", {})),
            s_run=replace(load_config("crawl_agar").kinematics.s_run, **ov_tr.get("s_run", {})),
        )
        preds_te = [_crawl_pred(_eval_postprocessed(tracks[i], kcfg)) for i in te]
        _, k_te, mae_te = _crawl_loss(preds_te, tracks, te)
        cv_pass += k_te
        cv_n += len(te)
        lovo_details[hold] = f"{k_te}/{len(te)} pass (MAE={mae_te:.2f})"
        log.info("  LOVO hold=%-6s -> %s | train_best=%s", hold, lovo_details[hold], json.dumps(ov_tr))

    zeros = [best_preds[i] for i, t in enumerate(tracks) if t.gt == 0]
    log.info(
        "最优真实爬行参数 (align_head=%s): pass=%d/%d (%.1f%%), MAE=%.2f, zeros=%s, LOVO=%d/%d | %s",
        align_head,
        best_pass,
        len(tracks),
        100.0 * best_pass / max(1, len(tracks)),
        best_mae,
        zeros,
        cv_pass,
        cv_n,
        json.dumps(best_ov),
    )

    doc = {
        "kinematics": best_ov,
        "_meta": {
            "align_head": align_head,
            "tracks": len(tracks),
            "in_sample_pass": f"{best_pass}/{len(tracks)}",
            "in_sample_mae": round(best_mae, 3),
            "lovo_pass": f"{cv_pass}/{cv_n}",
            "lovo_by_video": lovo_details,
            "gt_zero_preds": zeros,
            "engine": "KinematicsEngine.run_postprocessed",
        },
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return doc


def tune_real_swim(out_yaml: Path, align_head: bool = False) -> dict[str, Any]:
    """真实游泳视频调参：网格搜索 swing 阈值（骨架迟滞 + 免骨架双轨均由生产引擎统一执行）。"""
    tracks = _load_real_tracks("swim_liquid", REAL_SWIM_VIDEOS, align_head=align_head)
    log.info("已加载 %d 条真实游泳轨迹 (align_head=%s)", len(tracks), align_head)

    base_kcfg = load_config("swim_liquid").kinematics
    grid = {
        "bend_high": [0.65, 0.75, 0.85],
        "shape_aspect_prom": [0.04, 0.06, 0.08],
        "shape_motion_min": [1.0, 1.5, 2.0],
        "min_flip_interval_s": [0.06, 0.08, 0.10],
    }
    keys = list(grid)
    candidates: list[tuple[dict[str, Any], list[tuple[int, int]]]] = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        ov = dict(zip(keys, vals))
        kcfg = replace(base_kcfg, swing=replace(base_kcfg.swing, **ov))
        preds = [_swim_pred(_eval_postprocessed(t, kcfg, c_unit="cycle")) for t in tracks]
        candidates.append((ov, preds))

    all_idx = list(range(len(tracks)))
    best_ov, best_preds = min(candidates, key=lambda c: _swim_loss(c[1], tracks, all_idx)[0])
    _, best_pass, best_mae = _swim_loss(best_preds, tracks, all_idx)

    zeros = [best_preds[i] for i, t in enumerate(tracks) if t.gt == 0]
    log.info(
        "最优真实游泳参数 (align_head=%s): pass=%d/%d (%.1f%%), MAE=%.2f, zeros=%s | %s",
        align_head,
        best_pass,
        len(tracks),
        100.0 * best_pass / max(1, len(tracks)),
        best_mae,
        zeros,
        json.dumps(best_ov),
    )

    doc = {
        "kinematics": {"swing": best_ov},
        "_meta": {
            "align_head": align_head,
            "videos": REAL_SWIM_VIDEOS,
            "tracks": len(tracks),
            "in_sample_pass": f"{best_pass}/{len(tracks)}",
            "in_sample_mae": round(best_mae, 3),
            "gt_zero_preds": zeros,
            "engine": "KinematicsEngine.run_postprocessed",
        },
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return doc


def tune_crawl(dataset: list[_TrackData], out_yaml: Path) -> dict[str, Any]:
    """合成序列快速网格搜索（向后兼容入口，同样走生产引擎 run_postprocessed）。"""
    r_mins = [0.20, 0.25, 0.30]
    eps_mins = [0.35, 0.40, 0.50]
    disp_ratios = [0.08, 0.10, 0.15]
    post_ratios = [0.25, 0.35, 0.45]

    best_loss = 1e9
    best_params: dict[str, Any] = {}

    for r_m, e_m, d_r, p_r in itertools.product(r_mins, eps_mins, disp_ratios, post_ratios):
        overrides = {
            "kinematics": {
                "wave": {"r_min": r_m, "eps_min": e_m, "post_ratio_min": p_r},
                "s_run": {"min_disp_ratio": d_r},
            }
        }
        kcfg = load_config("crawl_agar", overrides=overrides).kinematics
        loss = 0.0
        zero_violated = False
        for track in dataset:
            rep = _eval_postprocessed(track, kcfg)
            if track.gt == 0 and rep.s_count > 1:
                zero_violated = True
                break
            loss += abs(rep.s_count - track.gt) / float(max(1, track.gt))
        if not zero_violated and loss < best_loss:
            best_loss = loss
            best_params = overrides

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(best_params, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("最优爬行参数已输出至 %s (loss=%.4f): %s", out_yaml, best_loss, best_params)
    return best_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="网格搜索调参工具（与生产引擎 run_postprocessed 同口径）")
    parser.add_argument("--profile", type=str, default="crawl_agar", choices=["crawl_agar", "swim_liquid"])
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth")
    parser.add_argument("--raw-head", action="store_true", help="真实视频调参时不模拟首帧头部对齐")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    out_path = Path(args.out) if args.out else (RE_ROOT / "benchmarks" / f"tuned_{args.profile}.yaml")
    if args.dataset == "real":
        if args.profile == "swim_liquid":
            tune_real_swim(out_path, align_head=False)
        else:
            tune_real_crawl(out_path, align_head=not args.raw_head)
    else:
        dataset = _build_synth_crawl_dataset()
        tune_crawl(dataset, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
