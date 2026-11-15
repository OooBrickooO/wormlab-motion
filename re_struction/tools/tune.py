"""T16 / T32 (07 §7): 运动学超参数网格搜索与留一视频交叉验证工具 (tune.py)。

支持两类数据集:
- `--dataset synth`: 内置 4 条合成序列快速搜索 (默认向后兼容)
- `--dataset real`: 读取 `data/cache/{10x,1.6x-1,1.6x-3}` 缓存与 `benchmarks/labels/` 真值，
  预计算滑窗特征与行波统计量后做全网格 + 留一视频交叉验证 (LOVO CV)，并输出
  `benchmarks/tuned_<profile>.yaml`（不自动修改 `default.yaml`）。
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
from wormlab.config import load_config  # noqa: E402
from wormlab.kinematics import KinematicsEngine  # noqa: E402
from wormlab.kinematics.features import compute_features  # noqa: E402
from wormlab.kinematics.modes import modal_phase  # noqa: E402
from wormlab.kinematics.s_runs import SRunCounter  # noqa: E402
from wormlab.kinematics.turns import TurnCounter  # noqa: E402
from wormlab.kinematics.wave import GateResult, wave_gate  # noqa: E402
from wormlab.pipeline.cache import load_cache  # noqa: E402
from wormlab.pipeline.late_entry import partition_tracks_by_uid, select_primary_uid_per_track_id  # noqa: E402
from wormlab.schemas import Pose, Quality  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("tune")

REAL_CRAWL_VIDEOS = ["10x", "1.6x-1", "1.6x-3"]


@dataclass
class _PrecomputedCrawlTrack:
    video_id: str
    label: str
    gt: int
    fps: float
    r_med: np.ndarray
    eps: np.ndarray
    amp_ant: np.ndarray
    amp_post: np.ndarray
    phi: list[float | None]
    center: list[np.ndarray | None]
    u: list[np.ndarray | None]
    L_ref: np.ndarray
    turn_or_bad: np.ndarray


def _iou_xyxy(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _build_synth_crawl_dataset() -> list[tuple[str, list[Pose], float, int]]:
    dataset: list[tuple[str, list[Pose], float, int]] = []
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
        dataset.append((kind, poses, 30.0, gt))
    return dataset


def _precompute_track(
    video_id: str,
    label: str,
    gt: int,
    poses_seq: list[Pose | None],
    fps: float,
    align_head: bool,
) -> _PrecomputedCrawlTrack:
    cfg = load_config("crawl_agar").kinematics

    if align_head:
        obs_feats = [
            compute_features(p.dense, cfg)
            for p in poses_seq
            if p is not None and p.dense is not None and p.quality != Quality.BAD
        ]
        obs_feats = [f for f in obs_feats if f is not None]
        if len(obs_feats) >= 2:
            dC = obs_feats[-1].center - obs_feats[0].center
            u_m = np.mean([f.u for f in obs_feats[: min(150, len(obs_feats))]], axis=0)
            if float(np.dot(dC, u_m)) < 0.0:
                poses_seq = [
                    None
                    if p is None or p.dense is None
                    else Pose(
                        dense=p.dense[::-1].copy(),
                        length=p.length,
                        chord=p.chord,
                        width=p.width,
                        area=p.area,
                        quality=p.quality,
                        flags=p.flags,
                    )
                    for p in poses_seq
                ]

    eng = KinematicsEngine(cfg=cfg, fps=fps, c_unit="cycle")
    n = len(poses_seq)
    for idx, p in enumerate(poses_seq):
        eng.valid_dense.append(p.dense if (p is not None and p.quality != Quality.BAD) else None)
    for i in range(n):
        eng._compute_feat_at(i)

    turn = TurnCounter(cfg.turn, fps=fps)
    r_med = np.zeros(n, dtype=np.float64)
    eps = np.zeros(n, dtype=np.float64)
    amp_ant = np.zeros(n, dtype=np.float64)
    amp_post = np.zeros(n, dtype=np.float64)
    phi: list[float | None] = [None] * n
    center: list[np.ndarray | None] = [None] * n
    u: list[np.ndarray | None] = [None] * n
    L_ref = np.full(n, 120.0, dtype=np.float64)
    turn_or_bad = np.zeros(n, dtype=bool)

    for i in range(n):
        f = eng.feats[i]
        lo = max(0, i - eng.h_wave)
        hi = min(n, i + eng.h_wave + 1)
        win = eng.feats[lo:hi]
        g: GateResult = wave_gate(win, cfg.wave, expected_len=len(win)) if f is not None else GateResult.off()
        r_med[i] = g.r_med
        eps[i] = g.eps
        amp_ant[i] = g.amp_ant
        amp_post[i] = g.amp_post
        L_ref[i] = eng._L_ref(i)
        turn.step(i, f.R if f is not None else 0.0, valid=(f is not None))
        turn_or_bad[i] = bool(turn.in_curl or eng._bad_too_long(i))
        if f is not None:
            phi[i] = modal_phase(f.a1, f.a2)
            center[i] = f.center
            u[i] = f.u

    return _PrecomputedCrawlTrack(
        video_id=video_id,
        label=label,
        gt=gt,
        fps=fps,
        r_med=r_med,
        eps=eps,
        amp_ant=amp_ant,
        amp_post=amp_post,
        phi=phi,
        center=center,
        u=u,
        L_ref=L_ref,
        turn_or_bad=turn_or_bad,
    )


def _build_real_crawl_dataset(align_head: bool) -> list[_PrecomputedCrawlTrack]:
    cfg0 = load_config("crawl_agar")
    tracks: list[_PrecomputedCrawlTrack] = []
    for vid in REAL_CRAWL_VIDEOS:
        cdir = RE_ROOT / "data" / "cache" / vid
        if not (cdir / "tracks.npz").exists():
            raise FileNotFoundError(f"缺少缓存 {cdir / 'tracks.npz'}，请先运行 eval_benchmark.py --rerun-pass-a")
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
            tracks.append(_precompute_track(vid, str(a["label"]), int(a["gt"]), seq, fps, align_head=align_head))
    return tracks


def _eval_fast(track: _PrecomputedCrawlTrack, kcfg: Any) -> int:
    w = kcfg.wave
    s_counter = SRunCounter(kcfg.s_run, fps=track.fps)
    n = len(track.phi)
    r_m = float(w.r_min)
    e_m = float(w.eps_min)
    pr_m = float(w.post_ratio_min)
    ap_m = float(w.amp_post_min)

    for i in range(n):
        if track.turn_or_bad[i]:
            s_counter.force_abort("curl_or_bad")
        gate_ok = bool(
            track.phi[i] is not None
            and track.r_med[i] >= r_m
            and track.eps[i] >= e_m
            and track.amp_post[i] >= pr_m * track.amp_ant[i]
            and track.amp_post[i] >= ap_m
        )
        s_counter.step(
            i=i,
            gate_ok=gate_ok,
            phi=track.phi[i],
            center=track.center[i],
            u=track.u[i],
            L_ref=float(track.L_ref[i]),
        )
    return int(s_counter.s_count)


def _crawl_pass(pred: int, gt: int) -> bool:
    return abs(pred - gt) <= 1 if gt <= 5 else abs(pred - gt) <= max(2, 0.2 * float(gt))


def _loss(preds: list[int], tracks: list[_PrecomputedCrawlTrack], idxs: list[int]) -> tuple[float, int, float]:
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


def tune_real_crawl(out_yaml: Path, align_head: bool = True) -> dict[str, Any]:
    tracks = _build_real_crawl_dataset(align_head=align_head)
    log.info("已加载 %d 条真实爬行轨迹 (align_head=%s)", len(tracks), align_head)
    grid = {
        "wave.r_min": [0.12, 0.15, 0.20],
        "wave.eps_min": [0.18, 0.20, 0.22],
        "wave.post_ratio_min": [0.35],
        "s_run.gap_hold_s": [0.8, 0.9],
        "s_run.min_disp_ratio": [0.04, 0.05],
        "s_run.dir_lock_rad": [0.8],
    }
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
        preds = [_eval_fast(t, kcfg) for t in tracks]
        candidates.append((ov, preds))

    all_idx = list(range(len(tracks)))
    best_ov, best_preds = min(candidates, key=lambda c: _loss(c[1], tracks, all_idx)[0])
    _, best_pass, best_mae = _loss(best_preds, tracks, all_idx)

    cv_pass = 0
    cv_n = 0
    lovo_details: dict[str, str] = {}
    for hold in REAL_CRAWL_VIDEOS:
        tr = [i for i, t in enumerate(tracks) if t.video_id != hold]
        te = [i for i, t in enumerate(tracks) if t.video_id == hold]
        ov_tr, preds_tr = min(candidates, key=lambda c: _loss(c[1], tracks, tr)[0])
        _, k_te, mae_te = _loss(preds_tr, tracks, te)
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
        },
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return doc


def tune_crawl(dataset: list[tuple[str, list[Pose], float, int]], out_yaml: Path) -> dict[str, Any]:
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
        for _, poses, fps, gt in dataset:
            rep, _ = KinematicsEngine.run(poses, fps=fps, cfg=kcfg)
            if gt == 0 and rep.s_count > 1:
                zero_violated = True
                break
            loss += abs(rep.s_count - gt) / float(max(1, gt))
        if not zero_violated and loss < best_loss:
            best_loss = loss
            best_params = overrides

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(best_params, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("最优爬行参数已输出至 %s (loss=%.4f): %s", out_yaml, best_loss, best_params)
    return best_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="网格搜索调参工具")
    parser.add_argument("--profile", type=str, default="crawl_agar")
    parser.add_argument("--dataset", choices=["synth", "real"], default="synth")
    parser.add_argument("--raw-head", action="store_true", help="真实视频调参时不模拟首帧头部对齐")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    out_path = Path(args.out) if args.out else (RE_ROOT / "benchmarks" / f"tuned_{args.profile}.yaml")
    if args.dataset == "real":
        tune_real_crawl(out_path, align_head=not args.raw_head)
    else:
        dataset = _build_synth_crawl_dataset()
        tune_crawl(dataset, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
