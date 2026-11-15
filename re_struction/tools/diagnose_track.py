"""T10 (02 §8): 轨迹姿态与几何度量诊断出图工具 (diagnose_track.py)。

支持直接从视频抽帧检测并提取指定虫（或面积/置信度第 N 的虫）的姿态序列，输出：
1. 6 帧拼图：原图 ROI + 掩码外轮廓 + 33 点中心线（头部绿点）+ 质量等级
2. 时间序列曲线：体长 length、面积 area、弦长 chord、舒展比 R 与质量色带
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = RE_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from wormlab.config import load_config  # noqa: E402
from wormlab.detection import Detector  # noqa: E402
from wormlab.pose import PoseExtractor  # noqa: E402
from wormlab.schemas import Detection, Pose, Quality  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("diagnose_track")


def run_diagnosis(
    video_path: Path,
    profile: str,
    track_idx: int,
    start_frame: int,
    end_frame: int,
    out_dir: Path,
) -> Path:
    cfg = load_config(profile)
    model_path = RE_ROOT / "backend" / "models" / cfg.detection.default_model
    detector = Detector(str(model_path), cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_f = min(total, end_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extractor = PoseExtractor(cfg.pose, fps=fps)
    anchor_center: np.ndarray | None = None

    poses: list[Pose] = []
    rois_vis: list[tuple[int, np.ndarray, Pose]] = []
    f_indices: list[int] = []

    sample_step = max(1, (end_f - start_frame) // 6)

    for fidx in range(start_frame, end_f):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        dets = detector.infer([frame])[0]
        chosen_det: Detection | None = None
        if dets:
            if anchor_center is None:
                idx = min(max(0, track_idx), len(dets) - 1)
                chosen_det = dets[idx]
                b = chosen_det.bbox
                anchor_center = np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float32)
            else:
                # 选距离上一帧中心最近的检测
                best_d = 1e9
                for d in dets:
                    b = d.bbox
                    c = np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float32)
                    dist = float(np.linalg.norm(c - anchor_center))
                    if dist < best_d and dist < 160.0:
                        best_d = dist
                        chosen_det = d
                if chosen_det is not None:
                    b = chosen_det.bbox
                    anchor_center = np.array([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float32)

        if chosen_det is not None:
            pose = extractor.process(chosen_det, frame_idx=fidx)
        else:
            pose = Pose(dense=None, length=0.0, chord=0.0, width=0.0, area=0.0, quality=Quality.MISSING)

        poses.append(pose)
        f_indices.append(fidx)

        if len(rois_vis) < 6 and (fidx - start_frame) % sample_step == 0 and chosen_det is not None:
            x0, y0 = chosen_det.mask_origin
            mh, mw = chosen_det.mask.shape[:2] if chosen_det.mask is not None else (80, 80)
            crop = frame[y0 : y0 + mh, x0 : x0 + mw].copy()
            if pose.contour is not None:
                c_local = np.round(pose.contour - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32)
                cv2.polylines(crop, [c_local.reshape(-1, 1, 2)], True, (255, 200, 0), 1, cv2.LINE_AA)
            if pose.dense is not None:
                d_local = np.round(pose.dense - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32)
                cv2.polylines(crop, [d_local.reshape(-1, 1, 2)], False, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.circle(crop, (int(d_local[0, 0]), int(d_local[0, 1])), 3, (0, 255, 0), -1, cv2.LINE_AA)
            rois_vis.append((fidx, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), pose))

    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 8), dpi=120)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.3])

    for k in range(6):
        ax = fig.add_subplot(gs[0, k])
        if k < len(rois_vis):
            fidx_k, img_rgb, p_k = rois_vis[k]
            ax.imshow(img_rgb)
            ax.set_title(f"F{fidx_k} [{p_k.quality.name}]\nL={p_k.length:.1f}px", fontsize=9)
        ax.axis("off")

    ax_len = fig.add_subplot(gs[1, :3])
    lengths = [p.length if p.dense is not None else np.nan for p in poses]
    chords = [p.chord if p.dense is not None else np.nan for p in poses]
    ax_len.plot(f_indices, lengths, label="Length (px)", color="tab:blue", lw=1.5)
    ax_len.plot(f_indices, chords, label="Chord (px)", color="tab:orange", lw=1.2)
    ax_len.set_title(f"{video_path.name} — Length & Chord")
    ax_len.set_xlabel("Frame")
    ax_len.legend(loc="upper right")
    ax_len.grid(True, alpha=0.3)

    ax_r = fig.add_subplot(gs[1, 3:])
    rs = [p.chord / max(1e-6, p.length) if p.dense is not None else np.nan for p in poses]
    quals = [int(p.quality) for p in poses]
    ax_r.plot(f_indices, rs, label="Extension R = D/L", color="tab:green", lw=1.5)
    ax_r.set_ylim(0.0, 1.05)
    ax_r.set_title("Extension Ratio R & Quality (0=OK,1=SUSPECT,2=BAD,3=MISSING)")
    ax_r.set_xlabel("Frame")
    ax_r2 = ax_r.twinx()
    ax_r2.step(f_indices, quals, color="tab:red", alpha=0.35, where="mid", label="Quality")
    ax_r2.set_ylim(-0.2, 3.5)
    ax_r.legend(loc="upper left")
    ax_r.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = out_dir / f"diag_{video_path.stem}_t{track_idx}.png"
    fig.savefig(str(out_png))
    plt.close(fig)
    log.info("已保存姿态诊断图: %s", out_png)
    return out_png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成线虫姿态与中心线诊断图")
    parser.add_argument("--video", type=str, required=True, help="视频路径")
    parser.add_argument("--profile", type=str, default="crawl_agar", help="crawl_agar 或 swim_liquid")
    parser.add_argument("--track-id", type=int, default=0, help="首帧目标序号 (默认 0)")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=120)
    parser.add_argument("--out", type=str, default=str(RE_ROOT / "data" / "diag"))
    args = parser.parse_args(argv)

    run_diagnosis(
        video_path=Path(args.video),
        profile=args.profile,
        track_idx=args.track_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        out_dir=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
