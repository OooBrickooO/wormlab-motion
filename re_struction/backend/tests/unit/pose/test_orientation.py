"""T08 单测：orient() 头尾连续性与用户头点对齐。"""
from __future__ import annotations

import numpy as np

from wormlab.pose.orientation import orient
from wormlab.schemas import N_DENSE


def test_orient_user_head_xy() -> None:
    # 起点在 (10, 0)，终点在 (110, 0)
    dense = np.stack([np.linspace(10.0, 110.0, N_DENSE), np.zeros(N_DENSE)], axis=1).astype(np.float32)
    # 用户标记头在 (108, 2) 附近 -> 应翻转使 dense[0] 在 110
    flipped = orient(dense, prev_dense=None, user_head_xy=(108.0, 2.0))
    assert flipped[0, 0] == 110.0
    assert flipped[-1, 0] == 10.0


def test_orient_sequence_recovery_and_hysteresis() -> None:
    base = np.stack([np.linspace(0.0, 100.0, N_DENSE), np.zeros(N_DENSE)], axis=1).astype(np.float32)
    prev = base.copy()
    for step in range(1, 10):
        cur = base + np.array([[step * 2.0, 0.0]], dtype=np.float32)
        raw = cur[::-1].copy() if step in (3, 6, 7) else cur.copy()
        out = orient(raw, prev_dense=prev)
        assert np.allclose(out, cur, atol=1e-4)
        prev = out

    # 完全对称且正反距离相等时，由于 margin=0.9 迟滞不翻转
    prev_sym = np.stack([np.linspace(-50.0, 50.0, N_DENSE), np.zeros(N_DENSE)], axis=1).astype(np.float32)
    cur_sym = np.stack([np.zeros(N_DENSE), np.linspace(-50.0, 50.0, N_DENSE)], axis=1).astype(np.float32)
    out_sym = orient(cur_sym, prev_dense=prev_sym)
    assert np.allclose(out_sym, cur_sym)


def test_orient_centered_across_translation_gap_and_reorient_track() -> None:
    from wormlab.pose.orientation import reorient_track_poses
    from wormlab.schemas import Pose, Quality

    s = np.linspace(0.0, 1.0, N_DENSE, dtype=np.float32)
    base = np.stack([s * 100.0, 18.0 * np.sin(np.pi * s) + 6.0 * s], axis=1).astype(np.float32)
    # 模拟跨越大位移/遮挡间隙后整体平移 (dx=90, dy=30) 且中心线端序保持正确
    shifted = base + np.array([[90.0, 30.0]], dtype=np.float32)
    out = orient(shifted, prev_dense=base)
    assert np.allclose(out, shifted, atol=1e-4)

    # 模拟中间断帧后偶发翻转 -> reorient_track_poses 应自动纠偏
    p0 = Pose(dense=base.copy(), length=100.0, chord=90.0, width=6.0, area=600.0, quality=Quality.OK)
    p2 = Pose(dense=shifted[::-1].copy(), length=100.0, chord=90.0, width=6.0, area=600.0, quality=Quality.OK)
    fixed = reorient_track_poses([p0, None, p2])
    assert fixed[2] is not None and fixed[2].dense is not None
    assert np.allclose(fixed[2].dense, shifted, atol=1e-4)


def test_reorient_coil_aware_flip_invariance_and_bridge_gaps() -> None:
    import sys
    from dataclasses import replace
    from pathlib import Path

    re_root = Path(__file__).resolve().parents[4]
    if str(re_root) not in sys.path:
        sys.path.insert(0, str(re_root))
    from tools.make_synthetic import Scn, make_series
    from wormlab.config import load_config
    from wormlab.pose.orientation import bridge_short_pose_gaps, reorient_track_poses_coil_aware
    from wormlab.schemas import Pose, Quality

    fps = 30.0
    cfg = load_config("crawl_agar").kinematics
    dense_seq, _ = make_series(Scn(kind="crawl_fwd", duration_s=8.0, fps=fps))
    poses: list[Pose | None] = []
    for i, d in enumerate(dense_seq):
        diffs = np.diff(d, axis=0)
        L = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
        chord = float(np.linalg.norm(d[0] - d[-1]))
        # 构造中间 5 帧深卷曲（chord/L < 0.45），且卷曲前后各自初始端点反向
        if 110 <= i < 115:
            poses.append(Pose(dense=d.copy(), length=L, chord=0.25 * L, width=7.0, area=840.0, quality=Quality.OK))
        else:
            d_use = d[::-1].copy() if (i == 0 or i >= 115) else d.copy()
            poses.append(Pose(dense=d_use, length=L, chord=chord, width=7.0, area=840.0, quality=Quality.OK))

    aligned = reorient_track_poses_coil_aware(poses, fps=fps, cfg=cfg, preserve_first_head=False)
    for idx in (10, 150):
        assert aligned[idx] is not None and aligned[idx].dense is not None
        # 前进正方向端点一致性：dense[0] 应与真实合成前进头部一致
        d_true_u = dense_seq[idx][0] - dense_seq[idx][-1]
        d_alg_u = aligned[idx].dense[0] - aligned[idx].dense[-1]
        assert float(np.dot(d_true_u, d_alg_u)) > 0.0

    # 短时缺口插值测试：缺失 6 帧 (0.2s <= 0.5s) 应被补全为 SUSPECT
    gapped = list(aligned)
    for k in range(30, 36):
        gapped[k] = None
    bridged = bridge_short_pose_gaps(gapped, fps=fps, max_gap_s=0.50)
    for k in range(30, 36):
        assert bridged[k] is not None and bridged[k].quality == Quality.SUSPECT
