"""T08 单测：PoseQualityJudge 解耦无效帧 (BAD) 与正常弯曲帧 (SUSPECT)。"""
from __future__ import annotations

import numpy as np

from wormlab.config import load_config
from wormlab.pose.quality import PoseQualityJudge
from wormlab.schemas import N_DENSE, Pose, Quality


def _dummy_pose(length: float = 120.0, width: float = 7.0, area: float = 800.0, flags: set[str] | None = None) -> Pose:
    dense = np.stack([np.linspace(0.0, length, N_DENSE), np.zeros(N_DENSE)], axis=1).astype(np.float32)
    return Pose(
        dense=dense,
        length=length,
        chord=length * 0.85,
        width=width,
        area=area,
        quality=Quality.OK,
        flags=frozenset(flags or set()),
    )


def test_quality_judge_merge_and_bend_suspect() -> None:
    cfg = load_config("crawl_agar").pose
    judge = PoseQualityJudge(cfg, fps=30.0)

    # 先喂 12 帧稳定 OK 样本建立中位数基准
    for _ in range(12):
        p = judge.judge(_dummy_pose(length=120.0, width=7.0, area=800.0))
        assert p.quality == Quality.OK

    # 插入面积×1.6 的两虫粘连帧 -> 判为 BAD 且带 merge 标记
    p_merge = judge.judge(_dummy_pose(length=130.0, width=9.0, area=800.0 * 1.6))
    assert p_merge.quality == Quality.BAD
    assert "merge" in p_merge.flags

    # 插入弯曲导致长度缩短 30% 但面积正常的帧 -> 判为 SUSPECT 且保留 dense
    p_bend = judge.judge(_dummy_pose(length=120.0 * 0.70, width=7.0, area=800.0))
    assert p_bend.quality == Quality.SUSPECT
    assert p_bend.dense is not None

    # 圆球碎片 (length < 3 * width) -> 判为 BAD
    p_blob = judge.judge(_dummy_pose(length=18.0, width=8.0, area=800.0))
    assert p_blob.quality == Quality.BAD
