"""02 §6: 姿态质量判定器 (PoseQualityJudge) —— 解耦"无效 (BAD/MISSING)"与"异常弯曲 (SUSPECT)"。

语义约定：
- OK / SUSPECT 帧均正常参与运动学计数；
- BAD 帧视为单帧缺失，由运动学层在 gap_max_s 内线性插值，超时才中止当前未完成周期；
- 任何质量等级均不复位已完成的 S/C 计数。
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from wormlab.config import PoseCfg
from wormlab.schemas import Pose, Quality


class PoseQualityJudge:
    """按轨迹维护历史面积与体长中位数，逐帧评估姿态质量等级。"""

    def __init__(self, cfg: PoseCfg, fps: float) -> None:
        self.cfg = cfg
        win_frames = max(1, int(round(float(cfg.ref_window_s) * float(fps))))
        self.area_hist: deque[float] = deque(maxlen=win_frames)
        self.len_hist: deque[float] = deque(maxlen=win_frames)

    def judge(self, pose: Pose) -> Pose:
        if pose.dense is None:
            return replace(pose, quality=Quality.MISSING)

        flags = set(pose.flags)
        bad = False
        suspect = False

        if len(self.area_hist) >= int(self.cfg.min_ref_samples):
            area_ref = float(np.median(self.area_hist))
            len_ref = float(np.median(self.len_hist))
            if area_ref > 1e-6:
                r = float(pose.area) / area_ref
                if r > float(self.cfg.bad_area_hi):
                    bad = True
                    flags.add("merge")
                elif r < float(self.cfg.bad_area_lo):
                    bad = True
                    flags.add("clipped")
            if len_ref > 1e-6:
                lr = float(pose.length) / len_ref
                if not (float(self.cfg.suspect_len_lo) <= lr <= float(self.cfg.suspect_len_hi)):
                    suspect = True

        if "branch" in flags and float(pose.branch_ratio) > float(self.cfg.branch_bad_ratio):
            bad = True

        if float(pose.length) < float(self.cfg.min_len_over_width) * max(1.0, float(pose.width)):
            bad = True

        if flags & {"loop", "fragment", "fallback"}:
            suspect = True

        quality = Quality.BAD if bad else (Quality.SUSPECT if suspect else Quality.OK)
        if quality == Quality.OK:
            self.area_hist.append(float(pose.area))
            self.len_hist.append(float(pose.length))

        return replace(pose, quality=quality, flags=frozenset(flags))
