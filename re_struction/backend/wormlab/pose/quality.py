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
        self.min_len_over_width = float(getattr(cfg, "min_len_over_width", 2.2))
        # 光照/焦距持续漂移保护：连续仅因面积比判 BAD 但形状健康的帧收集于此
        self._drift_win = max(1, int(round(1.5 * float(fps))))
        self._drift_areas: deque[float] = deque(maxlen=self._drift_win)
        self._drift_run: int = 0

    def judge(self, pose: Pose) -> Pose:
        if pose.dense is None:
            return replace(pose, quality=Quality.MISSING)

        flags = set(pose.flags)
        bad = False
        suspect = False
        bad_area_only = False

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

        if float(pose.length) < self.min_len_over_width * max(1e-6, float(pose.width)):
            bad = True

        if flags & {"loop", "fragment", "fallback"}:
            suspect = True

        quality = Quality.BAD if bad else (Quality.SUSPECT if suspect else Quality.OK)

        if quality == Quality.OK:
            self.area_hist.append(float(pose.area))
            self.len_hist.append(float(pose.length))
            self._drift_run = 0
        else:
            self._maybe_reset_drifted_area_hist(quality, flags, pose)

        return replace(pose, quality=quality, flags=frozenset(flags))

    def _maybe_reset_drifted_area_hist(self, quality: Quality, flags: set[str], pose: Pose) -> None:
        """持续光照/焦距漂移会导致 area_hist 参考基准永久锁死：检测到连续形状健康但面积比出界的
        BAD 帧后，用漂移期自身面积平滑重置参考（单帧双虫粘连不会被误重置——连续时长与形状一致性双门槛）。"""
        bad_area_only = (
            quality == Quality.BAD
            and bool(flags & {"merge", "clipped"})
            and not (flags & {"branch", "fragment"})
            and float(pose.length) >= self.min_len_over_width * max(1e-6, float(pose.width))
        )
        if not bad_area_only:
            self._drift_run = 0
            return
        self._drift_areas.append(float(pose.area))
        self._drift_run += 1
        if self._drift_run >= self._drift_win and len(self._drift_areas) >= max(4, self._drift_win // 2):
            arr = np.asarray(self._drift_areas, dtype=np.float64)
            med = float(np.median(arr))
            if med > 1e-6 and float(np.std(arr)) / med < 0.15:
                self.area_hist.clear()
                for a in arr:
                    self.area_hist.append(float(a))
                self._drift_areas.clear()
                self._drift_run = 0
