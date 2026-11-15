"""03 §1, §9: 延迟因果运动学引擎 (KinematicsEngine) —— 统一实时流式 push 与批量 run。"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Sequence

import numpy as np

from wormlab.config import KinCfg
from wormlab.schemas import Behavior, Pose, Quality

from .behavior import behavior_to_legacy_motion_type, classify_behavior
from .features import Feat, compute_features
from .metrics import LiveCounts, PerFrame, TrackReport, compute_spatial_stats
from .modes import modal_phase, project_trig_modes
from .s_runs import SRunCounter
from .swing import SwingCounter
from .turns import TurnCounter
from .wave import GateResult, wave_gate


class KinematicsEngine:
    """单轨迹延迟因果运动学引擎。

    保证逐帧 push() + finalize() 与一次性 run() 产出逐位完全一致的 TrackReport 与 PerFrame。
    """

    def __init__(
        self,
        cfg: KinCfg,
        fps: float,
        c_unit: str = "cycle",
        frame_height: int = 1080,
    ) -> None:
        self.cfg = cfg
        self.fps = float(fps)
        self.c_unit = c_unit if c_unit in ("cycle", "bend") else "cycle"
        self.frame_height = int(frame_height)

        self.h_smooth = max(0, int(round(float(cfg.smooth_s) * self.fps / 2.0)))
        self.max_gap_frames = max(1, int(round(float(cfg.gap_max_s) * self.fps)))
        self.h_wave = max(1, int(round(float(cfg.wave.win_s) * self.fps / 2.0)))
        self.h_speed = max(1, int(round(0.5 * self.fps)))
        self.ref_win_frames = max(1, int(round(float(cfg.ref_window_s) * self.fps)))
        self.gap_hold_frames = max(1, int(round(float(cfg.s_run.gap_hold_s) * self.fps)))
        self.act_win_frames = max(1, int(round(1.5 * self.fps)))

        # 原始输入序列
        self.raw_poses: list[Pose | None] = []
        self.valid_dense: list[np.ndarray | None] = []
        self.qualities: list[Quality] = []

        # 阶段 1: 时间平滑 + 特征缓存
        self.feats: list[Feat | None] = []
        self._s_feats: list[Feat | None] = []
        self._s_raw_a: list[tuple[float, float] | None] = []
        self._next_feat_idx: int = 0
        self._dc_win_frames = max(20, int(round(5.0 * self.fps)))
        self._dc_min_samples = max(15, int(round(2.5 * self.fps)))

        # 阶段 2: 门限 + 计数器步进
        self.s = SRunCounter(cfg.s_run, fps=self.fps)
        self.swing = SwingCounter(cfg.swing, fps=self.fps, gap_hold_s=cfg.s_run.gap_hold_s)
        self.turn = TurnCounter(cfg.turn, fps=self.fps)
        self._occ_hold_frames = max(self.gap_hold_frames, int(round(6.0 * self.fps)))
        slow_hold_s = float(getattr(cfg.s_run, "slowdown_hold_s", 2.5))
        self._slowdown_hold_frames = max(self.gap_hold_frames, int(round(slow_hold_s * self.fps)))
        self._occ_run: int = 0
        self._pause_run: int = 0
        self.per_frame: list[PerFrame] = []
        self.eps_history: list[float] = []
        self._next_step_idx: int = 0

    def _select_c_total(self, c_cycles: int, c_bends: int, c_turn: int) -> int:
        base = c_bends if self.c_unit == "bend" else c_cycles
        return int(base + c_turn)

    def _interp_dense_at(self, idx: int) -> np.ndarray | None:
        """获取第 idx 帧的中心线；若缺失但前后缺失间隙 <= max_gap_frames 则线性插值。"""
        d = self.valid_dense[idx]
        if d is not None:
            return d

        n = len(self.valid_dense)
        prev_i: int | None = None
        for j in range(idx - 1, max(-1, idx - self.max_gap_frames - 1), -1):
            if self.valid_dense[j] is not None:
                prev_i = j
                break
        if prev_i is None:
            return None

        next_i: int | None = None
        for j in range(idx + 1, min(n, idx + self.max_gap_frames + 1)):
            if self.valid_dense[j] is not None:
                next_i = j
                break
        if next_i is None or (next_i - prev_i - 1) > self.max_gap_frames:
            return None

        alpha = float(idx - prev_i) / float(next_i - prev_i)
        d0 = self.valid_dense[prev_i]
        d1 = self.valid_dense[next_i]
        assert d0 is not None and d1 is not None
        return ((1.0 - alpha) * d0 + alpha * d1).astype(np.float32)

    def _smoothed_dense(self, i: int) -> np.ndarray | None:
        """计算以 i 为中心 ±h_smooth 窗口的时间平滑中心线。"""
        center_d = self._interp_dense_at(i)
        if center_d is None:
            return None
        if self.h_smooth <= 0:
            return center_d

        n = len(self.valid_dense)
        lo = max(0, i - self.h_smooth)
        hi = min(n, i + self.h_smooth + 1)
        samples: list[np.ndarray] = []
        for k in range(lo, hi):
            dk = self._interp_dense_at(k)
            if dk is not None:
                samples.append(dk)
        if not samples:
            return center_d
        return np.mean(samples, axis=0).astype(np.float32)

    def _make_s_feat(self, feat: Feat | None) -> Feat | None:
        """为 S 波计数按 s_run.k_body 投影并去除慢时变静止弯曲偏移。"""
        if feat is None or not self.cfg.s_run.enabled:
            self._s_raw_a.append(None)
            return feat
        s_k = float(self.cfg.s_run.k_body)
        if abs(s_k - float(self.cfg.k_body)) > 1e-6:
            raw_a1, raw_a2 = project_trig_modes(feat.th, k_body=s_k)
        else:
            raw_a1, raw_a2 = float(feat.a1), float(feat.a2)
        self._s_raw_a.append((raw_a1, raw_a2))

        lo = max(0, len(self._s_raw_a) - self._dc_win_frames)
        vals = [p for p in self._s_raw_a[lo:] if p is not None]
        if len(vals) >= self._dc_min_samples:
            arr = np.asarray(vals, dtype=np.float64)
            m1, m2 = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
            s1, s2 = float(np.std(arr[:, 0])), float(np.std(arr[:, 1]))
            norm_s = math.hypot(s1, s2)
            if min(s1, s2) >= 0.16 and norm_s >= 0.35 and math.hypot(m1, m2) > 0.45 * norm_s:
                return replace(feat, a1=float(raw_a1 - m1), a2=float(raw_a2 - m2))
        if abs(s_k - float(self.cfg.k_body)) > 1e-6:
            return replace(feat, a1=float(raw_a1), a2=float(raw_a2))
        return feat

    def _compute_feat_at(self, i: int) -> None:
        dense_s = self._smoothed_dense(i)
        feat = compute_features(dense_s, self.cfg) if dense_s is not None else None
        self.feats.append(feat)
        self._s_feats.append(self._make_s_feat(feat))

    def _L_ref(self, i: int) -> float:
        lo = max(0, i - self.ref_win_frames)
        ls = [self.feats[k].L for k in range(lo, i + 1) if self.feats[k] is not None]
        return float(np.median(ls)) if ls else 120.0

    def _bad_too_long(self, i: int) -> bool:
        if i < self.gap_hold_frames:
            return False
        for k in range(i - self.gap_hold_frames, i + 1):
            if self.feats[k] is not None:
                return False
        return True

    def _speed_bls(self, i: int, L_ref: float) -> float:
        f_cur = self.feats[i]
        if f_cur is None or L_ref <= 1e-6:
            return 0.0
        lo = max(0, i - self.h_speed)
        hi = min(len(self.feats) - 1, i + self.h_speed)
        while lo < i and self.feats[lo] is None:
            lo += 1
        while hi > i and self.feats[hi] is None:
            hi -= 1
        if hi <= lo or self.feats[lo] is None or self.feats[hi] is None:
            return 0.0
        dt_s = float(hi - lo) / self.fps
        disp = float(np.linalg.norm(self.feats[hi].center - self.feats[lo].center))
        return disp / max(1e-3, dt_s) / L_ref

    def _step_frame(self, i: int) -> None:
        feat = self.feats[i]
        s_feat = self._s_feats[i] if self.cfg.s_run.enabled else feat
        lo = max(0, i - self.h_wave)
        hi = min(len(self.feats), i + self.h_wave + 1)
        win_feats = self._s_feats[lo:hi] if self.cfg.s_run.enabled else self.feats[lo:hi]
        # 使用窗口实际跨度作为边缘帧的有效率分母，使首尾帧与 proto_kinematics 保持一致
        gate = wave_gate(win_feats, self.cfg.wave, expected_len=len(win_feats)) if s_feat is not None else GateResult.off()
        if feat is not None and gate.r_med >= self.cfg.wave.r_min:
            self.eps_history.append(gate.eps)

        L_ref = self._L_ref(i)

        if self.cfg.turn.enabled:
            self.turn.step(i, feat.R if feat is not None else 0.0, valid=(feat is not None))

        if self.cfg.s_run.enabled:
            if self.turn.in_curl:
                if (i - self.turn.curl_start) >= self.turn.min_dwell_frames:
                    self.s.force_abort("curl_or_bad")
                    self._occ_run = 0
                    self._pause_run = 0
                else:
                    self.s.freeze()
            elif s_feat is None:
                self._occ_run += 1
                if self._occ_run > self._occ_hold_frames:
                    self.s.force_abort("bad_timeout")
                    self._pause_run = 0
                else:
                    self.s.freeze()
            elif not gate.ok:
                self._occ_run = 0
                self._pause_run += 1
                if self._pause_run > self._slowdown_hold_frames:
                    self.s.force_abort("gate_lost")
                else:
                    self.s.freeze()
            else:
                self._occ_run = 0
                self._pause_run = 0
                phi = modal_phase(s_feat.a1, s_feat.a2)
                self.s.step(
                    i=i,
                    gate_ok=True,
                    phi=phi,
                    center=s_feat.center,
                    u=s_feat.u,
                    L_ref=L_ref,
                )

        if self.cfg.swing.enabled:
            self.swing.step(i, feat.c if feat is not None else 0.0, valid=(feat is not None))

        speed_bls = self._speed_bls(i, L_ref)
        beh = classify_behavior(
            frame_idx=i,
            feat=feat,
            gate=gate,
            s_active=self.s.active,
            in_curl=self.turn.in_curl,
            swing_last_high=self.swing.last_high_frame,
            swing_last_flip=self.swing.last_flip,
            speed_bls=speed_bls,
            cfg=self.cfg,
            fps=self.fps,
        )

        # 计算最近 1.5s 活性分
        prev_behs = [pf.behavior for pf in self.per_frame[max(0, i - self.act_win_frames + 1) : i]] + [beh]
        active_set = {Behavior.CRAWL, Behavior.THRASH, Behavior.TURN, Behavior.MOVING}
        act_score = float(sum(1 for b in prev_behs if b in active_set)) / float(max(1, len(prev_behs)))

        c_cyc = self.swing.cycles
        c_bnd = self.swing.bends
        c_trn = self.turn.count
        c_tot = self._select_c_total(c_cyc, c_bnd, c_trn)

        obs_frames = sum(1 for k in range(i + 1) if self.feats[k] is not None)
        obs_sec = float(obs_frames) / self.fps
        avg_freq = float(self.s.s_count + c_tot) / obs_sec if obs_sec >= 0.5 else 0.0

        # 实时频率 (近 3s 内计数增量)
        win_3s = max(1, int(round(3.0 * self.fps)))
        if i >= int(round(1.0 * self.fps)) and len(self.per_frame) > 0:
            ref_pf = self.per_frame[max(0, i - win_3s)]
            dt_live = float(i - ref_pf.frame_idx + 1) / self.fps
            ds_cnt = (self.s.s_count - ref_pf.s_count) + (c_tot - ref_pf.c_count)
            live_freq = max(0.0, float(ds_cnt) / max(0.5, dt_live))
        else:
            live_freq = avg_freq

        phi_val = modal_phase(feat.a1, feat.a2) if feat is not None else None
        self.per_frame.append(
            PerFrame(
                frame_idx=i,
                behavior=beh,
                s_count=self.s.s_count,
                c_cycles=c_cyc,
                c_bends=c_bnd,
                c_turn=c_trn,
                c_count=c_tot,
                wave_ok=gate.ok,
                phi=phi_val,
                c=feat.c if feat is not None else 0.0,
                R=feat.R if feat is not None else 0.0,
                quality=self.qualities[i],
                activity_score=round(act_score, 4),
                live_freq=round(live_freq, 4),
                avg_freq=round(avg_freq, 4),
                bends_per_min=round(avg_freq * 60.0, 2),
            )
        )

    def _append_one_pose(self, pose: Pose | None) -> None:
        self.raw_poses.append(pose)
        if pose is None or pose.dense is None:
            self.valid_dense.append(None)
            self.qualities.append(Quality.MISSING if pose is None else pose.quality)
        elif pose.quality == Quality.BAD:
            self.valid_dense.append(None)
            self.qualities.append(Quality.BAD)
        else:
            self.valid_dense.append(pose.dense)
            self.qualities.append(pose.quality)

        n_raw = len(self.raw_poses)
        feat_lookahead = self.h_smooth + self.max_gap_frames
        while (n_raw - 1) >= (self._next_feat_idx + feat_lookahead):
            self._compute_feat_at(self._next_feat_idx)
            self._next_feat_idx += 1

        step_lookahead = max(self.h_wave, self.h_speed)
        n_feats = len(self.feats)
        while (n_feats - 1) >= (self._next_step_idx + step_lookahead):
            self._step_frame(self._next_step_idx)
            self._next_step_idx += 1

    def push(self, frame_idx: int, pose: Pose | None) -> None:
        """推入单帧姿态（流式因果延迟处理）。若 frame_idx 跳跃则自动填充缺失帧 None。"""
        f_i = int(frame_idx)
        expected_idx = len(self.raw_poses)
        if f_i < expected_idx:
            raise ValueError(f"frame_idx 必须单调递增，期望 >= {expected_idx}，收到 {f_i}")
        while len(self.raw_poses) < f_i:
            self._append_one_pose(None)
        self._append_one_pose(pose)

    def snapshot(self) -> LiveCounts:
        """读取目前已处理帧的最新实时计数快照（纯只读，不影响后续 finalize）。"""
        if not self.per_frame:
            return LiveCounts(
                s_count=0,
                c_cycles=0,
                c_bends=0,
                c_turn_count=0,
                c_count=0,
                c_unit=self.c_unit,
                behavior=Behavior.STATIC,
                motion_type="static",
                activity_score=0.0,
                is_static=True,
                live_freq=0.0,
                avg_freq=0.0,
                cycles=0,
                swings=0,
                bends_per_min=0.0,
                is_bidirectional=False,
            )
        last = self.per_frame[-1]
        mtype = behavior_to_legacy_motion_type(last.behavior)
        cycles_tot = last.s_count + last.c_count
        return LiveCounts(
            s_count=last.s_count,
            c_cycles=last.c_cycles,
            c_bends=last.c_bends,
            c_turn_count=last.c_turn,
            c_count=last.c_count,
            c_unit=self.c_unit,
            behavior=last.behavior,
            motion_type=mtype,
            activity_score=last.activity_score,
            is_static=(last.behavior == Behavior.STATIC),
            live_freq=last.live_freq,
            avg_freq=last.avg_freq,
            cycles=cycles_tot,
            swings=last.c_bends if last.c_bends > 0 else cycles_tot,
            bends_per_min=last.bends_per_min,
            is_bidirectional=False,
        )

    def finalize(self) -> tuple[TrackReport, list[PerFrame]]:
        """冲刷尾部延迟缓冲帧并生成终稿 TrackReport 与逐帧序列。"""
        n_raw = len(self.raw_poses)
        while self._next_feat_idx < n_raw:
            self._compute_feat_at(self._next_feat_idx)
            self._next_feat_idx += 1

        n_feats = len(self.feats)
        while self._next_step_idx < n_feats:
            self._step_frame(self._next_step_idx)
            self._next_step_idx += 1

        c_cyc = self.swing.cycles
        c_bnd = self.swing.bends
        c_trn = self.turn.count
        c_tot = self._select_c_total(c_cyc, c_bnd, c_trn)

        mode_counts = Counter(pf.behavior.value for pf in self.per_frame)
        mode_time = {b.value: round(float(mode_counts.get(b.value, 0)) / self.fps, 3) for b in Behavior}

        valid_feats = [f for f in self.feats if f is not None]
        obs_sec = float(len(valid_feats)) / self.fps
        freq_s = float(self.s.s_count) / obs_sec if obs_sec > 1e-3 else 0.0
        freq_c = float(c_tot) / obs_sec if obs_sec > 1e-3 else 0.0
        avg_freq_leg = float(self.s.s_count + c_tot) / obs_sec if obs_sec > 1e-3 else 0.0

        active_behs = {Behavior.CRAWL, Behavior.THRASH, Behavior.TURN, Behavior.MOVING}
        n_active = sum(1 for pf in self.per_frame if pf.behavior in active_behs)
        act_score = float(n_active) / float(max(1, len(valid_feats))) if valid_feats else 0.0

        # 主导行为状态映射
        if self.s.s_count > 0:
            dom_beh = Behavior.CRAWL
        elif c_tot > 0:
            dom_beh = Behavior.THRASH if self.cfg.swing.enabled else Behavior.TURN
        elif act_score < 0.15:
            dom_beh = Behavior.STATIC
        else:
            non_inv = [pf.behavior for pf in self.per_frame if pf.behavior != Behavior.INVALID]
            dom_beh = Counter(non_inv).most_common(1)[0][0] if non_inv else Behavior.STATIC

        q_counts = Counter(self.qualities)
        centers_list = [f.center for f in valid_feats]
        spatial = compute_spatial_stats(centers_list, frame_height=self.frame_height)

        report = TrackReport(
            s_count=int(self.s.s_count),
            s_cycles_float=round(float(self.s.s_cycles_float), 3),
            c_cycles=int(c_cyc),
            c_bends=int(c_bnd),
            c_turn_count=int(c_trn),
            c_count=int(c_tot),
            c_unit=self.c_unit,
            rejected=dict(self.s.rejected),
            aborted_runs=int(self.s.aborted),
            mode_time=mode_time,
            wave_eps_median=round(float(np.median(self.eps_history)), 4) if self.eps_history else 0.0,
            observed_seconds=round(obs_sec, 3),
            freq_s_hz=round(freq_s, 4),
            freq_c_hz=round(freq_c, 4),
            avg_freq_legacy=round(avg_freq_leg, 4),
            activity_score=round(float(np.clip(act_score, 0.0, 1.0)), 4),
            motion_type_legacy=behavior_to_legacy_motion_type(dom_beh),
            n_ok=int(q_counts.get(Quality.OK, 0)),
            n_suspect=int(q_counts.get(Quality.SUSPECT, 0)),
            n_bad=int(q_counts.get(Quality.BAD, 0)),
            n_missing=int(q_counts.get(Quality.MISSING, 0)),
            spatial=spatial,
        )
        return report, self.per_frame

    @staticmethod
    def run(
        poses: Sequence[Pose | None],
        fps: float,
        cfg: KinCfg,
        c_unit: str = "cycle",
        frame_height: int = 1080,
    ) -> tuple[TrackReport, list[PerFrame]]:
        engine = KinematicsEngine(cfg=cfg, fps=fps, c_unit=c_unit, frame_height=frame_height)
        for idx, p in enumerate(poses):
            engine.push(idx, p)
        return engine.finalize()

    @staticmethod
    def run_postprocessed(
        poses: Sequence[Pose | None],
        fps: float,
        cfg: KinCfg,
        c_unit: str = "cycle",
        frame_height: int = 1080,
    ) -> tuple[TrackReport, list[PerFrame]]:
        """Pass B 终稿运动学计算：在标准 run() 基础上按配置执行短缺口桥接、自适应体波数与文末余弦补全。"""
        if not cfg.s_run.enabled:
            return KinematicsEngine.run(poses, fps=fps, cfg=cfg, c_unit=c_unit, frame_height=frame_height)

        work_poses: Sequence[Pose | None] = poses
        bridge_s = float(getattr(cfg.s_run, "bridge_gap_s", 0.0))
        if bridge_s > 0.0:
            from wormlab.pose.orientation import bridge_short_pose_gaps

            work_poses = bridge_short_pose_gaps(work_poses, fps=fps, max_gap_s=bridge_s)

        eng = KinematicsEngine(cfg=cfg, fps=fps, c_unit=c_unit, frame_height=frame_height)
        for idx, p in enumerate(work_poses):
            eng.push(idx, p)
        rep, pfs = eng.finalize()

        valid_feats = [f for f in eng.feats if f is not None]
        if getattr(cfg.s_run, "adaptive_k_body", False) and len(valid_feats) >= 6:
            base_k = float(cfg.s_run.k_body)
            k_grid = (1.0, 1.15, 1.30, 1.45, 1.60)
            k_scores = {
                k: float(np.median([math.hypot(*project_trig_modes(f.th, k_body=k)) for f in valid_feats]))
                for k in k_grid
            }
            base_score = k_scores.get(base_k)
            if base_score is None:
                base_score = float(
                    np.median([math.hypot(*project_trig_modes(f.th, k_body=base_k)) for f in valid_feats])
                )
            top_k = max(k_grid, key=lambda k: k_scores[k])
            if base_score < 0.30 and k_scores[top_k] >= 1.20 * base_score and abs(top_k - base_k) > 1e-6:
                cfg_k = replace(cfg, s_run=replace(cfg.s_run, k_body=top_k))
                eng = KinematicsEngine(cfg=cfg_k, fps=fps, c_unit=c_unit, frame_height=frame_height)
                for idx, p in enumerate(work_poses):
                    eng.push(idx, p)
                rep, pfs = eng.finalize()
                valid_feats = [f for f in eng.feats if f is not None]

        tail_frac = getattr(cfg.s_run, "eof_tail_frac", None)
        if tail_frac is not None and eng.s.active and eng.s.direction > 0 and valid_feats:
            part = (eng.s.prog - eng.s.anchor_prog) / (2.0 * math.pi)
            if part >= float(tail_frac):
                last_f = next((f for f in reversed(eng.feats) if f is not None), None)
                if last_f is not None:
                    ax = eng.s._unit(eng.s.axis_sum)
                    p_tail = float(np.dot(last_f.center - eng.s.anchor_center, ax))
                    l_m = float(np.median([f.L for f in valid_feats]))
                    if p_tail >= float(eng.cfg.s_run.min_disp_ratio) * l_m:
                        new_s = int(rep.s_count) + 1
                        obs_sec = max(1e-3, float(rep.observed_seconds))
                        c_tot = int(rep.c_count)
                        new_freq_s = float(new_s) / obs_sec
                        new_avg_leg = float(new_s + c_tot) / obs_sec
                        rep = replace(
                            rep,
                            s_count=new_s,
                            freq_s_hz=round(new_freq_s, 4),
                            avg_freq_legacy=round(new_avg_leg, 4),
                            motion_type_legacy=behavior_to_legacy_motion_type(Behavior.CRAWL),
                        )
                        if pfs:
                            pfs = list(pfs)
                            pfs[-1] = replace(pfs[-1], s_count=new_s)

        return rep, pfs
