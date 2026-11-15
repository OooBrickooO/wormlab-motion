"""参考原型（非最终实现）：验证 03_SPEC_kinematics 中 S 计数逻辑与合成场景。

用途：给实现者一份"可运行的数学参照"；最终代码必须按规格拆到 wormlab/kinematics/* 并带单测。
运行：python proto_kinematics.py   → 打印各合成场景的 s_count / 拒绝原因。
"""
import numpy as np

N = 32  # 切线段数（33 点）
S_MID = (np.arange(N) + 0.5) / N
CFG = dict(k_body=1.0, win_s=3.0, r_min=0.25, eps_min=0.40, post_ratio_min=0.35, amp_post_min=0.10,
           dir_lock=0.8, reverse_phase_eps=0.3, reverse_disp_ratio=0.03, min_disp_ratio=0.10, gap_hold_s=0.3,
           count_backward=False)


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------- 合成 ----------------
def build_dense(theta, L, head_pos):
    """theta: (32,) 沿 head→tail 的切线角；返回 (33,2)，dense[0]=head。"""
    seg = L / N
    d = np.stack([np.cos(theta), np.sin(theta)], 1) * seg
    pts = np.vstack([[0, 0], np.cumsum(d, 0)])
    return pts + head_pos


def synth(kind, dur=20.0, fps=30, L=120.0, A=0.8, f=0.5, disp=0.4, heading=0.0):
    T = int(dur * fps); t = np.arange(T) / fps
    th0 = heading + np.pi                       # 切线(头→尾)方向 = 前进方向 + π
    fwd = np.array([np.cos(heading), np.sin(heading)])
    phase = np.zeros(T); pos = np.zeros((T, 2)); theta = np.zeros((T, N))
    if kind in ("crawl_fwd", "treadmill", "wrong_dir", "wobble", "standing", "head_wiggle"):
        for i, tt in enumerate(t):
            if kind == "wobble":
                ph = 0.4 * 2 * np.pi * np.sin(2 * np.pi * 0.2 * tt)
            else:
                ph = 2 * np.pi * f * tt
            if kind == "standing":
                th = A * np.sin(2 * np.pi * S_MID) * np.cos(2 * np.pi * f * tt)
            elif kind == "head_wiggle":
                th = np.where(S_MID < 0.25, 0.6 * np.sin(2 * np.pi * 1.5 * tt) * np.sin(np.pi * S_MID / 0.25), 0.0)
            else:
                th = A * np.sin(2 * np.pi * S_MID - ph)
            theta[i] = th0 + th
        v = {"crawl_fwd": disp, "treadmill": 0.0, "wrong_dir": -disp, "wobble": disp, "standing": disp, "head_wiggle": disp}[kind]
        pos = (t[:, None] * (v * L * f)) * fwd[None, :]
    elif kind == "crawl_bwd":
        for i, tt in enumerate(t):
            theta[i] = th0 + A * np.sin(2 * np.pi * S_MID + 2 * np.pi * f * tt)
        pos = -(t[:, None] * (disp * L * f)) * fwd[None, :]
    elif kind == "fwd_then_back":
        half = dur / 2; x = np.zeros(2)
        for i, tt in enumerate(t):
            if tt < half:
                theta[i] = th0 + A * np.sin(2 * np.pi * S_MID - 2 * np.pi * f * tt); x = x + fwd * disp * L * f / fps
            else:
                tb = tt - half
                theta[i] = th0 + A * np.sin(2 * np.pi * S_MID - 2 * np.pi * f * half + 2 * np.pi * f * tb)
                x = x - fwd * disp * L * f / fps
            pos[i] = x
    elif kind == "static_noise":
        for i in range(T):
            theta[i] = th0
    dense = np.stack([build_dense(theta[i], L, pos[i]) for i in range(T)])
    if kind == "static_noise":
        dense = dense + np.random.default_rng(0).normal(0, 0.5, dense.shape)
    return dense, fps


# ---------------- 特征 / 门限 / run ----------------
def feats(dense):
    d = np.diff(dense, axis=0)
    L = np.linalg.norm(d, axis=1).sum()
    theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    th = theta - theta.mean()
    a1 = 2 * np.mean(th * np.cos(2 * np.pi * CFG["k_body"] * S_MID))
    a2 = 2 * np.mean(th * np.sin(2 * np.pi * CFG["k_body"] * S_MID))
    u = dense[0] - dense[-1]; u = u / (np.linalg.norm(u) + 1e-9)
    return dict(L=L, theta=theta, a1=a1, a2=a2, u=u, center=dense.mean(0))


def gate(F, i, half):
    lo, hi = max(0, i - half), min(len(F), i + half + 1)
    W = F[lo:hi]
    if len(W) < 0.6 * (2 * half + 1):
        return False
    A = np.array([[f["a1"], f["a2"]] for f in W])
    r_med = np.median(np.hypot(A[:, 0], A[:, 1]))
    ev = np.linalg.eigvalsh(np.cov(A.T)); eps = np.sqrt(max(ev[0], 0) / max(ev[1], 1e-12))
    TH = np.stack([f["theta"] - np.median(f["theta"]) for f in W]); TH = TH - TH.mean(0)
    amp = lambda X: np.sqrt(np.mean(np.var(X, axis=0)))
    a_ant, a_post = amp(TH[:, :16]), amp(TH[:, 16:])
    return bool(r_med >= CFG["r_min"] and eps >= CFG["eps_min"]
                and a_post >= CFG["post_ratio_min"] * a_ant and a_post >= CFG["amp_post_min"])


def run_s(dense, fps):
    F = [feats(d) for d in dense]
    half = int(round(CFG["win_s"] * fps / 2)); gap_hold = int(round(CFG["gap_hold_s"] * fps))
    st = dict(active=False)
    out = dict(s=0, dir={1: 0, -1: 0}, rej={"no_disp": 0, "wrong_dir": 0, "backward": 0}, abort=0)

    def start(i):
        st.update(active=True, direction=0, prog=0.0, extreme=0.0, anchor=0.0, phi_prev=np.arctan2(F[i]["a2"], F[i]["a1"]),
                  a_center=F[i]["center"].copy(), axis=np.zeros(2), gap=0)

    for i, f in enumerate(F):
        ok = gate(F, i, half)
        if not ok:
            if st["active"]:
                st["gap"] += 1
                if st["gap"] > gap_hold:
                    st["active"] = False; out["abort"] += 1
            continue
        phi = np.arctan2(f["a2"], f["a1"])
        if not st["active"]:
            start(i); continue
        if st["gap"] > 0:
            st["phi_prev"] = phi; st["gap"] = 0
        dphi = wrap_pi(phi - st["phi_prev"]); st["phi_prev"] = phi; st["axis"] += f["u"]
        if st["direction"] == 0:
            st["prog"] += dphi
            if abs(st["prog"]) >= CFG["dir_lock"]:
                st["direction"] = int(np.sign(st["prog"])); st["prog"] = abs(st["prog"]); st["extreme"] = st["prog"]; st["anchor"] = 0.0
            continue
        st["prog"] += st["direction"] * dphi
        st["extreme"] = max(st["extreme"], st["prog"])
        Lref = np.median([g["L"] for g in F[max(0, i - 60):i + 1]])
        # 反向运动（甲方定义）：本周期相位进度回退到为负 且 质心沿体轴实际后退 → 作废未完成的半截周期，已完成的保留
        cyc_prog = st["prog"] - st["anchor"]
        if cyc_prog < -CFG["reverse_phase_eps"]:
            ax0 = st["axis"] / (np.linalg.norm(st["axis"]) + 1e-9)
            p0 = float(np.dot(f["center"] - st["a_center"], ax0))
            if np.sign(p0) == -st["direction"] and abs(p0) >= CFG["reverse_disp_ratio"] * Lref:
                st["active"] = False; out["abort"] += 1; start(i); continue
        while st["prog"] - st["anchor"] >= 2 * np.pi:
            ax = st["axis"] / (np.linalg.norm(st["axis"]) + 1e-9)
            p = float(np.dot(f["center"] - st["a_center"], ax))
            ok_dir = np.sign(p) == st["direction"]; ok_disp = abs(p) >= CFG["min_disp_ratio"] * Lref
            if ok_dir and ok_disp and (st["direction"] > 0 or CFG["count_backward"]):
                out["s"] += 1; out["dir"][st["direction"]] += 1
            elif ok_dir and ok_disp:
                out["rej"]["backward"] += 1            # 向后 S：甲方要求完全不统计
            else:
                out["rej"]["wrong_dir" if not ok_dir else "no_disp"] += 1
            st["anchor"] += 2 * np.pi; st["a_center"] = f["center"].copy(); st["axis"][:] = 0
    return out


if __name__ == "__main__":
    expect = {"crawl_fwd": "s=10 dir+", "crawl_bwd": "s=0 (全部为 backward 拒绝)", "treadmill": "s=0", "wrong_dir": "s=0", "head_wiggle": "s=0",
              "wobble": "s=0", "standing": "s=0", "fwd_then_back": "s=3 (仅前向；后向不计)", "static_noise": "s=0"}
    expect["fwd_then_back"] = "s=3 (仅前向；后向不计)"
    for k, e in expect.items():
        for fps in (30, 10):
            d, fp = synth(k, fps=fps, dur=(14.0 if k == "fwd_then_back" else 20.0))
            r = run_s(d, fp)
            print(f"{k:14s} fps={fps:2d} -> s={r['s']:2d} dir={r['dir']} rej={r['rej']} abort={r['abort']}   expect {e}")
