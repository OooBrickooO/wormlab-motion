import numpy as np
from worm_processing import TrackHistory

def test_s_shape():
    history = TrackHistory(max_len_frames=300, fps=10.0)
    # 模拟真实线虫全身正弦波向前爬行 (q_S 主导，3个完整周期)
    for i in range(100):
        cx = i * 2.5
        cy = 0.0
        phase = 2.0 * np.pi * (i / 28.0)
        pts = []
        for j in range(11):
            s = (5 - j) * 10.0  # 11个节点体长约 100px
            # 行波 y(s, t) = A * sin(phase - k * s)，产生反对称 S 曲率
            u = (j / 10.0) * 2.0 * np.pi
            py = cy + 12.0 * np.sin(phase - u)
            px = cx + s
            pts.append([px, py])
        history.append(pts)

    print(f"S Shape Test: C_count={history.c_count}, S_count={history.s_count}, Bi={history.is_bidirectional}, "
          f"motion={history.get_dominant_motion_type()}, act={history.get_activity_score():.2f}")
    assert history.s_count >= 2 and history.c_count == 0, f"Expected S>=2, C=0, got S={history.s_count}, C={history.c_count}"

def test_c_shape():
    history = TrackHistory(max_len_frames=300, fps=10.0)
    # 模拟液体中对称 C 形周期摆动 (Thrashing, q_C 主导)
    for i in range(100):
        cx = 100.0 + i * 0.4
        cy = 100.0
        bend = 0.95 * np.sin(2.0 * np.pi * (i / 10.0))  # 1.0 Hz C型摆动
        pts = []
        for j in range(11):
            u = (j - 5) / 5.0  # [-1, 1]
            # 全躯干同向抛物线/圆弧弯曲
            px = cx + u * 45.0 * (1.0 - 0.35 * (bend ** 2))
            py = cy + bend * 28.0 * (1.0 - u ** 2)
            pts.append([px, py])
        history.append(pts)

    print(f"C Shape Test: C_count={history.c_count}, S_count={history.s_count}, Bi={history.is_bidirectional}, "
          f"motion={history.get_dominant_motion_type()}, act={history.get_activity_score():.2f}")
    assert history.c_count >= 5 and history.s_count == 0, f"Expected C>=5, S=0, got C={history.c_count}, S={history.s_count}"

def test_static_frozen_s():
    history = TrackHistory(max_len_frames=300, fps=10.0)
    # 模拟保持 S 姿态但静止不动（仅有微小边缘噪声）的线虫 -> 应判定为 static, C=0, S=0
    rng = np.random.default_rng(42)
    for i in range(80):
        pts = []
        for j in range(11):
            u = (j / 10.0) * 2.0 * np.pi
            px = 200.0 + (5 - j) * 10.0 + rng.normal(0, 0.3)
            py = 200.0 + 12.0 * np.sin(u) + rng.normal(0, 0.3)
            pts.append([px, py])
        history.append(pts)

    print(f"Static Frozen-S Test: C_count={history.c_count}, S_count={history.s_count}, "
          f"motion={history.get_dominant_motion_type()}, act={history.get_activity_score():.2f}")
    assert history.c_count == 0 and history.s_count == 0 and history.get_dominant_motion_type() == 'static'

if __name__ == '__main__':
    test_s_shape()
    test_c_shape()
    test_static_frozen_s()
    print("All synthetic TrackHistory tests passed!")
