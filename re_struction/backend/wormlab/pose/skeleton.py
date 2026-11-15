"""02 §3: 距离变换加权骨架图构建、自接触闭环自动解环与测地最短路。"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import scipy.ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize


def _break_skeleton_loops(sk: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """若骨架中存在因 Ω 弯自接触产生的环路，自动剥离出 2-core 环并切断短桥（接触颈部）。"""
    ys, xs = np.nonzero(sk)
    n = len(xs)
    if n < 8:
        return sk

    h, w = sk.shape[:2]
    idx_grid = np.full((h, w), -1, dtype=np.int32)
    idx_grid[ys, xs] = np.arange(n, dtype=np.int32)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        y, x = int(ys[i]), int(xs[i])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    j = int(idx_grid[ny, nx])
                    if j >= 0:
                        adj[i].append(j)

    # 消除 3x3 团块中的伪三角小环：仅保留通过拓扑剥离后周长 >= 10 的真实环
    deg = np.array([len(nb) for nb in adj], dtype=np.int32)
    in_core = np.ones(n, dtype=bool)
    q = deque(i for i in range(n) if deg[i] <= 1)
    while q:
        u = q.popleft()
        in_core[u] = False
        for v in adj[u]:
            if in_core[v]:
                deg[v] -= 1
                if deg[v] == 1:
                    q.append(v)

    core_nodes = np.nonzero(in_core)[0]
    if len(core_nodes) < 10:
        return sk

    # 找出自由端分支（非 core 节点）连接到 core 的锚点（junctions）
    junctions: list[int] = []
    for u in core_nodes:
        if any(not in_core[v] for v in adj[u]):
            junctions.append(int(u))

    sk_out = sk.copy()
    dt_nodes = dt[ys, xs]

    if len(junctions) >= 2:
        # 取距离最远或最主要的两个分支锚点 u0, u1，在 core 子图上找最短桥并将其切断
        u0 = junctions[0]
        # 找离 u0 在 core 内跳数大于 2 的另一个锚点
        u1 = junctions[-1]
        for cand in junctions[1:]:
            if abs(int(ys[cand]) - int(ys[u0])) + abs(int(xs[cand]) - int(xs[u0])) > 3:
                u1 = cand
                break

        # BFS 找 core 内从 u0 到 u1 的最短接触桥
        prev = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1, dtype=np.int32)
        bfs_q: deque[int] = deque([u0])
        dist[u0] = 0
        while bfs_q:
            cur = bfs_q.popleft()
            if cur == u1:
                break
            for nb in adj[cur]:
                if in_core[nb] and dist[nb] < 0:
                    dist[nb] = dist[cur] + 1
                    prev[nb] = cur
                    bfs_q.append(nb)

        bridge: list[int] = []
        if dist[u1] > 0:
            cur = u1
            while cur != -1:
                bridge.append(int(cur))
                cur = int(prev[cur])
            bridge.reverse()

        if 2 <= len(bridge) < len(core_nodes) // 2:
            for b_node in bridge[1:-1]:
                sk_out[ys[b_node], xs[b_node]] = False
            if len(bridge) == 2:
                # 若两个锚点直接对角/正交相邻，移除其中一个非端点侧的桥接像素
                sk_out[ys[u0], xs[u0]] = False
            return sk_out

    # 若是无尾纯圆环或单交点 8 字环：在 core 上找远离锚点且 dt 最小的像素切开（仅移除最多 2 个相邻环像素以断开 8 邻域）
    candidates = [
        int(u)
        for u in core_nodes
        if all(abs(int(ys[u]) - int(ys[j])) + abs(int(xs[u]) - int(xs[j])) > 3 for j in junctions)
    ]
    if not candidates:
        candidates = [int(u) for u in core_nodes]
    cut_node = min(candidates, key=lambda idx: float(dt_nodes[idx]))
    sk_out[ys[cut_node], xs[cut_node]] = False
    for nb in adj[cut_node]:
        if in_core[nb]:
            sk_out[ys[nb], xs[nb]] = False
            break
    return sk_out


def extract_skeleton_path(
    mask: np.ndarray,
    center_penalty: float = 2.0,
    has_loop: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """从二值掩码提取骨架测地主干路径。"""
    m = np.asarray(mask, dtype=bool)
    dt = scipy.ndimage.distance_transform_edt(m).astype(np.float32)
    sk = skeletonize(m)
    if has_loop:
        sk = _break_skeleton_loops(sk, dt)

    ys, xs = np.nonzero(sk)
    n = len(xs)
    if n < 8:
        return None, np.zeros((0,), dtype=np.float32), 0.0

    h, w = m.shape[:2]
    idx_grid = np.full((h, w), -1, dtype=np.int32)
    idx_grid[ys, xs] = np.arange(n, dtype=np.int32)

    dt_nodes = dt[ys, xs]
    dt_max = max(float(dt_nodes.max()), 1e-6)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    geo_list: list[np.ndarray] = []
    cost_list: list[np.ndarray] = []

    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ny = ys + dy
        nx = xs + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        if not np.any(valid):
            continue
        a_idx = np.nonzero(valid)[0].astype(np.int32)
        b_idx = idx_grid[ny[valid], nx[valid]]
        edge_ok = b_idx >= 0
        if not np.any(edge_ok):
            continue
        a_idx = a_idx[edge_ok]
        b_idx = b_idx[edge_ok]

        geo_val = math.hypot(dy, dx)
        geo_arr = np.full(len(a_idx), geo_val, dtype=np.float64)
        mean_dt = 0.5 * (dt_nodes[a_idx] + dt_nodes[b_idx])
        cost_arr = geo_arr * (1.0 + float(center_penalty) * (1.0 - mean_dt / dt_max))

        rows_list.append(a_idx)
        cols_list.append(b_idx)
        geo_list.append(geo_arr)
        cost_list.append(cost_arr)

    if not rows_list:
        return None, np.zeros((0,), dtype=np.float32), 0.0

    r_all = np.concatenate(rows_list)
    c_all = np.concatenate(cols_list)
    g_all = np.concatenate(geo_list)
    w_all = np.concatenate(cost_list)

    rows = np.concatenate([r_all, c_all])
    cols = np.concatenate([c_all, r_all])
    geos = np.concatenate([g_all, g_all])
    costs = np.concatenate([w_all, w_all])

    g_geo = csr_matrix((geos, (rows, cols)), shape=(n, n))
    g_cost = csr_matrix((costs, (rows, cols)), shape=(n, n))

    n_comp, comp_labels = connected_components(g_geo, directed=False)
    if n_comp > 1:
        counts = np.bincount(comp_labels)
        main_comp = int(np.argmax(counts))
        start_node = int(np.nonzero(comp_labels == main_comp)[0][0])
    else:
        start_node = 0

    d0 = dijkstra(g_geo, directed=False, indices=start_node)
    a_node = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    da = dijkstra(g_geo, directed=False, indices=a_node)
    b_node = int(np.argmax(np.where(np.isfinite(da), da, -1.0)))

    if a_node == b_node or not np.isfinite(da[b_node]):
        return None, np.zeros((0,), dtype=np.float32), 0.0

    _, predecessors = dijkstra(g_cost, directed=False, indices=a_node, return_predecessors=True)

    path: list[int] = []
    cur = b_node
    visited_guard = 0
    while cur != a_node and cur >= 0 and visited_guard <= n:
        path.append(int(cur))
        cur = int(predecessors[cur])
        visited_guard += 1
    if cur != a_node:
        return None, np.zeros((0,), dtype=np.float32), 0.0
    path.append(int(a_node))
    path.reverse()

    if len(path) < 4:
        return None, np.zeros((0,), dtype=np.float32), 0.0

    path_arr = np.asarray(path, dtype=np.int32)
    path_xy = np.stack([xs[path_arr], ys[path_arr]], axis=1).astype(np.float64)
    path_dt = dt_nodes[path_arr].astype(np.float32)

    on_path = np.zeros_like(sk, dtype=bool)
    on_path[ys[path_arr], xs[path_arr]] = True
    near_path = scipy.ndimage.binary_dilation(on_path, iterations=2)
    branch_px = int(np.logical_and(sk, ~near_path).sum())
    branch_ratio = float(branch_px) / float(max(1, len(path_arr)))

    return path_xy, path_dt, branch_ratio
