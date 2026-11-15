"""04 §2: 严格受限与无限制双模式 ID 池 (IdPool)。

行为与旧 EnhancedWormTracker._alloc_id / _release_id / set_max_worm_id 保持 1:1 一致。
"""
from __future__ import annotations


class IdPool:
    """管理线虫显示序号 track_id 的分配、回收与上限动态切换。"""

    def __init__(self, max_id: int | None = None) -> None:
        self.max_id: int | None = int(max_id) if (max_id is not None and int(max_id) > 0) else None
        self.pool: list[int] | None = (
            list(range(1, self.max_id + 1)) if self.max_id is not None else None
        )
        self.next_id: int = max(self.max_id or 0, 0) + 1

    def alloc(self) -> int | None:
        """分配一个可用 ID；在受限模式且池已空时严格返回 None（绝不溢出分配）。"""
        if self.pool:
            return int(self.pool.pop(0))
        if self.max_id is not None:
            return None
        new_id = int(self.next_id)
        self.next_id += 1
        return new_id

    def release(self, tid: int) -> None:
        """在受限模式下回收位于 1..max_id 内的 ID 并保持升序；无限制模式下不回收复用。"""
        t_int = int(tid)
        if self.max_id is not None and 1 <= t_int <= self.max_id:
            if self.pool is not None and t_int not in self.pool:
                self.pool.append(t_int)
                self.pool.sort()

    def set_max(self, max_id: int | None, active_ids: set[int]) -> None:
        """运行中动态调整虫数上限（支持从受限切回无限制或修改上限）。"""
        norm_max = int(max_id) if (max_id is not None and int(max_id) > 0) else None
        self.max_id = norm_max
        if norm_max is not None:
            self.pool = [i for i in range(1, norm_max + 1) if i not in active_ids]
            self.pool.sort()
            self.next_id = max(self.next_id, norm_max + 1)
        else:
            self.pool = None
            if active_ids:
                self.next_id = max(self.next_id, max(active_ids) + 1)
