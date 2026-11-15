"""T17 单测：严格覆盖 04 §2 用例表全部 8 项行为。"""
from __future__ import annotations

from wormlab.tracking.id_pool import IdPool


def test_id_pool_all_table_cases() -> None:
    # 1. max_id=3: 连续 alloc 4 次 -> 1, 2, 3, None
    pool = IdPool(max_id=3)
    assert [pool.alloc() for _ in range(4)] == [1, 2, 3, None]
    # 8. 受限模式下 next_id 起点 max_id+1，且受限模式永不使用
    assert pool.next_id == 4

    # 2. release(2) 后 alloc -> 返回 2
    pool.release(2)
    assert pool.alloc() == 2

    # 3. release(7) (不在 1..max) -> 忽略，池不变
    pool.release(7)
    assert pool.pool == []

    # 4. 重复 release(2) -> 池中只有一个 2
    pool.release(2)
    pool.release(2)
    assert pool.pool == [2]

    # 5. max_id=None: alloc × 3 -> 1, 2, 3 单调递增
    u_pool = IdPool(max_id=None)
    assert [u_pool.alloc() for _ in range(3)] == [1, 2, 3]

    # 6. set_max(5, active={1, 2}) -> 池=[3, 4, 5], next_id=max(next_id, 6)
    u_pool.set_max(5, active_ids={1, 2})
    assert u_pool.pool == [3, 4, 5]
    assert u_pool.next_id == 6

    # 7. set_max(None, ...) -> 池清空，转无限制模式；已存在 ID 不回收
    u_pool.set_max(None, active_ids={1, 2})
    assert u_pool.pool is None
    assert u_pool.alloc() == 6
