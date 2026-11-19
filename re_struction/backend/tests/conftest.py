"""pytest 共享配置（backend/tests）。

Windows + DSH 沙箱环境兼容性修复：
CPython 在 Windows 上执行 ``os.mkdir(path, mode)`` 时会按 mode 调用 ``_wchmod`` 重写
目录 DACL，丢弃父目录继承的沙箱访问授权 ACE。pytest 创建 tmp_path / basetemp /
numbered dir 时固定使用 ``mode=0o700``，于是这些目录内后续创建文件与子目录会被
拒绝（WinError 5 / Permission denied），导致所有依赖 tmp_path 的单测无法运行。

这里仅在 Windows 下把测试进程内的 ``os.mkdir`` mode 归一化为 0o777，使 pytest
临时目录正常继承父目录 ACL。该补丁只在 pytest 测试进程内生效，不影响生产代码；
在非 Windows 平台上完全不生效。
"""
from __future__ import annotations

import os

if os.name == "nt":  # pragma: no cover - 平台特定分支
    _orig_mkdir = os.mkdir

    def _sandbox_safe_mkdir(path, mode=0o777, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return _orig_mkdir(path, 0o777, *args, **kwargs)

    os.mkdir = _sandbox_safe_mkdir  # type: ignore[assignment]
