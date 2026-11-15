"""T01: 环境与 GPU / YOLO 模型自检脚本。

检查项：
1. Python 版本与核心依赖导入
2. PyTorch 版本与 CUDA 可用性（若当前环境为 CPU 构建则打印明确警告或按 --allow-cpu 参数放行）
3. 加载 origin.pt 并在 imgsz=416 下完成 1 帧推理自检
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("check_env")


def resolve_origin_model(re_root: Path) -> Path:
    """按优先级解析 origin.pt 路径。"""
    candidates = [
        re_root / "backend" / "models" / "origin.pt",
        re_root.parent / "backend" / "models" / "origin.pt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"未找到 origin.pt，尝试路径: {candidates}")


def check_nvidia_smi() -> str:
    """读取宿主 nvidia-smi 摘要信息。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        return out
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi 不可用 ({exc})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 re_struction 运行环境与模型推理")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="允许在 CPU 版 PyTorch 下通过自检（仅当未安装 CUDA 轮子时回退）",
    )
    args = parser.parse_args(argv)

    re_root = Path(__file__).resolve().parents[1]
    log.info("Python: %s (%s)", sys.version.split()[0], sys.executable)
    log.info("PyTorch: %s | CUDA build: %s", torch.__version__, torch.version.cuda)
    log.info("Host GPU (nvidia-smi): %s", check_nvidia_smi())

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        dev_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        log.info("CUDA 可用: True | Device 0: %s | Compute Capability: %d.%d", dev_name, cap[0], cap[1])
        device = "cuda"
    else:
        if not args.allow_cpu:
            log.error("torch.cuda.is_available() == False！请安装 CUDA 12.8+ 版 PyTorch，或传 --allow-cpu。")
            return 1
        log.warning("torch.cuda.is_available() == False，因指定 --allow-cpu 回退使用 CPU 自检。")
        device = "cpu"

    model_path = resolve_origin_model(re_root)
    log.info("加载模型: %s (device=%s, imgsz=416)", model_path, device)
    model = YOLO(str(model_path))
    model.to(device)

    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    results = model([dummy_frame], verbose=False, device=device, imgsz=416, retina_masks=True)
    assert len(results) == 1, "推理返回结果长度异常"
    log.info("单帧推理自检成功 (imgsz=416, device=%s)。", device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
