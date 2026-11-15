"""05 §7: 后端单进程启动入口 (run.py) —— 默认监听 0.0.0.0:5001。"""
from __future__ import annotations

import logging
import uvicorn

from server.app_factory import create_app
from wormlab.config import load_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = load_config("crawl_agar")
    _, _, asgi_app = create_app(cfg)
    uvicorn.run(asgi_app, host=cfg.server.host, port=int(cfg.server.port), reload=False)


if __name__ == "__main__":
    main()
