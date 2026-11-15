"""T31 端到端冒烟：上传 Swim-4 走完整流程，记录到 docs/spec/SMOKE_RESULT.md。"""
import json, sys, time, threading
from pathlib import Path
import requests, socketio

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:5001"
video = ROOT / "referAnswer" / "Swim-4.mp4"

frames = {"n": 0}; events = []
sio = socketio.Client()
@sio.on("frame")
def _f(d): frames["n"] += 1
@sio.on("phase")
def _p(d): events.append(("phase", d.get("phase")))
@sio.on("complete")
def _c(d): events.append(("complete", None))
sio.connect(BASE)

t0 = time.time()
with open(video, "rb") as fh:
    r = requests.post(f"{BASE}/upload", files={"video": (video.name, fh, "video/mp4")},
                      data={"worm_count": "1", "profile": "swim_liquid", "c_unit": "cycle"})
r.raise_for_status()
tid = str(r.json()["task_id"])
sio.emit("join", {"task_id": tid})
last = None
while time.time() - t0 < 900:
    s = requests.get(f"{BASE}/task/{tid}/status").json()
    if s["status"] != last:
        print("status", s["status"], s.get("current_frame"), s.get("total_frames"), flush=True); last = s["status"]
    if s["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(2)
dt = time.time() - t0
sio.disconnect()
res = {"task_id": tid, "status": s["status"], "elapsed_s": round(dt, 1), "frames_via_socket": frames["n"],
       "events": events, "error": s.get("error_message"), "summary": s.get("id_pose_summary")}
chk = {}
for k in ("video_url", "log_url", "sub_videos_url", "summary_url", "csv_url"):
    u = s.get(k)
    chk[k] = (requests.get(BASE + u if u.startswith("/") else u, stream=True).status_code if u else None)
res["downloads"] = chk
res["sub_videos"] = requests.get(f"{BASE}/task/{tid}/sub_videos").json()
out = ROOT / "docs" / "spec" / "SMOKE_RESULT.md"
out.write_text("# T31 端到端冒烟结果\n\n- 视频: Swim-4.mp4 | profile=swim_liquid | worm_count=1 | c_unit=cycle\n"
               f"- 状态: **{res['status']}**，耗时 {res['elapsed_s']} s，socket 收到帧 {res['frames_via_socket']}\n"
               f"- 事件: {events[:10]}\n- 下载链接 HTTP 状态: {chk}\n- 子视频: {res['sub_videos']}\n\n"
               "## id_pose_summary\n```json\n" + json.dumps(res["summary"], ensure_ascii=False, indent=1) + "\n```\n", encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "summary"}, ensure_ascii=False))
print(json.dumps(res["summary"], ensure_ascii=False)[:1500])
sys.exit(0 if res["status"] == "completed" else 1)
