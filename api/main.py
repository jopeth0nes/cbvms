"""FastAPI server — camera discovery, selection, and live MJPEG stream."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.camera_manager import camera_manager, test_ip_camera
from api.camera_store import add_ip_camera, delete_ip_camera, get_saved_ip_cameras
from api.detection_service import detection_service

app = FastAPI(title="CBVMS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraTestPayload(BaseModel):
    url: str = Field(..., min_length=1)


class CameraSelectPayload(BaseModel):
    id: str
    type: str
    index: Optional[int] = None
    label: Optional[str] = None
    url: Optional[str] = None


class IpCameraPayload(BaseModel):
    label: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


@app.on_event("startup")
async def _startup() -> None:
    detection_service.set_event_loop(asyncio.get_running_loop())
    detection_service.start()
    camera_manager.restore_preference()


@app.get("/api/cameras/scan")
def scan_cameras() -> dict:
    return {"cameras": camera_manager.scan_all(), "active": camera_manager.active_camera}


@app.get("/api/cameras/active")
def get_active_camera() -> dict:
    return {"active": camera_manager.active_camera}


@app.post("/api/cameras/test")
def test_camera(payload: CameraTestPayload) -> dict:
    success, message = camera_manager.test_url(payload.url)
    return {"success": success, "message": message}


@app.post("/api/cameras/select")
def select_camera(payload: CameraSelectPayload) -> dict:
    try:
        return camera_manager.select(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cameras/ip/add")
def add_ip_camera_route(payload: IpCameraPayload) -> dict:
    try:
        entry = add_ip_camera(payload.label, payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "camera": {
            "id": f"ip_{entry['id']}",
            "type": "rj45",
            "label": entry["label"],
            "url": entry["url"],
            "status": "available" if test_ip_camera(entry["url"]) else "unreachable",
        }
    }


@app.delete("/api/cameras/ip/{camera_id}")
def delete_ip_camera_route(camera_id: str) -> dict:
    raw_id = camera_id.removeprefix("ip_")
    if not delete_ip_camera(raw_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_manager.clear_active_if(f"ip_{raw_id}")
    return {"success": True}


@app.get("/api/cameras/ip")
def list_ip_cameras() -> dict:
    cameras = []
    for cam in get_saved_ip_cameras():
        reachable = test_ip_camera(str(cam["url"]))
        cameras.append(
            {
                "id": f"ip_{cam['id']}",
                "type": "rj45",
                "label": cam.get("label", "IP Camera"),
                "url": cam["url"],
                "status": "available" if reachable else "unreachable",
            }
        )
    return {"cameras": cameras}


async def _mjpeg_generator():
    while True:
        jpeg = await asyncio.to_thread(camera_manager.read_frame_jpeg)
        if jpeg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        else:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.03)


@app.get("/api/cameras/stream")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/api/ws/detections")
async def detections_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    detection_service.register_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        detection_service.unregister_client(websocket)


@app.websocket("/ws/detections")
async def detections_ws_alias(websocket: WebSocket) -> None:
    """Alias for clients expecting /ws/detections."""
    await detections_ws(websocket)


@app.get("/api/detections/latest")
def latest_detections() -> dict:
    return {"detections": detection_service.get_latest()}


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
