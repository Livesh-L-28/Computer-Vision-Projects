"""
FastAPI Production Server for Computer Vision Intelligence Suite.
Serves web dashboard, REST API for running vision pipelines, live MJPEG streaming, and output downloads.
"""

import logging
import shutil
from collections.abc import Generator
from pathlib import Path

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Project imports
from air_combat_intelligence.src.main import AirCombatSystem
from airport_runway_intelligence.src.main import AirportRunwaySystem
from common.utils import generate_synthetic_test_video
from license_plate_intelligence.src.main import LicensePlateSystem

logger = logging.getLogger("CV-WEB")

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
UPLOADS_DIR = BASE_DIR / "outputs" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Computer Vision Intelligence Suite",
    version="2.0.0",
    description="Production Computer Vision & Visual Intelligence Platform",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory execution state
pipeline_state = {
    "status": "idle",
    "system": None,
    "progress": 0,
    "current_frame": 0,
    "total_frames": 0,
    "latest_output_video": None,
    "latest_report_csv": None,
    "latest_heatmap": None,
    "latest_dashboard": None,
    "metrics": {},
}


def run_pipeline_task(system_type: str, video_path: str, max_frames: int | None = 90):
    global pipeline_state
    pipeline_state["status"] = "processing"
    pipeline_state["system"] = system_type
    pipeline_state["metrics"] = {}

    try:
        if system_type == "combat":
            sys_obj = AirCombatSystem(source=video_path, conf=0.25, detect_every=1)
            out_file = sys_obj.run(max_frames=max_frames)
            pipeline_state["latest_output_video"] = f"/outputs/air_combat/{out_file.name}"
            pipeline_state["latest_report_csv"] = "/outputs/air_combat/combat_report.csv"
            pipeline_state["latest_heatmap"] = "/outputs/air_combat/airspace_heatmap.png"
            pipeline_state["latest_dashboard"] = "/outputs/air_combat/combat_dashboard.png"
            pipeline_state["metrics"] = {
                "Peak Threat": f"{sys_obj.peak_threat:.1f}",
                "Frames Processed": sys_obj.frame_num,
                "Active Targets": len(sys_obj.tracker.tracks),
            }

        elif system_type == "runway":
            sys_obj = AirportRunwaySystem(source=video_path, conf=0.25, detect_every=2)
            out_file = sys_obj.run(max_frames=max_frames)
            pipeline_state["latest_output_video"] = f"/outputs/airport_runway/{out_file.name}"
            pipeline_state["latest_report_csv"] = "/outputs/airport_runway/runway_report.csv"
            pipeline_state["latest_heatmap"] = "/outputs/airport_runway/tarmac_heatmap.png"
            pipeline_state["latest_dashboard"] = None
            pipeline_state["metrics"] = {
                "Surface Assets": len(sys_obj.tracker.tracks),
                "Frames Processed": sys_obj.frame_num,
            }

        elif system_type == "anpr":
            sys_obj = LicensePlateSystem(source=video_path, detect_every=2, plate_conf=0.25)
            out_file = sys_obj.run(max_frames=max_frames)
            pipeline_state["latest_output_video"] = f"/outputs/license_plate/{out_file.name}"
            pipeline_state["latest_report_csv"] = "/outputs/license_plate/plates.csv"
            pipeline_state["latest_heatmap"] = None
            pipeline_state["latest_dashboard"] = None
            pipeline_state["metrics"] = {
                "Vehicles Logged": sys_obj.total_vehicles,
                "Plates Read": len(sys_obj.plate_records),
                "Unique Plates": len(sys_obj.plate_counts),
            }

        pipeline_state["status"] = "completed"

    except Exception as e:
        logger.exception("Pipeline execution failed")
        pipeline_state["status"] = "error"
        pipeline_state["metrics"] = {"Error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/status")
async def get_status():
    return JSONResponse(pipeline_state)


@app.post("/api/run-synthetic")
async def run_synthetic(background_tasks: BackgroundTasks, system: str = Form("combat"), frames: int = Form(60)):
    if pipeline_state["status"] == "processing":
        raise HTTPException(status_code=400, detail="A pipeline task is currently running.")

    test_video = UPLOADS_DIR / f"synthetic_{system}.mp4"
    generate_synthetic_test_video(test_video, num_frames=frames, scenario=system)

    background_tasks.add_task(run_pipeline_task, system, str(test_video), frames)
    return JSONResponse({"status": "started", "system": system, "mode": "synthetic"})


@app.post("/api/run-upload")
async def run_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    system: str = Form("combat"),
    frames: int = Form(100),
):
    if pipeline_state["status"] == "processing":
        raise HTTPException(status_code=400, detail="A pipeline task is currently running.")

    upload_path = UPLOADS_DIR / file.filename
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_pipeline_task, system, str(upload_path), frames)
    return JSONResponse({"status": "started", "system": system, "filename": file.filename})


def stream_annotated_scenario(system: str) -> Generator[bytes, None, None]:
    """Generates continuous MJPEG frames for real-time browser preview."""
    sim_video = UPLOADS_DIR / f"stream_{system}.mp4"
    if not sim_video.exists():
        generate_synthetic_test_video(sim_video, num_frames=180, scenario=system)

    cap = cv2.VideoCapture(str(sim_video))
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Subtle live streaming watermark
        cv2.putText(
            frame,
            f"● LIVE STREAM: {system.upper()}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 242, 254),
            2,
            cv2.LINE_AA,
        )

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        cv2.waitKey(30)


@app.get("/api/stream/{system}")
async def live_stream(system: str):
    """Real-time live video stream endpoint for surveillance integration."""
    if system not in ("combat", "runway", "anpr"):
        raise HTTPException(status_code=404, detail="Unknown intelligence system")
    return StreamingResponse(stream_annotated_scenario(system), media_type="multipart/x-mixed-replace; boundary=frame")
