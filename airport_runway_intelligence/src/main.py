"""
Airport Runway Intelligence System (v2.0 Production Edition)
Tarmac Surveillance, Runway Incursion Monitoring, Surface Digital Twin & Risk Analytics.
"""

import datetime
import logging
import sys
import time
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from colorama import Fore, Style
from colorama import init as colorama_init
from tqdm import tqdm

# Project common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import Colors
from common.tracker import ObjectTracker, Track
from common.utils import (
    create_video_writer,
    generate_synthetic_test_video,
    get_optimal_device,
)
from common.visualization import (
    corner_ticks,
    draw_corner_frame,
    draw_dashboard_panel,
    draw_scanline,
    generate_activity_heatmap,
    glow_box,
    motion_trail,
    pill_label,
)

colorama_init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIRPORT-RUNWAY")

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║     AIRPORT RUNWAY INTELLIGENCE SYSTEM  v2.0 (Production)       ║
║     Surface Surveillance, Digital Twin & Runway Incursions       ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""


class Category(str, Enum):
    COMMERCIAL_AIRCRAFT = "commercial_aircraft"
    CARGO_AIRCRAFT = "cargo_aircraft"
    PRIVATE_JET = "private_jet"
    HELICOPTER = "helicopter"
    FUEL_TRUCK = "fuel_truck"
    SERVICE_VEHICLE = "service_vehicle"
    BAGGAGE_VEHICLE = "baggage_vehicle"
    MAINTENANCE_VEHICLE = "maintenance_vehicle"
    GROUND_CREW = "ground_crew"
    AIRPORT_STAFF = "airport_staff"
    UNKNOWN = "unknown"


CATEGORY_COLORS = {
    Category.COMMERCIAL_AIRCRAFT: Colors.NEON_CYAN,
    Category.CARGO_AIRCRAFT: Colors.NEON_ORANGE,
    Category.PRIVATE_JET: Colors.NEON_PINK,
    Category.HELICOPTER: Colors.NEON_PURPLE,
    Category.FUEL_TRUCK: Colors.NEON_YELLOW,
    Category.SERVICE_VEHICLE: Colors.NEON_GREEN,
    Category.BAGGAGE_VEHICLE: (80, 255, 170),
    Category.MAINTENANCE_VEHICLE: Colors.NEON_WHITE,
    Category.GROUND_CREW: (255, 120, 20),
    Category.AIRPORT_STAFF: (255, 200, 100),
    Category.UNKNOWN: (180, 180, 180),
}

COCO_TO_CAT = {
    4: Category.COMMERCIAL_AIRCRAFT,
    0: Category.GROUND_CREW,
    7: Category.SERVICE_VEHICLE,
    2: Category.MAINTENANCE_VEHICLE,
    5: Category.CARGO_AIRCRAFT,
    3: Category.BAGGAGE_VEHICLE,
}


def check_box_intersection(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    """Return True if bounding boxes overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


class AirportDigitalTwin:
    """Generates an overhead 2D radar digital twin display."""

    def __init__(self, width: int = 240, height: int = 160):
        self.width = width
        self.height = height

    def render(
        self,
        tracks: list[Track],
        zones: dict[str, tuple[int, int, int, int, tuple[int, int, int]]],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        twin = np.full((self.height, self.width, 3), (12, 10, 8), dtype=np.uint8)
        cv2.rectangle(twin, (0, 0), (self.width - 1, self.height - 1), Colors.BORDER_CYAN, 1)
        cv2.putText(twin, "RADAR DIGITAL TWIN", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, Colors.BORDER_CYAN, 1)

        # Scale and render defined airport zones
        for zname, (zx1, zy1, zx2, zy2, zcol) in zones.items():
            tx1 = int((zx1 / float(frame_w)) * self.width)
            ty1 = int((zy1 / float(frame_h)) * self.height)
            tx2 = int((zx2 / float(frame_w)) * self.width)
            ty2 = int((zy2 / float(frame_h)) * self.height)
            cv2.rectangle(twin, (tx1, ty1), (tx2, ty2), zcol, 1)

        # Render active assets
        for tr in tracks:
            cx, cy = tr.centroid()
            tx = int((cx / float(frame_w)) * self.width)
            ty = int((cy / float(frame_h)) * self.height)
            col = CATEGORY_COLORS.get(tr.category, Colors.NEON_WHITE)
            cv2.circle(twin, (tx, ty), 3, col, -1)

        return twin


class AirportRunwaySystem:
    """Production Airport Surface Perception & Runway Incursion System."""

    def __init__(
        self,
        source: str,
        model_path: str = "yolov8s.pt",
        conf: float = 0.25,
        detect_every: int = 2,
        output_dir: Path | None = None,
        device: str | None = None,
    ):
        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.detect_every = detect_every
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/airport_runway")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or get_optimal_device()
        self.tracker = ObjectTracker(max_disappeared=25, px_per_meter=8.0, fps=30.0)
        self.digital_twin = AirportDigitalTwin(width=230, height=150)
        self.heatmap_pts: deque = deque(maxlen=6000)
        self.csv_records: list[dict[str, Any]] = []
        self.frame_num = 0
        self._last_detections: list[tuple[tuple[int, int, int, int], str, float]] = []

        self._load_yolo()

    def _load_yolo(self) -> None:
        from ultralytics import YOLO

        logger.info(f"Target compute device: {self.device.upper()}")

        local_weight = Path(self.model_path)
        if not local_weight.exists():
            root_weight = Path(__file__).resolve().parent.parent.parent / "yolov8s.pt"
            if root_weight.exists():
                local_weight = root_weight

        logger.info(f"Loading YOLO weights from {local_weight}...")
        self.model = YOLO(str(local_weight))
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device, conf=self.conf)
        logger.info("YOLO airport detector ready.")

    def _detect(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
        res = self.model(frame, verbose=False, device=self.device, classes=list(COCO_TO_CAT.keys()), conf=self.conf)[0]
        dets = []
        for box in res.boxes:
            cid = int(box.cls[0])
            c = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cat = COCO_TO_CAT.get(cid, Category.UNKNOWN)
            dets.append(((x1, y1, x2, y2), cat, c))
        return dets

    def run(self, max_frames: int | None = None) -> Path:
        print(BANNER)

        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video stream: {self.source}")

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Define dynamic runway surface zones based on frame resolution
        zones = {
            "RUNWAY 09L/27R": (int(fw * 0.1), int(fh * 0.42), int(fw * 0.9), int(fh * 0.58), Colors.ZONE_RUNWAY_ACTIVE),
            "TAXIWAY ALPHA": (int(fw * 0.1), int(fh * 0.65), int(fw * 0.9), int(fh * 0.75), Colors.ZONE_TAXIWAY),
            "APRON ZONE": (int(fw * 0.15), int(fh * 0.78), int(fw * 0.85), int(fh * 0.92), Colors.ZONE_SAFE),
            "RESTRICTED AP": (int(fw * 0.4), int(fh * 0.35), int(fw * 0.6), int(fh * 0.42), Colors.ZONE_RESTRICTED),
        }

        out_video_path = self.output_dir / "annotated_runway.mp4"
        writer, actual_out_path = create_video_writer(out_video_path, fps_in, fw, fh)
        logger.info(f"Processing airport footage: {self.source} -> {actual_out_path}")

        pbar = tqdm(
            total=min(total_f, max_frames) if (total_f and max_frames) else (total_f or None),
            desc=f"{Fore.CYAN}RUNWAY PIPELINE{Style.RESET_ALL}",
            unit="f",
            dynamic_ncols=True,
            colour="cyan",
        )

        t_prev = time.perf_counter()
        log_interval = max(1, int(fps_in * 2))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_num += 1
                if max_frames and self.frame_num > max_frames:
                    break

                t_now = time.perf_counter()
                fps_disp = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                # Detection & Tracking step
                if self.frame_num % self.detect_every == 0 or self.frame_num == 1:
                    self._last_detections = self._detect(frame)

                tracks = self.tracker.update(self._last_detections, self.frame_num)

                # Render Runway Zones overlay
                zone_layer = frame.copy()
                for zname, (zx1, zy1, zx2, zy2, zcol) in zones.items():
                    cv2.rectangle(zone_layer, (zx1, zy1), (zx2, zy2), zcol, -1)
                cv2.addWeighted(zone_layer, 0.08, frame, 0.92, 0, dst=frame)

                for zname, (zx1, zy1, zx2, zy2, zcol) in zones.items():
                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), zcol, 1)
                    cv2.putText(frame, zname, (zx1 + 6, zy1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, zcol, 1, cv2.LINE_AA)

                # Process tracks and risk incursion detection
                counts = defaultdict(int)
                incursions = 0
                runway_box = zones["RUNWAY 09L/27R"][:4]

                for tr in tracks:
                    x1, y1, x2, y2 = tr.bbox
                    cx, cy = tr.centroid()
                    col = CATEGORY_COLORS.get(tr.category, Colors.NEON_WHITE)
                    self.heatmap_pts.append((cx, cy))

                    # Track geometry
                    glow_box(frame, x1, y1, x2, y2, col, self.frame_num)
                    corner_ticks(frame, tr.bbox, col)
                    motion_trail(frame, tr.history, col)

                    # Check for runway occupancy / incursion
                    on_runway = check_box_intersection(tr.bbox, runway_box)
                    is_aircraft = "aircraft" in str(tr.category).lower()

                    if on_runway:
                        counts["runway_occupied"] += 1
                        if not is_aircraft:
                            incursions += 1
                            risk_str, risk_col = "INCURSION ALERT", Colors.THREAT_HIGH
                        else:
                            risk_str, risk_col = "ACTIVE RUNWAY", Colors.NEON_YELLOW
                    else:
                        risk_str, risk_col = "TAXI/APRON", Colors.NEON_GREEN

                    if is_aircraft:
                        counts["aircraft"] += 1
                        pred_path = tr.predict_future(steps=6, gap=8)
                        for pt in pred_path:
                            cv2.circle(frame, pt, 2, col, -1)
                        lines = [
                            f"ACFT #{tr.track_id:02d} {tr.category.split('.')[-1].upper()}",
                            f"{tr.speed_kmh:4.1f} KM/H  HDG {tr.heading_deg:03.0f}°",
                            f"ZONE: {risk_str}",
                        ]
                    else:
                        counts["vehicles_crew"] += 1
                        lines = [
                            f"ID #{tr.track_id:02d} {tr.category.split('.')[-1].replace('_', ' ').upper()}",
                            f"{tr.speed_kmh:4.1f} KM/H",
                            f"ZONE: {risk_str}",
                        ]

                    base_y = y1 - 8 - 14 * (len(lines) - 1)
                    for idx, ln in enumerate(lines):
                        line_col = risk_col if idx == len(lines) - 1 else col
                        pill_label(frame, ln, x1, base_y + idx * 14, line_col, fs=0.30)

                # Overlay Digital Twin minimap
                twin_img = self.digital_twin.render(tracks, zones, fw, fh)
                tx, ty = 12, fh - twin_img.shape[0] - 16
                frame[ty : ty + twin_img.shape[0], tx : tx + twin_img.shape[1]] = twin_img

                # Telemetry dashboard
                status_color = Colors.THREAT_HIGH if incursions > 0 else Colors.THREAT_LOW
                status_text = "INCURSION WARNING" if incursions > 0 else "RUNWAY CLEAR"
                stats_rows = [
                    ("SURFACE FPS", f"{fps_disp:>5.1f}", Colors.TEXT_DIM),
                    ("FRAME", f"{self.frame_num:>6d}", Colors.TEXT_DIM),
                    ("AIRCRAFT ON GND", f"{counts['aircraft']:>4d}", Colors.BORDER_CYAN),
                    ("GROUND VEHICLES", f"{counts['vehicles_crew']:>4d}", Colors.NEON_GREEN),
                    (
                        "RUNWAY OCCUPANCY",
                        f"{counts['runway_occupied']:>4d}",
                        Colors.NEON_YELLOW if counts["runway_occupied"] else Colors.TEXT_DIM,
                    ),
                    ("INCURSION RISKS", f"{incursions:>4d}", Colors.THREAT_HIGH if incursions else Colors.TEXT_DIM),
                    ("TOWER STATUS", status_text, status_color),
                ]

                draw_scanline(frame, self.frame_num)
                draw_corner_frame(frame)
                draw_dashboard_panel(frame, "AIRPORT SURFACE TELEMETRY", stats_rows, width=285)

                if self.frame_num % log_interval == 0:
                    self.csv_records.append(
                        {
                            "frame": self.frame_num,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "aircraft_count": counts["aircraft"],
                            "vehicles_count": counts["vehicles_crew"],
                            "runway_occupied": counts["runway_occupied"],
                            "incursions": incursions,
                            "status": status_text,
                        }
                    )

                writer.write(frame)
                pbar.update(1)
                pbar.set_postfix({"acft": counts["aircraft"], "veh": counts["vehicles_crew"], "incursions": incursions})

        finally:
            pbar.close()
            cap.release()
            writer.release()

        self._export_reports(fw, fh)
        logger.info(f"Airport Runway pipeline complete. Processed {self.frame_num} frames.")
        return Path(actual_out_path)

    def _export_reports(self, width: int, height: int) -> None:
        if self.csv_records:
            csv_path = self.output_dir / "runway_report.csv"
            pd.DataFrame(self.csv_records).to_csv(csv_path, index=False)
            logger.info(f"Runway telemetry CSV -> {csv_path}")

        if self.heatmap_pts:
            hm_path = str(self.output_dir / "tarmac_heatmap.png")
            generate_activity_heatmap(
                self.heatmap_pts, width, height, sigma=25, save_path=hm_path, title="Airport Tarmac Movement Heatmap"
            )
            logger.info(f"Tarmac Heatmap -> {hm_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Airport Runway Intelligence System (v2.0 Production)")
    parser.add_argument("source_pos", nargs="?", default=None, help="Positional video source path")
    parser.add_argument("--input", "-i", dest="source_opt", default=None, help="Input video path or camera index")
    parser.add_argument("--synthetic", action="store_true", help="Generate and run on synthetic runway scenario")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--every", type=int, default=2, help="Inference frequency (every N frames)")
    parser.add_argument("--out-dir", default=None, help="Custom output directory")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    source = args.source_opt or args.source_pos
    if args.synthetic or not source:
        test_video = Path("outputs/synthetic_runway.mp4")
        generate_synthetic_test_video(test_video, num_frames=120, scenario="runway")
        source = str(test_video)

    system = AirportRunwaySystem(
        source=source,
        model_path=args.model,
        conf=args.conf,
        detect_every=args.every,
        output_dir=Path(args.out_dir) if args.out_dir else None,
    )
    system.run(max_frames=args.frames)


if __name__ == "__main__":
    main()
