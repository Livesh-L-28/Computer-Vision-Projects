"""
Air Combat Intelligence System (v2.0 Production Edition)
Aerial Threat Detection, Kinematic Tracking, Threat Convergence Scoring & Tactical HUD.
"""

import datetime
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    alpha_rect,
    direction_vector,
    draw_corner_frame,
    draw_dashboard_panel,
    draw_scanline,
    future_path,
    generate_activity_heatmap,
    glow_box,
    motion_trail,
    pill_label,
    radar_ring,
)

colorama_init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AIR-COMBAT")

BANNER = f"""
{Fore.RED}╔══════════════════════════════════════════════════════════════════╗
║        AIR COMBAT INTELLIGENCE SYSTEM   v2.0 (Production)       ║
║        Aerial Threat Detection & Tactical Airspace Tracking     ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""

# Aircraft sub-classification palette
CLASS_COLORS = {
    "fighter jet": Colors.NEON_RED,
    "military aircraft": Colors.NEON_CYAN,
    "helicopter": Colors.NEON_GREEN,
    "drone": Colors.NEON_YELLOW,
    "vehicle": Colors.NEON_PURPLE,
    "person": Colors.NEON_ORANGE,
}

# Relevant COCO classes
AIRCRAFT_COCO = {4: "aircraft"}
GROUND_COCO = {2: "vehicle", 5: "vehicle", 7: "vehicle", 0: "person"}
ALL_COCO_IDS = list(AIRCRAFT_COCO.keys()) + list(GROUND_COCO.keys())


def classify_aircraft(track: Track, frame_w: int, frame_h: int) -> str:
    """Heuristic aircraft sub-type assignment based on geometry and kinematics."""
    x1, y1, x2, y2 = track.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    area_frac = (w * h) / float(max(1, frame_w * frame_h))
    aspect = w / float(h)
    vx, vy = track.velocity_px()
    speed = math.hypot(vx, vy)

    if area_frac < 0.0025 and speed > 2.5:
        return "drone"
    if aspect > 2.0 and area_frac > 0.003:
        return "fighter jet"
    if speed < 0.8 and area_frac > 0.002:
        return "helicopter"
    return "military aircraft"


def compute_threat_score(track: Track, all_tracks: list[Track], frame_w: int, frame_h: int) -> float:
    """Composite 0-100 threat score from speed, center-proximity and path convergence."""
    cx, cy = track.centroid()
    vx, vy = track.velocity_px()
    speed = math.hypot(vx, vy)

    center_dx = abs(cx - frame_w / 2.0) / (frame_w / 2.0)
    center_dy = abs(cy - frame_h / 2.0) / (frame_h / 2.0)
    proximity = 1.0 - min(1.0, math.hypot(center_dx, center_dy))

    convergence = 0.0
    for other in all_tracks:
        if other.track_id == track.track_id:
            continue
        ox, oy = other.centroid()
        ovx, ovy = other.velocity_px()

        # Project 15 frames into the future
        fx1, fy1 = cx + vx * 15.0, cy + vy * 15.0
        fx2, fy2 = ox + ovx * 15.0, oy + ovy * 15.0
        d_now = math.hypot(cx - ox, cy - oy)
        d_future = math.hypot(fx1 - fx2, fy1 - fy2)

        if d_now > 1.0 and d_future < d_now:
            convergence = max(convergence, 1.0 - (d_future / max(d_now, 1.0)))

    score = (min(speed / 12.0, 1.0) * 40.0) + (proximity * 30.0) + (convergence * 30.0)
    return float(max(0.0, min(100.0, score)))


def get_threat_level(score: float) -> tuple[str, tuple[int, int, int]]:
    if score >= 66.0:
        return "HIGH", Colors.THREAT_HIGH
    if score >= 33.0:
        return "MEDIUM", Colors.THREAT_MED
    return "LOW", Colors.THREAT_LOW


class AirCombatSystem:
    """Production Air Combat Intelligence Engine."""

    def __init__(
        self,
        source: str,
        model_path: str = "yolov8s.pt",
        conf: float = 0.25,
        detect_every: int = 1,
        output_dir: Path | None = None,
        device: str | None = None,
    ):
        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.detect_every = detect_every
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/air_combat")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or get_optimal_device()
        self.tracker = ObjectTracker(max_disappeared=20, iou_threshold=0.2, fps=30.0)
        self.heatmap_pts: deque = deque(maxlen=6000)
        self.csv_records: list[dict[str, Any]] = []
        self.frame_num = 0
        self.peak_threat = 0.0
        self._last_detections: list[tuple[tuple[int, int, int, int], str, float]] = []

        self._load_yolo()

    def _load_yolo(self) -> None:
        from ultralytics import YOLO

        logger.info(f"Target compute device: {self.device.upper()}")

        # Check if local model file exists, otherwise download or use default
        local_weight = Path(self.model_path)
        if not local_weight.exists():
            root_weight = Path(__file__).resolve().parent.parent.parent / "yolov8s.pt"
            if root_weight.exists():
                local_weight = root_weight

        logger.info(f"Loading YOLO weights from {local_weight}...")
        self.model = YOLO(str(local_weight))

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device, conf=self.conf)
        logger.info("YOLO model initialized and ready.")

    def _detect(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
        res = self.model(frame, verbose=False, device=self.device, classes=ALL_COCO_IDS, conf=self.conf)[0]
        detections = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            c = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            category = "aircraft" if cls_id in AIRCRAFT_COCO else GROUND_COCO.get(cls_id, "vehicle")
            detections.append(((x1, y1, x2, y2), category, c))
        return detections

    def _draw_minimap(self, frame: np.ndarray, tracks: list[Track], fw: int, fh: int) -> None:
        mw, mh = 220, 150
        x, y = 12, frame.shape[0] - mh - 16
        alpha_rect(frame, x, y, x + mw, y + mh, Colors.PANEL_BG, 0.85)
        cv2.rectangle(frame, (x, y), (x + mw, y + mh), Colors.BORDER_CYAN, 1)
        cv2.putText(
            frame,
            "AIRSPACE DIGITAL TWIN",
            (x + 6, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            Colors.BORDER_CYAN,
            1,
            cv2.LINE_AA,
        )

        # Draw past tracks
        for px, py in list(self.heatmap_pts)[-250:]:
            mx = x + 8 + int((px / float(fw)) * (mw - 16))
            my = y + 22 + int((py / float(fh)) * (mh - 30))
            cv2.circle(frame, (mx, my), 1, (75, 75, 95), -1)

        # Draw current aircraft targets
        for tr in tracks:
            cx, cy = tr.centroid()
            mx = x + 8 + int((cx / float(fw)) * (mw - 16))
            my = y + 22 + int((cy / float(fh)) * (mh - 30))
            col = CLASS_COLORS.get(tr.metadata.get("disp_label", ""), Colors.BORDER_CYAN)
            cv2.circle(frame, (mx, my), 3, col, -1)

    def run(self, max_frames: int | None = None) -> Path:
        print(BANNER)

        # Open video source
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        out_video_path = self.output_dir / "annotated_combat.mp4"
        writer, actual_out_path = create_video_writer(out_video_path, fps_in, fw, fh)

        logger.info(f"Processing source: {self.source} [{fw}x{fh} @ {fps_in:.1f} FPS]")
        logger.info(f"Output video writing to: {actual_out_path}")

        pbar = tqdm(
            total=min(total_f, max_frames) if (total_f and max_frames) else (total_f or None),
            desc=f"{Fore.RED}COMBAT PIPELINE{Style.RESET_ALL}",
            unit="f",
            dynamic_ncols=True,
            colour="red",
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
                aircraft_tracks = [t for t in tracks if t.category == "aircraft"]
                ground_tracks = [t for t in tracks if t.category != "aircraft"]

                speeds = []
                scores = []
                drones_count = 0

                # Process aerial tracks
                for tr in aircraft_tracks:
                    label = classify_aircraft(tr, fw, fh)
                    tr.metadata["disp_label"] = label
                    score = compute_threat_score(tr, aircraft_tracks, fw, fh)
                    tr.metadata["threat_score"] = score
                    scores.append(score)

                    vx, vy = tr.velocity_px()
                    speed_kmh = math.hypot(vx, vy) * 18.0
                    speeds.append(speed_kmh)
                    if label == "drone":
                        drones_count += 1

                    color = CLASS_COLORS.get(label, Colors.NEON_CYAN)
                    x1, y1, x2, y2 = tr.bbox
                    cx, cy = tr.centroid()

                    # HUD visual elements
                    glow_box(frame, x1, y1, x2, y2, color, self.frame_num)
                    motion_trail(frame, tr.history, color)
                    radar_ring(frame, int(cx), int(cy), self.frame_num, color)
                    direction_vector(frame, cx, cy, vx, vy, color)
                    future_path(frame, cx, cy, vx, vy, color)
                    self.heatmap_pts.append((cx, cy))

                    # Tactical pill badges
                    bucket, bcol = get_threat_level(score)
                    lines = [
                        f"ID #{tr.track_id} {label.upper()}",
                        f"{speed_kmh:0.0f} KM/H  HDG {tr.heading_deg:0.0f}°",
                        f"THREAT: {bucket} ({score:.0f})",
                    ]
                    base_y = y1 - 8 - 14 * (len(lines) - 1)
                    for idx, ln in enumerate(lines):
                        line_col = bcol if idx == len(lines) - 1 else color
                        pill_label(frame, ln, x1, base_y + idx * 14, line_col, fs=0.32)

                # Process ground tracks
                for tr in ground_tracks:
                    label = tr.category
                    tr.metadata["disp_label"] = label
                    color = CLASS_COLORS.get(label, Colors.NEON_PURPLE)
                    x1, y1, x2, y2 = tr.bbox
                    glow_box(frame, x1, y1, x2, y2, color, self.frame_num)
                    pill_label(frame, f"ID #{tr.track_id} {label.upper()}", x1, y1 - 6, color, fs=0.32)

                # Aggregate mission metrics
                max_threat = max(scores) if scores else 0.0
                self.peak_threat = max(self.peak_threat, max_threat)
                top_target = aircraft_tracks[scores.index(max_threat)].track_id if scores else -1
                collisions = sum(1 for s in scores if s >= 66.0)
                density = min(100.0, (len(tracks) / 10.0) * 100.0)
                bucket_str, bucket_col = get_threat_level(max_threat)

                stats_rows = [
                    ("FPS", f"{fps_disp:>5.1f}", Colors.TEXT_DIM),
                    ("FRAME", f"{self.frame_num:>6d}", Colors.TEXT_DIM),
                    ("ACTIVE AIRCRAFT", f"{len(aircraft_tracks):>4d}", Colors.BORDER_CYAN),
                    ("ACTIVE DRONES", f"{drones_count:>4d}", Colors.NEON_YELLOW),
                    ("AVG SPEED", f"{np.mean(speeds) if speeds else 0:>5.0f} KM/H", Colors.TEXT_DIM),
                    ("PRIMARY THREAT", f"#{top_target}", bucket_col),
                    ("DEFCON STATUS", bucket_str, bucket_col),
                    ("AIRSPACE DENSITY", f"{density:>5.1f}%", Colors.TEXT_DIM),
                    ("INTERCEPT RISKS", f"{collisions:>4d}", Colors.THREAT_HIGH if collisions else Colors.TEXT_DIM),
                    ("SYSTEM STATUS", "OPERATIONAL", Colors.THREAT_LOW),
                ]

                # Tactical HUD layers
                draw_scanline(frame, self.frame_num)
                draw_corner_frame(frame)
                self._draw_minimap(frame, tracks, fw, fh)
                draw_dashboard_panel(frame, "TACTICAL AIRSPACE TELEMETRY", stats_rows, width=285)

                # Periodic CSV telemetry logging
                if self.frame_num % log_interval == 0:
                    self.csv_records.append(
                        {
                            "frame": self.frame_num,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "active_aircraft": len(aircraft_tracks),
                            "active_drones": drones_count,
                            "avg_speed_kmh": round(float(np.mean(speeds)) if speeds else 0.0, 1),
                            "max_threat_score": round(max_threat, 1),
                            "threat_level": bucket_str,
                            "primary_target_id": top_target,
                            "airspace_density_pct": round(density, 1),
                            "collision_risks": collisions,
                        }
                    )

                writer.write(frame)
                pbar.update(1)
                pbar.set_postfix({"targets": len(tracks), "threat": f"{max_threat:.0f}", "fps": f"{fps_disp:.1f}"})

        finally:
            pbar.close()
            cap.release()
            writer.release()

        # Save artifacts
        self._export_reports(fw, fh)
        logger.info(
            f"Air Combat pipeline complete. Processed {self.frame_num} frames. Peak threat: {self.peak_threat:.1f}"
        )
        return Path(actual_out_path)

    def _export_reports(self, width: int, height: int) -> None:
        # Save CSV
        if self.csv_records:
            csv_path = self.output_dir / "combat_report.csv"
            df = pd.DataFrame(self.csv_records)
            df.to_csv(csv_path, index=False)
            logger.info(f"Telemetry CSV -> {csv_path}")

            # Save 4-pane dashboard figure
            fig_path = self.output_dir / "combat_dashboard.png"
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0a0a14")
            for ax in axes.flat:
                ax.set_facecolor("#0a0a14")
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333")

            axes[0, 0].plot(df["frame"], df["active_aircraft"], color="#00d4ff", lw=2)
            axes[0, 0].set_title("Active Aircraft Volume", color="white")
            axes[0, 1].plot(df["frame"], df["max_threat_score"], color="#ff3030", lw=2)
            axes[0, 1].set_title("Threat Metric Progression", color="white")
            axes[1, 0].plot(df["frame"], df["avg_speed_kmh"], color="#30ff90", lw=2)
            axes[1, 0].set_title("Average Airspeed (km/h)", color="white")
            axes[1, 1].plot(df["frame"], df["airspace_density_pct"], color="#ffd400", lw=2)
            axes[1, 1].set_title("Airspace Density (%)", color="white")

            plt.tight_layout()
            plt.savefig(fig_path, dpi=120, facecolor="#0a0a14")
            plt.close()
            logger.info(f"Analytics Dashboard -> {fig_path}")

        # Save Heatmap
        if self.heatmap_pts:
            hm_path = str(self.output_dir / "airspace_heatmap.png")
            generate_activity_heatmap(
                self.heatmap_pts,
                width,
                height,
                sigma=20,
                save_path=hm_path,
                title="Airspace Cumulative Activity Heatmap",
            )
            logger.info(f"Airspace Heatmap -> {hm_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Air Combat Intelligence System (v2.0 Production)")
    parser.add_argument("source_pos", nargs="?", default=None, help="Positional video source path")
    parser.add_argument("--input", "-i", dest="source_opt", default=None, help="Input video path or camera index")
    parser.add_argument("--synthetic", action="store_true", help="Generate and run on a synthetic combat scenario")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--every", type=int, default=1, help="Inference frequency (every N frames)")
    parser.add_argument("--out-dir", default=None, help="Custom output directory")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    source = args.source_opt or args.source_pos
    if args.synthetic or not source:
        test_video = Path("outputs/synthetic_combat.mp4")
        generate_synthetic_test_video(test_video, num_frames=120, scenario="combat")
        source = str(test_video)

    system = AirCombatSystem(
        source=source,
        model_path=args.model,
        conf=args.conf,
        detect_every=args.every,
        output_dir=Path(args.out_dir) if args.out_dir else None,
    )
    system.run(max_frames=args.frames)


if __name__ == "__main__":
    main()
