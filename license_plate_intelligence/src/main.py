"""
License Plate Intelligence System (ANPR v4.0 Production Edition)
Dual-stage YOLO Detection (Vehicle + Plate), Optimized EasyOCR with Preprocessing & Analytics.
"""

import datetime
import logging
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from colorama import Fore, Style
from colorama import init as colorama_init
from tqdm import tqdm

# Project common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import MODELS_DIR, ANPRConfig, Colors
from common.tracker import ObjectTracker
from common.utils import (
    create_video_writer,
    generate_synthetic_test_video,
    get_optimal_device,
)
from common.visualization import (
    draw_corner_frame,
    draw_dashboard_panel,
    draw_scanline,
    glow_box,
    pill_label,
)

colorama_init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ANPR-SYSTEM")

BANNER = f"""
{Fore.GREEN}╔══════════════════════════════════════════════════════════════════╗
║     LICENSE PLATE INTELLIGENCE SYSTEM   v4.0 (Production)       ║
║     Dual-Stage YOLOv8 + EasyOCR Automatic Number Plate Engine   ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""


def ensure_plate_model(model_dir: Path = MODELS_DIR) -> str:
    """Ensure the specialized license-plate YOLO model is cached locally."""
    model_dir.mkdir(parents=True, exist_ok=True)
    target_path = model_dir / ANPRConfig.plate_model_name

    # Check if already present next to script or in models dir
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return str(target_path)

    script_parent_target = Path(__file__).resolve().parent.parent.parent / ANPRConfig.plate_model_name
    if script_parent_target.exists() and script_parent_target.stat().st_size > 1_000_000:
        return str(script_parent_target)

    logger.info(f"Downloading license plate detector weights to {target_path}...")
    try:
        import ssl

        try:
            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_context = ssl._create_unverified_context()

        req = urllib.request.Request(
            ANPRConfig.plate_model_url, headers={"User-Agent": "Mozilla/5.0 (Computer Vision Intelligence Suite)"}
        )
        with urllib.request.urlopen(req, context=ssl_context) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())

        if target_path.exists() and target_path.stat().st_size > 1_000_000:
            logger.info("Plate model downloaded successfully.")
            return str(target_path)
    except Exception as e:
        logger.warning(f"Download from primary URL failed ({e}). Using standard YOLO weights.")

    return "yolov8n.pt"


class EasyOCREngine:
    """Optical Character Recognition engine with contrast enhancing and whitelist cleaning."""

    def __init__(self, device: str = "cpu"):
        import easyocr

        use_gpu = device == "cuda"
        logger.info(f"Initializing EasyOCR (GPU={use_gpu})...")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        logger.info("EasyOCR engine initialized.")

    def read_plate(self, roi: np.ndarray) -> tuple[str, float]:
        """Extract alphanumeric license plate text from cropped ROI."""
        if roi is None or roi.size == 0:
            return "", 0.0
        h, w = roi.shape[:2]
        if h < 8 or w < 16:
            return "", 0.0

        # Preprocessing: resize to standard height for OCR
        target_h = 80
        scale = target_h / float(max(h, 1))
        roi_scaled = cv2.resize(roi, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)

        # Bilateral filter and adaptive threshold
        gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)

        results = self.reader.readtext(binary, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1)
        if not results:
            # Retry on grayscale if binary threshold had no characters
            results = self.reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1)

        if not results:
            return "", 0.0

        texts, confs = [], []
        for _, text, conf in results:
            clean = re.sub(r"[^A-Z0-9]", "", text.upper())
            if clean and conf > 0.20:
                texts.append(clean)
                confs.append(conf)

        if texts:
            return "".join(texts), float(np.mean(confs))
        return "", 0.0


class LicensePlateSystem:
    """Production License Plate Intelligence Pipeline."""

    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(
        self,
        source: str,
        detect_every: int = 2,
        half_res: bool = False,
        ocr_recheck_interval: int = 45,
        max_ocr_per_frame: int = 3,
        plate_conf: float = 0.25,
        output_dir: Path | None = None,
        device: str | None = None,
    ):
        self.source = source
        self.detect_every = detect_every
        self.half_res = half_res
        self.ocr_recheck_interval = ocr_recheck_interval
        self.max_ocr_per_frame = max_ocr_per_frame
        self.plate_conf = plate_conf

        self.output_dir = Path(output_dir) if output_dir else Path("outputs/license_plate")
        self.screenshots_dir = self.output_dir / "screenshots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or get_optimal_device()
        self.tracker = ObjectTracker(max_disappeared=30, iou_threshold=0.3, fps=30.0)
        self.plate_records: list[dict[str, Any]] = []
        self.plate_counts: dict[str, int] = defaultdict(int)
        self.total_vehicles = 0
        self.frame_num = 0
        self._last_detections: list[tuple[tuple[int, int, int, int], str, float]] = []

        self._load_models()
        self.ocr = EasyOCREngine(device=self.device)

    def _load_models(self) -> None:
        from ultralytics import YOLO

        logger.info(f"Target compute device: {self.device.upper()}")

        # Vehicle detection model
        self.vehicle_model = YOLO(ANPRConfig.vehicle_model_path)
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.vehicle_model(dummy, verbose=False, device=self.device)
        logger.info("Vehicle detector (YOLOv8n) ready.")

        # Plate detection model
        plate_weights = ensure_plate_model()
        self.plate_model = YOLO(plate_weights)
        self.plate_model(dummy, verbose=False, device=self.device)
        logger.info(f"License plate detector loaded ({plate_weights}).")

    def _detect_vehicles(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], str, float]]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2)) if self.half_res else frame
        scale = 2.0 if self.half_res else 1.0

        res = self.vehicle_model(
            small,
            verbose=False,
            device=self.device,
            classes=list(self.VEHICLE_CLASSES.keys()),
            conf=ANPRConfig.vehicle_conf,
        )[0]
        dets = []
        for box in res.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append(
                ((int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)), self.VEHICLE_CLASSES[cid], conf)
            )
        return dets

    def _detect_plates(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        res = self.plate_model(frame, verbose=False, device=self.device, conf=self.plate_conf)[0]
        plates = []
        for box in res.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1, y1, x2, y2, conf))
        return plates

    @staticmethod
    def _is_box_inside(inner: tuple[int, int, int, int, float], outer: tuple[int, int, int, int]) -> bool:
        ix1, iy1, ix2, iy2, _ = inner
        ox1, oy1, ox2, oy2 = outer
        cx, cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0
        return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

    def run(self, max_frames: int | None = None, enable_gui: bool = False) -> Path:
        print(BANNER)

        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        out_video_path = self.output_dir / "annotated_anpr.mp4"
        writer, actual_out_path = create_video_writer(out_video_path, fps_in, fw, fh)
        logger.info(f"Processing ANPR: {self.source} -> {actual_out_path}")

        pbar = tqdm(
            total=min(total_f, max_frames) if (total_f and max_frames) else (total_f or None),
            desc=f"{Fore.GREEN}ANPR PIPELINE{Style.RESET_ALL}",
            unit="f",
            dynamic_ncols=True,
            colour="green",
        )

        t_prev = time.perf_counter()

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

                # Detect vehicles & update tracker
                if self.frame_num % self.detect_every == 0 or self.frame_num == 1:
                    self._last_detections = self._detect_vehicles(frame)

                tracks = self.tracker.update(self._last_detections, self.frame_num)
                self.total_vehicles = max(self.total_vehicles, self.tracker.next_id - 1)

                # Detect plates
                plate_boxes = self._detect_plates(frame) if tracks else []
                ocr_budget = self.max_ocr_per_frame

                for tr in tracks:
                    x1, y1, x2, y2 = tr.bbox
                    glow_box(frame, x1, y1, x2, y2, Colors.NEON_GREEN, self.frame_num)
                    pill_label(frame, f"VEH #{tr.track_id:03d} {tr.category.upper()}", x1, y1 - 6, Colors.NEON_CYAN)

                    # Initialize track metadata for plate records
                    if "confirmed_plate" not in tr.metadata:
                        tr.metadata["confirmed_plate"] = ""
                        tr.metadata["confirmed_conf"] = 0.0
                        tr.metadata["last_ocr_frame"] = -999

                    # Find matching plate for this vehicle
                    matching_plates = [p for p in plate_boxes if self._is_box_inside(p, tr.bbox)]
                    for px1, py1, px2, py2, pconf in matching_plates:
                        # Draw plate bounding box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), Colors.NEON_YELLOW, 2)

                        # Throttle OCR execution
                        should_ocr = (ocr_budget > 0) and (
                            not tr.metadata["confirmed_plate"]
                            or (self.frame_num - tr.metadata["last_ocr_frame"]) >= self.ocr_recheck_interval
                        )

                        if should_ocr:
                            plate_roi = frame[max(0, py1) : min(fh, py2), max(0, px1) : min(fw, px2)]
                            text, conf = self.ocr.read_plate(plate_roi)
                            tr.metadata["last_ocr_frame"] = self.frame_num
                            ocr_budget -= 1

                            if text and len(text) >= 4:
                                if conf > tr.metadata["confirmed_conf"]:
                                    tr.metadata["confirmed_plate"] = text
                                    tr.metadata["confirmed_conf"] = conf
                                    self.plate_counts[text] += 1

                                self.plate_records.append(
                                    {
                                        "plate_text": text,
                                        "confidence": round(conf, 3),
                                        "vehicle_id": tr.track_id,
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "frame_number": self.frame_num,
                                    }
                                )

                    # Render plate badge if found
                    plate_display = tr.metadata["confirmed_plate"]
                    if plate_display:
                        badge_text = f"PLATE: {plate_display} ({tr.metadata['confirmed_conf']:.0%})"
                        pill_label(frame, badge_text, x1, y2 + 18, Colors.NEON_YELLOW, fs=0.38)

                # Telemetry dashboard
                top_plate = max(self.plate_counts, key=self.plate_counts.get) if self.plate_counts else "N/A"
                stats_rows = [
                    ("ANPR FPS", f"{fps_disp:>5.1f}", Colors.TEXT_DIM),
                    ("FRAME", f"{self.frame_num:>6d}", Colors.TEXT_DIM),
                    ("VEHICLES LOGGED", f"{self.total_vehicles:>5d}", Colors.NEON_GREEN),
                    ("READS RECORDED", f"{len(self.plate_records):>5d}", Colors.NEON_YELLOW),
                    ("UNIQUE PLATES", f"{len(self.plate_counts):>5d}", Colors.NEON_CYAN),
                    ("PRIMARY PLATE", f"{top_plate:<12}", Colors.NEON_PINK),
                    ("RECOGNITION", "ONLINE", Colors.THREAT_LOW),
                ]

                draw_scanline(frame, self.frame_num)
                draw_corner_frame(frame)
                draw_dashboard_panel(frame, "ANPR RECOGNITION TELEMETRY", stats_rows, width=285)

                writer.write(frame)
                pbar.update(1)
                pbar.set_postfix(
                    {"veh": self.total_vehicles, "plates": len(self.plate_records), "fps": f"{fps_disp:.1f}"}
                )

        finally:
            pbar.close()
            cap.release()
            writer.release()

        self._export_reports()
        logger.info(
            f"ANPR pipeline complete. Processed {self.frame_num} frames. Plates recorded: {len(self.plate_records)}"
        )
        return Path(actual_out_path)

    def _export_reports(self) -> None:
        if self.plate_records:
            csv_path = self.output_dir / "plates.csv"
            pd.DataFrame(self.plate_records).to_csv(csv_path, index=False)
            logger.info(f"ANPR plates CSV -> {csv_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="License Plate Intelligence System (ANPR v4.0 Production)")
    parser.add_argument("source_pos", nargs="?", default=None, help="Positional video source path")
    parser.add_argument("--input", "-i", dest="source_opt", default=None, help="Input video path or camera index")
    parser.add_argument("--synthetic", action="store_true", help="Generate and run on synthetic traffic scenario")
    parser.add_argument("--every", type=int, default=2, help="Vehicle detection frequency")
    parser.add_argument("--plate-conf", type=float, default=0.25, help="License plate confidence threshold")
    parser.add_argument("--out-dir", default=None, help="Custom output directory")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--gui", action="store_true", help="Launch interactive Tkinter desktop GUI")
    args = parser.parse_args()

    source = args.source_opt or args.source_pos
    if args.synthetic or not source:
        test_video = Path("outputs/synthetic_traffic.mp4")
        generate_synthetic_test_video(test_video, num_frames=120, scenario="anpr")
        source = str(test_video)

    system = LicensePlateSystem(
        source=source,
        detect_every=args.every,
        plate_conf=args.plate_conf,
        output_dir=Path(args.out_dir) if args.out_dir else None,
    )
    system.run(max_frames=args.frames, enable_gui=args.gui)


if __name__ == "__main__":
    main()
