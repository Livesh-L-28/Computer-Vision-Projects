"""
Utility functions: Video I/O, device selection, synthetic data generation, and reporting.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CV-SUITE")


def get_optimal_device() -> str:
    """Detect the best computing device: Apple Silicon (mps), CUDA GPU, or CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter, str]:
    """Create a VideoWriter attempting standard H.264 codecs with safe fallbacks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codecs_to_try = [
        ("avc1", ".mp4"),
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
    ]

    for codec, ext in codecs_to_try:
        try:
            target_file = str(output_path.with_suffix(ext))
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(target_file, fourcc, max(1.0, fps), (int(width), int(height)))
            if writer.isOpened():
                return writer, target_file
        except Exception as e:
            logger.debug(f"Codec {codec} failed: {e}")

    # Default fallback
    target_file = str(output_path.with_suffix(".mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(target_file, fourcc, max(1.0, fps), (int(width), int(height)))
    return writer, target_file


def generate_synthetic_test_video(
    output_path: Path, num_frames: int = 120, width: int = 1280, height: int = 720, scenario: str = "combat"
) -> str:
    """
    Generate an animated synthetic test video simulating aircraft or vehicles.
    Ensures end-to-end functionality can be verified immediately without external footage.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer, final_file = create_video_writer(output_path, 30.0, width, height)

    # Define simulated moving objects
    if scenario == "combat":
        objects = [
            {"pos": [150.0, 200.0], "vel": [6.0, 2.0], "size": (60, 30), "color": (120, 120, 180)},
            {"pos": [width - 150.0, 400.0], "vel": [-5.0, -1.5], "size": (50, 25), "color": (140, 140, 160)},
            {"pos": [300.0, 550.0], "vel": [3.0, -1.0], "size": (30, 20), "color": (100, 100, 100)},
        ]
    elif scenario == "runway":
        objects = [
            {"pos": [100.0, height // 2], "vel": [7.0, 0.0], "size": (120, 50), "color": (160, 160, 180)},
            {"pos": [width // 2, 220.0], "vel": [2.0, 1.0], "size": (45, 30), "color": (80, 180, 80)},
            {"pos": [width // 2 - 100, height - 150], "vel": [1.0, -0.5], "size": (20, 20), "color": (200, 200, 50)},
        ]
    else:  # traffic / anpr
        objects = [
            {"pos": [200.0, 250.0], "vel": [4.0, 3.0], "size": (110, 70), "color": (60, 80, 200)},
            {"pos": [700.0, 180.0], "vel": [5.0, 2.5], "size": (130, 80), "color": (180, 80, 60)},
        ]

    for f in range(num_frames):
        # Synthetic backdrop (sky / tarmac / road)
        if scenario == "combat":
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [45, 30, 20]  # Dark tactical airspace
            # Star/grid effect
            cv2.line(frame, (0, height // 2), (width, height // 2), (30, 35, 45), 1)
        elif scenario == "runway":
            frame = np.full((height, width, 3), (35, 40, 35), dtype=np.uint8)
            # Runway stripe
            cv2.rectangle(frame, (50, height // 2 - 40), (width - 50, height // 2 + 40), (60, 60, 60), -1)
            for x in range(100, width - 100, 80):
                cv2.rectangle(frame, (x, height // 2 - 4), (x + 40, height // 2 + 4), (200, 200, 200), -1)
        else:
            frame = np.full((height, width, 3), (40, 40, 45), dtype=np.uint8)
            cv2.rectangle(frame, (0, 200), (width, height - 100), (65, 65, 70), -1)

        for obj in objects:
            x, y = int(obj["pos"][0]), int(obj["pos"][1])
            w_box, h_box = obj["size"]
            x1, y1 = max(0, x - w_box // 2), max(0, y - h_box // 2)
            x2, y2 = min(width - 1, x + w_box // 2), min(height - 1, y + h_box // 2)

            # Draw synthetic object body
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj["color"], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # If ANPR, draw a small white rectangle resembling a license plate
            if scenario == "anpr":
                px1, py1 = x1 + 20, y2 - 18
                px2, py2 = min(x2 - 20, px1 + 70), y2 - 4
                if px2 > px1:
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (240, 240, 240), -1)
                    cv2.putText(frame, "KA01AB12", (px1 + 4, py2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)

            # Advance position
            obj["pos"][0] += obj["vel"][0]
            obj["pos"][1] += obj["vel"][1]
            if obj["pos"][0] < 50 or obj["pos"][0] > width - 50:
                obj["vel"][0] *= -1
            if obj["pos"][1] < 50 or obj["pos"][1] > height - 50:
                obj["vel"][1] *= -1

        writer.write(frame)

    writer.release()
    logger.info(f"Synthetic test video created: {final_file}")
    return final_file


def save_csv_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save record list as a structured CSV file."""
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Report saved to {output_path}")
