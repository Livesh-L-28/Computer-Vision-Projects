"""
Unified Object Tracking Engine for Computer Vision Intelligence Suite.
Supports both native YOLO ByteTrack/BoT-SORT and enhanced standalone tracking.
"""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(intersection / (area_a + area_b - intersection))


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    category: str
    conf: float
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: int = 0
    frames_seen: int = 0
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def velocity_px(self, lookback: int = 5) -> tuple[float, float]:
        """Compute pixel velocity vector (vx, vy) per frame over recent history."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        n = min(lookback, len(self.history))
        p0 = self.history[-n]
        p1 = self.history[-1]
        dt = float(n - 1)
        if dt <= 0.0:
            return (0.0, 0.0)
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)

    def update_motion(self, px_per_meter: float = 8.0, fps: float = 30.0) -> None:
        """Update speed and heading based on recent motion."""
        vx, vy = self.velocity_px()
        speed_px = math.hypot(vx, vy)

        # Heading (0 to 360 deg, 0=Right, 90=Down in screen space)
        if speed_px > 0.3:
            self.heading_deg = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0

        # Speed in km/h
        if px_per_meter > 0:
            speed_mps = (speed_px * fps) / px_per_meter
            self.speed_kmh = speed_mps * 3.6
        else:
            self.speed_kmh = speed_px * 18.0

    def predict_future(self, steps: int = 5, gap: int = 10) -> list[tuple[int, int]]:
        """Project future centroid coordinates."""
        cx, cy = self.centroid()
        vx, vy = self.velocity_px()
        points = []
        for i in range(1, steps + 1):
            points.append((int(cx + vx * gap * i), int(cy + vy * gap * i)))
        return points


class ObjectTracker:
    """Robust multi-object tracker with persistence, velocity smoothing, and IOU matching."""

    def __init__(
        self,
        max_disappeared: int = 25,
        iou_threshold: float = 0.25,
        px_per_meter: float = 8.0,
        fps: float = 30.0,
        max_history: int = 30,
    ):
        self.tracks: dict[int, Track] = {}
        self.next_id: int = 1
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.px_per_meter = px_per_meter
        self.fps = fps
        self.max_history = max_history

    def update(self, detections: list[tuple[tuple[int, int, int, int], str, float]], frame_num: int) -> list[Track]:
        """
        Update tracker with current frame detections.
        detections: list of ((x1, y1, x2, y2), category_name, confidence)
        """
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Match existing tracks with detections based on IoU
        for tid, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det_idx = -1

            for idx in unmatched_dets:
                bbox = detections[idx][0]
                iou = compute_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx != -1:
                bbox, category, conf = detections[best_det_idx]
                track.bbox = bbox
                track.category = category
                track.conf = conf
                track.last_seen = frame_num
                track.frames_seen += 1
                track.history.append(track.centroid())
                track.update_motion(self.px_per_meter, self.fps)
                matched_track_ids.add(tid)
                unmatched_dets.remove(best_det_idx)

        # Create new tracks for unmatched detections
        for idx in unmatched_dets:
            bbox, category, conf = detections[idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=bbox,
                category=category,
                conf=conf,
                last_seen=frame_num,
                frames_seen=1,
            )
            new_track.history.append(new_track.centroid())
            self.tracks[self.next_id] = new_track
            matched_track_ids.add(self.next_id)
            self.next_id += 1

        # Prune stale tracks
        expired = [tid for tid, tr in self.tracks.items() if (frame_num - tr.last_seen) > self.max_disappeared]
        for tid in expired:
            del self.tracks[tid]

        # Return active tracks visible in current frame
        return [self.tracks[tid] for tid in matched_track_ids if tid in self.tracks]
