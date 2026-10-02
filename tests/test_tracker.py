"""
Unit tests for ObjectTracker, Track kinematics, and IoU algorithms.
"""

import pytest

from common.tracker import ObjectTracker, Track, compute_iou


def test_compute_iou():
    # Identical boxes -> IoU = 1.0
    box_a = (10, 10, 50, 50)
    assert compute_iou(box_a, box_a) == pytest.approx(1.0, 0.01)

    # Disjoint boxes -> IoU = 0.0
    box_b = (100, 100, 150, 150)
    assert compute_iou(box_a, box_b) == 0.0

    # 50% overlap box
    box_c = (10, 10, 30, 50)  # Area 20x40 = 800
    box_d = (20, 10, 40, 50)  # Area 20x40 = 800, intersection 10x40 = 400
    # Union = 800 + 800 - 400 = 1200. IoU = 400/1200 = 1/3 ~ 0.333
    assert compute_iou(box_c, box_d) == pytest.approx(1.0 / 3.0, 0.01)


def test_track_centroid_and_motion():
    track = Track(track_id=1, bbox=(100, 100, 200, 200), category="aircraft", conf=0.9)
    assert track.centroid() == (150.0, 150.0)

    # Simulate movement history: 5 frames moving right
    track.history.append((100.0, 100.0))
    track.history.append((110.0, 100.0))
    track.history.append((120.0, 100.0))
    track.history.append((130.0, 100.0))
    track.history.append((140.0, 100.0))

    vx, vy = track.velocity_px()
    assert vx == pytest.approx(10.0, 0.01)
    assert vy == pytest.approx(0.0, 0.01)

    track.update_motion(px_per_meter=8.0, fps=30.0)
    assert track.heading_deg == pytest.approx(0.0, 0.1)  # Moving strictly right = 0 deg
    assert track.speed_kmh > 0.0


def test_object_tracker_lifecycle():
    tracker = ObjectTracker(max_disappeared=3, iou_threshold=0.25)

    # Frame 1: First detection
    dets_f1 = [((50, 50, 100, 100), "car", 0.85)]
    tracks = tracker.update(dets_f1, frame_num=1)
    assert len(tracks) == 1
    t1_id = tracks[0].track_id

    # Frame 2: Slightly moved detection
    dets_f2 = [((55, 52, 105, 102), "car", 0.88)]
    tracks = tracker.update(dets_f2, frame_num=2)
    assert len(tracks) == 1
    assert tracks[0].track_id == t1_id  # ID must persist

    # Frames 3, 4, 5, 6: No detections (object disappeared)
    tracker.update([], frame_num=3)
    tracker.update([], frame_num=4)
    tracker.update([], frame_num=5)
    tracks = tracker.update([], frame_num=6)

    # Track must be expired after max_disappeared=3
    assert t1_id not in tracker.tracks
