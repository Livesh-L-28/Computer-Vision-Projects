"""
Visualization & Tactical HUD drawing library for Computer Vision Intelligence Suite.
"""

import math

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .config import Colors

FONT = cv2.FONT_HERSHEY_SIMPLEX


def alpha_rect(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], alpha: float = 0.5
) -> None:
    """Draw a semi-transparent colored rectangle in-place."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    overlay = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, dst=sub)


def corner_ticks(
    frame: np.ndarray, bbox: Sequence[int], color: tuple[int, int, int], length: int = 14, thickness: int = 2
) -> None:
    """Draw corner brackets around a bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    length = min(length, max(4, (x2 - x1) // 3), max(4, (y2 - y1) // 3))

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def glow_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    fnum: int = 0,
    thickness: int = 2,
) -> None:
    """Draw a neon glowing bounding box with pulsating corner reticles."""
    alpha_rect(frame, x1, y1, x2, y2, color, alpha=0.15)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    corner_ticks(frame, (x1, y1, x2, y2), color, length=12 + int(3 * abs(math.sin(fnum * 0.15))), thickness=2)


def pill_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int] = Colors.PANEL_BG,
    fs: float = 0.38,
    th: int = 1,
    pad: int = 4,
) -> None:
    """Render a pill-shaped tactical text badge."""
    (tw, th_), base = cv2.getTextSize(text, FONT, fs, th)
    h, w = frame.shape[:2]

    x = max(pad, min(int(x), w - tw - pad))
    y = max(th_ + pad, min(int(y), h - pad))

    x1, y1 = x - pad, y - th_ - pad
    x2, y2 = x + tw + pad, y + base + pad

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fg, 1)
    cv2.putText(frame, text, (x, y), FONT, fs, fg, th, cv2.LINE_AA)


def radar_ring(frame: np.ndarray, cx: int, cy: int, fnum: int, color: tuple[int, int, int], max_r: int = 36) -> None:
    """Draw an expanding radar pulse ring."""
    r = 12 + (fnum * 2) % max_r
    alpha = max(0.0, 1.0 - r / float(max_r))
    if alpha <= 0.05:
        return
    ov = frame.copy()
    cv2.circle(ov, (int(cx), int(cy)), int(r), color, 1)
    cv2.addWeighted(ov, alpha * 0.7, frame, 1.0 - (alpha * 0.7), 0, dst=frame)


def motion_trail(frame: np.ndarray, history: Sequence[tuple[float, float]], color: tuple[int, int, int]) -> None:
    """Draw a fading motion trail through historical centroid positions."""
    pts = list(history)
    n = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        a = i / float(n)
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]), int(pts[i][1]))
        ov = frame.copy()
        cv2.line(ov, p1, p2, color, 2, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.2 + 0.5 * a, frame, 1.0 - (0.2 + 0.5 * a), 0, dst=frame)


def direction_vector(
    frame: np.ndarray, cx: float, cy: float, vx: float, vy: float, color: tuple[int, int, int], scale: float = 7.0
) -> None:
    """Draw a velocity direction arrow."""
    mag = math.hypot(vx, vy)
    if mag < 0.2:
        return
    ex = int(cx + vx * scale)
    ey = int(cy + vy * scale)
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), color, 2, tipLength=0.35)


def future_path(
    frame: np.ndarray,
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    color: tuple[int, int, int],
    steps: int = 5,
    gap: int = 10,
) -> None:
    """Draw extrapolated future motion waypoints."""
    mag = math.hypot(vx, vy)
    if mag < 0.2:
        return
    for i in range(1, steps + 1):
        fx = int(cx + vx * gap * i)
        fy = int(cy + vy * gap * i)
        if i % 2 == 0:
            cv2.circle(frame, (fx, fy), 3, color, -1)
        if i == steps:
            cv2.circle(frame, (fx, fy), 8, color, 1)
            cv2.putText(frame, "PRED", (fx + 10, fy), FONT, 0.30, color, 1, cv2.LINE_AA)


def draw_scanline(frame: np.ndarray, fnum: int, color: tuple[int, int, int] = Colors.BORDER_CYAN) -> None:
    """Simulate a subtle CRT/radar scanline effect."""
    h, w = frame.shape[:2]
    yy = (fnum * 3) % h
    ov = frame.copy()
    cv2.line(ov, (0, yy), (w, yy), color, 1)
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, dst=frame)


def draw_corner_frame(
    frame: np.ndarray, color: tuple[int, int, int] = Colors.BORDER_CYAN, size: int = 36, thickness: int = 2
) -> None:
    """Draw framing HUD brackets in all 4 screen corners."""
    h, w = frame.shape[:2]
    # Top-Left
    cv2.line(frame, (0, 0), (size, 0), color, thickness)
    cv2.line(frame, (0, 0), (0, size), color, thickness)
    # Top-Right
    cv2.line(frame, (w - 1, 0), (w - 1 - size, 0), color, thickness)
    cv2.line(frame, (w - 1, 0), (w - 1, size), color, thickness)
    # Bottom-Left
    cv2.line(frame, (0, h - 1), (size, h - 1), color, thickness)
    cv2.line(frame, (0, h - 1), (0, h - 1 - size), color, thickness)
    # Bottom-Right
    cv2.line(frame, (w - 1, h - 1), (w - 1 - size, h - 1), color, thickness)
    cv2.line(frame, (w - 1, h - 1), (w - 1, h - 1 - size), color, thickness)


def draw_dashboard_panel(
    frame: np.ndarray,
    title: str,
    rows: list[tuple[str, Any, tuple[int, int, int]]],
    x: int | None = None,
    y: int = 10,
    width: int = 280,
    accent: tuple[int, int, int] = Colors.BORDER_CYAN,
) -> None:
    """Render a tactical telemetry dashboard panel."""
    fh, fw = frame.shape[:2]
    if x is None:
        x = fw - width - 12
    height = 30 + len(rows) * 18

    alpha_rect(frame, x, y, x + width, y + height, Colors.PANEL_BG, 0.88)
    cv2.rectangle(frame, (x, y), (x + width, y + height), accent, 1)
    cv2.line(frame, (x + 1, y + 22), (x + width - 1, y + 22), accent, 1)
    cv2.putText(frame, title, (x + 8, y + 16), FONT, 0.36, accent, 1, cv2.LINE_AA)

    for i, (k, v, col) in enumerate(rows):
        ky = y + 36 + i * 18
        cv2.putText(frame, f" {k:<15}", (x + 6, ky), FONT, 0.30, Colors.TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, str(v), (x + 170, ky), FONT, 0.30, col, 1, cv2.LINE_AA)


def generate_activity_heatmap(
    heatmap_pts: Sequence[tuple[float, float]],
    width: int,
    height: int,
    sigma: int = 20,
    save_path: str | None = None,
    title: str = "Activity Heatmap",
) -> np.ndarray | None:
    """Generate and save a Gaussian-smoothed spatial density heatmap."""
    if not heatmap_pts:
        return None
    hm = np.zeros((height, width), dtype=np.float32)
    for x, y in heatmap_pts:
        xi, yi = int(x), int(y)
        if 0 <= yi < height and 0 <= xi < width:
            hm[yi, xi] += 1.0

    blurred = gaussian_filter(hm, sigma=sigma)
    if save_path:
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0a0a14")
        ax.set_facecolor("#0a0a14")
        im = ax.imshow(blurred, cmap="inferno", interpolation="bilinear")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Spatial Density", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        ax.set_title(title, color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, facecolor="#0a0a14")
        plt.close()
    return blurred
