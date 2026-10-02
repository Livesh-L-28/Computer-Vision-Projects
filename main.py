#!/usr/bin/env python3
"""
Computer Vision Intelligence Suite - Unified Production Runner
Supports both Command-Line Interface (CLI) and Web Dashboard Server.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from air_combat_intelligence.src.main import AirCombatSystem
from airport_runway_intelligence.src.main import AirportRunwaySystem
from common.utils import generate_synthetic_test_video
from license_plate_intelligence.src.main import LicensePlateSystem


def run_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn

    print(f"\n🚀 Launching Computer Vision Intelligence Suite Dashboard on http://localhost:{port}")
    uvicorn.run("web.server:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="Computer Vision Intelligence Suite (Production v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive web dashboard & API:
  python main.py --serve --port 8080

  # Run Air Combat Intelligence on synthetic scenario:
  python main.py --system combat --synthetic

  # Run Airport Runway Intelligence on custom video:
  python main.py --system runway --input airport.mp4

  # Run License Plate (ANPR) Intelligence on camera/video:
  python main.py --system anpr --input traffic.mp4
        """,
    )

    parser.add_argument("--serve", action="store_true", help="Launch production Web Dashboard server")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")

    parser.add_argument(
        "--system",
        choices=["combat", "runway", "anpr"],
        default="combat",
        help="Target intelligence system (combat, runway, anpr)",
    )
    parser.add_argument("--input", "-i", default=None, help="Input video path or camera device index")
    parser.add_argument("--synthetic", action="store_true", help="Run on a generated synthetic scenario")
    parser.add_argument("--model", default=None, help="Custom YOLO model weights file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--every", type=int, default=1, help="Inference frame frequency")
    parser.add_argument("--frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--out-dir", default=None, help="Custom output directory")
    parser.add_argument("--gui", action="store_true", help="Launch desktop Tkinter GUI (ANPR only)")

    args = parser.parse_args()

    if args.serve or (len(sys.argv) == 1):
        run_server(host=args.host, port=args.port)
        return

    # CLI Execution
    source = args.input
    if args.synthetic or not source:
        test_video = Path(f"outputs/synthetic_{args.system}.mp4")
        generate_synthetic_test_video(test_video, num_frames=args.frames or 90, scenario=args.system)
        source = str(test_video)

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.system == "combat":
        sys_obj = AirCombatSystem(
            source=source,
            model_path=args.model or "yolov8s.pt",
            conf=args.conf,
            detect_every=args.every,
            output_dir=out_dir,
        )
        sys_obj.run(max_frames=args.frames)

    elif args.system == "runway":
        sys_obj = AirportRunwaySystem(
            source=source,
            model_path=args.model or "yolov8s.pt",
            conf=args.conf,
            detect_every=args.every or 2,
            output_dir=out_dir,
        )
        sys_obj.run(max_frames=args.frames)

    elif args.system == "anpr":
        sys_obj = LicensePlateSystem(
            source=source, detect_every=args.every or 2, plate_conf=args.conf, output_dir=out_dir
        )
        sys_obj.run(max_frames=args.frames, enable_gui=args.gui)


if __name__ == "__main__":
    main()
