#!/usr/bin/env python3
"""
License Plate Intelligence System (ANPR)
Modular Production Entrypoint.
"""

import sys
from pathlib import Path

# Ensure root is on path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from license_plate_intelligence.src.main import LicensePlateSystem, main

if __name__ == "__main__":
    main()
