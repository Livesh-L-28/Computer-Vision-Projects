#!/usr/bin/env python3
"""
Air Combat Intelligence System
Modular Production Entrypoint.
"""

import sys
from pathlib import Path

# Ensure root is on path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from air_combat_intelligence.src.main import AirCombatSystem, main

if __name__ == "__main__":
    main()
