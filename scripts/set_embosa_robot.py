#!/usr/bin/env python3
"""Point the system-wide embosa config at a named robot from config/embosa_robots.json.

embosa (the Galbot SDK's comms layer) reads its peer IPs from a hardcoded path,
/data/config/embosa_ip_config.json, with no env var or API override -- so
switching which robot the SDK talks to means overwriting that file. This
script keeps the known robots' IPs versioned in config/embosa_robots.json and
does the overwrite via `sudo tee`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROBOTS_FILE = Path(__file__).resolve().parent.parent / "config" / "embosa_robots.json"
TARGET = Path("/data/config/embosa_ip_config.json")


def load_robots() -> dict:
    return json.loads(ROBOTS_FILE.read_text())["robots"]


def print_robots(robots: dict) -> None:
    print(f"Known robots ({ROBOTS_FILE}):")
    for name, entry in robots.items():
        print(f"  {name}: local={entry['local_interface']} peers={entry['peer_lists']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", nargs="?", help="Robot name from config/embosa_robots.json")
    parser.add_argument("--list", action="store_true", help="List known robots and exit")
    args = parser.parse_args()

    robots = load_robots()

    if args.list or not args.robot:
        print_robots(robots)
        return 0

    if args.robot not in robots:
        print(f"[ERROR] Unknown robot '{args.robot}'. Known: {', '.join(robots)}", file=sys.stderr)
        return 1

    entry = robots[args.robot]
    payload = {
        "embosa_ip": {
            "local_interface": entry["local_interface"],
            "peer_lists": entry["peer_lists"],
        }
    }
    content = json.dumps(payload, indent=4) + "\n"

    print(f"Writing {TARGET} for robot '{args.robot}':\n{content}")
    result = subprocess.run(
        ["sudo", "tee", str(TARGET)], input=content, text=True, stdout=subprocess.DEVNULL, check=False
    )
    if result.returncode != 0:
        print("[ERROR] Failed to write config (sudo failed or was cancelled)", file=sys.stderr)
        return 1
    print(f"[OK] embosa now targets '{args.robot}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
