from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate Galbot Motion collision boxes in point clouds")
    parser.add_argument("point_cloud", nargs="?", help="PCD, PLY, VTK or VTP point cloud file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    application = QApplication(sys.argv)
    window = MainWindow(args.point_cloud)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

