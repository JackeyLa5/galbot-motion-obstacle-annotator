#!/usr/bin/env python3
"""Interactively delete obstacles currently loaded in Galbot Motion."""

from __future__ import annotations

import argparse
import time

import galbot_sdk.g1 as gm
from galbot_sdk.g1 import GalbotMotion, GalbotRobot

PROTECTED_OBSTACLE_IDS = {"ground"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="循环删除 Galbot Motion 中的障碍物，并保留默认地面 ground。"
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("selection", nargs="?", help="首次删除的列表序号或障碍物 ID")
    target.add_argument("--all", action="store_true", help="删除除 ground 外的全部障碍物")
    parser.add_argument("--yes", action="store_true", help="使用 --all 时跳过确认提示")
    return parser


def print_obstacles(obstacle_ids: list[str]) -> None:
    print(f"\n当前共有 {len(obstacle_ids)} 个障碍物：")
    for index, obstacle_id in enumerate(obstacle_ids, start=1):
        suffix = "  [默认地面，受保护]" if obstacle_id in PROTECTED_OBSTACLE_IDS else ""
        print(f"{index}. {obstacle_id!r}{suffix}")


def resolve_obstacle_id(selection: str, obstacle_ids: list[str]) -> str | None:
    """Resolve a full obstacle ID or a one-based list index."""
    selection = selection.strip()
    if selection in obstacle_ids:
        return selection

    if selection.isdigit():
        index = int(selection)
        if 1 <= index <= len(obstacle_ids):
            return obstacle_ids[index - 1]

    return None


def clear_all_obstacles(motion: GalbotMotion, skip_confirmation: bool) -> int:
    obstacle_ids = list(motion.get_built_obstacles_list())
    removable_ids = [
        obstacle_id
        for obstacle_id in obstacle_ids
        if obstacle_id not in PROTECTED_OBSTACLE_IDS
    ]
    if not removable_ids:
        print("\n没有可删除的障碍物；默认地面 ground 已保留")
        return 0

    if not skip_confirmation:
        confirmation = input(
            f"\n确认删除除 ground 外的 {len(removable_ids)} 个障碍物？请输入 yes："
        ).strip().lower()
        if confirmation != "yes":
            print("已取消删除操作")
            return 0

    failed_ids = []
    for obstacle_id in removable_ids:
        status = motion.remove_obstacle(obstacle_id)
        if status == gm.MotionStatus.SUCCESS:
            print(f"[OK] 已删除障碍物: {obstacle_id!r}")
        else:
            print(f"[ERROR] 删除障碍物 {obstacle_id!r} 失败，status={status}")
            failed_ids.append(obstacle_id)

    print("[INFO] 默认地面 'ground' 已保留")
    return 1 if failed_ids else 0


def delete_one_obstacle(motion: GalbotMotion, selection: str, obstacle_ids: list[str]) -> bool:
    obstacle_id = resolve_obstacle_id(selection, obstacle_ids)
    if obstacle_id is None:
        print(f"[ERROR] 无效的序号或 SDK 中不存在障碍物: {selection!r}")
        return False

    if obstacle_id in PROTECTED_OBSTACLE_IDS:
        print(f"[WARNING] {obstacle_id!r} 是 Motion 默认地面碰撞体，不允许删除")
        return False

    status = motion.remove_obstacle(obstacle_id)
    if status == gm.MotionStatus.SUCCESS:
        print(f"[OK] 已删除障碍物: {obstacle_id!r}")
        return True
    if status == gm.MotionStatus.INVALID_INPUT:
        print(f"[ERROR] 删除失败，障碍物不存在或输入无效: {obstacle_id!r}")
        return False

    print(f"[ERROR] 删除障碍物 {obstacle_id!r} 失败，status={status}")
    return False


def run_delete_loop(motion: GalbotMotion, initial_selection: str | None) -> int:
    selection = initial_selection

    while True:
        obstacle_ids = list(motion.get_built_obstacles_list())
        if not obstacle_ids:
            print("\n当前没有障碍物，程序退出")
            return 0

        if all(obstacle_id in PROTECTED_OBSTACLE_IDS for obstacle_id in obstacle_ids):
            print_obstacles(obstacle_ids)
            print("\n只剩默认地面 ground，程序退出")
            return 0

        print_obstacles(obstacle_ids)

        if selection is None:
            selection = input(
                "\n请输入序号或障碍物 ID；输入 q 退出，输入 all 删除全部非默认障碍物："
            ).strip()
        else:
            selection = selection.strip()

        if selection.lower() in {"q", "quit", "exit"}:
            print("已退出删除程序")
            return 0

        if selection.lower() == "all":
            result = clear_all_obstacles(motion, skip_confirmation=False)
            if result != 0:
                return result
        elif selection:
            delete_one_obstacle(motion, selection, obstacle_ids)
        else:
            print("[ERROR] 输入不能为空")

        selection = None
        time.sleep(0.5)


def main() -> int:
    args = build_parser().parse_args()
    motion = GalbotMotion()
    robot = GalbotRobot()
    robot_initialized = False

    try:
        print("正在初始化 GalbotMotion...")
        if not motion.init():
            print("[ERROR] GalbotMotion 初始化失败")
            return 1

        print("正在初始化 GalbotRobot...")
        robot_initialized = robot.init()
        if not robot_initialized:
            print("[ERROR] GalbotRobot 初始化失败")
            return 1

        time.sleep(2)

        if args.all:
            obstacle_ids = list(motion.get_built_obstacles_list())
            print_obstacles(obstacle_ids)
            if not obstacle_ids:
                return 0
            return clear_all_obstacles(motion, args.yes)

        return run_delete_loop(motion, args.selection)

    except KeyboardInterrupt:
        print("\n用户中断程序")
        return 130
    except Exception as error:
        print(f"[ERROR] 删除障碍物失败: {error}")
        return 1
    finally:
        if robot_initialized:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
