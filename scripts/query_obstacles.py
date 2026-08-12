#!/usr/bin/env python3
"""Print obstacle IDs currently loaded in Galbot Motion."""

from __future__ import annotations

import time

from galbot_sdk.g1 import GalbotMotion, GalbotRobot


def main() -> int:
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
        print(f"检测到的障碍物 ID：{motion.get_built_obstacles_list()}")
        return 0

    except KeyboardInterrupt:
        print("\n用户中断程序")
        return 130
    except Exception as error:
        print(f"[ERROR] 查询障碍物失败: {error}")
        return 1
    finally:
        if robot_initialized:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
