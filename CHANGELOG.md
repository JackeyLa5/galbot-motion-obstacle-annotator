# Changelog

本项目的更新记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，目前尚未按 SemVer 发版，条目按提交时间倒序排列。

## [未发布]

工作区未提交的改动（`feat/arm_workspace` 分支）。让可插拔规划架构里的 Galbot Motion 规划器真正可用：

### 新增
- `compare_planners`：新增「对比 PyRoki / Galbot Motion」按钮，用同一个 TCP 目标分别跑两个规划器，叠加显示两条末端路径（青色 PyRoki / 橙色 Galbot Motion），用于判断 Galbot Motion 自身规划器是否也认为该目标可达。
- `scripts/set_embosa_robot.py` + `config/embosa_robots.json`：一键切换 embosa（Galbot SDK 通信层）指向的机器人 IP，替代手动 `sudo vi` 系统文件 `/data/config/embosa_ip_config.json`。

## [837623e] - 2026-08-18 — 可达工作空间可视化与 UI 重构

- **重构**：将 2300 行的 `main_window.py` 上帝类拆分为 `main_window/` 包，按关注点拆成点云、选中、障碍物、机器人、TCP、规划、导出、工作空间等 mixin，便于维护。
- **新增**：可达工作空间点云（Monte-Carlo 关节采样），支持点击预览对应姿态；安装 PyRoki 时启用碰撞感知采样（自碰撞 + 障碍物碰撞，并以机器人当前姿态为基准校准，避免结构上本就相邻、恒定接触的部件被误判为碰撞），未安装则退化为纯运动学采样。
- **修复**：TCP gizmo 拖拽/旋转此前跟随机器人底盘坐标系，现在正确跟随末端执行器自身的局部坐标系。
- **UI**：重新设计 Qt 界面，采用与 3D 场景强调色一致的深色主题、可折叠的控制面板分区、主/次按钮层级；修复了因新主题变得可见的关节滑块控件泄漏问题（`deleteLater()` 在渲染密集的场景下不会及时触发）。
- **清理**：修复全仓库 ruff 检查项（import 顺序、类型注解现代化），并为有意的宽泛异常边界和外部输入校验补充逐文件 ignore 说明。

## [d7f8957] - 2026-08-13 — PyRoki 规划与机器人/TCP 编辑体验改进

- **新增**：`planning/` 规划器抽象层（`protocol.py` / `registry.py` / `models.py`），支持插拔式规划后端；接入 PyRoki 规划器 (`planning/pyroki.py`) 与 Galbot Motion 规划器 (`planning/galbot_motion.py`)。
- **新增**：`scripts/test_pyroki_plan.py` 独立测试脚本；`docs/research/pluggable_motion_planning.md` 记录可插拔规划的调研。
- **新增**：`robot_state.py` 统一管理机器人关节状态。
- 改进机器人模型加载与 TCP 编辑交互体验。

## [45187c4] - 2026-08-13 — 改进障碍物标注流程

- 改进障碍物的导入/导出逻辑（`importers.py`、`exporters.py`）与主窗口交互。
- 新增 `tools/generate_test_pcd.py` 及配套测试点云数据 `tests/data/test.pcd`，用于生成可重复的测试用点云。

## [dcbf892] - 2026-08-12 — 加载和编辑障碍物 JSON

- 新增 `importers.py`：支持加载已有的障碍物 JSON 并在标注器中编辑。

## [2bfcc1b] - 2026-08-12 — 障碍物查询与删除脚本

- 新增 `scripts/query_obstacles.py`：查询 Galbot Motion 中已加载的障碍物 ID。
- 新增 `scripts/delete_obstacle.py`：按 ID 删除已加载的障碍物。

## [31c59c7] - 2026-08-12 — 初始提交

- 项目骨架：基于 PySide6 + PyVista + VTK 的点云碰撞体标注工具。
- 核心能力：加载/裁剪点云、右键逐点采样、自动拟合 Box/Sphere/Cylinder、3D 交互编辑（拖动/旋转/缩放/精确数值编辑）、多障碍物管理、URDF 机器人模型加载与底盘位姿设置、导出 Galbot JSON 与可运行的 Python 脚本模板。
