# 可插拔运动规划研究

该研究分支用于探索机器人运动可达性比较与 PyVista 轨迹可视化，任何规划结果都不能发送给真实机器人执行。

## 目标

- 在 PyVista 场景中选择笛卡尔目标位姿。
- 使用相同目标、起始状态和障碍物调用一个或多个规划后端。
- 比较可达性、规划耗时和关节路径。
- 使用已加载的 URDF 模型播放规划轨迹。
- 禁止向机器人控制层发送轨迹。

## 插件边界

通用类型位于 `galbot_motion_obstacle_annotator.planning`：

- `PoseTarget`：目标链、目标坐标系、位置和四元数。
- `PlanRequest`：显式起始状态、场景障碍物和后端选项。
- `PlanResult`：规划状态和每条运动链的关节轨迹。
- `MotionPlannerPlugin`：所有规划后端必须实现的协议。
- `PlannerRegistry`：内置插件注册和 Python entry point 自动发现。

外部 Python 包可以通过以下 entry point 注册新的规划插件：

```toml
[project.entry-points."galbot_motion_obstacle_annotator.planners"]
my-planner = "my_planner_package:MyPlanner"
```

插件类必须提供 `metadata`、`is_available()` 和 `plan(request)`。

## Galbot Motion 源码结论

首个内置后端是 `GalbotMotionPlanner`。已检查 `/home/ubuntu/JJJ/SDK/galbot_g1_sdk_source` 中的公共头文件、实现、pybind 和测试：

- `PlannerConfig.is_direct_execute` 默认是 `false`，SDK 注释将其定义为只规划、不执行，适用于预览和验证。
- `motion_plan()` 返回 `dict[str, list[list[float]]]`，字典键是运动链名称。
- Python 返回结果只保留逐帧关节位置；内部 SingoriX 轨迹的时间戳在转换过程中被丢弃。
- 环境碰撞检测只使用通过 `add_obstacle()` 显式加载的障碍物，不会自动同步实时感知。
- 高层只规划路径在没有完整 `reference_robot_states` 时可能读取真实机器人当前状态。
- 当前实现中，只要传入 `reference_robot_states`，它就成为权威状态；单独的 `start` 不会覆盖其中的目标链，和 API 文档描述存在差异。
- 当前机器安装的 Galbot SDK 1.10.0 Python `Parameter` 没有暴露 `enable_env_collision_check`，而当前源码已经包含该字段。

因此当前适配器执行以下强制策略：

1. 无条件将 `is_direct_execute` 设置为 `False`。
2. 只调用 `motion_plan()`，不调用任何移动或轨迹执行 API。
3. 强制要求 `galbot_whole_body_joint_positions` 和 `galbot_base_pose`。
4. 不传分链 `start`，避免 SDK 回退读取真实机器人状态。
5. 加载障碍物时同样显式传入 whole-body joints 和 base pose。
6. 障碍物使用唯一临时 ID，规划完成后只删除本次添加的临时障碍物。
7. 不调用 `clear_obstacle()`，避免删除 Motion 服务中不属于当前工具的障碍物。
8. 如果 SDK 没暴露环境碰撞开关且请求要求环境碰撞检测，直接返回错误，不静默生成忽略障碍物的结果。

适配器仍需要调用 `GalbotMotion.init()` 并连接可用的 Galbot Motion 规划服务，但不会初始化 `GalbotRobot`，也不会请求机器人运动。

## PyRoki 后端

第二个内置后端是 `PyrokiPlanner`，实现依据 PyRoki 官方仓库提交
`388e43e1fc0d0ee382968d3dd72970fd62a0450c` 中的 `examples/pyroki_snippets/_trajopt.py`。
该后端只在本地加载 URDF、构建 JAX/JAXLS 优化问题并返回关节轨迹，不包含机器人连接、控制或执行入口。

PyRoki 使用懒加载，并作为项目的可选依赖提供。普通 `uv sync` 或现有的
`uv run galbot-motion-obstacle-annotator ...` 不会安装它；只有需要使用 PyRoki 时才执行：

```bash
uv sync --extra pyroki
```

该可选组会从 PyRoki 官方 Git 仓库安装 PyRoki 及其 JAX/JAXLS 相关依赖。

调用方必须在 `PlanRequest.options` 中显式提供：

- `pyroki_urdf_path`：URDF 文件路径。
- `pyroki_joint_names`：严格按照 PyRoki 解析出的 URDF actuated joint 顺序排列的关节名。
- `pyroki_start_joint_positions`：按关节名提供的完整起始状态映射。
- `pyroki_active_joint_names`：允许优化的关节；界面默认只放开当前目标手臂的 7 个关节，另一只手臂、腿部、头部、轮子和夹爪固定在当前可视化环境状态。只有明确勾选腿部规划时，才额外放开腿部 5 个关节，共 12 个自由度。
- `pyroki_target_link`：URDF 中的目标末端 link；未提供时使用 `PoseTarget.frame_id`。

可选参数包括 `pyroki_timesteps`、`pyroki_dt`、`pyroki_position_tolerance` 和
`pyroki_orientation_tolerance`。适配器会检查最终 FK 位姿是否落在容差内，优化器返回数值结果但未到达目标时不会标记为可达。

碰撞策略如下：

- `collision_check=True` 时启用机器人自碰撞代价。
- `environment_collision_check=True` 时使用相邻轨迹帧间的扫掠胶囊约束检查场景障碍物。
- box 使用带姿态的 PyRoki box，sphere 使用 sphere。
- PyRoki 当前没有 cylinder 几何，因此 cylinder 使用相同半径、高度和姿态的 capsule 近似；结果诊断中会明确记录该近似。
- 当前只接受 `world` 坐标系中的目标和障碍物，不进行隐式 TF 查询或坐标变换。

PyRoki 首次运行以及障碍物数量或轨迹维度变化时可能触发 JAX 编译，因此首次规划耗时不应直接和已经热身的后续规划比较。

## PyVista 集成顺序

后续 UI 建议分阶段实现：

1. 增加独立的目标位姿选择模式和目标坐标轴 Actor。
2. 使用 `PlannerRegistry` 填充规划器选择框。
3. 从当前显示的机器人状态和障碍物构造 `PlanRequest`。
4. 在 Qt 主线程外运行规划插件。
5. 为不同插件使用不同颜色绘制末端路径。
6. 缓存 URDF 网格 Actor，逐帧应用 FK 变换播放关节轨迹。
7. 增加播放控制和规划结果比较表。

当前主窗口已经提供工具 TCP 抓取姿态编辑：

- 在场景中右键选择一个点后，会显示对应手臂的夹爪视觉模型，而不是抽象坐标轴或额外机器人模型。
- 夹爪模型来自 URDF 中 `end_effector_mount_link` 的后代，并转换到对应的 `*_gripper_tcp_link` 局部坐标。
- 拖动夹爪上的变换控件或修改 XYZ/RPY 后，保存的位姿矩阵就是工具 TCP 在 `world` 中的位姿。
- Galbot Motion 适配器固定设置 `params.is_tool_pose=True`，并将内部目标 frame 标准化为 `TCP`；因此不会把 TCP 姿态误当成法兰姿态。
- `PoseTarget.frame_id` 仍保留真实 URDF TCP link 名称，便于 PyRoki 等直接使用 URDF 的规划器；Galbot Motion 的 SDK 特殊 frame 映射只发生在适配器内部。
- 抓取点尚未选择时不显示原点夹爪；选择成功后才显示腕部法兰、末端安装座、夹爪、腕部相机和三维姿态操控器。预览从对应的 `arm_link7` 整个末端子树生成，避免相机与夹爪之间出现视觉断裂。
- 可选择 Galbot Motion 或 PyRoki 检查可达性，并分别控制自碰撞和场景障碍物检查。旧版 Galbot SDK 不支持环境碰撞开关时，界面会明确报告能力不足，用户可以关闭场景障碍物检查后只验证运动学和自碰撞。
- 成功轨迹会缓存每帧 URDF link 变换，显示 TCP 路径，并支持播放、暂停和从第一帧重播；该过程只更新 PyVista actor，不发送任何执行命令。
- 规划起点统一来自主窗口维护的 `RobotEnvironmentState`，机器人渲染、TCP 预览、Galbot Motion 和 PyRoki 共用同一份当前关节状态。代码不再内置一组 G1 关节角；首次加载 URDF 时只为环境中尚不存在的关节建立零值状态，已经由仿真、配置或外部同步提供的关节值会原样保留。后续从真实机器人同步关节角时，只需调用 `update_environment_joint_positions()`，不需要修改规划器。
- 底盘位姿使用独立 Gizmo 编辑：X/Y 箭头改变 `base_pose.x/y`，Z 轴旋转环改变 yaw；底盘 Z、roll、pitch 被锁定。Gizmo 与机器人本体共享同一个基座变换。为避免大型点云场景卡顿，拖动过程中不执行整机重建或点云过滤，鼠标释放或点击“应用机器人位姿”后才刷新。
- Galbot Motion 始终只规划当前目标手臂，腿部、另一只手臂、头部、轮子和夹爪均固定；其 whole-body reference 按 SDK 固定格式仍包含腿部 5、头部 2、左右臂各 7 个关节，共 21 维。21 维表示完整参考状态，不表示 21 个关节都参与优化。
- PyRoki 默认也只规划当前目标手臂；勾选“允许腿部参与规划”后才使用腿部 5 + 目标手臂 7 的 12 自由度模式。另一只手臂和头部不会因为使用完整 URDF 而自动参与规划。

机器人可视化必须缓存网格和 Actor。若每个轨迹帧都重新调用 `load_urdf_visuals()`，会反复解析 URDF 和加载网格，不适合动画播放。

## 尚待确认

- G1 whole-body 21 维关节向量的权威顺序，以及如何由界面中的 URDF 关节字典稳定生成该向量。
- Galbot Python 轨迹没有时间戳，比较时应按原始帧编号还是归一化进度重采样。
- 桌面环境可使用哪种 Galbot Motion 规划服务部署方式。
- FK 是继续扩展现有 URDF 实现，还是采用规划插件可共享的外部运动学库。
- PyRoki 与 Galbot Motion 的碰撞模型和安全边距不同，比较结果应展示后端配置，不能只显示成功或失败。
