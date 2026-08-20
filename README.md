# Galbot Motion Obstacle Annotator

Galbot Motion Obstacle Annotator 是一个基于 PySide6、PyVista 和 VTK 的点云碰撞体标注工具。它用于在建图点云上交互式拟合 `box`、`sphere` 和 `cylinder` 障碍物，并导出 Galbot Motion `add_obstacle()` 可参考的障碍物信息设置。

## ✨ Highlights

- 加载从机器人导出的 `/var/maps/cur/global_cloud_cleaned.pcd`
- 按机器人位置裁剪大范围点云，减少无效区域
- 独立设置 XY 显示半范围以及 Z 最低/最高高度
- 鼠标右键逐点采样，不扫描整个视锥点云
- 自动拟合 Box、Sphere 和 Cylinder
- 鼠标拖动、旋转和缩放 3D Box
- 精确编辑中心、RPY 和长宽高
- 管理多个碰撞体
- 加载 URDF 机器人模型并设置底盘位姿
- 使用场景中的底盘 Gizmo 沿 X/Y 平移或绕 Z 轴旋转
- 导出 Galbot JSON
- 导出可直接运行、结构参考 `add_obstacle.py` 的 Galbot Motion 障碍物添加 Python 脚本
- 显示世界坐标轴和点云边界

## ⚙️ 安装与校验

推荐使用 Python 3.10 或更新版本，并统一使用 `uv` 管理环境。

### 1. 下载项目

```bash
git clone https://github.com/JackeyLa5/galbot-motion-obstacle-annotator.git
cd galbot-motion-obstacle-annotator
```

### 2. 安装依赖

```bash
uv sync --dev
```

PyRoki 不是默认必需依赖。只有需要使用 PyRoki 规划器时，额外安装：

```bash
uv sync --extra pyroki
```

不使用 PyRoki 时不要加该参数，默认环境不会安装 JAX、JAXLS 等额外依赖。

Ubuntu 使用 Qt 6/X11 时，还需要安装光标运行库：

```bash
sudo apt-get install libxcb-cursor0
```

### 3. 使用测试点云快速校验

仓库已经提供可直接使用的室内测试点云：

```bash
uv run galbot-motion-obstacle-annotator tests/data/test.pcd
```

正常情况下会打开标注界面，并显示餐桌、餐椅、沙发、茶几、矮柜、纸箱、垃圾桶、圆凳、盆栽、落地灯和行李箱等常见物体。世界原点附近预留了机器人活动空间，点云中不会额外放置机器人。

可以使用该场景快速检查：

- 点云是否正常加载、过滤和按高度着色。
- 鼠标右键是否能够选取点云中的点。
- Box、Sphere 和 Cylinder 是否能够正常拟合和编辑。
- JSON 和 Python 脚本是否能够正常导出。

测试点云由 `tools/generate_test_pcd.py` 确定性生成。如需重新生成：

```bash
python3 tools/generate_test_pcd.py
```

## 🚀 快速开始

完成安装和测试点云校验后，可以按照以下步骤加载真实机器人模型和地图点云并开始标注。

### 1. 准备机器人模型

```bash
git clone https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description.git
```

建议将 `galbot_one_golf_description` 放在当前项目根目录下，程序会自动查找：

```text
./galbot_one_golf_description/urdf/galbot_one_golf.urdf
```

也可以在界面中手动选择其他 URDF 文件或机器人模型目录。

### 2. 下载地图点云

```bash
scp galbot@robot_ip:/var/maps/cur/global_cloud_cleaned.pcd ./
```

### 3. 启动标注工具

```bash
uv run galbot-motion-obstacle-annotator global_cloud_cleaned.pcd
```

也可以使用模块入口启动：

```bash
uv run python -m galbot_motion_obstacle_annotator.app global_cloud_cleaned.pcd
```

### 4. 开始标注

1. 选择碰撞体类型。
2. 点击“开始逐点选择”，在点云上右键选择若干具有代表性的点。
3. 可以从不同视角选择边界点，并使用“撤销一点”或“清空点”。
4. 点击“生成碰撞体”，程序根据采样点拟合几何参数。
5. 如有需要，再通过右侧表单或 Box 控制柄微调，最后导出 JSON 或 Python 脚本。

逐点选择每次只执行一次点拾取，不再对几十万点执行矩形视锥选择，因此更适合大型建图 PCD。左键仍用于旋转视角，右键用于采样。Box 至少需要 2 点，Sphere 至少需要 2 点，Cylinder 至少需要 3 点。

第一次采样点不会再创建新的 VTK Actor：红色采样点 Actor 在窗口初始化时已经建立，后续只更新点数据。

## 📦 碰撞体类型

Galbot Motion SDK 本身还支持 `point_cloud` 和 `depth_image`，但标注工具输入场景本身已经是点云，再把完整点云作为碰撞体容易重复、过密并降低规划性能，因此界面只保留以下 3 种常用几何类型：

| 类型 | Scale 含义 | 标注方式 |
|---|---|---|
| `box` | `[length, width, height]` | 采样边界点，拟合 AABB/OBB |
| `sphere` | `[radius, 0, 0]` | 采样表面点，拟合中心和半径 |
| `cylinder` | `[radius, height, 0]` | 至少 3 点，PCA 拟合轴向、半径和高度 |

所有类型都会导出 `pose`、`scale` 和 `target_frame`。这个工具只负责场景几何标注，不再在界面里填写 `key`、`safe_margin`、`resolution` 这类更偏机器人或规划侧的参数。

逐点选择在 Qt 层直接拦截并消费右键事件，再手动调用 VTK Point Picker 吸附到点云顶点。Hardware Picker 虽然适合大型表面模型，但对仅由顶点组成的 PCD 可能只命中 Actor 而无法返回点，因此不用于这里。

红色采样点使用同一个长期存在的 PolyData/Actor。新增、撤销或清空时会同时更新点坐标和 Vertex Cell 拓扑，因此所有已选点都会显示，而不是只显示第一个点。

## 🎯 碰撞体显示与选择

- 碰撞体使用带边框的半透明实体显示，而不是仅显示线框。
- 未选中 `box` 使用青蓝色，未选中 `sphere` 使用紫色，未选中 `cylinder` 使用绿色，透明度均为 `0.28`。
- 当前选中碰撞体使用黄色，透明度为 `0.42`。
- Box 选中时会额外显示三轴拖动、旋转和缩放控制柄。
- 点击右侧“取消选中”后会清除当前列表选择，并移除当前变换控制柄；碰撞体本身仍以半透明实体保留在场景中。

## 🌐 点云显示过滤

大范围建图 PCD 会保留在内存中，但三维窗口默认只显示机器人附近的区域：

```text
中心：机器人基座的 X、Y
X：robot_x - 10m ～ robot_x + 10m
Y：robot_y - 10m ～ robot_y + 10m
Z：-2m ～ 2m
```

也就是默认显示以机器人为中心、XY 半范围为 10m 的正方形区域，并过滤过高和过低的点。参数可以在右侧“点云显示范围”中修改：

- `XY 半范围`：机器人当前位置向 X/Y 正负方向延伸的距离。
- `Z 最低/Z 最高`：点云绝对 Z 高度范围。
- `点云颜色`：默认对所有已加载点云按 Z 高度使用柔和色图着色，也可以切换为低亮度单色显示。
- `点大小`：调整点的屏幕尺寸，默认值为 `2.0`，大范围密集点云可适当调小。
- `点云透明度`：调整点云整体透明度，默认值为 `0.72`，降低密集点叠加造成的过亮和模糊。
- `应用显示过滤`：立即重新裁剪显示点云。

颜色、点大小和透明度只会重绘当前显示区域，不会重新遍历和裁剪完整地图，因此可以在大型点云上实时调整。这套显示规则统一适用于 PCD、PLY、VTK 和 VTP，不依赖测试点云或文件自带颜色。

过滤只影响显示和逐点选择，不会修改原始 `global_cloud_cleaned.pcd` 文件。

## 🤖 机器人模型

也可以在右侧“机器人模型”面板选择其他机器人模型目录或 URDF 文件。URDF 中的视觉 Mesh、Box、Sphere 和 Cylinder 会被加载到点云场景中。对于机器人描述仓库里的视觉网格，程序会优先使用配套的 OBJ 和贴图资源；没有贴图的部位会使用纯白色显示。

机器人基座初始位姿格式为：

```text
[x, y, z, qx, qy, qz, qw]
```

默认值为：

```text
[0, 0, 0, 0, 0, 0, 1]
```

修改右侧位置和四元数后，点击“应用机器人位姿”。四元数会在应用时自动归一化。场景中的底盘 Gizmo 支持沿 X/Y 平移和绕 Z 轴旋转；鼠标释放后才同步机器人模型、点云显示中心和 Galbot Motion 的 `galbot_base_pose`，拖动过程中不会重复重建整机或过滤点云。底盘 Z、roll 和 pitch 保持不变。机器人模型设置为不可选取，因此不会干扰点云逐点选择。

机器人默认关节姿态不是全零，而是采用适合场景标注的展开姿态：

```text
腿部： [0.6, 1.8, 1.2, 0.0, 0.0]
头部： [0.0, 0.0]
左臂： [1.9, -1.5, -0.6, -2.1, 0.0, -0.25, 0.1]
右臂： [-1.9, 1.5, 0.6, 2.1, 0.0, 0.25, -0.1]
```

右臂各关节角是左臂对应关节角的相反数。URDF 加载器会根据关节原点和旋转轴执行正向运动学，而不是仅移动视觉网格。

## 🧭 机械臂工作空间

在“工具 TCP 抓取姿态”面板下方的“机械臂工作空间”里，可以在选点/规划之前先大致判断当前运动链够不够得着：

- 设置“采样点数”（默认 1500）后点击“计算工作空间”，会按当前“运动链”（左臂/右臂）关节限位随机采样并做正向运动学，其余关节固定在当前状态，在后台线程计算，不会卡住界面。
- 安装了 PyRoki 时（`uv sync --extra pyroki`），采样会自动过滤自碰撞姿态——比如手臂穿过自己脑袋这种明显不合理的构型不会出现在结果里；勾选“启用场景障碍物检查”后还会一并过滤和已标注障碍物的碰撞。自碰撞判定会以机器人当前姿态为基准做校准（很多相邻部件在正常姿态下本来就贴得很近，不能算碰撞），只有比当前状态更差的接触才会被判定为碰撞。可行姿态较少时可能找不满“采样点数”指定的数量，状态栏会如实报告实际找到的个数和尝试次数，不会假装凑够了。未安装 PyRoki 时退回纯运动学采样，不做任何碰撞检查，状态栏会明确提示。
- 计算完成后场景里会显示一团半透明黄色点云，即末端大致可达的位置范围；这只反映**位置**是否可达，不代表任意抓取朝向在该位置都可行。
- “显示/隐藏工作空间”只是切换显示，不会重新计算。
- 机器人关节、底盘位姿或运动链发生变化后，之前算好的工作空间会自动隐藏并提示需要重新计算，避免展示过时、可能误导判断的范围。
- 点击“点击可达点查看姿态”进入点选模式后，右键点击黄色点云中的某个点，会把整条运动链摆到"末端到达该点"时刻的姿态，方便直观判断这个可达点是不是通过一个别扭、不合理的姿态够到的。这只是**预览**，不会写回当前机器人关节状态，也不会影响已经计算好的工作空间；再次点击按钮可退出预览模式。

## 📤 导出格式

```json
{
  "version": 1,
  "source_point_cloud": "global_cloud_cleaned.pcd",
  "obstacles": [
    {
      "obstacle_id": "obstacle_001",
      "obstacle_type": "box",
      "target_frame": "world",
      "pose": [1.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0],
      "scale": [1.2, 0.8, 1.5]
    }
  ]
}
```

## 🧭 坐标系注意事项

标注工具不会自动推断 PCD 的坐标系。导出的 `target_frame` 必须与点云坐标一致。如果 PCD 位于 LiDAR 坐标系，应先把点云变换到 `world`，或者确保 Motion 能正确解析对应目标坐标系。

## ⚠️ 当前限制

- 暂不支持 `binary_compressed` PCD
- 当前界面尚未提供逐关节角编辑，机器人使用代码中配置的默认关节姿态
- 机器人无贴图部位当前统一使用纯白色显示，材质表现不会完全等同于原始模型文件
- 如果当前项目根目录下没有机器人资产仓库，程序不会自动加载机器人，只会在状态栏提示缺少资产

导出的 Python 文件是可直接运行的脚本模板，内部会初始化 `GalbotMotion` 和 `GalbotRobot`，然后逐个调用 `motion.add_obstacle()`。

## 🌐 切换机器人 embosa 网络配置

`GalbotMotion`（真实 SDK 规划器）通过 embosa 与跑在机器人 Orin/XCU 上的 `service_motion_plan` 服务通信，peer IP 读取自系统级文件 `/data/config/embosa_ip_config.json`；这个路径写死在 `libembosa.so` 里，SDK 本身不提供环境变量或参数覆盖。

为了不用每次都手动 `sudo vi` 那个文件，已知机器人的 IP 记在本仓库的 `config/embosa_robots.json` 里（按机器人名分组，可持续增补），切换时跑：

```bash
python3 scripts/set_embosa_robot.py --list          # 查看已知机器人
python3 scripts/set_embosa_robot.py <robot_name>     # 切换到该机器人（会 sudo 覆盖系统文件，需要密码）
```

新机器人直接在 `config/embosa_robots.json` 里加一条（`local_interface` 是这台 PC 在能连到该机器人时应使用的 IP，`peer_lists` 是机器人 Orin/XCU 的 IP）。

## 🧹 SDK 障碍物管理脚本

查询 Motion 中已加载的障碍物 ID：

```bash
python3 scripts/query_obstacles.py
```

交互式循环删除障碍物：

```bash
python3 scripts/delete_obstacle.py
```

可以输入列表序号或完整 ID。删除后会继续显示剩余障碍物；输入 `q` 退出，输入 `all` 删除全部非默认障碍物。

`ground` 是 Motion 默认地面碰撞体，脚本会标记并保留它。

## ❓ 常见问题

### Qt 无法加载 xcb 平台插件

首次启动时可能出现以下错误：

```text
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

这是 PySide6/Qt 6 在 Ubuntu X11 环境中缺少 `libxcb-cursor0`，不是标注器或 PyVista 代码错误。安装系统运行库即可：

```bash
sudo apt-get update
sudo apt-get install -y libxcb-cursor0
```

安装后直接使用 `uv` 启动：

```bash
uv run galbot-motion-obstacle-annotator global_cloud_cleaned.pcd
```
