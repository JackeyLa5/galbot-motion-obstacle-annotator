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
- 加载 URDF 机器人模型并设置 7D 初始位姿
- 导出 Galbot JSON
- 导出可直接运行、结构参考 `add_obstacle.py` 的 Galbot Motion 障碍物添加 Python 脚本
- 显示世界坐标轴和点云边界

## ⚙️ 安装

推荐使用 Python 3.10 或更新版本，并统一使用 `uv`：

```bash
uv sync --dev
```

Ubuntu 使用 Qt 6/X11 时，还需要安装光标运行库：

```bash
sudo apt-get install libxcb-cursor0
```

## 🚀 快速开始

1. 下载机器人模型资产：

```bash
git clone https://github.com/GalaxyGeneralRobotics/galbot_one_golf_description.git
```

建议将 `galbot_one_golf_description` 放在当前项目根目录下，这样程序会自动查找：

```text
./galbot_one_golf_description/urdf/galbot_one_golf.urdf
```

2. 从机器人下载当前地图点云：

```bash
scp galbot@robot_ip:/var/maps/cur/global_cloud_cleaned.pcd ./
```

3. 在本地安装依赖：

```bash
uv sync --dev
```

4. 启动标注工具：

```bash
uv run galbot-motion-obstacle-annotator global_cloud_cleaned.pcd
```

或：

```bash
uv run python -m galbot_motion_obstacle_annotator.app global_cloud_cleaned.pcd
```

5. 开始标注：

1. 选择碰撞体类型。Box 还可以选择 `AABB` 或 `OBB`。
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
- 未选中碰撞体使用蓝色，透明度为 `0.28`。
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
- `应用显示过滤`：立即重新裁剪显示点云。

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

修改右侧位置和四元数后，点击“应用机器人位姿”。四元数会在应用时自动归一化。机器人模型设置为不可选取，因此不会干扰点云逐点选择。

机器人默认关节姿态不是全零，而是采用适合场景标注的展开姿态：

```text
腿部： [0.6, 1.8, 1.2, 0.0, 0.0]
头部： [0.0, 0.0]
左臂： [1.9, -1.5, -0.6, -2.1, 0.0, -0.25, 0.1]
右臂： [-1.9, 1.5, 0.6, 2.1, 0.0, 0.25, -0.1]
```

右臂各关节角是左臂对应关节角的相反数。URDF 加载器会根据关节原点和旋转轴执行正向运动学，而不是仅移动视觉网格。

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
