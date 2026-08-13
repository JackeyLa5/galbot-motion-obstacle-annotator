from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from .geometry import rpy_to_matrix

FloatArray = NDArray[np.float64]


@dataclass
class RobotVisual:
    name: str
    mesh: pv.PolyData
    transform: FloatArray
    texture: pv.Texture | None = None
    link_name: str = ""
    local_transform: FloatArray | None = None


def _numbers(value: str | None, length: int, default: tuple[float, ...]) -> FloatArray:
    if value is None:
        return np.asarray(default, dtype=float)
    result = np.fromstring(value, sep=" ", dtype=float)
    if result.shape != (length,):
        raise ValueError(f"Expected {length} values, got {value!r}")
    return result


def _origin(element) -> FloatArray:
    matrix = np.eye(4, dtype=float)
    if element is None:
        return matrix
    matrix[:3, :3] = rpy_to_matrix(*_numbers(element.get("rpy"), 3, (0.0, 0.0, 0.0)))
    matrix[:3, 3] = _numbers(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    return matrix


def _axis_rotation(axis: FloatArray, angle: float) -> FloatArray:
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(4, dtype=float)
    x, y, z = axis / norm
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    return matrix


def _resolve_mesh_path(filename: str, description_dir: Path) -> Path:
    package_prefix = "package://"
    if filename.startswith(package_prefix):
        relative = filename[len(package_prefix) :]
        parts = Path(relative).parts
        if parts and parts[0] == description_dir.name:
            relative = str(Path(*parts[1:]))
        return description_dir / relative
    path = Path(filename)
    return path if path.is_absolute() else description_dir / path


def _flatten_mesh(data) -> list[pv.PolyData]:
    if isinstance(data, pv.MultiBlock):
        meshes: list[pv.PolyData] = []
        for block in data:
            if block is not None:
                meshes.extend(_flatten_mesh(block))
        return meshes
    if isinstance(data, pv.PolyData):
        return [data]
    return [data.extract_surface()]


def _visual_texture_for_path(path: Path, visual_root: Path) -> pv.Texture | None:
    search_dirs: list[Path] = []
    current = path.parent
    while True:
        search_dirs.append(current)
        if current == visual_root or current.parent == current:
            break
        current = current.parent
    for directory in search_dirs:
        candidates = sorted(directory.glob("M_*.png")) + sorted(directory.glob("M_*.jpg"))
        if candidates:
            texture = pv.read_texture(candidates[0])
            texture.interpolate = True
            return texture
    return None


def _mjcf_visual_meshes(glb_path: Path, description_dir: Path, scale: FloatArray) -> list[tuple[pv.PolyData, pv.Texture | None]]:
    relative = glb_path.relative_to(description_dir / "meshes" / "visual")
    visual_root = description_dir / "mjcf" / "meshes" / "meshes" / "visual"
    stem_path = visual_root / relative.with_suffix("")
    obj_candidates: list[Path] = []
    single_obj = stem_path.with_suffix(".obj")
    if single_obj.exists():
        obj_candidates.append(single_obj)
    if stem_path.is_dir():
        obj_candidates.extend(sorted(stem_path.glob("*.obj")))
    meshes: list[tuple[pv.PolyData, pv.Texture | None]] = []
    for obj_path in obj_candidates:
        texture = _visual_texture_for_path(obj_path, visual_root)
        for mesh in _flatten_mesh(pv.read(obj_path)):
            meshes.append((mesh.scale(scale, inplace=False), texture))
    return meshes


def _geometry_meshes(geometry, description_dir: Path) -> list[tuple[pv.PolyData, pv.Texture | None]]:
    mesh_element = geometry.find("mesh")
    if mesh_element is not None:
        path = _resolve_mesh_path(mesh_element.attrib["filename"], description_dir)
        if not path.exists():
            return []
        scale = _numbers(mesh_element.get("scale"), 3, (1.0, 1.0, 1.0))
        if path.suffix.lower() in {".glb", ".gltf"}:
            mjcf_meshes = _mjcf_visual_meshes(path, description_dir, scale)
            # Prefer the OBJ export for visual meshes and never fall back to
            # GLB, whose material/texture handling is inconsistent in VTK.
            return mjcf_meshes
        meshes = _flatten_mesh(pv.read(path))
        return [(mesh.scale(scale, inplace=False), None) for mesh in meshes]

    box = geometry.find("box")
    if box is not None:
        size = _numbers(box.get("size"), 3, (1.0, 1.0, 1.0))
        return [(pv.Box(bounds=tuple(value for extent in size for value in (-extent / 2.0, extent / 2.0))), None)]

    sphere = geometry.find("sphere")
    if sphere is not None:
        return [(pv.Sphere(radius=float(sphere.attrib["radius"])), None)]

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return [(
            pv.Cylinder(
                radius=float(cylinder.attrib["radius"]),
                height=float(cylinder.attrib["length"]),
                direction=(0.0, 0.0, 1.0),
            ),
            None,
        )]
    return []


def load_urdf_visuals(
    urdf_path: str | Path, joint_positions: dict[str, float] | None = None
) -> list[RobotVisual]:
    robot, description_dir, links, child_joints, link_transforms = _load_urdf_tree(
        urdf_path, joint_positions
    )
    return _visuals_for_links(links, link_transforms, description_dir, set(links))


def load_tcp_gripper_visuals(
    urdf_path: str | Path,
    mount_link: str,
    tcp_link: str,
    joint_positions: dict[str, float] | None = None,
    auxiliary_root_links: tuple[str, ...] = (),
    excluded_links: tuple[str, ...] = (),
) -> list[RobotVisual]:
    _, description_dir, links, child_joints, link_transforms = _load_urdf_tree(
        urdf_path, joint_positions
    )
    if mount_link not in links:
        raise ValueError(f"URDF mount link does not exist: {mount_link}")
    if tcp_link not in links:
        raise ValueError(f"URDF TCP link does not exist: {tcp_link}")

    descendant_links = {mount_link}
    pending = [mount_link]
    while pending:
        parent = pending.pop()
        for joint in child_joints.get(parent, []):
            child = joint.find("child").attrib["link"]
            if child not in descendant_links:
                descendant_links.add(child)
                pending.append(child)
    if tcp_link not in descendant_links:
        raise ValueError(f"URDF TCP link {tcp_link!r} is not below mount link {mount_link!r}")

    for auxiliary_root in auxiliary_root_links:
        if auxiliary_root not in links:
            raise ValueError(f"URDF auxiliary link does not exist: {auxiliary_root}")
        descendant_links.add(auxiliary_root)
        pending = [auxiliary_root]
        while pending:
            parent = pending.pop()
            for joint in child_joints.get(parent, []):
                child = joint.find("child").attrib["link"]
                if child not in descendant_links:
                    descendant_links.add(child)
                    pending.append(child)

    tcp_inverse = np.linalg.inv(link_transforms[tcp_link])
    local_transforms = {
        link_name: tcp_inverse @ link_transforms[link_name] for link_name in descendant_links
    }
    included_links = descendant_links - set(excluded_links)
    return _visuals_for_links(links, local_transforms, description_dir, included_links)


def _load_urdf_tree(
    urdf_path: str | Path, joint_positions: dict[str, float] | None
):
    urdf_path = Path(urdf_path).resolve()
    description_dir = urdf_path.parent.parent
    robot = ElementTree.parse(urdf_path).getroot()

    links = {element.attrib["name"]: element for element in robot.findall("link")}
    child_joints: dict[str, list] = {}
    child_links: set[str] = set()
    for joint in robot.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        child_joints.setdefault(parent, []).append(joint)
        child_links.add(child)

    roots = set(links) - child_links
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, got {sorted(roots)}")

    joint_positions = joint_positions or {}
    link_transforms: dict[str, FloatArray] = {}
    pending = [(roots.pop(), np.eye(4, dtype=float))]
    while pending:
        link_name, link_transform = pending.pop()
        link_transforms[link_name] = link_transform
        for joint in child_joints.get(link_name, []):
            child = joint.find("child").attrib["link"]
            joint_transform = _origin(joint.find("origin"))
            if joint.attrib.get("type") in {"revolute", "continuous"}:
                axis_element = joint.find("axis")
                axis = _numbers(
                    axis_element.get("xyz") if axis_element is not None else None,
                    3,
                    (1.0, 0.0, 0.0),
                )
                joint_transform = joint_transform @ _axis_rotation(
                    axis, float(joint_positions.get(joint.attrib["name"], 0.0))
                )
            pending.append((child, link_transform @ joint_transform))

    return robot, description_dir, links, child_joints, link_transforms


def _visuals_for_links(
    links: dict,
    link_transforms: dict[str, FloatArray],
    description_dir: Path,
    included_links: set[str],
) -> list[RobotVisual]:

    visuals: list[RobotVisual] = []
    for link_name, link in links.items():
        if link_name not in included_links:
            continue
        for visual_index, visual in enumerate(link.findall("visual")):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            local_transform = _origin(visual.find("origin"))
            visual_transform = link_transforms[link_name] @ local_transform
            for mesh_index, (mesh, texture) in enumerate(_geometry_meshes(geometry, description_dir)):
                visuals.append(
                    RobotVisual(
                        name=f"{link_name}_{visual_index}_{mesh_index}",
                        mesh=mesh,
                        transform=visual_transform,
                        texture=texture,
                        link_name=link_name,
                        local_transform=local_transform,
                    )
                )
    return visuals


def load_urdf_link_transforms(
    urdf_path: str | Path, joint_positions: dict[str, float] | None = None
) -> dict[str, FloatArray]:
    return _load_urdf_tree(urdf_path, joint_positions)[4]


def load_urdf_actuated_joint_names(urdf_path: str | Path) -> tuple[str, ...]:
    robot = ElementTree.parse(Path(urdf_path).resolve()).getroot()
    return tuple(
        joint.attrib["name"]
        for joint in robot.findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
        and joint.find("mimic") is None
    )


def load_urdf_joint_limits(urdf_path: str | Path) -> dict[str, tuple[float, float]]:
    """Return slider/planner limits for non-mimic actuated URDF joints."""
    robot = ElementTree.parse(Path(urdf_path).resolve()).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in robot.findall("joint"):
        if joint.attrib.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        if joint.find("mimic") is not None:
            continue
        name = joint.attrib["name"]
        limit = joint.find("limit")
        if joint.attrib.get("type") == "continuous":
            lower, upper = -np.pi, np.pi
        elif limit is None or limit.get("lower") is None or limit.get("upper") is None:
            lower, upper = (-np.pi, np.pi) if joint.attrib.get("type") == "revolute" else (-1.0, 1.0)
        else:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid limits for URDF joint {name!r}: {lower}, {upper}")
        limits[name] = (lower, upper)
    return limits
