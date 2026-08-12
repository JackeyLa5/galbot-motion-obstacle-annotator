from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PcdHeader:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    points: int
    data: str


def _parse_header(stream) -> PcdHeader:
    values: dict[str, list[str]] = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD header does not contain a DATA entry")
        decoded = line.decode("ascii", errors="strict").strip()
        if not decoded or decoded.startswith("#"):
            continue
        key, *parts = decoded.split()
        values[key.upper()] = parts
        if key.upper() == "DATA":
            break

    fields = tuple(values.get("FIELDS", values.get("FIELD", [])))
    sizes = tuple(map(int, values.get("SIZE", [])))
    types = tuple(value.upper() for value in values.get("TYPE", []))
    counts = tuple(map(int, values.get("COUNT", ["1"] * len(fields))))
    points = int(values.get("POINTS", ["0"])[0])
    if not points:
        width = int(values.get("WIDTH", ["0"])[0])
        height = int(values.get("HEIGHT", ["1"])[0])
        points = width * height

    if not fields or not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("Invalid PCD field metadata")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PCD file must contain x, y and z fields")

    return PcdHeader(fields, sizes, types, counts, points, values["DATA"][0].lower())


def _numpy_type(type_code: str, size: int) -> str:
    kind = {"F": "f", "I": "i", "U": "u"}.get(type_code)
    if kind is None or size not in {1, 2, 4, 8}:
        raise ValueError(f"Unsupported PCD field type: {type_code}{size}")
    return f"<{kind}{size}"


def _load_ascii(stream, header: PcdHeader) -> NDArray[np.float64]:
    data = np.loadtxt(stream, dtype=float, ndmin=2)
    offsets = np.cumsum((0, *header.counts[:-1]))
    indices = [int(offsets[header.fields.index(axis)]) for axis in ("x", "y", "z")]
    return np.asarray(data[:, indices], dtype=float)


def _load_binary(stream, header: PcdHeader) -> NDArray[np.float64]:
    dtype = np.dtype(
        [
            (name, _numpy_type(type_code, size), count)
            if count > 1
            else (name, _numpy_type(type_code, size))
            for name, size, type_code, count in zip(header.fields, header.sizes, header.types, header.counts)
        ]
    )
    data = np.frombuffer(stream.read(header.points * dtype.itemsize), dtype=dtype, count=header.points)
    return np.column_stack((data["x"], data["y"], data["z"])).astype(float, copy=False)


def load_pcd(path: str | Path) -> NDArray[np.float64]:
    with Path(path).open("rb") as stream:
        header = _parse_header(stream)
        if header.data == "ascii":
            points = _load_ascii(stream, header)
        elif header.data == "binary":
            points = _load_binary(stream, header)
        elif header.data == "binary_compressed":
            raise ValueError("binary_compressed PCD is not supported; convert it to ASCII or binary first")
        else:
            raise ValueError(f"Unsupported PCD DATA mode: {header.data}")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if not len(points):
        raise ValueError("PCD file does not contain any finite XYZ points")
    return points

