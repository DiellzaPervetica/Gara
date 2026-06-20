from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .models import Instance, Street, Vehicle


_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore

        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    except Exception:
        pass

    with zipfile.ZipFile(path) as zf:
        raw = zf.read("word/document.xml")
    root = ET.fromstring(raw)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines: list[str] = []
    for para in root.iter(ns + "p"):
        text = "".join(t.text or "" for t in para.iter(ns + "t")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def read_text(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".docx":
        return _extract_docx(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _looks_like_coordinate(line: str) -> bool:
    parts = line.split()
    return len(parts) == 2 and all(_NUM.match(x) for x in parts)


def parse_instance(path: str | Path) -> Instance:
    p = Path(path)
    lines = [ln.strip().lstrip("\ufeff") for ln in read_text(p).splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{p}: empty instance")

    header = lines[0].split()
    if len(header) < 6:
        raise ValueError(f"{p}: invalid header")
    n, m, time_limit, vehicle_count, depot = map(int, header[:5])
    alpha = float(header[5])

    offset = 1
    coordinates = None
    if len(lines) >= 1 + n + m + 1 and all(_looks_like_coordinate(x) for x in lines[1 : 1 + n]):
        coordinates = []
        for line in lines[1 : 1 + n]:
            x, y = line.split()
            coordinates.append((float(x), float(y)))
        offset += n

    streets: list[Street] = []
    for i in range(m):
        if offset + i >= len(lines):
            raise ValueError(f"{p}: expected {m} street lines, got {i}")
        parts = lines[offset + i].split()
        if len(parts) != 7:
            raise ValueError(f"{p}: invalid street line {i}: {lines[offset + i]!r}")
        a, b, d, travel_time, length = map(int, parts[:5])
        category = parts[5]
        requirement = int(parts[6])
        if category not in {"M", "O", "C"}:
            raise ValueError(f"{p}: invalid category {category!r} on street {i}")
        streets.append(Street(i, a, b, d, travel_time, length, category, requirement))

    vehicle_line_index = offset + m
    if vehicle_line_index >= len(lines):
        raise ValueError(f"{p}: missing vehicle line")
    kinds = lines[vehicle_line_index].split()
    if len(kinds) != vehicle_count:
        raise ValueError(f"{p}: expected {vehicle_count} vehicles, got {len(kinds)}")
    vehicles = [Vehicle(i, kind) for i, kind in enumerate(kinds)]

    return Instance(
        name=p.stem,
        n=n,
        m=m,
        time_limit=time_limit,
        depot=depot,
        alpha=alpha,
        streets=streets,
        vehicles=vehicles,
        coordinates=coordinates,
    )
