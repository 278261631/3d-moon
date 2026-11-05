"""Validate the exported STLs with FreeCAD's Mesh workbench.

Run with FreeCADCmd from a FreeCAD installation, for example:
  FreeCADCmd scripts/validate_freecad.py
"""
from __future__ import annotations

import json
from pathlib import Path

import FreeCAD  # type: ignore
import Mesh  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "export"
OUT = ROOT / "data/freecad_validation.json"


def main() -> None:
    rows = []
    paths = sorted(EXPORT.glob("*_icosphere.stl")) + sorted(EXPORT.glob("*_part_*.stl"))
    for path in paths:
        m = Mesh.Mesh(str(path))
        bbox = m.BoundBox
        row = {
            "file": str(path.relative_to(EXPORT)).replace("\\", "/"),
            "facets": m.CountFacets,
            "points": m.CountPoints,
            "is_solid": bool(m.isSolid()),
            "has_nonmanifolds": bool(m.hasNonManifolds()),
            "has_self_intersections": bool(m.hasSelfIntersections()),
            "invalid_neighbourhood": bool(m.hasInvalidNeighbourhood()),
            "components": m.countComponents(),
            "non_uniform_oriented_facets": m.countNonUniformOrientedFacets(),
            "volume_mm3": round(float(m.Volume), 2),
            "area_mm2": round(float(m.Area), 2),
            "bbox_mm": [round(bbox.XLength, 3), round(bbox.YLength, 3), round(bbox.ZLength, 3)],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
