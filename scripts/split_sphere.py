"""Hollow the geodesic spheres (3.2 mm shell), then cut at z=0 and z=90."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mapbox_earcut
import numpy as np

from planet_cup import DATA, EXPORT, SPECS, edge_check, signed_volume, write_stl

WALL = 3.2


def clip_open(verts: np.ndarray, faces: np.ndarray, cut: float, keep: str):
    """Clip a surface by z=cut; return the open clipped surface + cut ring."""
    eps = 1e-7
    new_v = [v.copy() for v in verts]
    cache: dict[tuple[int, int], int] = {}
    out: list[list[int]] = []
    cut_ids: set[int] = set()

    def inside(z: float) -> bool:
        return z >= cut - eps if keep == "above" else z <= cut + eps

    def interp(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        za, zb = verts[a][2], verts[b][2]
        t = (cut - za) / (zb - za) if abs(zb - za) > 1e-15 else 0.0
        p = verts[a] + t * (verts[b] - verts[a])
        p[2] = cut
        idx = len(new_v)
        new_v.append(p)
        cache[key] = idx
        return idx

    for f in faces:
        poly: list[int] = []
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            ia, ib = inside(verts[a][2]), inside(verts[b][2])
            if ia:
                poly.append(a)
            if ia != ib:
                poly.append(interp(a, b))
        clean: list[int] = []
        for idx in poly:
            if not clean or clean[-1] != idx:
                clean.append(idx)
        if len(clean) > 1 and clean[0] == clean[-1]:
            clean.pop()
        if len(clean) < 3:
            continue
        for k in range(1, len(clean) - 1):
            tri = [clean[0], clean[k], clean[k + 1]]
            if len(set(tri)) == 3:
                out.append(tri)
        for idx in clean:
            if abs(new_v[idx][2] - cut) < 1e-6:
                cut_ids.add(idx)

    v = np.asarray(new_v, dtype=np.float64)
    f = np.asarray(out, dtype=np.int32)
    ring = sorted(cut_ids, key=lambda i: math.atan2(v[i, 1], v[i, 0]))
    return v, f, ring


def weld(v: np.ndarray, f: np.ndarray, tol: float = 1e-3):
    key = np.round(v / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    out = np.zeros((int(inv.max()) + 1, 3), dtype=np.float64)
    cnt = np.zeros(len(out), dtype=np.float64)
    np.add.at(out, inv, v)
    np.add.at(cnt, inv, 1.0)
    out /= cnt[:, None]
    f2 = inv[f]
    keep = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 2] != f2[:, 0])
    return out, f2[keep].astype(np.int32)


def ring_on(v: np.ndarray, f: np.ndarray, cut: float) -> np.ndarray:
    used = np.unique(f)
    ids = used[np.abs(v[used, 2] - cut) < 1e-5]
    return np.asarray(sorted(ids, key=lambda i: math.atan2(v[i, 1], v[i, 0])), dtype=np.int32)


def annulus(v: np.ndarray, outer: np.ndarray, inner: np.ndarray, cut: float, nz: float):
    """Triangulate the wall rim between outer and inner cut loops."""
    inner_rev = inner[::-1]
    coords = np.vstack([v[outer, :2], v[inner_rev, :2]])
    ends = np.asarray([len(outer), len(outer) + len(inner_rev)], dtype=np.uint32)
    tri = mapbox_earcut.triangulate_float64(coords, ends).reshape(-1, 3)
    faces = []
    for a, b, c in tri:
        ia = int(outer[a]) if a < len(outer) else int(inner_rev[a - len(outer)])
        ib = int(outer[b]) if b < len(outer) else int(inner_rev[b - len(outer)])
        ic = int(outer[c]) if c < len(outer) else int(inner_rev[c - len(outer)])
        p0, p1, p2 = v[ia], v[ib], v[ic]
        cross = np.cross(p1 - p0, p2 - p0)[2]
        if cross * nz < 0:
            ib, ic = ic, ib
        faces.append([ia, ib, ic])
    return np.asarray(faces, dtype=np.int32)


def disk(v: np.ndarray, ring: np.ndarray, cut: float, nz: float):
    """Full flat disk closing a circular opening; returns faces + center index."""
    center = len(v)
    v = np.vstack([v, [[0.0, 0.0, cut]]])
    faces = []
    for a, b in zip(ring, np.roll(ring, -1)):
        p0, p1, p2 = v[center], v[int(a)], v[int(b)]
        cross = np.cross(p1 - p0, p2 - p0)[2]
        if cross * nz < 0:
            a, b = b, a
        faces.append([center, int(a), int(b)])
    return v, np.asarray(faces, dtype=np.int32)


def build_middle(outer_v, outer_f, inner_v, inner_f):
    """Hollow middle with a 3.2 mm solid floor at z=0 and an open top."""
    n = len(outer_v)
    v = np.vstack([outer_v, inner_v])
    f = np.vstack([outer_f, inner_f + n])

    # bottom outer face (full disk), then inner floor top at z=3.2
    ro = ring_on(v, f, 0.0)
    ro = ro[ro < n]
    v, cap = disk(v, ro, 0.0, -1.0)
    f = np.vstack([f, cap])

    rf = ring_on(v, f, 3.2)
    rf = rf[rf >= n]
    v, cap = disk(v, rf, 3.2, 1.0)
    f = np.vstack([f, cap])

    # open top: annular wall rim at z=90
    ro = ring_on(v, f, 90.0)
    ro = ro[ro < n]
    ri = ring_on(v, f, 90.0)
    ri = ri[ri >= n]
    f = np.vstack([f, annulus(v, ro, ri, 90.0, 1.0)])

    vol = signed_volume(v, f)
    if vol < 0:
        f = f[:, ::-1].copy()
    return v, f


def build_part(outer_v, outer_f, inner_v, inner_f, caps):
    """Combine outer/inner surfaces and add annular rims at the cut planes."""
    n = len(outer_v)
    v = np.vstack([outer_v, inner_v])
    f = np.vstack([outer_f, inner_f + n])
    for cut, nz in caps:
        ro = ring_on(v, f, cut)
        # only outer-surface indices (first block)
        ro = ro[ro < n]
        ri = ring_on(v, f, cut)
        ri = ri[ri >= n]
        if len(ro) < 3 or len(ri) < 3:
            continue
        f = np.vstack([f, annulus(v, ro, ri, cut, nz)])
    vol = signed_volume(v, f)
    if vol < 0:
        f = f[:, ::-1].copy()
    return v, f


def stats(path: Path, v: np.ndarray, f: np.ndarray) -> dict:
    _, bad = edge_check(f)
    vol = signed_volume(v, f)
    used = np.unique(f)
    bbox = v[used].max(0) - v[used].min(0)
    return {
        "file": path.name,
        "triangles": int(len(f)),
        "vertices": int(len(v)),
        "non_manifold_edges": int(bad),
        "volume_mm3": round(float(vol), 2),
        "bbox_mm": [round(float(x), 3) for x in bbox],
        "stl_bytes": path.stat().st_size,
    }


def main() -> None:
    rows = []
    for s in SPECS:
        d = np.load(DATA / f"{s.slug}_icosphere.npz")
        outer_all = d["vertices"].astype(np.float64)
        faces = d["faces"].astype(np.int32)
        zc = s.height / 2.0
        unit = outer_all.copy()
        unit[:, 2] -= zc
        r = np.linalg.norm(unit, axis=1)
        unit /= r[:, None]
        inner_all = unit * (r - WALL)[:, None]
        inner_all[:, 2] += zc
        inner_faces = faces[:, ::-1].copy()

        # outer/inner clipped sets for each part
        o_bot, f_ob, _ = clip_open(outer_all, faces, 0.0, "below")
        i_bot, f_ib, _ = clip_open(inner_all, inner_faces, 0.0, "below")
        o_top, f_ot, _ = clip_open(outer_all, faces, 90.0, "above")
        i_top, f_it, _ = clip_open(inner_all, inner_faces, 90.0, "above")

        o_mid, f_om, _ = clip_open(outer_all, faces, 0.0, "above")
        o_mid, f_om, _ = clip_open(o_mid, f_om, 90.0, "below")
        i_mid, f_im, _ = clip_open(inner_all, inner_faces, 3.2, "above")
        i_mid, f_im, _ = clip_open(i_mid, f_im, 90.0, "below")



        parts = (
            ("bottom", o_bot, f_ob, i_bot, f_ib, [(0.0, 1.0)]),
            ("middle", o_mid, f_om, i_mid, f_im, []),
            ("top", o_top, f_ot, i_top, f_it, [(90.0, -1.0)]),
        )
        for name, ov, of, iv, iff, caps in parts:
            v, f = build_middle(ov, of, iv, iff) if name == "middle" else build_part(ov, of, iv, iff, caps)
            path = EXPORT / f"{s.slug}_part_{name}.stl"
            write_stl(path, v, f)
            st = stats(path, v, f)
            st["slug"] = s.slug
            st["part"] = name
            rows.append(st)
            print(json.dumps(st, ensure_ascii=False))

    (DATA / "parts_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote hollow parts")


if __name__ == "__main__":
    main()
