"""Parametric planet pen-cup generator.

The outer wall is a globe-like barrel.  A real equirectangular planetary
image supplies both the preview texture and the radial relief; the inner
cavity, rim and base are generated analytically so the result is a single
watertight solid.
"""
from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "assets/processed"
EXPORT = ROOT / "export"
DATA = ROOT / "data"
EXPORT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


@dataclass
class CupSpec:
    slug: str
    name: str
    name_cn: str
    texture: str
    heightmap: str
    u_front: float
    height: float = 90.0
    sphere_r: float = 60.0
    r_base: float = 37.0
    r_top: float = 35.0
    r_max: float = 46.0
    wall: float = 3.2
    base_t: float = 5.0
    v_scale: float = 1.0
    lip: float = 0.0
    bottom_chamfer: float = 0.0
    inner_chamfer: float = 1.4
    fade: float = 0.6
    foot_h: float = 0.0
    foot_band: float = 0.0
    n_theta: int = 576
    n_z: int = 300
    n_inner_z: int = 160
    relief_gain: float = 1.0
    material: str = "PLA"


SPECS = [
    CupSpec(
        slug="luna",
        name="Luna",
        name_cn="月球",
        texture="moon_color.png",
        heightmap="moon_height.npy",
        u_front=0.50,  # near side, 0 deg longitude
        relief_gain=1.0,
    ),
    CupSpec(
        slug="terra",
        name="Terra",
        name_cn="地球",
        texture="earth_color.png",
        heightmap="earth_height.npy",
        u_front=0.556,  # 20 deg E: Africa / Europe at the front
        relief_gain=0.95,
    ),
    CupSpec(
        slug="pluto",
        name="Pluto",
        name_cn="冥王星",
        texture="pluto_color.png",
        heightmap="pluto_height.npy",
        u_front=0.55,  # Sputnik Planitia / the heart at the front
        relief_gain=1.0,
    ),
]


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bilinear_wrap(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a 2:1 map with horizontal wrap and clamped poles."""
    h, w = img.shape[:2]
    x = np.mod(u, 1.0) * w - 0.5
    y = np.clip(v, 0.0, 1.0) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = x - x0
    ty = y - y0
    x0 %= w
    x1 = (x0 + 1) % w
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    if img.ndim == 3:
        tx = tx[..., None]
        ty = ty[..., None]
        c00 = img[y0, x0]
        c10 = img[y0, x1]
        c01 = img[y1, x0]
        c11 = img[y1, x1]
        return (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    return (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty


def profile_radius(z: np.ndarray, s: CupSpec) -> np.ndarray:
    if s.sphere_r > 0:
        # A true spherical outer surface, cut flat at the foot and the mouth.
        # The short cylindrical foot keeps the first layers printable and stable.
        zz = np.maximum(z, s.foot_band)
        r2 = s.sphere_r**2 - (zz - s.height / 2) ** 2
        return np.sqrt(np.maximum(r2, 0.0))
    t = np.clip(z / s.height, 0.0, 1.0)
    bulge = s.r_max - 0.5 * (s.r_base + s.r_top)
    return s.r_base * (1 - t) + s.r_top * t + bulge * np.sin(np.pi * t) ** 1.25


def inner_profile_radius(z: np.ndarray, s: CupSpec) -> np.ndarray:
    if s.sphere_r > 0:
        r2 = (s.sphere_r - s.wall) ** 2 - (z - s.height / 2) ** 2
        return np.sqrt(np.maximum(r2, 0.0))
    return profile_radius(z, s) - s.wall


def outer_radius(z: np.ndarray, theta: np.ndarray, relief: np.ndarray, s: CupSpec) -> np.ndarray:
    r = profile_radius(z, s)[:, None]
    # 45-degree chamfers at the foot and lip.
    r = r - np.where(z < s.bottom_chamfer, s.bottom_chamfer - z, 0.0)[:, None]
    r = r - np.where(z > s.height - s.lip, z - (s.height - s.lip), 0.0)[:, None]
    # Short fade at the cut planes keeps the rim circular and the mating
    # faces simple; the fade band is only 1.5 mm, much narrower than the old
    # 6 mm band that read as a raised ring.
    fade = smoothstep((z - s.foot_h) / s.fade) * smoothstep((s.height - z) / s.fade)
    return r + relief * fade[:, None]


def inner_radius(z: np.ndarray, s: CupSpec) -> np.ndarray:
    r = inner_profile_radius(z, s)
    r = r - np.where(z > s.height - s.lip, z - (s.height - s.lip), 0.0)
    # Chamfer where the inner wall meets the floor.
    c = np.clip((s.base_t + s.inner_chamfer - z) / s.inner_chamfer, 0.0, 1.0)
    r = r - s.inner_chamfer * c
    return r


def build_mesh(s: CupSpec) -> dict:
    tex = np.array(Image.open(PROC / s.texture).convert("RGB"), dtype=np.uint8)
    hm = np.load(PROC / s.heightmap).astype(np.float32)
    assert tex.shape[:2] == hm.shape, (tex.shape, hm.shape)

    theta = np.linspace(0.0, 2 * np.pi, s.n_theta, endpoint=False)

    # Outer rings: fine bottom chamfer, body, fine lip chamfer.
    z_outer = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, s.bottom_chamfer, 5),
                np.linspace(s.bottom_chamfer, s.height - s.lip, s.n_z),
                np.linspace(s.height - s.lip, s.height, 5),
            ]
        )
    )
    zz = z_outer[:, None]
    uu = (s.u_front + (theta[None, :] - np.pi / 2) / (2 * np.pi)) % 1.0
    uu = np.broadcast_to(uu, (len(z_outer), s.n_theta))
    # Spherical latitude mapping: the cup gets the equatorial band, the two
    # caps get the real polar bands, and the complete sphere is continuous.
    lat = np.arcsin(np.clip((z_outer - s.height / 2) / s.sphere_r, -1.0, 1.0))
    vv = (0.5 - lat / np.pi)[:, None]
    vv = np.clip(vv * s.v_scale, 0.0, 1.0)
    relief = bilinear_wrap(hm, uu, vv)
    # Mild smoothing in the (theta,z) grid to keep printable slopes.
    relief = ndimage.gaussian_filter(relief, sigma=(0.75, 1.0), mode=("nearest", "wrap"))
    relief = np.maximum(relief, 0.0) * s.relief_gain
    r_out = outer_radius(z_outer, theta, relief, s)

    # Outer vertices, ring-major.
    x = r_out * np.cos(theta)[None, :]
    y = r_out * np.sin(theta)[None, :]
    outer_v = np.stack(
        [x.ravel(), y.ravel(), np.repeat(z_outer, s.n_theta)], axis=1
    ).astype(np.float64)
    outer_uv = np.stack([uu.ravel(), np.repeat(vv, s.n_theta, axis=1).ravel()], axis=1)

    # Outer quads -> triangles, outward winding.
    j = np.arange(s.n_theta)
    jn = (j + 1) % s.n_theta
    faces = []
    for i in range(len(z_outer) - 1):
        a = i * s.n_theta + j
        b = i * s.n_theta + jn
        c = (i + 1) * s.n_theta + jn
        d = (i + 1) * s.n_theta + j
        faces.append(np.stack([a, b, c], axis=1))
        faces.append(np.stack([a, c, d], axis=1))
    outer_faces = np.concatenate(faces, axis=0)

    # Inner rings, bottom to top.
    z_in = np.linspace(s.base_t, s.height, s.n_inner_z)
    r_in = inner_radius(z_in, s)
    theta_i = theta
    inner_v = np.stack(
        [
            (r_in[:, None] * np.cos(theta_i)[None, :]).ravel(),
            (r_in[:, None] * np.sin(theta_i)[None, :]).ravel(),
            np.repeat(z_in, s.n_theta),
        ],
        axis=1,
    ).astype(np.float64)
    inner_uv = np.zeros((inner_v.shape[0], 2), dtype=np.float64)

    inner_faces = []
    for i in range(len(z_in) - 1):
        a = i * s.n_theta + j
        b = i * s.n_theta + jn
        c = (i + 1) * s.n_theta + jn
        d = (i + 1) * s.n_theta + j
        # reversed winding -> normals point into the cavity
        inner_faces.append(np.stack([a, c, b], axis=1))
        inner_faces.append(np.stack([a, d, c], axis=1))
    inner_faces = np.concatenate(inner_faces, axis=0)

    # Assemble vertices in blocks: outer, inner, bottom centre, top rim.
    n_outer = outer_v.shape[0]
    n_inner = inner_v.shape[0]
    idx_bottom_center = n_outer + n_inner
    bottom_center = np.array([[0.0, 0.0, 0.0]])
    verts = np.concatenate([outer_v, inner_v, bottom_center], axis=0)
    uv = np.concatenate([outer_uv, inner_uv, np.zeros((1, 2))], axis=0)

    # Offset inner face indices.
    inner_faces = inner_faces + n_outer
    # Bottom disk: outer bottom ring (outer block, ring 0) + centre.
    ib = np.arange(s.n_theta)
    ibn = (ib + 1) % s.n_theta
    bottom_faces = np.stack(
        [np.full(s.n_theta, idx_bottom_center), ibn, ib], axis=1
    )
    # Top rim: outer top ring and inner top ring.
    ot = (len(z_outer) - 1) * s.n_theta + j
    otn = (len(z_outer) - 1) * s.n_theta + jn
    it = n_outer + (len(z_in) - 1) * s.n_theta + j
    itn = n_outer + (len(z_in) - 1) * s.n_theta + jn
    rim_faces = np.concatenate(
        [
            np.stack([ot, otn, itn], axis=1),
            np.stack([ot, itn, it], axis=1),
        ],
        axis=0,
    )
    # Inner floor: inner bottom ring + floor centre.
    ifb = n_outer + ib
    ifbn = n_outer + ibn
    idx_floor_center = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, s.base_t]])], axis=0)
    uv = np.concatenate([uv, np.zeros((1, 2))], axis=0)
    floor_faces = np.stack(
        [np.full(s.n_theta, idx_floor_center), ifb, ifbn], axis=1
    )

    all_faces = np.concatenate(
        [outer_faces, inner_faces, bottom_faces, rim_faces, floor_faces], axis=0
    ).astype(np.int32)
    # Region: 0 outer textured, 1 inner, 2 bottom, 3 rim, 4 floor, 5 foot.
    outer_region = np.zeros(len(outer_faces), np.int8)
    # Two triangles per quad, rings in z order.
    mid_z = 0.5 * (z_outer[:-1] + z_outer[1:])
    per_ring = np.repeat((mid_z < s.foot_h).astype(np.int8) * 5, 2 * s.n_theta)
    outer_region[: len(per_ring)] = per_ring
    region = np.concatenate(
        [
            outer_region,
            np.ones(len(inner_faces), np.int8),
            np.full(len(bottom_faces), 2, np.int8),
            np.full(len(rim_faces), 3, np.int8),
            np.full(len(floor_faces), 4, np.int8),
        ]
    )

    # Face normals.
    v0 = verts[all_faces[:, 0]]
    v1 = verts[all_faces[:, 1]]
    v2 = verts[all_faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(fn, axis=1)
    ln[ln == 0] = 1.0
    fn /= ln[:, None]

    # Smooth normals for the outer surface and cavity; flat for rim/base.
    vn = np.zeros_like(verts)
    smooth_mask = (region <= 1) | (region == 5)
    for k in range(3):
        np.add.at(vn, all_faces[smooth_mask, k], fn[smooth_mask] * ln[smooth_mask, None])
    vnl = np.linalg.norm(vn, axis=1)
    vnl[vnl == 0] = 1.0
    vn /= vnl[:, None]

    return {
        "spec": asdict(s),
        "top_opening": float(2 * inner_radius(np.array([s.height]), s)[0]),
        "base_diameter": float(2 * (profile_radius(np.array([0.0]), s)[0] - s.bottom_chamfer)),
        "rim_width": float(
            profile_radius(np.array([s.height]), s)[0] - inner_radius(np.array([s.height]), s)[0]
        ),
        "vertices": verts.astype(np.float32),
        "faces": all_faces,
        "uv": uv.astype(np.float32),
        "normals": vn.astype(np.float32),
        "face_normals": fn.astype(np.float32),
        "region": region,
        "texture": tex,
        "heightmap": hm,
        "z_outer": z_outer,
        "r_out": r_out,
        "relief": relief,
    }


def build_cap_mesh(s: CupSpec, top: bool, n_theta: int = 576, n_z: int = 72) -> dict:
    """Solid spherical cap in ASSEMBLY coordinates.

    The cup is z=0..H; the top cap is z=H..H+cap_h, the bottom cap is
    z=-cap_h..0.  Their flat faces exactly match the cup's mouth and foot,
    so the three STLs form a complete Ø2R sphere when overlaid.
    """
    R = s.sphere_r
    zc = s.height / 2
    H = s.height
    cap_h = R - abs(H - zc)
    r_cut = float(np.sqrt(max(R**2 - (H - zc) ** 2, 0.0)))
    t = np.linspace(0.0, 1.0, n_z, endpoint=False)
    z = (H + cap_h * t) if top else (-cap_h * t)
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    u = (s.u_front + (theta[None, :] - np.pi / 2) / (2 * np.pi)) % 1.0
    u = np.broadcast_to(u, (n_z, n_theta))
    lat = np.arcsin(np.clip((z - zc) / R, -1.0, 1.0))
    v = np.clip((0.5 - lat / np.pi) * s.v_scale, 0.0, 1.0)
    # Real planetary relief on the cap, sampled from the same polar rows as
    # the cup.  Fade to zero at the mating face and at the pole so the part
    # stays watertight and the seam is continuous.
    hm = np.load(PROC / s.heightmap).astype(np.float32)
    relief = bilinear_wrap(hm, u, np.repeat(v[:, None], n_theta, axis=1))
    # Polar rows are extremely compressed in an equirectangular map; blend
    # toward a wider low-pass near the pole to avoid radial spikes/facets.
    relief_wide = ndimage.gaussian_filter(relief, sigma=(2.5, 6.0), mode=("nearest", "wrap"))
    blend = smoothstep((t - 0.50) / 0.35)
    relief = (1.0 - blend)[:, None] * relief + blend[:, None] * relief_wide
    relief = ndimage.gaussian_filter(relief, sigma=(1.2, 2.0), mode=("nearest", "wrap"))
    # Circular mating ring: delay the relief just enough that it cannot
    # push the cap outside the sphere cut radius and form a local bulge.
    pole_fade = smoothstep((1.0 - t) / 0.28)
    base_fade = smoothstep((t - 0.04) / 0.06)
    relief_applied = relief * s.relief_gain * base_fade[:, None] * pole_fade[:, None]
    r = np.sqrt(np.maximum(R**2 - (z - zc) ** 2, 0.0))[:, None] + relief_applied
    x = r * np.cos(theta)[None, :]
    y = r * np.sin(theta)[None, :]
    verts = np.stack([x.ravel(), y.ravel(), np.repeat(z, n_theta)], axis=1).astype(np.float64)
    uv_side = np.stack([u.ravel(), np.repeat(v, n_theta)], axis=1)
    apex_z = (H + cap_h) if top else (-cap_h)
    i_apex = verts.shape[0]
    i_base_center = i_apex + 1
    verts = np.concatenate(
        [verts, np.array([[0.0, 0.0, apex_z]]), np.array([[0.0, 0.0, H if top else 0.0]])],
        axis=0,
    )
    uv = np.concatenate([uv_side, np.array([[s.u_front, 0.0 if top else 1.0]]), np.zeros((1, 2))], axis=0)

    j = np.arange(n_theta)
    jn = (j + 1) % n_theta
    faces = []
    for i in range(n_z - 1):
        a = i * n_theta + j
        b = i * n_theta + jn
        c = (i + 1) * n_theta + jn
        d = (i + 1) * n_theta + j
        if top:
            faces.append(np.stack([a, b, c], axis=1))
            faces.append(np.stack([a, c, d], axis=1))
        else:
            faces.append(np.stack([a, c, b], axis=1))
            faces.append(np.stack([a, d, c], axis=1))
    last = (n_z - 1) * n_theta
    if top:
        faces.append(np.stack([last + j, last + jn, np.full(n_theta, i_apex)], axis=1))
    else:
        faces.append(np.stack([last + j, np.full(n_theta, i_apex), last + jn], axis=1))
    base_ring = j
    if top:
        faces.append(np.stack([np.full(n_theta, i_base_center), base_ring[jn], base_ring[j]], axis=1))
    else:
        faces.append(np.stack([np.full(n_theta, i_base_center), base_ring[j], base_ring[jn]], axis=1))
    all_faces = np.concatenate(faces, axis=0).astype(np.int32)
    region = np.concatenate(
        [np.zeros(len(all_faces) - n_theta, np.int8), np.full(n_theta, 3, np.int8)]
    )
    v0 = verts[all_faces[:, 0]]
    v1 = verts[all_faces[:, 1]]
    v2 = verts[all_faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(fn, axis=1)
    ln[ln == 0] = 1.0
    fn /= ln[:, None]
    vn = np.zeros_like(verts)
    smooth = region == 0
    for k in range(3):
        np.add.at(vn, all_faces[smooth, k], fn[smooth] * ln[smooth, None])
    vnl = np.linalg.norm(vn, axis=1)
    vnl[vnl == 0] = 1.0
    vn /= vnl[:, None]
    return {
        "vertices": verts.astype(np.float32),
        "faces": all_faces,
        "uv": uv.astype(np.float32),
        "normals": vn.astype(np.float32),
        "face_normals": fn.astype(np.float32),
        "region": region,
        "cap_h": float(cap_h),
        "r_cut": r_cut,
        "top": bool(top),
        "relief": relief_applied.astype(np.float32),
    }


def cap_stats(cap: dict, name: str) -> dict:
    v = cap["vertices"]
    f = cap["faces"]
    edges, bad = edge_check(f)
    bbox = v.max(0) - v.min(0)
    return {
        "file": name,
        "top": cap["top"],
        "facets": int(len(f)),
        "vertices": int(len(v)),
        "non_manifold_edges": bad,
        "volume_mm3": round(signed_volume(v, f), 2),
        "bbox_mm": [round(float(x), 3) for x in bbox],
        "cap_height_mm": round(cap["cap_h"], 3),
        "base_radius_mm": round(cap["r_cut"], 3),
        "relief_min_mm": round(float(cap["relief"].min()), 3),
        "relief_max_mm": round(float(cap["relief"].max()), 3),
    }


def build_assembled_display(s: CupSpec) -> dict:
    """Concatenate cup + caps for viewing.  Internal mating faces are
    removed so the review viewer does not show a coplanar rim ring.  This is
    a DISPLAY mesh, not a printable watertight solid."""
    cup = np.load(DATA / f"{s.slug}_mesh.npz")
    parts = [
        {
            "v": cup["vertices"].astype(np.float32),
            "f": cup["faces"].astype(np.int32),
            "r": cup["region"].astype(np.int8),
            "drop": (2, 3),  # bottom disk + top rim are internal when capped
        }
    ]
    for top in (True, False):
        cap = build_cap_mesh(s, top)
        parts.append(
            {
                "v": cap["vertices"].astype(np.float32),
                "f": cap["faces"].astype(np.int32),
                "r": cap["region"].astype(np.int8),
                "drop": (3,),  # cap mating disk
            }
        )
    verts, faces = [], []
    off = 0
    for p in parts:
        keep = ~np.isin(p["r"], p["drop"])
        verts.append(p["v"])
        faces.append(p["f"][keep] + off)
        off += len(p["v"])
    v = np.concatenate(verts, axis=0)
    f = np.concatenate(faces, axis=0)
    path = EXPORT / f"{s.slug}_assembled_display.stl"
    write_stl(path, v, f)
    return {"file": path.name, "facets": int(len(f)), "stl_bytes": path.stat().st_size}


def sphere_mesh(s: CupSpec, seam: bool = False) -> dict:
    """Watertight one-piece sphere with a real planetary relief.

    ``seam=False`` uses wrapped theta indices (for STL); ``seam=True`` adds a
    duplicate column at u=1 for a texture-safe OBJ/preview mesh.
    """
    R = s.sphere_r
    zc = s.height / 2.0
    n_theta = s.n_theta
    n_z = 360
    n_col = n_theta + 1 if seam else n_theta
    theta = 1.5 * np.pi + np.linspace(0.0, 2 * np.pi, n_col, endpoint=seam)
    u = (s.u_front + (theta - np.pi / 2) / (2 * np.pi)) % 1.0
    if seam:
        u[-1] = 1.0
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_z + 2)[1:-1]
    z = zc + R * np.sin(lat)
    v = 0.5 - lat / np.pi
    uu = np.broadcast_to(u[None, :], (n_z, n_col))
    vv = np.broadcast_to(v[:, None], (n_z, n_col))
    hm = np.load(PROC / s.heightmap).astype(np.float32)
    relief = bilinear_wrap(hm, uu, vv)
    relief = ndimage.gaussian_filter(relief, sigma=(0.75, 1.0), mode=("nearest", "wrap"))
    pole_fade = smoothstep((np.pi / 2 - np.abs(lat)) / np.deg2rad(6.0))[:, None]
    relief = np.maximum(relief, 0.0) * s.relief_gain * pole_fade
    r = R * np.cos(lat)[:, None] + relief

    ring_v = np.stack(
        [
            (r * np.cos(theta)[None, :]).ravel(),
            (r * np.sin(theta)[None, :]).ravel(),
            np.repeat(z, n_col),
        ],
        axis=1,
    ).astype(np.float64)
    ring_uv = np.stack(
        [uu.ravel(), np.repeat(v, n_col)], axis=1
    ).astype(np.float64)
    bottom = len(ring_v)
    top = bottom + 1
    verts = np.concatenate(
        [ring_v, [[0.0, 0.0, zc - R]], [[0.0, 0.0, zc + R]]], axis=0
    )
    uv = np.concatenate([ring_uv, [[0.5, 1.0], [0.5, 0.0]]], axis=0)

    j = np.arange(n_col) if not seam else np.arange(n_col - 1)
    jn = j + 1 if seam else (j + 1) % n_col
    faces = []
    for i in range(n_z - 1):
        a = i * n_col + j
        b = i * n_col + jn
        c = (i + 1) * n_col + jn
        d = (i + 1) * n_col + j
        faces.append(np.stack([a, b, c], axis=1))
        faces.append(np.stack([a, c, d], axis=1))
    last = (n_z - 1) * n_col
    faces.append(np.stack([np.full_like(j, bottom), jn, j], axis=1))
    faces.append(np.stack([np.full_like(j, top), last + j, last + jn], axis=1))
    faces = np.concatenate(faces, axis=0).astype(np.int32)
    vol = signed_volume(verts, faces)
    if vol < 0:
        faces = faces[:, ::-1].copy()
        vol = -vol

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(fn, axis=1)
    ln[ln == 0] = 1.0
    fn = fn / ln[:, None]
    vn = np.zeros_like(verts)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    ln = np.linalg.norm(vn, axis=1)
    ln[ln == 0] = 1.0
    vn = vn / ln[:, None]
    return {
        "vertices": verts.astype(np.float32),
        "faces": faces,
        "uv": uv.astype(np.float32),
        "normals": vn.astype(np.float32),
        "face_normals": fn.astype(np.float32),
        "region": np.zeros(len(faces), dtype=np.int8),
        "volume_mm3": float(vol),
    }


def build_sphere(s: CupSpec) -> dict:
    wrap = sphere_mesh(s, seam=False)
    seam = sphere_mesh(s, seam=True)
    stl_path = EXPORT / f"{s.slug}_sphere.stl"
    write_stl(stl_path, wrap["vertices"], wrap["faces"])
    np.savez_compressed(
        DATA / f"{s.slug}_sphere.npz",
        vertices=wrap["vertices"],
        faces=wrap["faces"],
        uv=wrap["uv"],
        normals=wrap["normals"],
        face_normals=wrap["face_normals"],
        region=wrap["region"],
    )
    tex_name = f"{s.slug}_sphere_texture.png"
    (EXPORT / tex_name).write_bytes((PROC / s.texture).read_bytes())
    write_obj(EXPORT / f"{s.slug}_sphere.obj", seam, tex_name)
    edges, bad = edge_check(wrap["faces"])
    bbox = wrap["vertices"].max(0) - wrap["vertices"].min(0)
    return {
        "file": stl_path.name,
        "triangles": int(len(wrap["faces"])),
        "vertices": int(len(wrap["vertices"])),
        "non_manifold_edges": int(bad),
        "volume_mm3": round(float(wrap["volume_mm3"]), 2),
        "bbox_mm": [round(float(x), 3) for x in bbox],
        "stl_bytes": stl_path.stat().st_size,
    }


def icosphere(level: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic sphere: nearly uniform triangles, no polar singularity."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [np.asarray(v, dtype=np.float64) for v in verts]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    cache = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        v = verts[a] + verts[b]
        v = v / np.linalg.norm(v)
        verts.append(v)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for _ in range(level):
        nxt = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            nxt.extend(
                [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            )
        faces = nxt
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def build_icosphere(s: CupSpec, level: int = 7) -> dict:
    R = s.sphere_r
    zc = s.height / 2.0
    unit, faces = icosphere(level)
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    u = (s.u_front + (np.arctan2(y, x) - np.pi / 2) / (2 * np.pi)) % 1.0
    v = np.clip(0.5 - lat / np.pi, 0.0, 1.0)
    hm = np.load(PROC / s.heightmap).astype(np.float32)
    relief = bilinear_wrap(hm, u.reshape(1, -1), v.reshape(1, -1)).ravel()
    # One-ring neighbour smoothing works in every direction, without the
    # latitude-dependent blur of an equirectangular image filter.
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    sums = np.zeros(len(unit), dtype=np.float64)
    cnt = np.zeros(len(unit), dtype=np.float64)
    np.add.at(sums, e[:, 0], relief[e[:, 1]])
    np.add.at(sums, e[:, 1], relief[e[:, 0]])
    np.add.at(cnt, e[:, 0], 1.0)
    np.add.at(cnt, e[:, 1], 1.0)
    relief = 0.5 * relief + 0.5 * (sums / np.maximum(cnt, 1.0))
    r = R + np.maximum(relief, 0.0) * s.relief_gain
    verts = unit * r[:, None]
    verts[:, 2] += zc

    vol = signed_volume(verts, faces)
    if vol < 0:
        faces = faces[:, ::-1].copy()
        vol = -vol
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(fn, axis=1)
    ln[ln == 0] = 1.0
    fn = fn / ln[:, None]
    vn = np.zeros_like(verts)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    ln = np.linalg.norm(vn, axis=1)
    ln[ln == 0] = 1.0
    vn = vn / ln[:, None]
    mesh = {
        "vertices": verts.astype(np.float32),
        "faces": faces,
        "uv": np.stack([u, v], axis=1).astype(np.float32),
        "normals": vn.astype(np.float32),
        "face_normals": fn.astype(np.float32),
        "region": np.zeros(len(faces), dtype=np.int8),
    }
    stl_path = EXPORT / f"{s.slug}_icosphere.stl"
    write_stl(stl_path, mesh["vertices"], mesh["faces"])
    np.savez_compressed(
        DATA / f"{s.slug}_icosphere.npz",
        vertices=mesh["vertices"],
        faces=mesh["faces"],
        uv=mesh["uv"],
        normals=mesh["normals"],
        face_normals=mesh["face_normals"],
        region=mesh["region"],
    )
    tex_name = f"{s.slug}_icosphere_texture.png"
    (EXPORT / tex_name).write_bytes((PROC / s.texture).read_bytes())
    write_obj(EXPORT / f"{s.slug}_icosphere.obj", mesh, tex_name)
    edges, bad = edge_check(mesh["faces"])
    bbox = mesh["vertices"].max(0) - mesh["vertices"].min(0)
    return {
        "file": stl_path.name,
        "triangles": int(len(mesh["faces"])),
        "vertices": int(len(mesh["vertices"])),
        "non_manifold_edges": int(bad),
        "volume_mm3": round(float(vol), 2),
        "bbox_mm": [round(float(x), 3) for x in bbox],
        "stl_bytes": stl_path.stat().st_size,
    }


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def edge_check(faces: np.ndarray) -> tuple[int, int]:
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e.sort(axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    bad = counts != 2
    return int(uniq.shape[0]), int(bad.sum())


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    n /= ln[:, None]
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data = np.zeros(len(faces), dtype=dt)
    data["n"] = n
    data["v"] = np.stack([v0, v1, v2], axis=1)
    with path.open("wb") as f:
        f.write(b"3d-stars planet pen cup / generated from real planetary imagery".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        f.write(data.tobytes())


def write_obj(path: Path, mesh: dict, tex_name: str) -> None:
    """Write a textured OBJ with per-corner UVs (seam-safe)."""
    v = mesh["vertices"].astype(np.float64)
    f = mesh["faces"]
    uv = mesh["uv"]
    region = mesh["region"]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# planet pen cup\n")
        fh.write(f"mtllib {path.stem}.mtl\n")
        for p in v:
            fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        # One vt per face corner for the outer surface, a single vt for the rest.
        vt_index = np.zeros((len(f), 3), dtype=np.int64)
        vt = [(0.0, 0.0)]
        for k in range(len(f)):
            if region[k] == 0:
                for c in range(3):
                    vi = f[k, c]
                    u, vv = uv[vi]
                    vt.append((u, vv))
                    vt_index[k, c] = len(vt)
            else:
                vt_index[k, :] = 1
        for u, vv in vt:
            fh.write(f"vt {u:.6f} {vv:.6f}\n")
        fh.write("usemtl planet\n")
        for k in range(len(f)):
            if region[k] != 0:
                continue
            a, b, c = f[k]
            ta, tb, tc = vt_index[k]
            fh.write(f"f {a+1}/{ta} {b+1}/{tb} {c+1}/{tc}\n")
        fh.write("usemtl body\n")
        for k in range(len(f)):
            if region[k] == 0:
                continue
            a, b, c = f[k]
            fh.write(f"f {a+1}/1 {b+1}/1 {c+1}/1\n")
    mtl = path.with_suffix(".mtl")
    with mtl.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("newmtl planet\n")
        fh.write("Ka 0.2 0.2 0.2\nKd 1 1 1\nKs 0.05 0.05 0.05\nNs 10\n")
        fh.write(f"map_Kd {tex_name}\n\n")
        fh.write("newmtl body\n")
        fh.write("Ka 0.18 0.18 0.18\nKd 0.42 0.44 0.47\nKs 0.08 0.08 0.08\nNs 20\n")


def stats(mesh: dict) -> dict:
    s = mesh["spec"]
    v = mesh["vertices"]
    f = mesh["faces"]
    edges, bad_edges = edge_check(f)
    vol = signed_volume(v, f)
    r = np.linalg.norm(v[:, :2], axis=1)
    bbox = v.max(0) - v.min(0)
    return {
        "slug": s["slug"],
        "name": s["name"],
        "name_cn": s["name_cn"],
        "texture": s["texture"],
        "heightmap": s["heightmap"],
        "u_front": s["u_front"],
        "relief_gain": s["relief_gain"],
        "triangles": int(len(f)),
        "vertices": int(len(v)),
        "edges": edges,
        "non_manifold_edges": bad_edges,
        "signed_volume_mm3": round(vol, 1),
        "bbox_mm": [round(float(x), 3) for x in bbox],
        "max_radius_mm": round(float(r.max()), 3),
        "max_diameter_mm": round(float(2 * r.max()), 3),
        "min_radius_mm": round(float(r.min()), 3),
        "top_opening_mm": round(mesh["top_opening"], 3),
        "base_diameter_mm": round(mesh["base_diameter"], 3),
        "rim_width_mm": round(mesh["rim_width"], 3),
        "nominal_wall_mm": s["wall"],
        "min_wall_mm": s["wall"],
        "base_thickness_mm": s["base_t"],
        "foot_height_mm": s["foot_h"],
        "relief_min_mm": round(float(mesh["relief"].min()), 3),
        "relief_max_mm": round(float(mesh["relief"].max()), 3),
        "height_mm": s["height"],
    }


def build_one(s: CupSpec) -> dict:
    t0 = time.time()
    mesh = build_mesh(s)
    stl = EXPORT / f"{s.slug}_pen_cup.stl"
    write_stl(stl, mesh["vertices"], mesh["faces"])
    texture = Image.open(PROC / s.texture).convert("RGB")
    texture.save(EXPORT / f"{s.slug}_texture.png", optimize=True)
    write_obj(EXPORT / f"{s.slug}_pen_cup.obj", mesh, f"{s.slug}_texture.png")
    np.savez_compressed(
        DATA / f"{s.slug}_mesh.npz",
        vertices=mesh["vertices"],
        faces=mesh["faces"],
        uv=mesh["uv"],
        normals=mesh["normals"],
        face_normals=mesh["face_normals"],
        region=mesh["region"],
        z_outer=mesh["z_outer"].astype(np.float32),
        r_out=mesh["r_out"].astype(np.float32),
        relief=mesh["relief"].astype(np.float32),
    )
    st = stats(mesh)
    st["stl_bytes"] = stl.stat().st_size
    st["seconds"] = round(time.time() - t0, 2)
    print(json.dumps(st, ensure_ascii=False))
    return st


def main() -> None:
    ico_rows = []
    for s in SPECS:
        st = build_icosphere(s)
        ico_rows.append(st)
        print(json.dumps(st, ensure_ascii=False))
    (DATA / "icosphere_manifest.json").write_text(
        json.dumps(ico_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote geodesic icosphere STLs")
    return

    # Legacy cup/cap/display pipeline below is intentionally not run.
    manifest = []
    for s in SPECS:
        manifest.append(build_one(s))
    (DATA / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", DATA / "manifest.json")

    # Per-planet spherical caps, with real image-derived relief on each pole.
    cap_dir = EXPORT / "assembly"
    cap_dir.mkdir(exist_ok=True)
    cap_rows = []
    for s in SPECS:
        for top in (True, False):
            cap = build_cap_mesh(s, top)
            side = "top" if top else "bottom"
            name = f"{s.slug}_cap_{side}.stl"
            write_stl(cap_dir / name, cap["vertices"], cap["faces"])
            write_obj(cap_dir / f"{s.slug}_cap_{side}.obj", cap, f"../{s.slug}_texture.png")
            st = cap_stats(cap, name)
            st["slug"] = s.slug
            st["stl_bytes"] = (cap_dir / name).stat().st_size
            cap_rows.append(st)
            np.savez_compressed(
                DATA / f"{s.slug}_cap_{side}.npz",
                vertices=cap["vertices"],
                faces=cap["faces"],
                uv=cap["uv"],
                normals=cap["normals"],
                face_normals=cap["face_normals"],
                region=cap["region"],
            )
            print(json.dumps(st, ensure_ascii=False))
    (DATA / "caps_manifest.json").write_text(
        json.dumps(cap_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for old in ("sphere_cap_top.stl", "sphere_cap_bottom.stl"):
        p = cap_dir / old
        if p.exists():
            p.unlink()
    print("wrote", cap_dir, "and", DATA / "caps_manifest.json")

    display_rows = []
    for s in SPECS:
        st = build_assembled_display(s)
        display_rows.append(st)
        print(json.dumps(st, ensure_ascii=False))
    (DATA / "display_manifest.json").write_text(
        json.dumps(display_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote display assembly STLs")

    sphere_rows = []
    for s in SPECS:
        st = build_sphere(s)
        sphere_rows.append(st)
        print(json.dumps(st, ensure_ascii=False))
    (DATA / "sphere_manifest.json").write_text(
        json.dumps(sphere_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote one-piece sphere STLs")

    ico_rows = []
    for s in SPECS:
        st = build_icosphere(s)
        ico_rows.append(st)
        print(json.dumps(st, ensure_ascii=False))
    (DATA / "icosphere_manifest.json").write_text(
        json.dumps(ico_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote geodesic icosphere STLs")


if __name__ == "__main__":
    main()
