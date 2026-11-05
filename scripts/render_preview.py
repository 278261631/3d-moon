"""Software-rendered turntable / product previews for the planet cups.

No GPU or 3D package is required: a small Numba z-buffer rasteriser samples
the real planetary texture on the displaced mesh and applies studio lighting.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = ROOT / "assets/processed"
OUT = ROOT / "preview"
OUT.mkdir(exist_ok=True)


@njit(cache=True, fastmath=True)
def _sample(tex, u, v):
    h, w, _ = tex.shape
    x = (u % 1.0) * w - 0.5
    y = min(max(v, 0.0), 1.0) * h - 0.5
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    tx = x - x0
    ty = y - y0
    x0 = x0 % w
    x1 = (x0 + 1) % w
    y0 = min(max(y0, 0), h - 1)
    y1 = min(y0 + 1, h - 1)
    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]
    out = np.empty(3, np.float32)
    for k in range(3):
        out[k] = (c00[k] * (1 - tx) + c10[k] * tx) * (1 - ty) + (c01[k] * (1 - tx) + c11[k] * tx) * ty
    return out


@njit(cache=True, fastmath=True)
def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


@njit(cache=True, fastmath=True)
def _render(vertices, faces, uv, vn, fn, region, tex, cam, right, up, fwd, f, W, H, clay, exposure):
    col = np.zeros((H, W, 3), np.float32)
    depth = np.full((H, W), 1e30, np.float32)
    alpha = np.zeros((H, W), np.float32)

    L1 = -0.55 * right + 0.55 * up - 0.90 * fwd
    L2 = 0.70 * right + 0.15 * up - 0.55 * fwd
    L3 = 0.10 * right - 0.30 * up + 0.95 * fwd
    for L in (L1, L2, L3):
        n = math.sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2])
        L[0] /= n
        L[1] /= n
        L[2] /= n

    for k in range(faces.shape[0]):
        i0 = faces[k, 0]
        i1 = faces[k, 1]
        i2 = faces[k, 2]
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        # back-face culling
        cx = (p0[0] + p1[0] + p2[0]) / 3.0
        cy = (p0[1] + p1[1] + p2[1]) / 3.0
        cz = (p0[2] + p1[2] + p2[2]) / 3.0
        vx = cam[0] - cx
        vy = cam[1] - cy
        vz = cam[2] - cz
        if fn[k, 0] * vx + fn[k, 1] * vy + fn[k, 2] * vz <= 0.0:
            continue

        r0 = p0 - cam
        r1 = p1 - cam
        r2 = p2 - cam
        z0 = r0[0] * fwd[0] + r0[1] * fwd[1] + r0[2] * fwd[2]
        z1 = r1[0] * fwd[0] + r1[1] * fwd[1] + r1[2] * fwd[2]
        z2 = r2[0] * fwd[0] + r2[1] * fwd[1] + r2[2] * fwd[2]
        if z0 <= 0.5 or z1 <= 0.5 or z2 <= 0.5:
            continue
        x0 = f * (r0[0] * right[0] + r0[1] * right[1] + r0[2] * right[2]) / z0 + W * 0.5
        x1 = f * (r1[0] * right[0] + r1[1] * right[1] + r1[2] * right[2]) / z1 + W * 0.5
        x2 = f * (r2[0] * right[0] + r2[1] * right[1] + r2[2] * right[2]) / z2 + W * 0.5
        y0 = -f * (r0[0] * up[0] + r0[1] * up[1] + r0[2] * up[2]) / z0 + H * 0.5
        y1 = -f * (r1[0] * up[0] + r1[1] * up[1] + r1[2] * up[2]) / z1 + H * 0.5
        y2 = -f * (r2[0] * up[0] + r2[1] * up[1] + r2[2] * up[2]) / z2 + H * 0.5

        minx = max(0, int(math.floor(min(x0, min(x1, x2)))))
        maxx = min(W - 1, int(math.ceil(max(x0, max(x1, x2)))))
        miny = max(0, int(math.floor(min(y0, min(y1, y2)))))
        maxy = min(H - 1, int(math.ceil(max(y0, max(y1, y2)))))
        if minx > maxx or miny > maxy:
            continue
        area = _edge(x0, y0, x1, y1, x2, y2)
        if abs(area) < 1e-9:
            continue
        inv_area = 1.0 / area

        u0 = uv[i0, 0]
        u1 = uv[i1, 0]
        u2 = uv[i2, 0]
        if u1 - u0 > 0.5:
            u1 -= 1.0
        elif u1 - u0 < -0.5:
            u1 += 1.0
        if u2 - u0 > 0.5:
            u2 -= 1.0
        elif u2 - u0 < -0.5:
            u2 += 1.0
        v0 = uv[i0, 1]
        v1 = uv[i1, 1]
        v2 = uv[i2, 1]
        reg = region[k]

        for yy in range(miny, maxy + 1):
            py = yy + 0.5
            for xx in range(minx, maxx + 1):
                px = xx + 0.5
                w0 = _edge(x1, y1, x2, y2, px, py) * inv_area
                w1 = _edge(x2, y2, x0, y0, px, py) * inv_area
                w2 = 1.0 - w0 - w1
                if w0 < -1e-5 or w1 < -1e-5 or w2 < -1e-5:
                    continue
                invz = w0 / z0 + w1 / z1 + w2 / z2
                z = 1.0 / invz
                if z >= depth[yy, xx]:
                    continue
                depth[yy, xx] = z
                alpha[yy, xx] = 1.0

                u = (w0 * u0 / z0 + w1 * u1 / z1 + w2 * u2 / z2) / invz
                v = (w0 * v0 / z0 + w1 * v1 / z1 + w2 * v2 / z2) / invz

                if reg == 0:
                    alb = _sample(tex, u, v)
                elif reg == 1:
                    alb = np.array([0.105, 0.110, 0.118], np.float32)
                elif reg == 2:
                    alb = np.array([0.055, 0.055, 0.058], np.float32)
                elif reg == 3:
                    alb = np.array([0.55, 0.53, 0.50], np.float32)
                elif reg == 5:
                    alb = np.array([0.115, 0.098, 0.082], np.float32)
                else:
                    alb = np.array([0.05, 0.05, 0.055], np.float32)
                if clay:
                    if reg == 0:
                        alb = np.array([0.62, 0.62, 0.63], np.float32)
                    else:
                        alb = np.array([0.45, 0.45, 0.46], np.float32)

                nx = w0 * vn[i0, 0] + w1 * vn[i1, 0] + w2 * vn[i2, 0]
                ny = w0 * vn[i0, 1] + w1 * vn[i1, 1] + w2 * vn[i2, 1]
                nz = w0 * vn[i0, 2] + w1 * vn[i1, 2] + w2 * vn[i2, 2]
                ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                if ln < 1e-9:
                    nx, ny, nz = fn[k, 0], fn[k, 1], fn[k, 2]
                else:
                    nx /= ln
                    ny /= ln
                    nz /= ln
                if reg >= 2 and reg != 5:
                    nx, ny, nz = fn[k, 0], fn[k, 1], fn[k, 2]

                # view vector
                ax = cam[0] - (w0 * p0[0] + w1 * p1[0] + w2 * p2[0])
                ay = cam[1] - (w0 * p0[1] + w1 * p1[1] + w2 * p2[1])
                az = cam[2] - (w0 * p0[2] + w1 * p1[2] + w2 * p2[2])
                vln = math.sqrt(ax * ax + ay * ay + az * az)
                ax /= vln
                ay /= vln
                az /= vln

                d1 = nx * L1[0] + ny * L1[1] + nz * L1[2]
                d2 = nx * L2[0] + ny * L2[1] + nz * L2[2]
                d3 = nx * L3[0] + ny * L3[1] + nz * L3[2]
                if d1 < 0.0:
                    d1 = 0.0
                if d2 < 0.0:
                    d2 = 0.0
                if d3 < 0.0:
                    d3 = 0.0
                if clay:
                    amb = 0.11 if reg == 0 else (0.09 if reg == 1 else 0.06)
                    r = alb[0] * (amb + 1.45 * d1 + 0.16 * d2)
                    g = alb[1] * (amb + 1.45 * d1 + 0.16 * d2)
                    b = alb[2] * (amb + 1.45 * d1 + 0.16 * d2)
                    r += 0.05 * d3
                    g += 0.055 * d3
                    b += 0.07 * d3
                else:
                    amb = 0.15 if reg == 0 else (0.085 if reg == 1 else 0.05)
                    r = alb[0] * (amb + 1.05 * d1 + 0.34 * d2)
                    g = alb[1] * (amb + 1.05 * d1 + 0.34 * d2)
                    b = alb[2] * (amb + 1.05 * d1 + 0.34 * d2)
                    # cool rim / warm key mix
                    r += 0.10 * d3
                    g += 0.12 * d3
                    b += 0.17 * d3
                    # specular
                    hx = L1[0] + ax
                    hy = L1[1] + ay
                    hz = L1[2] + az
                    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
                    ndh = (nx * hx + ny * hy + nz * hz) / max(hn, 1e-9)
                    if ndh > 0.0:
                        sp = ndh ** 42 * 0.22
                        r += sp
                        g += sp
                        b += sp
                    # fresnel
                    ndv = nx * ax + ny * ay + nz * az
                    if ndv < 0.0:
                        ndv = 0.0
                    fr = (1.0 - ndv) ** 4 * 0.12
                    r += fr * 0.75
                    g += fr * 0.82
                    b += fr * 1.0

                # filmic-ish tonemap + sRGB
                r *= exposure
                g *= exposure
                b *= exposure
                r = 1.0 - math.exp(-r * 1.25)
                g = 1.0 - math.exp(-g * 1.25)
                b = 1.0 - math.exp(-b * 1.25)
                r = r ** (1.0 / 2.2)
                g = g ** (1.0 / 2.2)
                b = b ** (1.0 / 2.2)
                # ordered dither
                dith = ((xx * 7 + yy * 13) % 7) / 255.0 - 0.012
                col[yy, xx, 0] = min(max(r + dith, 0.0), 1.0)
                col[yy, xx, 1] = min(max(g + dith, 0.0), 1.0)
                col[yy, xx, 2] = min(max(b + dith, 0.0), 1.0)
    return col, alpha


def camera_basis(az_deg: float, el_deg: float, target: np.ndarray, dist: float):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cam = target + dist * np.array(
        [math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)],
        dtype=np.float64,
    )
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return cam, right, up, fwd


def background(w: int, h: int) -> Image.Image:
    y = np.linspace(0, 1, h)[:, None, None]
    top = np.array([24, 27, 33], np.float32)
    bot = np.array([8, 9, 12], np.float32)
    arr = top * (1 - y) + bot * y
    arr = np.broadcast_to(arr, (h, w, 3)).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.43
    r = np.sqrt(((xx - cx) / (w * 0.62)) ** 2 + ((yy - cy) / (h * 0.60)) ** 2)
    glow = np.clip(1 - r, 0, 1) ** 2.2
    tint = np.array([70, 86, 112], np.float32)
    arr = arr + glow[..., None] * tint * 0.38
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def composite(col: np.ndarray, alpha: np.ndarray, w: int, h: int) -> Image.Image:
    bg = background(w, h)
    # contact shadow from the silhouette, offset down and blurred
    am = Image.fromarray((alpha * 255).astype(np.uint8), "L")
    shadow = Image.new("L", (w, h), 0)
    shadow.paste(am, (0, int(h * 0.012)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(6, w // 45)))
    shadow = shadow.point(lambda p: int(p * 0.55))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    bg = Image.composite(dark, bg, shadow)
    # tighter contact shadow
    shadow2 = am.filter(ImageFilter.GaussianBlur(max(2, w // 160)))
    shadow2 = shadow2.point(lambda p: int(p * 0.45))
    bg = Image.composite(dark, bg, shadow2)
    fg = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGB")
    fg.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))
    return Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")


TEXTURES = {"luna": "moon_color.png", "terra": "earth_color.png", "pluto": "pluto_color.png"}


def render_arrays(verts, faces, uv, vn, fn, region, tex, az, el, dist, w, h, clay, fov=27.0, exposure=1.0):
    target = np.array([0.0, 0.0, 0.5 * (float(verts[:, 2].min()) + float(verts[:, 2].max()))])
    cam, right, up, fwd = camera_basis(az, el, target, dist)
    f = 0.5 * h / math.tan(math.radians(fov) / 2)
    col, alpha = _render(verts, faces, uv, vn, fn, region, tex, cam, right, up, fwd, f, w, h, clay, exposure)
    return composite(col, alpha, w, h)


def render(slug: str, az: float, el: float, dist: float, w: int, h: int, clay: bool, fov=27.0, exposure=1.0):
    npz = np.load(DATA / f"{slug}_mesh.npz")
    tex = np.array(Image.open(PROC / TEXTURES[slug]).convert("RGB"), np.float32) / 255.0
    return render_arrays(
        npz["vertices"].astype(np.float32),
        npz["faces"].astype(np.int32),
        npz["uv"].astype(np.float32),
        npz["normals"].astype(np.float32),
        npz["face_normals"].astype(np.float32),
        npz["region"].astype(np.int8),
        tex ** 2.2,
        az, el, dist, w, h, clay, fov, exposure,
    )


def render_assembly(slug: str, exploded: bool, w: int, h: int, dist: float, az=66, el=10, exposure=1.0):
    from planet_cup import SPECS, build_cap_mesh

    spec = next(s for s in SPECS if s.slug == slug)
    cup = np.load(DATA / f"{slug}_mesh.npz")
    parts = [
        {
            "vertices": cup["vertices"].astype(np.float32),
            "faces": cup["faces"].astype(np.int32),
            "uv": cup["uv"].astype(np.float32),
            "normals": cup["normals"].astype(np.float32),
            "face_normals": cup["face_normals"].astype(np.float32),
            "region": cup["region"].astype(np.int8),
        }
    ]
    for top in (True, False):
        cap = build_cap_mesh(spec, top)
        if exploded:
            cap["vertices"] = cap["vertices"].copy()
            cap["vertices"][:, 2] += 30.0 if top else -30.0
        parts.append(cap)

    verts, faces, uv, vn, fn, region = [], [], [], [], [], []
    off = 0
    for p in parts:
        verts.append(p["vertices"])
        faces.append(p["faces"] + off)
        uv.append(p["uv"])
        vn.append(p["normals"])
        fn.append(p["face_normals"])
        region.append(p["region"])
        off += len(p["vertices"])
    tex = np.array(Image.open(PROC / TEXTURES[slug]).convert("RGB"), np.float32) / 255.0
    return render_arrays(
        np.concatenate(verts).astype(np.float32),
        np.concatenate(faces).astype(np.int32),
        np.concatenate(uv).astype(np.float32),
        np.concatenate(vn).astype(np.float32),
        np.concatenate(fn).astype(np.float32),
        np.concatenate(region).astype(np.int8),
        tex ** 2.2,
        az, el, dist, w, h, False, 27.0, exposure,
    )


def font(size: int, bold=False, cn=False):
    if cn:
        path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    else:
        path = "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def card(slug: str, cn: str, spec: dict, img: Image.Image) -> Image.Image:
    w, h = 1080, 1420
    out = Image.new("RGB", (w, h), (13, 14, 18))
    out.paste(img.resize((1000, 1250), Image.LANCZOS), (40, 0))
    d = ImageDraw.Draw(out)
    d.text((52, 1268), f"{spec['name'].upper()}  /  {cn}笔筒", font=font(46, True, cn=True), fill=(238, 238, 240))
    d.text(
        (54, 1332),
        f"Ø {spec['max_diameter_mm']:.1f} × H {spec['height_mm']:.0f} mm   开口 Ø {spec['top_opening_mm']:.1f} mm",
        font=font(24, cn=True),
        fill=(160, 168, 180),
    )
    d.text(
        (54, 1370),
        f"{spec['triangles']:,} 三角面 · 壁厚 {spec['nominal_wall_mm']} mm · 底厚 {spec['base_thickness_mm']} mm",
        font=font(22, cn=True),
        fill=(110, 118, 130),
    )
    return out


def collection(cards: list[Image.Image], out_path: Path):
    w, h = 2460, 1560
    bg = background(w, h)
    d = ImageDraw.Draw(bg)
    d.text((90, 64), "SPHERICAL PLANET PEN CUPS", font=font(66, True), fill=(245, 245, 248))
    d.text((94, 156), "真实行星影像 · 真实高程浮雕 · 可直接 3D 打印", font=font(34, cn=True), fill=(160, 178, 205))
    d.text((94, 206), "LRO / LOLA  ·  NASA Blue Marble + ETOPO1  ·  New Horizons", font=font(24), fill=(105, 118, 138))
    x = 60
    for c in cards:
        c2 = c.resize((760, 1000), Image.LANCZOS)
        bg.paste(c2, (x, 280))
        x += 790
    d.text((94, 1435), "Ø122–124 mm  ·  H90 mm  ·  3.2 mm wall  ·  watertight STL  ·  0.4 mm nozzle, 0.16–0.20 mm layers", font=font(23), fill=(120, 130, 145))
    d.text((94, 1480), "NASA / USGS / LROC / ETOPO1 data, public domain; Pluto colour map via Solar System Scope (CC-BY 4.0).", font=font(20), fill=(85, 94, 108))
    bg.save(out_path)


def annotate_exploded(img: Image.Image) -> Image.Image:
    w, h = img.size
    out = Image.new("RGB", (w + 300, h + 80), (13, 14, 18))
    out.paste(img, (0, 80))
    d = ImageDraw.Draw(out)
    d.text((24, 18), "3-PART SPHERICAL ASSEMBLY  /  Ø120 mm", font=font(30, True), fill=(240, 242, 246))
    d.text(
        (26, 53),
        "装配坐标：下盖 −15…0 · 杯体 0…90 · 上盖 90…105 mm；两盖含真实极区纹理/浮雕，仅作拼球展示",
        font=font(17, cn=True),
        fill=(150, 162, 180),
    )
    labels = [
        (190, "TOP CAP  /  上盖", "15 mm · 平面朝下"),
        (720, "CUP  /  笔筒", "90 mm · 开口 Ø69.3"),
        (1130, "BOTTOM CAP  /  下盖", "15 mm · 平面朝上"),
    ]
    for y, title, sub in labels:
        d.line([(w + 10, y), (w - 170, y)], fill=(110, 125, 150), width=2)
        d.text((w + 18, y - 26), title, font=font(22, True, cn=True), fill=(225, 230, 238))
        d.text((w + 18, y + 2), sub, font=font(16, cn=True), fill=(145, 158, 176))
    return out


def render_sphere(slug: str, w: int, h: int, dist: float = 350.0, az: float = 66.0, el: float = 18.0, clay: bool = False, exposure: float = 1.0):
    npz = np.load(DATA / f"{slug}_sphere.npz")
    tex = np.array(Image.open(PROC / TEXTURES[slug]).convert("RGB"), np.float32) / 255.0
    return render_arrays(
        npz["vertices"].astype(np.float32),
        npz["faces"].astype(np.int32),
        npz["uv"].astype(np.float32),
        npz["normals"].astype(np.float32),
        npz["face_normals"].astype(np.float32),
        npz["region"].astype(np.int8),
        tex ** 2.2,
        az, el, dist, w, h, clay, exposure=exposure,
    )


def render_icosphere(slug: str, w: int, h: int, dist: float = 350.0, az: float = 66.0, el: float = 18.0, clay: bool = False, exposure: float = 1.0):
    npz = np.load(DATA / f"{slug}_icosphere.npz")
    tex = np.array(Image.open(PROC / TEXTURES[slug]).convert("RGB"), np.float32) / 255.0
    return render_arrays(
        npz["vertices"].astype(np.float32),
        npz["faces"].astype(np.int32),
        npz["uv"].astype(np.float32),
        npz["normals"].astype(np.float32),
        npz["face_normals"].astype(np.float32),
        npz["region"].astype(np.int8),
        tex ** 2.2,
        az, el, dist, w, h, clay, exposure=exposure,
    )


def main():
    for slug in ("luna", "terra", "pluto"):
        render_icosphere(slug, 1000, 1000, clay=False).save(OUT / f"icosphere_{slug}.png")
        render_icosphere(slug, 900, 900, clay=True).save(OUT / f"icosphere_{slug}_clay.png")
    print("wrote icosphere previews")
    return

    # Legacy cup/cap preview pipeline below is intentionally not run.
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cards-only", action="store_true")
    args = ap.parse_args()
    manifest = {m["slug"]: m for m in json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))}
    if args.cards_only:
        cards = []
        for slug, cn in [("luna", "月球"), ("terra", "地球"), ("pluto", "冥王星")]:
            hero = Image.open(OUT / f"{slug}_hero.png").convert("RGB")
            c = card(slug, cn, manifest[slug], hero)
            c.save(OUT / f"{slug}_card.png")
            cards.append(c)
        collection(cards, OUT / "collection.png")
        print("wrote", OUT / "collection.png")
        return
    cards = []
    exposures = {"luna": 1.0, "terra": 1.0, "pluto": 0.72}
    for slug, cn in [("luna", "月球"), ("terra", "地球"), ("pluto", "冥王星")]:
        print("render", slug, flush=True)
        ex = exposures[slug]
        hero = render(slug, 66, 18, 350, 1000, 1250, False, exposure=ex)
        hero.save(OUT / f"{slug}_hero.png")
        front = render(slug, 90, 20, 345, 900, 1125, False, exposure=ex)
        front.save(OUT / f"{slug}_front.png")
        back = render(slug, 250, 18, 350, 900, 1125, False, exposure=ex)
        back.save(OUT / f"{slug}_back.png")
        clay = render(slug, 55, 18, 350, 900, 1125, True)
        clay.save(OUT / f"{slug}_clay.png")
        if not args.quick:
            detail = render(slug, 72, 6, 240, 1000, 1000, False, fov=31.0, exposure=ex)
            detail.save(OUT / f"{slug}_detail.png")
            c = card(slug, cn, manifest[slug], hero)
            c.save(OUT / f"{slug}_card.png")
            cards.append(c)
        print("ok", slug, flush=True)
    for slug in ("luna", "terra", "pluto"):
        print("sphere", slug, flush=True)
        render_sphere(slug, 1000, 1000, clay=False, exposure=exposures[slug]).save(
            OUT / f"sphere_{slug}.png"
        )
        render_sphere(slug, 900, 900, clay=True).save(OUT / f"sphere_{slug}_clay.png")
        print("ok sphere", slug, flush=True)

    for slug in ("luna", "terra", "pluto"):
        print("icosphere", slug, flush=True)
        render_icosphere(slug, 1000, 1000, clay=False, exposure=exposures[slug]).save(
            OUT / f"icosphere_{slug}.png"
        )
        render_icosphere(slug, 900, 900, clay=True).save(
            OUT / f"icosphere_{slug}_clay.png"
        )
        print("ok icosphere", slug, flush=True)

    if not args.quick:
        collection(cards, OUT / "collection.png")
        print("wrote", OUT / "collection.png")
        for slug in ("luna", "terra", "pluto"):
            print("assembly", slug, flush=True)
            render_assembly(slug, False, 1000, 1000, 300, exposure=exposures[slug]).save(
                OUT / f"assembly_{slug}.png"
            )
        annotate_exploded(render_assembly("luna", True, 1200, 1400, 470)).save(
            OUT / "assembly_exploded.png"
        )
        print("wrote assembly previews")


if __name__ == "__main__":
    main()
