"""Build a two-column provenance sheet: real colour image + derived relief."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "assets/processed"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def font(size: int, bold=False):
    path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def height_image(name: str) -> Image.Image:
    a = np.load(PROC / name).astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    g = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
    return Image.fromarray((g * 255).astype(np.uint8), "L").convert("RGB")


def tile(img: Image.Image, title: str, sub: str, accent: tuple[int, int, int]) -> Image.Image:
    w, h = 720, 360
    out = Image.new("RGB", (w, h + 66), (18, 20, 25))
    out.paste(img.resize((w, h), Image.LANCZOS), (0, 66))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, 6, h + 66], fill=accent)
    d.text((18, 10), title, font=font(25, True), fill=(238, 240, 244))
    d.text((18, 41), sub, font=font(16), fill=(150, 160, 175))
    return out


def main():
    rows = [
        (
            "月球  MOON",
            "LROC WAC 真实彩色影像",
            PROC / "moon_color.png",
            "LOLA 真实高程",
            "moon_height.npy",
            (222, 178, 92),
        ),
        (
            "地球  EARTH",
            "NASA Blue Marble 2002",
            PROC / "earth_color.png",
            "NOAA ETOPO1 真实高程",
            "earth_height.npy",
            (74, 150, 210),
        ),
        (
            "冥王星  PLUTO",
            "New Horizons 真实彩色影像（+90…−25°）",
            PROC / "pluto_color.png",
            "影像/地形派生浮雕（真实覆盖区）",
            "pluto_height.npy",
            (196, 132, 108),
        ),
    ]
    tiles = []
    for name, csub, cpath, hsub, hname, acc in rows:
        c = Image.open(cpath).convert("RGB")
        h = height_image(hname)
        if "PLUTO" in name:
            # The model uses a mirrored south fill; show only the real
            # New Horizons coverage in the provenance sheet.
            keep = int(round(c.height * (90.0 + 25.0) / 180.0))
            c = c.crop((0, 0, c.width, keep))
            h = h.crop((0, 0, h.width, keep))
        tiles.append(tile(c, f"{name} · 颜色", csub, acc))
        tiles.append(tile(h, f"{name} · 高度场", hsub, acc))

    gap = 18
    w = 2 * 720 + 3 * gap + 60
    h = 130 + 3 * (360 + 66) + 2 * gap + 76
    sheet = Image.new("RGB", (w, h), (11, 12, 15))
    d = ImageDraw.Draw(sheet)
    d.text((gap + 24, 26), "REAL PLANETARY IMAGERY → CARVED SURFACE", font=font(42, True), fill=(244, 246, 250))
    d.text(
        (gap + 26, 82),
        "左：真实影像颜色（仅用于预览/贴图）    右：由真实影像或高程数据生成的可打印浮雕",
        font=font(21),
        fill=(160, 172, 190),
    )
    for i, t in enumerate(tiles):
        r, c = divmod(i, 2)
        x = gap + 12 + c * (720 + gap)
        y = 130 + r * (360 + 66 + gap)
        sheet.paste(t, (x, y))
    d.text(
        (gap + 24, h - 52),
        "NASA SVS / LROC / LOLA · NASA Blue Marble · NOAA ETOPO1 · NASA New Horizons / USGS · Pluto colour map via Solar System Scope (CC-BY 4.0) · Pluto south mirror-filled in the model, not shown here",
        font=font(15),
        fill=(105, 115, 132),
    )
    out = DOCS / "source_map.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
