"""Three-panel clay preview of the cut parts (Luna)."""
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_preview import render_arrays

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "export"
OUT = ROOT / "preview"


def read_stl(path: Path):
    with path.open("rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        d = np.fromfile(f, dtype=dt, count=n)
    v = d["v"].reshape(-1, 3).astype(np.float64)
    f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(fn, axis=1)
    ln[ln == 0] = 1.0
    fn = (fn / ln[:, None]).astype(np.float32)
    vn = np.repeat(fn, 3, axis=0)
    uv = np.zeros((len(v), 2), dtype=np.float32)
    region = np.zeros(len(f), dtype=np.int8)
    return v.astype(np.float32), f, uv, vn, fn, region


def font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", size)
    except OSError:
        return ImageFont.load_default()


def main():
    specs = [
        ("top", "上盖 TOP", "z 90…105", 300, -30),
        ("middle", "中段 MIDDLE", "z 0…90", 430, 45),
        ("bottom", "下盖 BOTTOM", "z −15…0", 300, 35),
    ]
    tex = np.ones((1, 1, 3), np.float32)
    panels = []
    for name, label, ztext, dist, el in specs:
        v, f, uv, vn, fn, region = read_stl(EXPORT / f"luna_part_{name}.stl")
        im = render_arrays(v, f, uv, vn, fn, region, tex, 66, el, dist, 560, 560, True).convert("RGB")
        panels.append((im, label, ztext))

    canvas = Image.new("RGB", (1780, 760), (14, 16, 20))
    d = ImageDraw.Draw(canvas)
    d.text((28, 20), "LUNA / 3-PART CUT", font=font(36, True), fill=(242, 244, 248))
    d.text((28, 64), "同一个 geodesic 球，在 z=0 / z=90 各切一刀；中段带 3.2 mm 实心底", font=font(22), fill=(175, 190, 210))
    x = 20
    for im, label, ztext in panels:
        canvas.paste(im, (x, 110))
        d.text((x + 12, 680), label, font=font(25, True), fill=(240, 242, 246))
        d.text((x + 12, 712), ztext, font=font(18), fill=(150, 164, 184))
        x += 580
    canvas.save(OUT / "parts_luna.png")
    print(OUT / "parts_luna.png")


if __name__ == "__main__":
    main()
