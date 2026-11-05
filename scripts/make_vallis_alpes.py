"""Mark the annotated lunar feature: Vallis Alpes (the Alpine Valley)."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "assets/processed"
OUT = ROOT / "docs/vallis_alpes.png"


def font(size: int, bold=False):
    path = "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def main():
    color = Image.open(PROC / "moon_color.png").convert("RGB")
    h = np.load(PROC / "moon_height.npy").astype(np.float32)
    lo, hi = np.percentile(h, [1, 99])
    g = np.clip((h - lo) / (hi - lo), 0, 1)
    relief = Image.fromarray((g * 255).astype(np.uint8), "L").convert("RGB")
    # Crop around 35..62N, 14W..20E; annotation point is 46.6N, 3.9E.
    W, H = color.size
    x0, x1 = int((0.46) * W), int((0.56) * W)
    y0, y1 = int((0.15) * H), int((0.33) * H)
    color_c = color.crop((x0, y0, x1, y1)).resize((720, 520), Image.LANCZOS)
    relief_c = relief.crop((x0, y0, x1, y1)).resize((720, 520), Image.LANCZOS)

    # point pixel in the crop
    u = 0.5108
    v = (90.0 - 46.6) / 180.0
    px = (u * W - x0) / (x1 - x0) * 720
    py = (v * H - y0) / (y1 - y0) * 520

    out = Image.new("RGB", (1520, 680), (14, 16, 20))
    out.paste(color_c, (20, 108))
    out.paste(relief_c, (780, 108))
    d = ImageDraw.Draw(out)
    d.text((24, 18), "THE 'GROOVE' IS REAL LUNAR TERRAIN", font=font(34, True), fill=(242, 244, 248))
    d.text((24, 64), "Vallis Alpes / 阿尔卑斯月谷南端附近，不是接缝/建模凹槽", font=font(24, True), fill=(180, 194, 214))
    d.text((24, 96), "LROC WAC colour (left)  ·  LOLA relief (right)  ·  annotation point 46.6°N, 3.9°E", font=font(15), fill=(125, 138, 156))
    for ox in (20, 780):
        d.ellipse([ox + px - 13, 108 + py - 13, ox + px + 13, 108 + py + 13], outline=(255, 80, 55), width=4)
        d.line([ox + px + 15, 108 + py - 15, ox + px + 110, 108 + py - 75], fill=(255, 80, 55), width=2)
    d.text((20 + px + 115, 108 + py - 92), "Vallis Alpes", font=font(22, True), fill=(255, 120, 95))
    d.text((20 + px + 115, 108 + py - 64), "阿尔卑斯月谷（南端附近）", font=font(18), fill=(255, 170, 150))
    out.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
