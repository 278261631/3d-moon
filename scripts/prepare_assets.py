"""Normalise the raw planetary data into 2:1 north-up textures and
millimetre-scale relief fields used by the pen-cup builder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "assets/raw"
OUT = ROOT / "assets/processed"
OUT.mkdir(parents=True, exist_ok=True)
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

SIZE = (2048, 1024)


def save_rgb(a: np.ndarray, name: str) -> None:
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    Image.fromarray(a, "RGB").resize(SIZE, Image.LANCZOS).save(OUT / name)


def save_height(h: np.ndarray, name: str) -> None:
    h = np.array(Image.fromarray(h.astype(np.float32), mode="F").resize(SIZE, Image.BICUBIC))
    np.save(OUT / name, h)
    lo, hi = np.percentile(h, [1, 99])
    g = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
    Image.fromarray((g * 255).astype(np.uint8)).resize(SIZE, Image.LANCZOS).save(
        DOCS / name.replace(".npy", "_preview.png")
    )


def norm01(a: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if lo is None or hi is None:
        lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def prep_moon() -> None:
    color = np.array(Image.open(RAW / "moon_color_2k.tif").convert("RGB"))
    dem = np.array(Image.open(RAW / "moon_dem_1440.tif"), dtype=np.float32)  # km
    assert dem.shape == (720, 1440)
    # LOLA DEM is north-up, 0.25 deg/px, centred on 0 longitude.
    save_rgb(color, "moon_color.png")
    t = norm01(dem, -6.5, 7.0)
    relief = 0.22 + 1.55 * t**1.05
    relief = ndimage.gaussian_filter(relief, 0.9)
    save_height(relief, "moon_height.npy")
    print("moon relief", relief.min(), relief.max(), relief.shape)


def prep_earth() -> None:
    color = np.array(Image.open(RAW / "earth_blue_marble_1920.png").convert("RGB"))
    save_rgb(color, "earth_color.png")
    df = pd.read_csv(RAW / "earth_etopo1_1440.csv", skiprows=[1])
    print(df.shape, df.head(2).to_dict("records"))
    # ERDDAP order: latitude ascending -90..90, longitude ascending -180..180.
    elev = df["altitude"].to_numpy(np.float32).reshape(721, 1441)
    elev = elev[:-1, :-1][::-1, :]  # north-up, 720x1440
    assert elev.shape == (720, 1440)
    # Compress the huge bathymetric range so the cup remains a cup.
    t = np.clip((elev - (-10900.0)) / (8300.0 + 10900.0), 0, 1)
    relief = 0.15 + 2.05 * t**1.35
    relief = ndimage.gaussian_filter(relief, 1.1)
    save_height(relief, "earth_height.npy")
    print("earth relief", relief.min(), relief.max(), relief.shape, "elev", elev.min(), elev.max())


def _fill_black(rgb: np.ndarray) -> np.ndarray:
    """Fill the un-imaged black part of a planetary mosaic by nearest valid
    pixels, so a 3D-printed surface has no blank patch."""
    valid = rgb.max(axis=2) > 8
    if valid.all():
        return rgb
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return rgb[tuple(idx)]


def prep_pluto() -> None:
    # Real New Horizons global colour map (Solar System Scope texture pack,
    # derived from NASA imagery).  New Horizons imaged roughly +90..-25 deg;
    # south of that we complete the 2:1 map by mirroring the last real rows.
    # This keeps the caps textured without pretending the south was observed.
    src = Image.open(RAW / "pluto_color_2k.jpg").convert("RGB")
    lat_min = -25.0
    bottom = int(round(src.height * (90.0 - lat_min) / 180.0))
    crop = np.array(src.crop((0, 0, src.width, bottom)), dtype=np.float32) / 255.0
    crop = _fill_black((crop * 255).astype(np.uint8)).astype(np.float32) / 255.0
    full_h = int(round(crop.shape[0] * 180.0 / (90.0 - lat_min)))
    extra = max(0, full_h - crop.shape[0])
    mirror = crop[-extra:][::-1] if extra else crop[:0]
    full = np.concatenate([crop, mirror], axis=0) if extra else crop
    # A gentle S-curve restores the contrast lost in the browse texture.
    full = np.clip((full - 0.5) * 1.18 + 0.53, 0.0, 1.0)
    save_rgb(full * 255.0, "pluto_color.png")
    c = full
    lum = c @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # Sputnik Planitia (the bright heart) is a low, smooth nitrogen-ice basin;
    # the dark equatorial tholin belt is rough and topographically higher.
    macro = norm01(ndimage.gaussian_filter(lum, 12.0))
    fine = ndimage.gaussian_filter(lum, 0.7) - ndimage.gaussian_filter(lum, 3.5)
    fine = norm01(fine)
    relief = 0.18 + 1.10 * (1.0 - macro) ** 1.10 + 0.55 * fine
    relief = ndimage.gaussian_filter(relief, 0.55)
    save_height(relief, "pluto_height.npy")
    print("pluto relief", relief.min(), relief.max(), relief.shape, "mirror rows", extra)


if __name__ == "__main__":
    prep_moon()
    prep_earth()
    prep_pluto()
