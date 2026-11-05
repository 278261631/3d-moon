# Data attribution

This project combines public-domain NASA/NOAA data with one CC-BY 4.0 texture.
If you redistribute the models or previews, keep the following credits.

## Moon

- Colour: NASA Scientific Visualization Studio, **CGI Moon Kit**, LROC WAC natural colour.
  <https://svs.gsfc.nasa.gov/4720/>
- Elevation: LOLA global DEM (4 pixels/degree), from the same CGI Moon Kit.
- Credit: NASA / GSFC / Arizona State University / LROC / LOLA.

## Earth

- Colour: NASA **Blue Marble 2002**, accessed via Wikimedia Commons.
  <https://commons.wikimedia.org/wiki/File:Blue_Marble_2002.png>
- Elevation: NOAA **ETOPO1** 1 arc-minute Ice Surface, accessed through ERDDAP
  `etopo180` and downsampled to 0.25 degrees.
  <https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.html>
- Credit: NASA Earth Observatory / NOAA NCEI.

## Pluto

- Colour: **Solar System Scope** Solar Textures, Pluto map (5926 x 2963),
  obtained through the PyVista `vtk-data` mirror.
  <https://www.solarsystemscope.com/textures/>
  <https://github.com/pyvista/vtk-data>
- The texture pack is licensed **CC-BY 4.0**; credit "Solar System Scope".
- Underlying imagery: NASA New Horizons LORRI/MVIC. The south polar region was
  not imaged during the flyby. For the 3D model, the south of -25 deg is
  completed by mirroring the last real image rows; it is a same-source mirror
  fill, not an observation.
- Reference product: USGS Astrogeology, Pluto New Horizons LORRI-MVIC Global
  Mosaic/DEM 300 m.
  <https://astrogeology.usgs.gov/search/map/pluto_new_horizons_lorri_mvic_global_mosaic_300m>

## Tools

- Mesh generation: Python + NumPy/SciPy; optional validation with FreeCAD Mesh workbench.
- Preview rendering: custom NumPy/Numba renderer in `scripts/render_preview.py`.
- No third-party viewer code is bundled in this repository.
