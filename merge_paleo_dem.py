#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_paleo_dem.py
==================
Fuse a modern topography-bathymetry DEM with an interpreted palaeo-surface
(e.g. the Messinian Erosional Surface, MES) into a single, feathered
palaeo-DEM for perspective rendering.

Reproduces the core trick of Madof et al. (Nat. Geosci. 2026, Fig. 4): take
the modern topo-bathy as the regional canvas and warp/blend the interpreted
palaeo-surface in where you have it, with a feathered (distance-weighted) seam.

Handles the common real-world case where the modern DEM is GEOGRAPHIC
(e.g. GEBCO in degrees) and the MES is PROJECTED (metres): the modern DEM is
reprojected into a metric CRS so that FEATHER_M and the Blender scale are
correct.

Outputs (in OUT_DIR):
  merged_paleo_dem.tif  georeferenced float32 DEM   (QGIS / BlenderGIS)
  heightmap_16bit.png   normalised 16-bit heightmap (Blender displace)
  heightmap_meta.json   real-world z/extent mapping  (read by Blender script)
  hillshade_qc.png      quick hillshade for visual QC

Sign convention internally: ELEVATION in metres, sea negative, land positive.
Deps: rasterio numpy scipy pillow matplotlib pandas imageio tqdm
"""

import os
import json
import numpy as np

# =====================================================================
# CONFIG  --  edit this block, then run:  python merge_paleo_dem.py
# =====================================================================

# ---- MODERN DEM (the regional canvas; defines output grid & CRS) ----
MODERN_DEM_PATH   = "data/gebco_topobathy.tif"
MODERN_DEM_FORMAT = "geotiff"        # "geotiff" | "xyz"
MODERN_DEM_CRS    = None             # for xyz, OR fallback if a GeoTIFF has no embedded CRS
MODERN_XYZ_COLS   = (0, 1, 2)        # x,y,z column indices for xyz/csv
MODERN_XYZ_DELIM  = None             # None = whitespace, "," = CSV
MODERN_IS_DEPTH   = False            # GEBCO is elevation -> leave False

# ---- PALAEO SURFACE (e.g. MES; patched in where it exists) ----
MES_PATH    = "data/mes_grid_20m.tif"
MES_FORMAT  = "geotiff"              # "geotiff" | "xyz"
MES_CRS     = "EPSG:32633"           # MES GeoTIFF carries no embedded CRS -> supplied here (UTM33N)
MES_XYZ_COLS  = (0, 1, 2)
MES_XYZ_DELIM = None                 # None = whitespace, "," = CSV
MES_IS_DEPTH  = True                 # True: depth (positive down) -> elevation = -value  (check hillshade!)

# ---- RENDER MODE ----
# False: blend MES into the modern DEM (regional/context shot, GEBCO as canvas).
# True : render the MES grid ALONE (no GEBCO, no seam) - best for the canyon hero shot.
MES_ONLY = False

# ---- TARGET GRID ----
TARGET_CRS = None                    # None = auto (picks EPSG:32633 from MES_CRS, since GEBCO is geographic)
TARGET_RES = None                    # None = auto (~450 m from GEBCO); or metres (e.g. 100) for a canyon crop
CROP_BBOX  = (380000, 3980000, 660000, 4180000)                 # None = full extent; or (xmin,ymin,xmax,ymax) in TARGET CRS (metres!)

# ---- SEAM FEATHER ----
FEATHER_M = 100000.0                   # width (m) of the blend ramp from modern (MES edge) to MES (inside)

# ---- BACKGROUND WARP (alternative to the feather; for wide / context shots) ----
# True: warp the modern DEM toward the MES at the seam and decay that offset
# outward over WARP_LENGTH_M, so the canyon blends into a smoothly adjusted
# background with NO hard rectangle. The MES interior stays at full detail.
# NOTE: the warp is isotropic, so it also pulls the *landward* DEM toward MES
# depths over the decay length -> shorten WARP_LENGTH_M if you don't want that.
WARP_BACKGROUND_TO_MES = False
WARP_LENGTH_M = 50000.0              # offset decay length (m) into the GEBCO region
WARP_SMOOTH_M = 4000.0               # Gaussian smoothing of the offset field (m); removes facets

# ---- OPTIONAL FIRST-ORDER PALAEO CORRECTION (loading subsidence) ----
# Raises the deep landing zone back toward its 5.33 Ma elevation (subsidence
# sank the basin, so we ADD uplift to deep cells).
APPLY_SUBSIDENCE      = True
SUBSIDENCE_M          = 200.0        # max uplift applied in the deepest part (m)
SUBSIDENCE_ELEV_TOP   = -1000.0      # above this elevation: no correction
SUBSIDENCE_ELEV_FULL  = -2300.0      # below this elevation: full correction (smooth ramp)

# ---- PALAEO SEA LEVEL (metadata for the render water plane) ----
PALEO_SEA_LEVEL_M = -2000.0          # desiccated lowstand relative to present sea level

# ---- OUTPUT ----
OUT_DIR           = "output"
OUT_GEOTIFF       = "merged_paleo_dem.tif"
OUT_HEIGHTMAP_PNG = "heightmap_16bit.png"
OUT_META_JSON     = "heightmap_meta.json"
OUT_HILLSHADE     = "hillshade_qc.png"
OUT_COLORMAP_PNG  = "colormap.png"
VE_HINT           = 3.0              # suggested vertical exaggeration for the render (written to meta)

# ---- BAKED COLOUR (recommended): a hypsometric tint x hillshade PNG that Blender
# uses directly as the surface colour (shadeless), avoiding ColorRamp/lighting issues.
WRITE_COLORMAP  = True
COLORMAP_SCHEME = "desiccated"       # "desiccated" (dry drylands) | "bathymetric" (blue sea, green land)

# ---- FLOOD along the canyon (hero shot): the megaflood as a real RIVER network
# (D8 flow accumulation along the thalweg) plus the flooded basin, baked into the
# colormap. The river is dendritic: tributaries converge and run down to the basin.
# Flow accumulation uses numba if available (Anaconda ships it); without numba it
# falls back to a simpler concave-channel flood.
FLOOD_ALONG_CANYON     = True
FLOOD_MIN_DRAINAGE_KM2 = 6.0         # river = cells draining > this area; smaller -> more/finer tributaries
FLOOD_BASIN_LEVEL      = -2200.0     # also flood everything below this (the basin the flood pours into)
FLOOD_WATER_RGB        = (0.10, 0.34, 0.55)
FLOOD_RIVERS_IN_MES_ONLY = True      # in blend mode: draw rivers ONLY inside the MES coverage (not across GEBCO)
# fallback params (only used if numba is unavailable):
FLOOD_CONCAVITY_M      = 35.0
FLOOD_SMOOTH_M         = 1500.0

# ---- PERFORMANCE ----
NUM_THREADS = 1                      # 0 = all cores; parallelises the GDAL warp/resample

# =====================================================================
# implementation
# =====================================================================
 
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import (reproject, Resampling, transform as warp_transform,
                           calculate_default_transform)
from rasterio.windows import from_bounds as window_from_bounds, Window, transform as win_transform
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter
 
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                     # no-op decorator fallback
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco
 
try:
    from skimage.morphology import reconstruction as _sk_reconstruction
    _HAVE_SKIMAGE = True
except Exception:
    _HAVE_SKIMAGE = False
 
 
def _num_threads():
    return NUM_THREADS if (NUM_THREADS and NUM_THREADS > 0) else (os.cpu_count() or 1)
 
 
# ---- flow-accumulation river (D8) ----
def _fill_depressions(dem):
    """Fill pits so flow routes to the domain edge (greyscale reconstruction by erosion)."""
    seed = dem.copy()
    seed[1:-1, 1:-1] = dem.max()
    return _sk_reconstruction(seed, dem, method="erosion").astype("float32")
 
 
@njit(cache=True)
def _d8_accum(filled, ri, ci):
    nr, nc = filled.shape
    accum = np.ones((nr, nc), dtype=np.float64)
    di = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dj = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    dd = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])
    for k in range(ri.size):
        i = ri[k]; j = ci[k]
        z = filled[i, j]
        best = 0.0; bi = -1; bj = -1
        for m in range(8):
            ii = i + di[m]; jj = j + dj[m]
            if 0 <= ii < nr and 0 <= jj < nc:
                slope = (z - filled[ii, jj]) / dd[m]
                if slope > best:
                    best = slope; bi = ii; bj = jj
        if bi >= 0:
            accum[bi, bj] += accum[i, j]
    return accum
 
 
def river_mask(dem, pixel_m, min_drainage_km2):
    """River = cells whose upstream drainage area exceeds the threshold; trunk wider than tributaries."""
    from scipy.ndimage import binary_dilation
    filled = _fill_depressions(dem)
    order = np.argsort(filled.ravel())[::-1]                  # high -> low
    ri, ci = np.unravel_index(order, filled.shape)
    accum = _d8_accum(filled, ri.astype(np.int64), ci.astype(np.int64))
    area = accum * (pixel_m ** 2) / 1e6                       # upstream drainage area, km^2
    out = area > min_drainage_km2
    for mult, it in [(8, 2), (60, 3), (400, 4)]:              # widen the bigger channels
        big = area > (min_drainage_km2 * mult)
        if big.any():
            out = out | binary_dilation(big, iterations=it)
    return out
 
try:
    from tqdm import tqdm
except Exception:                                     # graceful fallback
    class tqdm:
        def __init__(self, *a, **k): self.n = 0
        def update(self, *a, **k): pass
        def set_description(self, *a, **k): pass
        def write(self, m): print(m)
        def close(self): pass
 
 
def _read_xyz(path, cols, delim, is_depth):
    import pandas as pd
    df = pd.read_csv(path, sep=(delim if delim else r"\s+"), header=None,
                     engine="python", comment="#")
    x = df.iloc[:, cols[0]].to_numpy("float64")
    y = df.iloc[:, cols[1]].to_numpy("float64")
    z = df.iloc[:, cols[2]].to_numpy("float64")
    if is_depth:
        z = -z
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[good], y[good], z[good]
 
 
def _read_band_as_float(ds, is_depth):
    """Read band 1 as float32, NaN nodata. Avoids the int-dtype fill_value crash."""
    arr = ds.read(1).astype("float32")
    nod = ds.nodata
    if nod is not None and np.isfinite(nod):
        arr[arr == np.float32(nod)] = np.nan
    if is_depth:
        arr = -arr
    return arr
 
 
def _reproject_array(arr, src_transform, src_crs, dst_crs, res=None):
    bounds = rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], src_transform)
    kw = dict(resolution=res) if res else {}
    dst_transform, w, h = calculate_default_transform(
        src_crs, dst_crs, arr.shape[1], arr.shape[0], *bounds, **kw)
    dst = np.full((h, w), np.nan, "float32")
    reproject(arr, dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear,
              num_threads=_num_threads())
    return dst, dst_transform
 
 
def resolve_target_crs(pbar):
    """Pick a metric target CRS. If modern DEM is geographic, fall back to the MES CRS."""
    if TARGET_CRS:
        return CRS.from_string(TARGET_CRS)
    if MODERN_DEM_FORMAT == "geotiff":
        with rasterio.open(MODERN_DEM_PATH) as ds:
            mcrs = ds.crs
    else:
        mcrs = CRS.from_string(MODERN_DEM_CRS) if MODERN_DEM_CRS else None
    if mcrs is not None and mcrs.is_geographic:
        proj = None
        if MES_FORMAT == "geotiff":
            with rasterio.open(MES_PATH) as ds:
                mes_crs = ds.crs
            if mes_crs is None and MES_CRS:
                mes_crs = CRS.from_string(MES_CRS)
            if mes_crs is not None and mes_crs.is_projected:
                proj = mes_crs
        elif MES_CRS:
            c = CRS.from_string(MES_CRS)
            proj = c if c.is_projected else None
        if proj is not None:
            pbar.write(f"[merge] modern DEM is geographic -> using metric MES CRS {proj} as target")
            return proj
        pbar.write("[merge] WARNING: target CRS is geographic (degrees). FEATHER_M and the "
                   "Blender scale assume metres. Set TARGET_CRS to a projected CRS (e.g. EPSG:32633).")
    return mcrs
 
 
def load_modern(target_crs, pbar):
    if MODERN_DEM_FORMAT == "geotiff":
        with rasterio.open(MODERN_DEM_PATH) as ds:
            src = _read_band_as_float(ds, MODERN_IS_DEPTH)
            src_crs, src_transform = ds.crs, ds.transform
        if src_crs is None:
            if not MODERN_DEM_CRS:
                raise ValueError("Modern GeoTIFF has no embedded CRS. Set MODERN_DEM_CRS.")
            src_crs = CRS.from_string(MODERN_DEM_CRS)
        same = (target_crs is None) or (src_crs == target_crs)
        if same and TARGET_RES is None:
            return src, src_transform, src_crs
        dst, dst_transform = _reproject_array(src, src_transform, src_crs, target_crs, TARGET_RES)
        return dst, dst_transform, target_crs
 
    elif MODERN_DEM_FORMAT == "xyz":
        if MODERN_DEM_CRS is None or TARGET_RES is None:
            raise ValueError("xyz modern DEM requires MODERN_DEM_CRS and TARGET_RES.")
        x, y, z = _read_xyz(MODERN_DEM_PATH, MODERN_XYZ_COLS, MODERN_XYZ_DELIM, MODERN_IS_DEPTH)
        src_crs = CRS.from_string(MODERN_DEM_CRS)
        ncols = int(np.ceil((x.max() - x.min()) / TARGET_RES))
        nrows = int(np.ceil((y.max() - y.min()) / TARGET_RES))
        transform = from_origin(x.min(), y.max(), TARGET_RES, TARGET_RES)
        gx, gy = _cell_centers(transform, nrows, ncols)
        arr = griddata((x, y), z, (gx, gy), method="linear").astype("float32")
        if target_crs is not None and target_crs != src_crs:
            arr, transform = _reproject_array(arr, transform, src_crs, target_crs, TARGET_RES)
            src_crs = target_crs
        return arr, transform, src_crs
    else:
        raise ValueError(f"Unknown MODERN_DEM_FORMAT: {MODERN_DEM_FORMAT}")
 
 
def _cell_centers(transform, nrows, ncols):
    cols = np.arange(ncols) + 0.5
    rows = np.arange(nrows) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    gx = transform.c + cc * transform.a + rr * transform.b
    gy = transform.f + cc * transform.d + rr * transform.e
    return gx, gy
 
 
def load_mes_on_grid(transform, shape, crs):
    nrows, ncols = shape
    if MES_FORMAT == "geotiff":
        with rasterio.open(MES_PATH) as ds:
            src = _read_band_as_float(ds, MES_IS_DEPTH)
            mes_crs, mes_transform = ds.crs, ds.transform
        if mes_crs is None:
            if not MES_CRS:
                raise ValueError("MES GeoTIFF has no embedded CRS. Set MES_CRS (e.g. 'EPSG:32633').")
            mes_crs = CRS.from_string(MES_CRS)
        dst = np.full(shape, np.nan, "float32")
        reproject(src, dst, src_transform=mes_transform, src_crs=mes_crs,
                  dst_transform=transform, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear,
                  num_threads=_num_threads())
        return dst
    elif MES_FORMAT == "xyz":
        x, y, z = _read_xyz(MES_PATH, MES_XYZ_COLS, MES_XYZ_DELIM, MES_IS_DEPTH)
        src_crs = CRS.from_string(MES_CRS)
        if src_crs != crs:
            xt, yt = warp_transform(src_crs, crs, list(x), list(y))
            x, y = np.asarray(xt), np.asarray(yt)
        gx, gy = _cell_centers(transform, nrows, ncols)
        dst = griddata((x, y), z, (gx, gy), method="linear")
        return dst.astype("float32")
    else:
        raise ValueError(f"Unknown MES_FORMAT: {MES_FORMAT}")
 
 
def fill_nan_nearest(arr):
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]
 
 
def feather_blend(modern, mes, pixel_m, pbar):
    coverage = np.isfinite(mes)
    if not coverage.any():
        pbar.write("[merge] WARNING: MES does not overlap the modern grid (0% coverage) - "
                   "check CRS/extent/tile. Returning modern DEM only.")
        return modern.copy(), coverage
    px = max(FEATHER_M / pixel_m, 1e-6)
    inside_dist = distance_transform_edt(coverage)
    w = np.clip(inside_dist / px, 0.0, 1.0)
    w[~coverage] = 0.0
    mes_filled = np.where(coverage, mes, modern)
    merged = w * mes_filled + (1.0 - w) * modern
    return merged.astype("float32"), coverage
 
 
def warp_blend(modern, mes, pixel_m, pbar):
    """Warp the modern DEM toward the MES at the seam, decaying the offset outward.
 
    merged = MES                              inside coverage   (full detail kept)
           = modern + offset * decay(dist)    outside coverage
    where offset == (MES - modern) at the nearest seam point and decay falls from
    1 at the boundary to 0 at WARP_LENGTH_M (smoothstep). At the boundary this is
    exactly MES, so there is no step; far out it is pure modern.
    """
    coverage = np.isfinite(mes)
    if not coverage.any():
        pbar.write("[merge] WARNING: MES does not overlap the modern grid (0% coverage). "
                   "Returning modern DEM only.")
        return modern.copy(), coverage
 
    # offset where the MES exists, then nearest-extrapolate it to the whole grid
    delta = np.where(coverage, mes - modern, np.nan).astype("float32")
    idx = distance_transform_edt(~coverage, return_distances=False, return_indices=True)
    delta_ext = delta[tuple(idx)]                      # nearest seam offset everywhere
 
    # smooth the offset field to remove nearest-neighbour facets
    sigma_px = max(WARP_SMOOTH_M / pixel_m, 0.0)
    if sigma_px > 0.3:
        delta_ext = gaussian_filter(delta_ext, sigma_px)
 
    # decay outward from the seam over WARP_LENGTH_M (smoothstep: 1 -> 0)
    dist_out = distance_transform_edt(~coverage) * pixel_m
    t = np.clip(dist_out / max(WARP_LENGTH_M, 1e-6), 0.0, 1.0)
    decay = 1.0 - (3.0 * t**2 - 2.0 * t**3)
 
    merged = np.where(coverage, mes, modern + delta_ext * decay)
    pbar.write(f"[merge] warp: seam offset median {np.nanmedian(np.abs(delta[coverage])):.0f} m, "
               f"decay length {WARP_LENGTH_M/1000:.0f} km")
    return merged.astype("float32"), coverage
 
 
def apply_subsidence(elev):
    if not APPLY_SUBSIDENCE:
        return elev
    denom = (SUBSIDENCE_ELEV_TOP - SUBSIDENCE_ELEV_FULL)
    ramp = np.clip((SUBSIDENCE_ELEV_TOP - elev) / denom, 0.0, 1.0)
    return (elev + SUBSIDENCE_M * ramp).astype("float32")
 
 
def write_geotiff(path, arr, transform, crs):
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                   compress="deflate")
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
 
 
def write_heightmap_png(path, arr):
    finite = arr[np.isfinite(arr)]
    zmin, zmax = float(finite.min()), float(finite.max())
    rng = (zmax - zmin) if (zmax > zmin) else 1.0
    u16 = np.clip((arr - zmin) / rng * 65535.0, 0, 65535).astype(np.uint16)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, u16)
    except Exception:
        from PIL import Image
        Image.fromarray(u16.astype(np.int32), mode="I").save(path)
    return zmin, zmax
 
 
def write_hillshade(path, arr, pixel_m, ve=2.0, azdeg=315, altdeg=45):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    rgb = ls.shade(arr, cmap=plt.cm.gist_earth, blend_mode="soft",
                   vert_exag=ve, dx=pixel_m, dy=pixel_m)
    h, w = arr.shape
    fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(rgb)
    fig.savefig(path, dpi=150); plt.close(fig)
 
 
# hypsometric stops (absolute elevation m -> RGB); MUST match render_waterfall.py
_COLORMAP_STOPS = {
    "desiccated": [(-4000, (0.28, 0.24, 0.22)), (-2000, (0.42, 0.34, 0.27)),
                   (-600, (0.60, 0.49, 0.35)), (0, (0.72, 0.60, 0.42)),
                   (700, (0.50, 0.46, 0.30)), (1700, (0.40, 0.31, 0.22)), (3000, (0.93, 0.92, 0.92))],
    "bathymetric": [(-4000, (0.04, 0.10, 0.30)), (-2000, (0.07, 0.22, 0.48)),
                    (-400, (0.20, 0.55, 0.62)), (0, (0.78, 0.70, 0.48)),
                    (600, (0.36, 0.52, 0.26)), (1600, (0.45, 0.36, 0.24)), (3000, (0.95, 0.95, 0.96))],
}
 
 
def write_colormap(path, arr, pixel_m, ve, scheme, coverage=None,
                   azdeg=315, altdeg=45, shade_lo=0.45):
    """Bake hypsometric tint x hillshade (+ optional flood) into an 8-bit RGB PNG.
 
    coverage (optional bool mask): where the MES exists. If FLOOD_RIVERS_IN_MES_ONLY,
    rivers are drawn only inside it (the GEBCO context stays river-free), while flow
    accumulation is still computed on the full grid so canyon rivers get their full
    upstream catchment.
    """
    from matplotlib.colors import LightSource
    stops = _COLORMAP_STOPS.get(scheme, _COLORMAP_STOPS["desiccated"])
    es = np.array([s[0] for s in stops], dtype="float64")
    cols = np.array([s[1] for s in stops], dtype="float64")
    tint = np.dstack([np.interp(arr, es, cols[:, k]) for k in range(3)])
    hs = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(
        arr, vert_exag=ve, dx=pixel_m, dy=pixel_m)[..., None]
    rgb = tint * (shade_lo + (1.0 - shade_lo) * hs)
 
    if FLOOD_ALONG_CANYON:
        if _HAVE_NUMBA and _HAVE_SKIMAGE:
            channel = river_mask(arr, pixel_m, FLOOD_MIN_DRAINAGE_KM2)      # dendritic D8 river
        else:                                                              # fallback: concave channels
            from scipy.ndimage import binary_closing, label
            sm = gaussian_filter(arr, FLOOD_SMOOTH_M / pixel_m)
            channel = (sm - arr) > FLOOD_CONCAVITY_M
            channel = binary_closing(channel, iterations=2)
            lab, n = label(channel)
            if n:
                sizes = np.bincount(lab.ravel()); sizes[0] = 0
                channel = (sizes >= (2e6 / pixel_m ** 2))[lab]
        if coverage is not None and FLOOD_RIVERS_IN_MES_ONLY:
            channel = channel & coverage                                   # rivers only in the MES part
        water = channel | (arr < FLOOD_BASIN_LEVEL)
        wrgb = np.array(FLOOD_WATER_RGB).reshape(1, 1, 3) * (0.55 + 0.45 * hs)
        rgb = np.where(water[..., None], wrgb, rgb)
 
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, u8)
    except Exception:
        from PIL import Image
        Image.fromarray(u8).save(path)
 
 
def load_mes_standalone(target_crs, pbar):
    """MES_ONLY mode: load the MES grid as the surface itself (no modern DEM, no blend)."""
    res = TARGET_RES if TARGET_RES else 25.0
    if TARGET_RES is None:
        pbar.write("[merge] MES_ONLY: TARGET_RES not set -> defaulting to 25 m "
                   "(native 5 m is too large for Blender).")
    if MES_FORMAT == "geotiff":
        with rasterio.open(MES_PATH) as ds:
            src = _read_band_as_float(ds, MES_IS_DEPTH)        # masks nodata, negates depth
            mes_crs, mes_transform = ds.crs, ds.transform
        if mes_crs is None:
            if not MES_CRS:
                raise ValueError("MES GeoTIFF has no embedded CRS. Set MES_CRS (e.g. 'EPSG:32633').")
            mes_crs = CRS.from_string(MES_CRS)
        dst_crs = target_crs or mes_crs
        arr, transform = _reproject_array(src, mes_transform, mes_crs, dst_crs, res)
        return arr, transform, dst_crs
    elif MES_FORMAT == "xyz":
        if MES_CRS is None:
            raise ValueError("xyz MES requires MES_CRS.")
        x, y, z = _read_xyz(MES_PATH, MES_XYZ_COLS, MES_XYZ_DELIM, MES_IS_DEPTH)
        src_crs = CRS.from_string(MES_CRS)
        ncols = int(np.ceil((x.max() - x.min()) / res))
        nrows = int(np.ceil((y.max() - y.min()) / res))
        transform = from_origin(x.min(), y.max(), res, res)
        gx, gy = _cell_centers(transform, nrows, ncols)
        arr = griddata((x, y), z, (gx, gy), method="linear").astype("float32")
        dst_crs = target_crs or src_crs
        if dst_crs != src_crs:
            arr, transform = _reproject_array(arr, transform, src_crs, dst_crs, res)
        return arr, transform, dst_crs
    else:
        raise ValueError(f"Unknown MES_FORMAT: {MES_FORMAT}")
 
 
def _crop(arr, transform):
    if CROP_BBOX is None:
        return arr, transform
    win = window_from_bounds(*CROP_BBOX, transform=transform)
    r0, r1 = max(int(round(win.row_off)), 0), min(int(round(win.row_off + win.height)), arr.shape[0])
    c0, c1 = max(int(round(win.col_off)), 0), min(int(round(win.col_off + win.width)), arr.shape[1])
    arr = arr[r0:r1, c0:c1]
    return arr, win_transform(Window(c0, r0, c1 - c0, r1 - r0), transform)
 
 
def finalize(paleo, transform, crs, pbar, coverage=None):
    """Write GeoTIFF + 16-bit heightmap + metadata + QC hillshade."""
    pixel_m = float(abs(transform.a))
 
    pbar.set_description("write GeoTIFF")
    write_geotiff(os.path.join(OUT_DIR, OUT_GEOTIFF), paleo, transform, crs); pbar.update(1)
 
    pbar.set_description("write heightmap")
    zmin, zmax = write_heightmap_png(os.path.join(OUT_DIR, OUT_HEIGHTMAP_PNG), paleo); pbar.update(1)
 
    if WRITE_COLORMAP:
        write_colormap(os.path.join(OUT_DIR, OUT_COLORMAP_PNG), paleo, pixel_m,
                       VE_HINT, COLORMAP_SCHEME, coverage=coverage)
 
    pbar.set_description("write metadata")
    meta = dict(zmin_m=zmin, zmax_m=zmax,
                width_m=paleo.shape[1] * pixel_m, height_m=paleo.shape[0] * float(abs(transform.e)),
                ncols=int(paleo.shape[1]), nrows=int(paleo.shape[0]), pixel_m=pixel_m,
                paleo_sea_level_m=float(PALEO_SEA_LEVEL_M), ve_hint=float(VE_HINT),
                crs=str(crs), heightmap_png=OUT_HEIGHTMAP_PNG)
    if WRITE_COLORMAP:
        meta["colormap_png"] = OUT_COLORMAP_PNG
    with open(os.path.join(OUT_DIR, OUT_META_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    pbar.update(1)
 
    pbar.set_description("write hillshade")
    write_hillshade(os.path.join(OUT_DIR, OUT_HILLSHADE), paleo, pixel_m, ve=VE_HINT); pbar.update(1)
    pbar.close()
 
    print(f"\n[merge] done. {paleo.shape[1]}x{paleo.shape[0]} px @ {pixel_m:.0f} m | "
          f"z {zmin:.0f}..{zmax:.0f} m | palaeo sea level {PALEO_SEA_LEVEL_M:.0f} m | VE {VE_HINT}")
    print(f"[merge] -> check {os.path.join(OUT_DIR, OUT_HILLSHADE)} (canyon clear? seam OK?)")
    print(f"[merge] -> then load in Blender via render_waterfall.py")
 
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
 
    # -------------------- MES-only (canyon hero shot) --------------------
    if MES_ONLY:
        pbar = tqdm(total=6, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        pbar.set_description("load MES (standalone)")
        target_crs = CRS.from_string(TARGET_CRS) if TARGET_CRS else None
        paleo, transform, crs = load_mes_standalone(target_crs, pbar)
        paleo, transform = _crop(paleo, transform)
        pbar.write(f"[merge] MES-only grid: {paleo.shape[1]}x{paleo.shape[0]} px @ "
                   f"{abs(transform.a):.0f} m  CRS={crs}")
        pbar.update(1)
        pbar.set_description("fill nodata"); paleo = fill_nan_nearest(paleo); pbar.update(1)
        if APPLY_SUBSIDENCE:
            pbar.write(f"[merge] subsidence correction (max {SUBSIDENCE_M:.0f} m)")
        paleo = apply_subsidence(paleo)
        finalize(paleo, transform, crs, pbar)
        return
 
    # -------------------- modern + MES blend (context shot) --------------
    stages = ["resolve CRS", "load modern DEM", "regrid MES", "fill nodata",
              "feather-blend", "subsidence", "write GeoTIFF", "write heightmap",
              "write metadata", "write hillshade"]
    pbar = tqdm(total=len(stages), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
 
    pbar.set_description("resolve CRS")
    target_crs = resolve_target_crs(pbar); pbar.update(1)
 
    pbar.set_description("load modern DEM")
    modern, transform, crs = load_modern(target_crs, pbar)
    modern, transform = _crop(modern, transform)
    pixel_m = float(abs(transform.a))
    pbar.write(f"[merge] grid: {modern.shape[1]}x{modern.shape[0]} px @ {pixel_m:.1f} m  CRS={crs}")
    pbar.update(1)
 
    pbar.set_description("regrid MES")
    mes = load_mes_on_grid(transform, modern.shape, crs)
    pbar.write(f"[merge] MES coverage of grid: {np.isfinite(mes).mean()*100:.1f} %")
    pbar.update(1)
 
    pbar.set_description("fill nodata"); modern = fill_nan_nearest(modern); pbar.update(1)
    pbar.set_description("warp-blend" if WARP_BACKGROUND_TO_MES else "feather-blend")
    if WARP_BACKGROUND_TO_MES:
        merged, coverage = warp_blend(modern, mes, pixel_m, pbar)
    else:
        merged, coverage = feather_blend(modern, mes, pixel_m, pbar)
    pbar.update(1)
    pbar.set_description("subsidence")
    paleo = fill_nan_nearest(apply_subsidence(merged)); pbar.update(1)
 
    finalize(paleo, transform, crs, pbar, coverage=coverage)
 
 
if __name__ == "__main__":
    main()
 