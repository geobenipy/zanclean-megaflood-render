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

# REGIONAL overview (the 280x200 km context shot) - swap the comments to go back:
TARGET_RES = 25
CROP_BBOX  = (380000, 3980000, 660000, 4180000)   # (xmin,ymin,xmax,ymax) target-CRS metres

# CANYON HERO shot: tight 70x60 km box around the waterfall (lip at ~532600/4074000),
# finer grid so the 2.2 km drop carries the frame. Pair with VE 5 in the render.
#TARGET_RES = 12.5
#CROP_BBOX  = (495000, 4040000, 565000, 4100000)

# ---- GENTLE TERRAIN SMOOTHING (softens MES data terraces) ----
# The interpolated MES surface has staircase contour terraces that show as horizontal
# steps on the dry canyon walls. A gentle Gaussian on the FINAL terrain rounds them.
# Keep it SMALL: the terrain is extracted data, so over-smoothing flattens real relief
# and falsifies the model. Applied consistently to the mesh, hillshade AND the flood bed.
# 0 = OFF (raw data, sharp terraces). ~40-60 m = gentle. Raise cautiously if terraces
# still read; drop to 0 to compare against the untouched data.
TERRAIN_SMOOTH_M = 50.0

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
OUT_DIR           = "output_v8"       # regional overview outputs
#OUT_DIR          = "output_hero_v6"  # canyon hero outputs
OUT_GEOTIFF       = "merged_paleo_dem.tif"
OUT_HEIGHTMAP_PNG = "heightmap_16bit.png"
OUT_META_JSON     = "heightmap_meta.json"
OUT_HILLSHADE     = "hillshade_qc.png"
OUT_COLORMAP_PNG  = "colormap.png"
OUT_COLORMAP_DRY_PNG = "colormap_dry.png"       # same but WITHOUT the megaflood painted in:
                                                # terrain colour for the flood_advance animation,
                                                # where the 3D water sheet alone carries the flood
OUT_FLOOD_CORRIDOR_PNG = "flood_corridor.png"   # RGB: R=corridor coverage, G=along-path 0..1, B=foam
OUT_FLOOD_SURFACE_PNG  = "flood_surface_16bit.png"   # 16-bit flood water-surface elevation (Blender flood sheet)
# VE_HINT         = 3.0              # regional overview
VE_HINT           = 5.0              # canyon hero: stronger relief so the 2.2 km drop reads

# ---- BAKED COLOUR (recommended): a hypsometric tint x hillshade PNG that Blender
# uses directly as the surface colour (shadeless), avoiding ColorRamp/lighting issues.
WRITE_COLORMAP  = True
COLORMAP_SCHEME = "desiccated"       # "desiccated" (dry drylands) | "bathymetric" (blue sea, green land)
COLORMAP_HILLSHADE = False            # bake hillshade into colour (map look). False for PHOTOREAL.

# ---- FLOOD along the canyon (hero shot): the megaflood as a real RIVER network
# (D8 flow accumulation along the thalweg) plus the flooded basin, baked into the
# colormap. The river is dendritic: tributaries converge and run down to the basin.
# Flow accumulation uses numba if available (Anaconda ships it); without numba it
# falls back to a simpler concave-channel flood.
FLOOD_ALONG_CANYON     = False       # OFF: the dendritic D8 tributaries painted thin blue
                                     # threads down the canyon walls + side ridges. The
                                     # flood must run ONLY through the megaflood corridor
                                     # (canyon + inflow + basin), so the feeder rivers off.
FLOOD_MIN_DRAINAGE_KM2 = 4.0         # river = cells draining > this area; smaller -> more/finer tributaries
FLOOD_BASIN_LEVEL      = -2200.0     # also flood everything below this (the basin the flood pours into)
FLOOD_WATER_RGB        = (0.10, 0.34, 0.55)
FLOOD_RIVERS_IN_MES_ONLY = True      # in blend mode: draw rivers ONLY inside the MES coverage (not across GEBCO)
# fallback params (only used if numba is unavailable):
FLOOD_CONCAVITY_M      = 35.0
FLOOD_SMOOTH_M         = 1500.0

# ---- ZANCLEAN MEGAFLOOD VISUALISATION ----
# Overlays a wide, turbid flood corridor on top of the D8 rivers to simulate
# the Zanclean megaflood entering the desiccated Mediterranean basin.
# The corridor enters from the western grid edge (Gibraltar direction) and
# narrows toward the canyon.  Restricted to MES coverage in blend mode.
ZANCLEAN_FLOOD          = False     # OFF: the legacy D8-channel "waterfall zone" overlay
                                    # (superseded by MEGAFLOOD_CORRIDOR); it also painted
                                    # stray blue on side slopes. Only the hand-traced
                                    # corridor should carry the flood now. RGB constants
                                    # below are still used by the corridor colours.
ZANCLEAN_ENTRY_ELEV_M   = -500.0    # cells below this elevation in the eastern basin portion seed the corridor
ZANCLEAN_ENTRY_FRACTION = 0.45      # fraction of grid width covered by the entry zone (measured from east/right edge)
ZANCLEAN_CHANNEL_KM     = 12.0      # full width (km) of the flood corridor within the MES
ZANCLEAN_DEEP_NAVY      = (0.04, 0.10, 0.30)   # deep basin (below FLOOD_BASIN_LEVEL)
ZANCLEAN_ATLANTIC_RGB   = (0.09, 0.26, 0.54)   # clean basin water at entry (east edge = Ionian Sea)
ZANCLEAN_TURBID_RGB     = (0.17, 0.23, 0.30)   # muddy blue-grey: sediment TINT, but the
                                               # flood must still read as WATER (full brown
                                               # made it look like a salt flat)
ZANCLEAN_FOAM_RGB       = (0.72, 0.85, 0.92)   # white-blue foam/spray at the cascade lip
ZANCLEAN_CASCADE_SLOPE  = 0.80      # gradient threshold (m/m) for the cascade/foam zone

# ---- ZANCLEAN MEGAFLOOD CORRIDOR (mask-driven, hand-traced route) ----
# The D8 rivers above look like normal rivers because they ARE (flow accumulation).
# The Zanclean flood was the largest flood in Earth history - one colossal torrent,
# not a dendritic network. So instead of guessing the route from the DEM, you TRACE
# it yourself: open colormap.png, read the waypoints of the route you want off the
# image, list them below. The script rasterises a wide, tapering corridor along that
# polyline: clean Atlantic blue at the inlet -> turbid sediment toward the canyon ->
# white foam where it plunges over the escarpment. This is the megaflood "mask".
# It is painted ON TOP of everything else (D8 rivers stay as fine feeder streams;
# set FLOOD_ALONG_CANYON=False if you want ONLY the trunk).
MEGAFLOOD_CORRIDOR = True

# How you give the path:
#   "image" = fractions of colormap.png, (fx, fy) with (0,0)=TOP-LEFT, (1,1)=BOTTOM-
#             RIGHT. Easiest: eyeball the pixel of each bend, divide by width/height.
#   "crs"   = target-CRS metres (x_east, y_north), read precisely off QGIS.
MEGAFLOOD_PATH_UNITS = "crs"    # CRS metres: crop-independent, same route for the
                                # regional view AND the canyon hero crop

# Waypoints ALONG the flood, ordered SOURCE (Gibraltar/inlet) -> PLUNGE. The default
# traces the trunk visible in your render: enters lower-left, runs NE across the dry
# basin, converges at the canyon head, plunges E into the Ionian. RETRACE THESE to
# match the blue route you drew - this is just a starting point.
# Route traced along the ACTUAL deep corridor (probed from the merged DEM): enters
# from the NW (Gibraltar direction along SE Sicily), runs down the -700..-890 m
# trough, crosses the shallow flats hugging their northern side, swings into the
# canyon head and plunges SE into the Ionian. (x_east, y_north) EPSG:32633 metres.
MEGAFLOOD_PATH = [
    (366000, 4096000),  # NW inlet OUTSIDE the grid (start-cap artefacts stay off-map)
    (391200, 4090000),  # enters the frame (~-400 m)
    (408000, 4076000),  # upper trough (~-700 m)
    (427600, 4064000),  # trough axis (~-850 m, deepest)
    (452800, 4060000),  # trough exit onto the flats (~-120 m)
    (475200, 4063000),  # flats, northern side / bay (~-30 m)
    (497600, 4066000),  # flats (~-80 m)
    (514400, 4070000),  # approach (~-150 m)
    (520000, 4072000),  # canyon head / convergence
    (532600, 4074000),  # escarpment lip (the waterfall)
    (538200, 4076100),  # plunge down the canyon axis (ENE, ~-2450 m)
    (545000, 4078000),  # plume into the deep Ionian (~-2690 m)
    (552000, 4080000),  # plume runs out into the basin
]

# Corridor HALF-width in km at each waypoint (list same length as MEGAFLOOD_PATH, OR
# a single number for constant width). This is the RIVER width of the overspill: the
# flood is clipped to this envelope, so the west stays a desiccated basin with one
# broad torrent running through it (not an inland sea). The shoreline INSIDE the
# envelope is topographic (waterline); the envelope edge is feathered. Half-widths,
# so 35 -> a 70 km wide torrent at the inlet, pinching toward the canyon head.
# Per-waypoint half-width (km). THIS is the main knob for WHERE the water is:
#   waypoints 0-4 = far upstream before/along Sicily (NW inlet -> deep trough);
#   waypoints 5-6 = the flats/bay; 7-9 = approach -> canyon head -> lip (the inflow into
#   the canyon); 10-12 = the plunge + plume down the canyon. Lowered the first entries
#   (less water before Sicily) and raised 7-11 (more water pouring into and down the
#   canyon). To push further: shrink 0-4 more, grow 7-11 more.
MEGAFLOOD_HALFWIDTH_KM = [15, 15, 16, 17, 16, 17, 17, 16, 12, 10, 14, 17, 19]

# Soft edge (km) feathered onto the corridor rim so it reads as a torrent, not a
# canal. Generous: within the feather the SHORELINE comes from the terrain (via the
# sinking surface); beyond halfwidth+feather the flood is guaranteed zero.
MEGAFLOOD_FEATHER_KM = 15.0

# Lower the water surface CONTINUOUSLY beyond the corridor rim, by this many metres
# per km of distance outside the half-width. The sinking surface intersects the
# terrain at some organic contour, so there is never a straight mask cut: flats next
# to the corridor fall dry within a few km, deep source basins to the west keep
# flooding for tens of km, and the falling cascade profile cannot drape sideways
# along the escarpment. 0 = hard envelope clip (weight * waterline).
MEGAFLOOD_EDGE_DROP_PER_KM = 120.0  # steep enough that the sinking surface cuts the
                                    # near-flat southern shelf within ~0.5 km - at 20 the
                                    # grazing intersection was tens of km of milky film,
                                    # at 50 the descending ramp still rendered as a
                                    # multi-km glassy levee along the corridor rim

# Whitewater band (km) painted upslope of the basin rim where the flood goes over.
# Only used by the legacy (non-confined) mode. The confined mode keys foam to the
# slope of the WATER SURFACE instead (see MEGAFLOOD_FOAM_SURF_SLOPE).
MEGAFLOOD_FOAM_BAND_KM = 4.0

# Confined-mode foam: whitewater where the water surface itself drops steeply (the
# cascade), ramping to full white at 3x this gradient (m/m). 0.05 = 5% surface drop.
# Must stay above EDGE_DROP_M / (FEATHER_KM*1000) or the corridor rim foams too.
MEGAFLOOD_FOAM_SURF_SLOPE = 0.05
# ... plus where a THIN racing sheet (depth below this, in m) skims a steep bed
# (> ZANCLEAN_CASCADE_SLOPE) - catches the lip itself without foaming deep wet slopes.
MEGAFLOOD_FOAM_THIN_DEPTH_M = 40.0

# Lateral feather (km) of the CASCADE section only: the falling jet is confined to
# halfwidth + this, instead of the generous MEGAFLOOD_FEATHER_KM of the lake/flats -
# otherwise the thin falling film paints a curtain along the whole escarpment face.
MEGAFLOOD_FALLS_FEATHER_KM = 3.6   # slightly wider cascade (was 3.0) -> a fuller
                                   # waterfall in the canyon without curtaining sideways

# Water-surface slope (m/m) above which the envelope hands over from the generous
# lake/flats feather to the tight falls feather. This replaces the old pooledness
# switch: a broad shallow sheet RACING across the flats keeps the full corridor
# width, only genuinely FALLING water (cascade, escarpment drape) is pinched to
# the canyon. Full handover at 3x this slope.
MEGAFLOOD_FALLS_SLOPE = 0.02

# Falling-water streaks: modulate the whitewater with noise smeared ALONG the flow
# direction (line-integral-convolution style). Turns the flat foam sheet into
# directional strands running down the cascade - the "it's actually falling" cue.
MEGAFLOOD_FOAM_STREAKS   = True
MEGAFLOOD_STREAK_LEN_M   = 900.0   # smear length (m); longer = longer strands
MEGAFLOOD_STREAK_MIN     = 0.45    # darkest strand keeps this fraction of the foam
MEGAFLOOD_STREAK_LANE_M  = 260.0   # WIDTH (m) of a current lane: the streak noise is
                                   # pre-smoothed to this scale so the flow texture is
                                   # broad water bands, not 1-px pencil hatching. Bigger
                                   # = smoother/wider lanes; ~200-350 reads as a torrent.

# The flood BODY has to read as a raging torrent, not a calm bay:
# - current lanes: streak-modulate the water colour over the WHOLE flood
# - rapids: streaky whitecaps wherever the water races (non-pooled sections)
# - churning margins: thin foam fringe along the waterline (the flood front)
MEGAFLOOD_BODY_STREAK_GAIN = 0.08  # +-8% colour modulation along flow lanes (0 = off):
                                   # subtle directional current sheen on the flood body
MEGAFLOOD_BODY_STREAK_LEN_M = 3500.0  # lane length (m); longer than the foam strands
MEGAFLOOD_RAPIDS_FOAM      = 0.0   # whitecaps on the flood BODY. 0 = off: the broad lanes
                                   # smeared foam as light-blue BLOBS over the calm shallow
                                   # bay (a pooled lake behind the sill -> should read glassy
                                   # like the sea, not speckled). Whitewater drama now lives
                                   # only at the cascade (foam_surf/foam_thin), where it belongs.
MEGAFLOOD_EDGE_FOAM        = 0.10  # churn-fringe intensity at the waterline (0 = off)
MEGAFLOOD_EDGE_FOAM_DEPTH  = 12.0  # fringe fades out below this water depth (m)
MEGAFLOOD_FOAM_MAX         = 0.72  # cap on visible foam whiteness: the 2 km cascade
                                   # face is steep everywhere -> foam saturated to a
                                   # flat white blob. Capping lets the blue water show
                                   # through, so the plunge reads as blue-white streaked
                                   # whitewater (with the broad foam strands) not paint.

# Plunge mist: foam-rich cells may dip this many metres BELOW the palaeo sea level
# before the coverage cutoff removes them -> a white spray zone where the cascade
# hits the sea instead of a hard cut at the waterline.
MEGAFLOOD_PLUNGE_MIST_M  = 250.0

# --- TOPOGRAPHIC CONFINEMENT (the flood follows the terrain) ---
# Instead of a flat mask laid over hills and valleys alike, the flood is confined by
# a WATER-SURFACE elevation that slopes down the corridor toward the sea. The flood
# only appears where the terrain lies BELOW that surface: it fills valleys, leaves
# ridges/hills dry (islands), and drops where the land drops. This also fixes the 3D
# "blue tarp" look - the water sheet now sits on the water surface, not on the ground.
MEGAFLOOD_CONFINE_TO_TERRAIN = True   # False = old behaviour (overtop everything, flat sheet)

# Water depth (m) above the channel floor along the route (auto profile). With
# MEGAFLOOD_POOLING this is effectively the HEAD above the highest downstream sill:
# the water level in the corridor = sill elevation + this depth. The route sill sits
# at ~+45 m, so 60 m -> overspill level ~ +105 m. Downstream of the sill it is the
# sheet thickness pouring over the escarpment. Raise for a wider/deeper torrent.
MEGAFLOOD_FILL_DEPTH_M = 48.0   # head over the sill NOTCH (bed-min window): 48 puts
                                # a solid 30-55 m sheet across the flats passage so the
                                # shallow terraced sill crossing floods deep enough to
                                # read as solid water (at 30 the braid bars dominated
                                # and the crossing read as "water vanishes")

# Cross-route notch window (km): the water profile samples the bed as the MINIMUM
# terrain along the PERPENDICULAR cross-section of this width at each route sample,
# so the pooled level is set by the deepest gap of the actual sill (the notch the
# water pours through), not by whatever terrain the polyline happens to clip.
# Lowers the upstream lake to a physical level and keeps the cascade in the deepest
# groove. 0 = sample exactly on the polyline (old behaviour, level ~20 m too high).
MEGAFLOOD_BED_MIN_WINDOW_KM = 2.0

# Pool the flood behind downstream sills (physically-correct overflow): the water
# surface is forced NON-INCREASING downstream (suffix max of bed+fill). The basin
# west of the sill fills up to sill+fill and pours over -> a huge upstream lake, a
# broad fast sheet across the sill, and the 2+ km plunge at the escarpment. False =
# old behaviour (surface hugs the bed, so water visibly "climbs" uphill over sills).
MEGAFLOOD_POOLING = True

# Drop wet patches that are not hydrologically connected to the flood route (flood
# fill on depth>0): kills stray puddles behind ridges inside the envelope. The flood
# extent itself is corridor envelope INTERSECTED with terrain-below-water-surface.
MEGAFLOOD_CONNECTED_FILL = True

# Optional floor for the water surface (m): the surface is not allowed to fall below
# this. None = let the terrain take it down all the way (recommended). Set e.g. to a
# value near PALEO_SEA_LEVEL_M if you want the flood to pool at the sea level.
MEGAFLOOD_WATER_OUTLET_M = None

# OPTIONAL full manual control: water-surface elevation (m) at each waypoint, same
# length as MEGAFLOOD_PATH. None = auto (channel floor + fill depth at the inlet,
# grading to the outlet level). The profile is always forced non-increasing downstream.
MEGAFLOOD_WATER_ELEV_M = None

# Smoothing of the water surface (m) and depth (m) over which the shoreline fades.
MEGAFLOOD_WATER_SMOOTH_M       = 4000.0    # smoothing of the propagated water surface (m)
# Terrain reference for the RACING-sheet water surface (cascade/flats). Smoothed so the
# falling water sheet does not inherit the MES contour terraces as horizontal foam/geometry
# "barcode" steps. Water surface only - the displayed terrain mesh is NOT smoothed.
MEGAFLOOD_SHEET_TERRAIN_SMOOTH_M = 160.0
# Final light smooth of the water surface itself (m): rounds residual terrace steps so
# foam, colour and the 3D sheet read as one coherent moving body. Keep gentle (100-160).
MEGAFLOOD_SURFACE_SMOOTH_M       = 130.0
MEGAFLOOD_WATERLINE_FEATHER_M  = 10.0   # near-binary: water paints FULL blue close to
                                        # its edge - much larger and wide shallow
                                        # margins render as milky semi-transparent film.

# Smooth the water DEPTH used for the shoreline/extent by this many metres (mask-only:
# the terrain heightmap arr is NOT touched, so the canyon/ridges stay sharp). The
# interpolated MES has contour terraces, so depth = surface - terrain crosses zero in a
# sawtooth -> jagged pixel shorelines and stipple holes. Smoothing the EXTENT (not the
# geometry) gives one clean water body. 0 = raw (jagged). 200-400 m is the useful range.
MEGAFLOOD_EXTENT_SMOOTH_M = 420.0
# Radius (m) up to which dry speckle / small islands inside the flood are closed so the
# body reads solid (not a mottled leopard pattern in the shallow terraced bay). Larger
# also swallows genuine small islands; ~600 keeps big ridges, kills terrace speckle even
# in the wide shallow sill crossing.
MEGAFLOOD_HOLE_FILL_M = 650.0

# Shore->deep colour: shallow water is lighter, deep water tends to the navy. Scale (m)
# over which the flood colour deepens with water depth.
MEGAFLOOD_DEEP_SCALE_M = 220.0    # shore->deep colour scale; higher keeps the turbid
                                  # tone visible over deep water, lower goes navy fast.
                                  # 220: the 20-40 m sill passage reads as solid water
                                  # (at 550 it painted as pale milky film)
# Smooth the water depth used for the shore->deep COLOUR ramp (m). Wide so broad MES
# terraces don't mottle the shallow bay light/dark; the large-scale shore->deep gradient
# survives. Colour only - extent/alpha use the raw depth.
MEGAFLOOD_COLOR_DEPTH_SMOOTH_M = 700.0

# Clean Atlantic -> turbid gradient along the corridor (False = uniform turbid).
MEGAFLOOD_TURBID_RAMP = True

# Also carve the corridor DOWN into the heightmap so the 3D render shows an incised
# scour megachannel. 0 = paint colour only, no geometry change (safe default).
# A value like 150-400 m gives a visible flood-scoured trough. Tapered by corridor weight.
MEGAFLOOD_SCOUR_M = 0.0

# Export flood_corridor.png (RGB: R=coverage, G=along-path 0..1, B=foam) so the
# Blender script can build a real 3D flood water sheet + a "flood rolls in" animation.
MEGAFLOOD_WRITE_MASK = True

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
    """River mask with megaflood-scale trunk width.

    Uses distance_transform_edt (O(n), fast) instead of many binary_dilation
    iterations.  Progressive widening by drainage area:
      minor tributaries → 1 km  |  main tributaries → 3 km
      canyon trunk      → 8 km  |  megaflood trunk  → 16 km
    """
    from scipy.ndimage import distance_transform_edt
    filled = _fill_depressions(dem)
    order = np.argsort(filled.ravel())[::-1]
    ri, ci = np.unravel_index(order, filled.shape)
    accum = _d8_accum(filled, ri.astype(np.int64), ci.astype(np.int64))
    area = accum * (pixel_m ** 2) / 1e6                       # upstream area, km²
    out = area > min_drainage_km2                              # base channel mask
    # (drainage threshold km², half-width m)
    for thresh_mult, half_w_m in [(8, 200), (60, 600), (400, 1500), (2000, 3000)]:
        big = area > (min_drainage_km2 * thresh_mult)
        if big.any():
            dist_m = distance_transform_edt(~big) * pixel_m
            out = out | (dist_m < half_w_m)
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
 
 

# =====================================================================
# mask-driven megaflood corridor  (validated in _validate_corridor.py)
# =====================================================================
def _megaflood_points_px(shape, transform=None):
    """Path waypoints -> (row, col) pixel coordinates."""
    H, W = shape
    out = []
    if MEGAFLOOD_PATH_UNITS == "image":
        for fx, fy in MEGAFLOOD_PATH:
            out.append((fy * (H - 1), fx * (W - 1)))
    elif MEGAFLOOD_PATH_UNITS == "crs":
        if transform is None:
            raise ValueError("MEGAFLOOD_PATH_UNITS='crs' needs the grid transform.")
        inv = ~transform
        for x, y in MEGAFLOOD_PATH:
            col, row = inv * (x, y)
            out.append((row, col))
    else:
        raise ValueError("MEGAFLOOD_PATH_UNITS must be 'image' or 'crs'")
    return out


def _megaflood_halfwidths_px(pixel_m, n):
    """Per-waypoint half-widths (km in config) -> pixels; resampled to n points."""
    hw = MEGAFLOOD_HALFWIDTH_KM
    hw = [float(hw)] * n if np.isscalar(hw) else list(map(float, hw))
    if len(hw) != n:
        xi = np.linspace(0, len(hw) - 1, n)
        hw = list(np.interp(xi, np.arange(len(hw)), hw))
    return [w * 1000.0 / pixel_m for w in hw]


def _densify_polyline(pts, halfw_px):
    """Sample the polyline at ~1 px spacing, carrying along-param + half-width."""
    pts = np.asarray(pts, float)
    hw  = np.asarray(halfw_px, float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    tv  = cum / max(cum[-1], 1e-9)                    # along-param 0..1 at vertices
    rows, cols, alo, wid = [], [], [], []
    for i in range(len(pts) - 1):
        n = max(int(seg[i]) + 1, 2)
        s = np.linspace(0.0, 1.0, n)
        rows.append(pts[i, 0] + s * (pts[i + 1, 0] - pts[i, 0]))
        cols.append(pts[i, 1] + s * (pts[i + 1, 1] - pts[i, 1]))
        alo.append(tv[i]  + s * (tv[i + 1]  - tv[i]))
        wid.append(hw[i]  + s * (hw[i + 1]  - hw[i]))
    return (np.concatenate(rows), np.concatenate(cols),
            np.concatenate(alo),  np.concatenate(wid))


def rasterize_megaflood_corridor(shape, pixel_m, transform=None):
    """Rasterise the flood path into (weight, along).

    weight: float 0..1 corridor coverage, 1 in the trunk, feathering to 0 at the rim.
    along : float 0..1 source->plunge (drives the turbidity gradient), NaN outside.
    Densify the polyline, stamp along-param + half-width at each sample pixel, then
    propagate nearest via distance_transform_edt (O(n), handles self-crossings).
    """
    H, W = shape
    pts = _megaflood_points_px(shape, transform)
    halfw_px = _megaflood_halfwidths_px(pixel_m, len(pts))
    rr, cc, tt, ww = _densify_polyline(pts, halfw_px)
    ri = np.clip(np.round(rr).astype(int), 0, H - 1)
    ci = np.clip(np.round(cc).astype(int), 0, W - 1)

    seed_along = np.full((H, W), np.nan, "float32")
    seed_halfw = np.full((H, W), np.nan, "float32")
    seed_along[ri, ci] = tt.astype("float32")
    seed_halfw[ri, ci] = ww.astype("float32")

    seeded = np.isfinite(seed_along)
    dist_px, idx = distance_transform_edt(~seeded, return_distances=True,
                                          return_indices=True)
    along   = seed_along[tuple(idx)]
    halfw_m = seed_halfw[tuple(idx)] * pixel_m
    dist_m  = dist_px * pixel_m

    feather_m = max(MEGAFLOOD_FEATHER_KM * 1000.0, 1e-6)
    weight = np.clip((halfw_m + feather_m - dist_m) / feather_m, 0.0, 1.0)
    weight = np.where(dist_m <= halfw_m, 1.0, weight).astype("float32")
    # along stays defined EVERYWHERE (nearest path sample): the flood can extend
    # beyond the envelope and still needs the source->plunge parameter there.
    # over_m: distance (m) beyond the local corridor half-width (0 inside).
    over_m = np.clip(dist_m - halfw_m, 0.0, None).astype("float32")
    return weight, along.astype("float32"), over_m


def megaflood_foam(arr, pixel_m, basin):
    """Whitewater field 0..1: basin-rim band (robust) unioned with steep local slope."""
    band_m = max(MEGAFLOOD_FOAM_BAND_KM * 1000.0, 1e-6)
    dist_rim_m = distance_transform_edt(~basin) * pixel_m
    lip = np.clip(1.0 - dist_rim_m / band_m, 0.0, 1.0) * (~basin)
    gy, gx = np.gradient(arr, pixel_m)
    slope = np.clip((np.hypot(gx, gy) - ZANCLEAN_CASCADE_SLOPE) /
                    max(ZANCLEAN_CASCADE_SLOPE, 1e-6), 0.0, 1.0)
    return np.maximum(lip, slope).astype("float32")


def _save_u8_rgb(path, arr01):
    u8 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, u8)
    except Exception:
        from PIL import Image
        Image.fromarray(u8).save(path)


def _megaflood_water_profile(arr, rr, cc, tt, pixel_m):
    """Water-surface elevation (m) at each densified path sample, non-increasing.

    Auto: inlet level = channel floor near the source + MEGAFLOOD_FILL_DEPTH_M, graded
    linearly along the path to the outlet (sea level by default). Manual: interpolate
    MEGAFLOOD_WATER_ELEV_M. Always forced non-increasing downstream (water flows down).
    Returns (zw, pooled01): surface and per-sample pooledness (both float32).
    """
    H, W = arr.shape
    if MEGAFLOOD_BED_MIN_WINDOW_KM and MEGAFLOOD_BED_MIN_WINDOW_KM > 0:
        # bed = deepest terrain along the PERPENDICULAR cross-section (+-window/2):
        # the pooled level is controlled by the lowest NOTCH of each sill, not by
        # whatever shoulder the polyline clips, and the cascade sheet hugs the
        # deepest groove. Strictly cross-route (a line, not a disk): an isotropic
        # window on the cascade sees far-DOWNSLOPE cells, over-deepens the surface
        # below the local channel floor and cuts km-scale dry gaps into the falls.
        k = max(int(round(MEGAFLOOD_BED_MIN_WINDOW_KM * 1000.0 / pixel_m / 2.0)), 1)
        dr = np.gradient(rr); dc = np.gradient(cc)
        seglen = np.hypot(dr, dc) + 1e-9
        nr, nc = -dc / seglen, dr / seglen                  # unit normal per sample
        zbed = np.full(rr.shape, np.inf, "float64")
        for j in range(-k, k + 1):
            rj = np.clip(np.round(rr + j * nr).astype(int), 0, H - 1)
            cj = np.clip(np.round(cc + j * nc).astype(int), 0, W - 1)
            zbed = np.minimum(zbed, arr[rj, cj])
    else:
        ri = np.clip(np.round(rr).astype(int), 0, H - 1)
        ci = np.clip(np.round(cc).astype(int), 0, W - 1)
        zbed = arr[ri, ci].astype("float64")                # bed along the path
    if MEGAFLOOD_WATER_ELEV_M is not None:
        wp_t = np.linspace(0.0, 1.0, len(MEGAFLOOD_WATER_ELEV_M))
        zw = np.interp(tt, wp_t, np.asarray(MEGAFLOOD_WATER_ELEV_M, float))
    else:
        # channel floor + fill depth: the water surface follows the terrain down the
        # route, so the flood covers the basin (~fill depth deep) and plunges where
        # the floor drops at the escarpment. Path should follow the thalweg.
        zw = zbed + MEGAFLOOD_FILL_DEPTH_M
        if MEGAFLOOD_POOLING:
            # overflow physics: water can't slope uphill downstream. Suffix max pools
            # the surface behind the highest downstream sill (upstream lake), keeps
            # the fill-depth sheet on the sill itself and on the escarpment drop.
            zw = np.maximum.accumulate(zw[::-1])[::-1]
        # pooledness from the RAW pooled profile, BEFORE the 1D smoothing below:
        # the smoothing cuts the corner at the escarpment lip (surface rides above
        # bed+fill there), which would misclassify the upper cascade as "pooled" -
        # foam damped to zero and no thin-sheet cap right where the falls start.
        pooled01 = np.clip((zw - (zbed + MEGAFLOOD_FILL_DEPTH_M)) / 50.0,
                           0.0, 1.0).astype("float32")
        # smooth the 1D profile ALONG the route (~1 km): the suffix-max staircase
        # (a small step wherever the bed-min envelope drops) propagates as regular
        # stripes perpendicular to the route in the racing sections. Killing the
        # steps at the source beats smoothing the propagated 2D field harder.
        from scipy.ndimage import gaussian_filter1d
        samp_m = max(pixel_m, 1e-6)                # densified samples are ~1 px apart
        zw = gaussian_filter1d(zw, 1000.0 / samp_m, mode="nearest")
        if MEGAFLOOD_POOLING:
            zw = np.maximum.accumulate(zw[::-1])[::-1]     # re-impose monotony
        if MEGAFLOOD_WATER_OUTLET_M is not None:
            zw = np.maximum(zw, float(MEGAFLOOD_WATER_OUTLET_M))   # don't fall below outlet
    if MEGAFLOOD_WATER_ELEV_M is not None:
        pooled01 = np.clip((zw - (zbed + MEGAFLOOD_FILL_DEPTH_M)) / 50.0,
                           0.0, 1.0).astype("float32")
    return zw.astype("float32"), pooled01


def _flow_streaks(zwater, pixel_m, length_m, seed=12345):
    """Line-integral-convolution style streak texture: white noise smeared along
    the water-surface gradient (the flow direction). 0..1, mean ~0.5."""
    from scipy.ndimage import map_coordinates
    H, W = zwater.shape
    gy, gx = np.gradient(zwater, pixel_m)
    mag = np.hypot(gx, gy) + 1e-9
    fx = (gx / mag).astype("float32")
    fy = (gy / mag).astype("float32")
    rng = np.random.default_rng(seed)
    noise = rng.random((H, W), dtype=np.float32)
    # pre-smooth the base noise so the smeared result is BROAD current bands
    # (~lane_m wide) rather than 1-px hairline scratches. Per-pixel white noise
    # smeared along the flow gave pencil-hatching that read as scribbles, not
    # water; broad lanes read as a moving surface. The std-normalisation below
    # restores contrast, so the lanes stay legible - we then apply them at low gain.
    lane_px = max(MEGAFLOOD_STREAK_LANE_M / pixel_m, 0.7)
    noise = gaussian_filter(noise, lane_px).astype("float32")
    yy, xx = np.mgrid[0:H, 0:W].astype("float32")
    n_steps = max(int(length_m / pixel_m / 2), 1)
    # step in ~lane-sized hops so a long strand needs few map_coordinates calls
    step = max(lane_px * 0.5, 1.0)
    acc = noise.copy()
    cnt = 1.0
    k = step
    while k <= n_steps:
        for s in (float(k), -float(k)):
            acc += map_coordinates(noise, [yy + s * fy, xx + s * fx],
                                   order=1, mode="nearest")
            cnt += 1.0
        k += step
    acc /= cnt
    a = (acc - acc.mean()) / (acc.std() + 1e-9)
    out = np.clip(0.5 + 0.5 * a, 0.0, 1.0).astype("float32")
    # gentle post-smooth kills any residual per-step aliasing along the strands
    return gaussian_filter(out, max(lane_px * 0.5, 0.5)).astype("float32")


def compute_megaflood(arr, pixel_m, transform):
    """Topographically-constrained megaflood: corridor AND (terrain below water surface).

    Returns (coverage, along, zwater, depth, foam):
      coverage 0..1 flood extent (feathered at the shoreline); along 0..1 source->plunge;
      zwater water-surface elevation (m) over the grid; depth = zwater-terrain (>=0 m);
      foam 0..1 whitewater at the plunge.
    """
    H, W = arr.shape
    weight, along, over_m = rasterize_megaflood_corridor(arr.shape, pixel_m, transform)
    # smooth the along-parameter: raw EDT nearest-sample propagation carries fan
    # discontinuities that band the turbidity colour ramp (hatching artefacts)
    along = gaussian_filter(along, 2000.0 / pixel_m).astype("float32")

    if not MEGAFLOOD_CONFINE_TO_TERRAIN:                      # legacy overtop sheet
        zwater = (arr + MEGAFLOOD_FILL_DEPTH_M).astype("float32")
        depth  = np.full_like(arr, MEGAFLOOD_FILL_DEPTH_M, dtype="float32")
        foam   = megaflood_foam(arr, pixel_m, arr < FLOOD_BASIN_LEVEL) * (weight > 0)
        return weight, along, zwater, depth, foam.astype("float32"), None

    pts = _megaflood_points_px(arr.shape, transform)
    halfw_px = _megaflood_halfwidths_px(pixel_m, len(pts))
    rr, cc, tt, _ = _densify_polyline(pts, halfw_px)
    ri = np.clip(np.round(rr).astype(int), 0, H - 1)
    ci = np.clip(np.round(cc).astype(int), 0, W - 1)
    # profile + per-sample POOLEDNESS (0..1 ramp): how far the raw pooled profile
    # stands above its own bed+fill, i.e. genuinely ponded water (lake behind a
    # sill) vs a racing thin sheet (sill crest, cascade, plume). Propagated by the
    # same EDT lookup - NO along-interpolation, which re-imported profile jumps at
    # every fan collision. Computed inside the profile fn from the same (min-
    # window) bed and the PRE-smoothing profile.
    zw_path, pooled01 = _megaflood_water_profile(arr, rr, cc, tt, pixel_m)

    seed = np.full((H, W), np.nan, "float32")
    seed[ri, ci] = zw_path
    idx = distance_transform_edt(~np.isfinite(seed), return_distances=False,
                                 return_indices=True)
    nearest = tuple(idx)
    zwater = seed[nearest]
    seed_p = np.full((H, W), np.nan, "float32")
    seed_p[ri, ci] = pooled01
    pooledness = seed_p[nearest]

    zw_raw = zwater.copy()
    if MEGAFLOOD_WATER_SMOOTH_M and MEGAFLOOD_WATER_SMOOTH_M > 0:
        zwater = gaussian_filter(zwater, MEGAFLOOD_WATER_SMOOTH_M / pixel_m).astype("float32")
        pooledness = gaussian_filter(pooledness, MEGAFLOOD_WATER_SMOOTH_M / pixel_m).astype("float32")
        # the smoothing must never SINK the surface far below the raw pooled
        # profile: at the canyon head the 4 km gaussian mixed the 2 km cascade
        # drop into the sill zone and pulled the surface under the terrain - a dry
        # gap right at the lip ("water vanishes before the canyon and reappears
        # inside it"). But only correct SIGNIFICANT deficits (>8 m, full by 20 m):
        # a hard max() re-imported the raw EDT staircase, which flickered around
        # zero depth on the shallow sill flats as a wet/dry "barcode" of stripes.
        deficit = zw_raw - zwater
        wdef = np.clip((deficit - 8.0) / 12.0, 0.0, 1.0)
        zwater = (zwater + wdef * deficit).astype("float32")
        del deficit, wdef
    del zw_raw

    # terrain reference for the racing-sheet water surface: SMOOTHED terrain, so the
    # cascade water surface does not inherit the MES contour terraces as horizontal
    # "barcode" steps (the falling sheet is water, not the terraced bed). This smooths
    # the WATER SURFACE only - the displayed terrain mesh (raw heightmap) is untouched.
    if MEGAFLOOD_SHEET_TERRAIN_SMOOTH_M and MEGAFLOOD_SHEET_TERRAIN_SMOOTH_M > 0:
        arr_sheet = gaussian_filter(arr, MEGAFLOOD_SHEET_TERRAIN_SMOOTH_M / pixel_m).astype("float32")
    else:
        arr_sheet = arr
    # cap the racing-sheet sections at bed+fill AFTER the smoothing: kills the
    # hanging skirts that smoothing builds at the cliff and at fan collisions
    # (phantom water over the open sea next to the falls). Ponded sections keep
    # their standing level.
    zw_thin = np.minimum(zwater, arr_sheet + MEGAFLOOD_FILL_DEPTH_M)
    zwater = (pooledness * zwater + (1.0 - pooledness) * zw_thin).astype("float32")

    # fallsiness from the PRE-edge-drop surface: only genuinely FALLING water (the
    # cascade, the escarpment drape) hands the envelope over to the tight falls
    # feather below. Keyed to surface slope, NOT pooledness: the broad shallow
    # sheet racing across the flats is non-pooled but must keep the full corridor
    # width, or the bay flood pinches to a 3 km ribbon.
    gyf, gxf = np.gradient(zwater, pixel_m)
    s0f = max(MEGAFLOOD_FALLS_SLOPE, 1e-6)
    falls01 = np.hypot(gxf, gyf); del gyf, gxf
    falls01 = np.clip((falls01 - s0f) / (2.0 * s0f), 0.0, 1.0).astype("float32")
    # de-speckle: residual propagation seams flicker falls01 0<->1, which combs the
    # envelope (weight vs w_falls) into teeth along the corridor taper
    falls01 = gaussian_filter(falls01, 500.0 / pixel_m).astype("float32")

    if MEGAFLOOD_EDGE_DROP_PER_KM and MEGAFLOOD_EDGE_DROP_PER_KM > 0:
        # sink the surface continuously beyond the corridor rim: the waterline forms
        # an organic terrain contour wherever the sinking surface meets the ground.
        # Flats next to the corridor fall dry within a few km, deep source basins
        # keep flooding for tens of km, and the falling cascade profile cannot drape
        # a phantom water skin sideways along the escarpment.
        zwater = (zwater - MEGAFLOOD_EDGE_DROP_PER_KM / 1000.0 * over_m).astype("float32")

    # final light smooth of the water SURFACE (~130 m): rounds any residual terrace
    # steps carried into the surface, so foam/colour/3D-sheet all read as a coherent
    # moving water body. Gentle vs the 4 km main smooth - it rounds the lip corner
    # (good for a cascade) without flattening the 2 km drop. Surface only, not terrain.
    if MEGAFLOOD_SURFACE_SMOOTH_M and MEGAFLOOD_SURFACE_SMOOTH_M > 0:
        zwater = gaussian_filter(zwater, MEGAFLOOD_SURFACE_SMOOTH_M / pixel_m).astype("float32")

    depth = np.clip(zwater - arr, 0.0, None).astype("float32")
    feather = max(MEGAFLOOD_WATERLINE_FEATHER_M, 1e-6)
    # shoreline/extent from a SMOOTHED depth (mask-only, terrain arr untouched): the
    # terraced MES makes raw depth cross zero in a sawtooth -> jagged pixel shorelines
    # and stipple holes. depth_ext gives one clean waterline contour; raw depth still
    # drives the colour deepening so real shallows stay lighter.
    if MEGAFLOOD_EXTENT_SMOOTH_M and MEGAFLOOD_EXTENT_SMOOTH_M > 0:
        depth_ext = gaussian_filter(depth, MEGAFLOOD_EXTENT_SMOOTH_M / pixel_m).astype("float32")
    else:
        depth_ext = depth
    waterline = np.clip(depth_ext / feather, 0.0, 1.0)       # soft shoreline (0 at edge -> 1)

    # waterline gives the organic terrain shoreline (with the sinking rim surface);
    # the envelope guarantees ZERO flood beyond halfwidth+feather. The racing-sheet
    # sections (non-pooled: cascade, plume) use their own tight feather so the
    # falling jet stays in the canyon instead of curtaining along the escarpment.
    w_falls = np.clip(1.0 - over_m / max(MEGAFLOOD_FALLS_FEATHER_KM * 1000.0, 1e-6),
                      0.0, 1.0).astype("float32")
    envelope = (1.0 - falls01) * weight + falls01 * w_falls
    coverage = (envelope * waterline).astype("float32")

    # cascade foam BEFORE the sea cutoff (the mist grace below needs it): white
    # where the water surface drops steeply (falls) or a thin sheet races over a
    # steep bed (the lip). Wide ramp -> graded whitewater, not a saturated blob.
    # Damped by pooledness: no foam on standing lake margins.
    gyw, gxw = np.gradient(zwater, pixel_m)
    s0 = max(MEGAFLOOD_FOAM_SURF_SLOPE, 1e-6)
    foam_surf = np.clip((np.hypot(gxw, gyw) - s0) / (9.0 * s0), 0.0, 1.0)
    gy, gx = np.gradient(arr_sheet, pixel_m)    # smoothed bed: no per-terrace foam bands
    foam_thin = np.clip((np.hypot(gx, gy) - ZANCLEAN_CASCADE_SLOPE) /
                        max(ZANCLEAN_CASCADE_SLOPE, 1e-6), 0.0, 1.0) \
        * (depth < MEGAFLOOD_FOAM_THIN_DEPTH_M)
    foam_raw = (np.maximum(foam_surf, foam_thin)
                * np.clip(1.0 - pooledness, 0.0, 1.0)).astype("float32")

    streaks = None
    if MEGAFLOOD_FOAM_STREAKS:
        # falling-water strands: smear noise along the flow and modulate the foam
        streaks = _flow_streaks(zwater, pixel_m, MEGAFLOOD_STREAK_LEN_M)
        foam_raw = (foam_raw * (MEGAFLOOD_STREAK_MIN
                                + (1.0 - MEGAFLOOD_STREAK_MIN) * streaks)).astype("float32")

    # rapids: SPARSE whitecap filaments on racing (non-pooled) water - only the
    # brightest streak lanes turn white. A dense streak**2 wash painted the whole
    # racing section as milky frost instead of white threads on blue water.
    if MEGAFLOOD_RAPIDS_FOAM > 0 and streaks is not None:
        lane = np.clip((streaks - 0.55) / 0.45, 0.0, 1.0) ** 2
        rapids = (MEGAFLOOD_RAPIDS_FOAM * np.clip(1.0 - pooledness, 0.0, 1.0)
                  * lane).astype("float32")
        foam_raw = np.maximum(foam_raw, rapids)


    # the flood sheet only exists ABOVE the receiving palaeo sea - EXCEPT that
    # foam-rich cells get a grace depth (plunge mist/spray where the cascade hits
    # the water). The visible surface of those cells is clamped onto the sea plane
    # below, so the spray floats as a white apron on the sea.
    grace = MEGAFLOOD_PLUNGE_MIST_M * np.clip(2.0 * foam_raw, 0.0, 1.0)
    above_sea = np.clip((zwater - PALEO_SEA_LEVEL_M + grace) / 150.0, 0.0, 1.0)
    coverage = (coverage * above_sea).astype("float32")
    if MEGAFLOOD_CONNECTED_FILL:
        # keep only wet cells hydrologically connected to the flood route: kills
        # stray puddles behind ridges that happen to lie below the water surface
        from scipy.ndimage import label
        lab, _n = label(coverage > 0)
        path_labs = np.unique(lab[ri, ci])
        path_labs = path_labs[path_labs != 0]
        conn = np.isin(lab, path_labs)
        coverage = (coverage * conn).astype("float32")

    # SOLIDIFY the flood interior so it reads as ONE coherent water body, not a mottled
    # leopard pattern. In the shallow terraced bay the coverage dips fleckily (terrace
    # tops graze the waterline) -> the semi-transparent sheet shows tan bed through the
    # gaps. grey_closing fills dark speckle + dry islands smaller than HOLE_FILL_M while
    # preserving the large-scale shoreline and genuine big islands; binary_fill_holes
    # then closes any fully-enclosed basin the closing didn't reach. Mask + water-surface
    # only - the terrain mesh is never modified.
    from scipy.ndimage import binary_fill_holes, grey_closing
    r = max(int(round(MEGAFLOOD_HOLE_FILL_M / pixel_m)), 3)
    cov_solid = grey_closing(coverage, size=r).astype("float32")
    filled = binary_fill_holes(cov_solid > 0.15)
    newly = (cov_solid > coverage) | (filled & (coverage <= 0.15))
    coverage = np.where(filled, np.maximum(coverage, cov_solid), coverage).astype("float32")
    # solidified/filled cells sit below the water surface -> ensure a real depth so the
    # 3D sheet covers them (surface already = max(zwater, terrain) there)
    depth = np.where(newly & (coverage > 0.15),
                     np.maximum(depth, feather), depth).astype("float32")
    # along stays defined wherever the (final, solidified) flood is wet
    along = np.where(coverage > 0, along, np.nan).astype("float32")

    # churning margins: thin foam fringe exactly AT the waterline. Edge-detected on
    # the BINARY wet mask - the soft coverage field has interior gradients all over
    # shallow flats and would paint the whole flat as frost.
    if MEGAFLOOD_EDGE_FOAM > 0:
        bnd = (coverage > 0.02).astype("float32")
        gcy, gcx = np.gradient(bnd)
        fringe = np.clip(np.hypot(gcx, gcy) * 2.0, 0.0, 1.0)
        fringe = gaussian_filter(fringe, 1.2).astype("float32") * MEGAFLOOD_EDGE_FOAM
        if streaks is not None:
            fringe = fringe * (0.4 + 0.6 * streaks)
        foam_raw = np.maximum(foam_raw, fringe.astype("float32"))

    # cap the VISIBLE foam (grace/mist above still uses the uncapped foam_raw): the
    # cascade face is steep everywhere so foam saturates to a flat white blob; capping
    # lets blue water show through the broad foam strands -> blue-white whitewater.
    foam = (np.clip(foam_raw, 0.0, MEGAFLOOD_FOAM_MAX) * (coverage > 0)).astype("float32")
    # lift the below-sea (mist) part of the sheet onto the sea plane
    zwater = np.where(coverage > 0,
                      np.maximum(zwater, PALEO_SEA_LEVEL_M + 2.0),
                      zwater).astype("float32")

    # long current lanes for the flood-body colour (separate, longer LIC pass)
    body_streaks = None
    if MEGAFLOOD_BODY_STREAK_GAIN > 0:
        body_streaks = _flow_streaks(zwater, pixel_m, MEGAFLOOD_BODY_STREAK_LEN_M,
                                     seed=777)
    return coverage, along, zwater, depth, foam, body_streaks


def _save_u16_norm(path, arr_m, zmin, zmax):
    """Save a metre-valued array as a 16-bit PNG normalised with the given z-range."""
    rng = (zmax - zmin) if (zmax > zmin) else 1.0
    u16 = np.clip((arr_m - zmin) / rng * 65535.0, 0, 65535).astype(np.uint16)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, u16)
    except Exception:
        from PIL import Image
        Image.fromarray(u16.astype(np.int32), mode="I").save(path)


def write_colormap(path, arr, pixel_m, ve, scheme, coverage=None,
                   azdeg=315, altdeg=45, shade_lo=0.45, transform=None):
    """Bake hypsometric tint × hillshade + flood overlay into an 8-bit RGB PNG.

    Flood layers (applied in order, last wins):
      1. D8 river channels (wide, megaflood-scale trunk)
      2. Deep basin below FLOOD_BASIN_LEVEL  →  dark navy
      3. Zanclean corridor from Gibraltar (west edge)  →  turbid/Atlantic gradient
      4. Cascade/foam at steepest descent within corridor  →  white-blue

    coverage: bool mask where MES exists; restricts flood to MES area in blend mode.
    """
    from matplotlib.colors import LightSource
    from scipy.ndimage import distance_transform_edt

    stops = _COLORMAP_STOPS.get(scheme, _COLORMAP_STOPS["desiccated"])
    es   = np.array([s[0] for s in stops], dtype="float64")
    cols = np.array([s[1] for s in stops], dtype="float64")
    tint = np.dstack([np.interp(arr, es, cols[:, k]) for k in range(3)])
    hs   = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(
        arr, vert_exag=ve, dx=pixel_m, dy=pixel_m)[..., None]   # (H, W, 1)
    if not COLORMAP_HILLSHADE:
        hs = np.ones_like(hs)
    rgb = tint * (shade_lo + (1.0 - shade_lo) * hs)

    # ------------------------------------------------------------------
    # Layer 1: D8 river channels + deep basin
    # ------------------------------------------------------------------
    channel = None   # may be set below; needed by layer 2
    if FLOOD_ALONG_CANYON:
        if _HAVE_NUMBA and _HAVE_SKIMAGE:
            channel = river_mask(arr, pixel_m, FLOOD_MIN_DRAINAGE_KM2)
        else:                                                    # fallback: concave channels
            from scipy.ndimage import binary_closing, label
            sm = gaussian_filter(arr, FLOOD_SMOOTH_M / pixel_m)
            channel = (sm - arr) > FLOOD_CONCAVITY_M
            channel = binary_closing(channel, iterations=2)
            lab, n = label(channel)
            if n:
                sizes = np.bincount(lab.ravel()); sizes[0] = 0
                channel = (sizes >= (10e6 / pixel_m ** 2))[lab]
        if coverage is not None and FLOOD_RIVERS_IN_MES_ONLY:
            channel = channel & coverage

        basin    = arr < FLOOD_BASIN_LEVEL
        deep_rgb = np.array(ZANCLEAN_DEEP_NAVY if ZANCLEAN_FLOOD else (0.04, 0.10, 0.30))
        river_rgb   = np.array(FLOOD_WATER_RGB)

        wrgb = np.where(basin[..., None],
                        deep_rgb.reshape(1, 1, 3),
                        river_rgb.reshape(1, 1, 3))
        wrgb = wrgb * (0.50 + 0.50 * hs)
        rgb  = np.where((channel | basin)[..., None], wrgb, rgb)
    else:
        basin = arr < FLOOD_BASIN_LEVEL

    # ------------------------------------------------------------------
    # Layer 2: Zanclean waterfall zone
    # Canyon rivers flow across the dry MES and PLUNGE over the escarpment
    # into the deep basin — not the other way around.
    # We paint the zone directly ABOVE the basin rim where rivers spill over.
    # ------------------------------------------------------------------
    if ZANCLEAN_FLOOD:
        # How far upslope from the basin boundary to paint (half-width in m)
        half_w_m = ZANCLEAN_CHANNEL_KM * 500.0
        dist_to_basin_m = distance_transform_edt(~basin) * pixel_m  # distance upslope from basin

        # Waterfall zone: upslope fringe of the basin rim, not in the basin itself
        waterfall_zone = (dist_to_basin_m < half_w_m) & ~basin
        # Restrict to where rivers actually flow (canyon thalweg, not full rim)
        if channel is not None:
            waterfall_zone = waterfall_zone & channel
        # Stay in MES area to avoid GEBCO background shelf artefacts
        if coverage is not None:
            waterfall_zone = waterfall_zone & coverage

        if waterfall_zone.any():
            gy, gx = np.gradient(arr, pixel_m)
            steep  = (np.hypot(gx, gy) > ZANCLEAN_CASCADE_SLOPE) & waterfall_zone

            water_col = np.array(ZANCLEAN_ATLANTIC_RGB, dtype="float32")
            foam_col  = np.array(ZANCLEAN_FOAM_RGB,     dtype="float32")

            casc_rgb = np.where(steep[..., None],
                                 foam_col[None, None, :] * 0.65 + water_col[None, None, :] * 0.35,
                                 water_col[None, None, :])
            casc_rgb = casc_rgb * (0.50 + 0.50 * hs)

            rgb = np.where(waterfall_zone[..., None], casc_rgb, rgb)

    # ------------------------------------------------------------------
    # Layer 3: mask-driven megaflood corridor (the big Zanclean trunk).
    # Painted LAST so the hand-traced torrent dominates over the D8 feeders.
    # ------------------------------------------------------------------
    if MEGAFLOOD_CORRIDOR:
        # pre-flood terrain colour (desiccated floor + D8 rivers + remnant basin
        # lakes, but NO megaflood): the flood_advance animation shows this where
        # the 3D water sheet hasn't arrived yet - a painted-in flood would spoil
        # the "flood rolls in" illusion.
        u8_dry = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        try:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(OUT_DIR, OUT_COLORMAP_DRY_PNG), u8_dry)
        except Exception:
            from PIL import Image
            Image.fromarray(u8_dry).save(os.path.join(OUT_DIR, OUT_COLORMAP_DRY_PNG))
        coverage, along, zwater, depth, foam_w, body_streaks = \
            compute_megaflood(arr, pixel_m, transform)

        atl  = np.array(ZANCLEAN_ATLANTIC_RGB, "float32")
        turb = np.array(ZANCLEAN_TURBID_RGB,   "float32")
        deep = np.array(ZANCLEAN_DEEP_NAVY,    "float32")
        foam = np.array(ZANCLEAN_FOAM_RGB,     "float32")

        a = np.nan_to_num(along, nan=0.0)[..., None]             # 0 source -> 1 lip
        if MEGAFLOOD_TURBID_RAMP:
            base = atl.reshape(1, 1, 3) * (1.0 - a) + turb.reshape(1, 1, 3) * a
        else:
            base = np.broadcast_to(turb.reshape(1, 1, 3), arr.shape + (3,)).copy()

        # shore -> deep by water depth. sqrt ramp: even shallow racing water darkens
        # fast (a linear ramp left the wide shallow shelves so light that the foam
        # lanes read as milky frost instead of whitecaps on dark water).
        # Smooth the depth for the COLOUR only: the interpolated MES has broad contour
        # terraces, and each terrace step varies the water depth -> the shore->deep colour
        # ramp printed them as a light/dark "leopard" mottle in the shallow bay. A wide
        # smooth (~700 m) makes the shallow bay one uniform mid-blue while the large-scale
        # shore->deep gradient survives. Extent/alpha keep the raw depth.
        depth_c = gaussian_filter(depth, MEGAFLOOD_COLOR_DEPTH_SMOOTH_M / pixel_m)
        dt = np.sqrt(np.clip(depth_c / max(MEGAFLOOD_DEEP_SCALE_M, 1e-6), 0.0, 1.0))[..., None]
        watercol = base * (1.0 - dt) + deep.reshape(1, 1, 3) * dt
        if body_streaks is not None:
            # current lanes: subtle brightness modulation along the flow direction
            g = MEGAFLOOD_BODY_STREAK_GAIN
            watercol = watercol * (1.0 - g + 2.0 * g * body_streaks[..., None])
        fw = foam_w[..., None]
        cor_rgb = watercol * (1.0 - fw) + foam.reshape(1, 1, 3) * fw
        # FLAT shading for the flood body: the bed hillshade (hs) is computed from the
        # terraced MES terrain, so at 0.45 weight the terrace light/dark banding baked
        # into the flood colour and mottled through the semi-transparent 3D sheet as a
        # "leopard" pattern in the shallow bay. Water is flat - it must not wear the
        # bed's relief. A faint 0.15 keeps a hint of macro-relief; the 3D sheet supplies
        # the real water lighting (sun + sky) on top.
        cor_rgb = cor_rgb * (0.85 + 0.15 * hs)

        w3 = coverage[..., None]
        rgb = rgb * (1.0 - w3) + cor_rgb * w3

        if MEGAFLOOD_WRITE_MASK:
            corr = np.zeros(arr.shape + (3,), "float32")
            corr[..., 0] = coverage
            corr[..., 1] = np.nan_to_num(along, nan=0.0) * (coverage > 0)
            corr[..., 2] = foam_w * (coverage > 0)
            _save_u8_rgb(os.path.join(OUT_DIR, OUT_FLOOD_CORRIDOR_PNG), corr)
            # 16-bit water-surface heightmap for the 3D flood sheet (same z-range as the
            # terrain heightmap, so Blender displaces both identically). CONTINUOUS at
            # the waterline: max(zwater, terrain) meets the terrain exactly where the
            # water surface intersects it, instead of jumping from lake level to ground
            # at the coverage edge - that jump aliased against the coarser sheet mesh
            # as comb-teeth terraces along every shoreline. Where dry, the sheet drapes
            # the terrain invisibly (alpha = coverage = 0).
            zmn = float(np.nanmin(arr)); zmx = float(np.nanmax(arr))
            surf = np.clip(np.maximum(zwater, arr), zmn, zmx).astype("float32")
            # light smooth of the exported 3D sheet surface (~80 m): where a terrace
            # riser of the raw terrain pokes above the (smoothed) water level, max()
            # picked the terraced bed and the sheet showed stair-steps. Smoothing the
            # sheet geometry removes those without touching the terrain mesh; alpha
            # (coverage) fades at the shore so the tiny lift there is invisible.
            surf = gaussian_filter(surf, 80.0 / pixel_m).astype("float32")
            # frame-edge treatment: where the flood is cut by the grid border (the
            # inlet enters off-map from the west), drop the surface to zmin on the
            # outermost ring -> the sheet dives as a vertical WATER WALL exactly at
            # the frame edge, like the sea plane cut. (The old 2-px alpha zeroing
            # removed the water there instead, leaving a see-under gap + glowing
            # facet along the edge.) Ring must be wider than the displace-vertex
            # texel spacing (~grid/2400) or the wall is missed by the mesh.
            rw = max(2, int(round(arr.shape[1] / 2400.0)) + 1)
            ring = np.zeros(arr.shape, bool)
            ring[:rw, :] = True; ring[-rw:, :] = True
            ring[:, :rw] = True; ring[:, -rw:] = True
            surf = np.where(ring & (coverage > 0.02), zmn, surf).astype("float32")
            _save_u16_norm(os.path.join(OUT_DIR, OUT_FLOOD_SURFACE_PNG), surf, zmn, zmx)

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

    if TERRAIN_SMOOTH_M and TERRAIN_SMOOTH_M > 0:
        # gentle smooth of the FINAL terrain: rounds the MES data terraces on the canyon
        # walls. Consistent across mesh + hillshade + flood bed (compute_megaflood reads
        # this same array). Keep small - the terrain is data (see TERRAIN_SMOOTH_M note).
        paleo = gaussian_filter(paleo, TERRAIN_SMOOTH_M / pixel_m).astype("float32")
        pbar.write(f"[merge] terrain smoothing: Gaussian sigma {TERRAIN_SMOOTH_M:.0f} m "
                   f"(softens MES data terraces; set TERRAIN_SMOOTH_M=0 to disable)")

    if MEGAFLOOD_CORRIDOR and MEGAFLOOD_SCOUR_M > 0:
        _w, _, _ = rasterize_megaflood_corridor(paleo.shape, pixel_m, transform)
        paleo = (paleo - MEGAFLOOD_SCOUR_M * _w).astype("float32")
        pbar.write(f"[merge] megaflood scour: carved up to {MEGAFLOOD_SCOUR_M:.0f} m "
                   f"into the corridor (incised megachannel for the 3D render)")
 
    pbar.set_description("write GeoTIFF")
    write_geotiff(os.path.join(OUT_DIR, OUT_GEOTIFF), paleo, transform, crs); pbar.update(1)
 
    pbar.set_description("write heightmap")
    zmin, zmax = write_heightmap_png(os.path.join(OUT_DIR, OUT_HEIGHTMAP_PNG), paleo); pbar.update(1)
 
    if WRITE_COLORMAP:
        write_colormap(os.path.join(OUT_DIR, OUT_COLORMAP_PNG), paleo, pixel_m,
                       VE_HINT, COLORMAP_SCHEME, coverage=coverage, transform=transform)
 
    pbar.set_description("write metadata")
    meta = dict(zmin_m=zmin, zmax_m=zmax,
                width_m=paleo.shape[1] * pixel_m, height_m=paleo.shape[0] * float(abs(transform.e)),
                ncols=int(paleo.shape[1]), nrows=int(paleo.shape[0]), pixel_m=pixel_m,
                paleo_sea_level_m=float(PALEO_SEA_LEVEL_M), ve_hint=float(VE_HINT),
                crs=str(crs), heightmap_png=OUT_HEIGHTMAP_PNG)
    if WRITE_COLORMAP:
        meta["colormap_png"] = OUT_COLORMAP_PNG
    if WRITE_COLORMAP and MEGAFLOOD_CORRIDOR:
        meta["colormap_dry_png"] = OUT_COLORMAP_DRY_PNG
    if WRITE_COLORMAP and MEGAFLOOD_CORRIDOR and MEGAFLOOD_WRITE_MASK:
        meta["flood_corridor_png"] = OUT_FLOOD_CORRIDOR_PNG
        meta["flood_surface_png"] = OUT_FLOOD_SURFACE_PNG
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