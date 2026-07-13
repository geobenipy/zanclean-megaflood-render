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

# CANYON HERO shot: box around the waterfall (lip at ~532600/4074000), finer grid so the
# 2.2 km drop carries the frame. Pair with VE 5 in the render.
# WIDENED from the old 70x60 km box (495000,4040000,565000,4100000): the camera looks WSW
# from 80 km with a 62 mm lens, so it saw straight over the old crop border - the DEM ended
# mid-frame as a straight cliff with a lit vertical cut face on the far right. 92x68 km puts
# that border out of shot (and the hero haze below softens whatever is left). Res 15 instead
# of 12.5 keeps the pixel count (and the ~5 min merge) at the old level despite the bigger box.
#TARGET_RES = 15.0
#CROP_BBOX  = (478000, 4036000, 570000, 4104000)   # (xmin,ymin,xmax,ymax) target-CRS metres

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
OUT_DIR           = "output_v17"       # REGIONAL / panel a. v13-v14 = open envelope -> the flood's true
                                       # EQUILIBRIUM extent (an inland lake). v15 = back to the RIVER, on
                                       # the measured low route (an advancing-torrent snapshot), but it ran
                                       # out on the southern shelf and lapped the Malta escarpment rim.
                                       # v16 = the river re-measured hugging SICILY + corridor trimmed, so
                                       # the escarpment south of the canyon stays dry. v17 = the signed-depth
                                       # extent fix: no more water film painted up the dry banks.
#OUT_DIR          = "output_hero_v20"  # canyon hero / panel b - MUST be re-merged whenever the route or
                                       # the envelope changes, or the two panels show different floods.
#OUT_DIR          = "output_hero_v18"  # v11 = first cascade relief; v12 = gate
                                       # moved pre-edge-drop; v13 = gate slope raised; v14 = gate keyed
                                       # to the LARGE-SCALE fall (the fix that stopped the relief
                                       # rippling the flat approach bay); v15 = whitewater on the crests;
                                       # v16 = spray plume mask; v17 = de-periodised rollers;
                                       # v18 = same open envelope + lake swell as the regional panel,
                                       # so both panels of the figure now show ONE flood
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
OUT_FLOOD_RELIEF_PNG   = "flood_relief_16bit.png"    # 16-bit SIGNED cascade relief (metres, +-MEGAFLOOD_RELIEF_AMP_M)
OUT_FLOOD_MIST_PNG     = "flood_mist.png"            # 8-bit spray/mist density 0..1 (Blender volume)
VE_HINT           = 3.0              # regional overview
#VE_HINT          = 5.0              # canyon hero: stronger relief so the 2.2 km drop reads

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
# RE-ROUTED TWICE, both times MEASURED (never eyeballed).
#
# (1) The original middle waypoints crossed the flats at (475200,4063000) / (497600,4066000) /
#     (514400,4070000), where the bed CLIMBS to +46 m. That was a self-inflicted sill: the water
#     had to pond ~100 m deep just to get over ground it never needed to cross - THAT is where
#     the inland-lake look came from. A minimax search (Dijkstra where the path cost is the
#     HIGHEST cell crossed) found a -61 m corridor out on the southern shelf. Sill -> -48 m.
#
# (2) That southern-shelf line ran far from Sicily, out in the middle of the frame. The route is
#     now the least-rise channel CONSTRAINED to y >= 4052000, i.e. forced to hug the Sicilian
#     margin - the same anastomosing system that is visible in the DEM and that the flood would
#     have followed off the Sicilian slope. Waypoints 4-13 ARE that measured channel (Dijkstra
#     path, resampled + smoothed), not a hand-drawn curve, which is why it meanders.
#
# Hugging Sicily costs head: the constrained sill is +6 m against -48 m out on the shelf, so the
# ponded surface upstream stands ~54 m higher. That is the honest price of the river's position
# and it is paid on purpose (see MEGAFLOOD_HALFWIDTH_KM: the corridor is narrowed to match).
#
# Waypoints 12-13 climb the trough at x ~ 511000 and turn north WEST OF THE MALTA ESCARPMENT
# CREST (which runs at x ~ 514000-516000 between y = 4050000 and 4071000). The old route
# instead ran NNE along that crest at (516500,4060000) with a 13 km half-width, so the corridor
# lapped over the rim and the flood poured down the escarpment face as a SECOND cascade,
# 15 km south of the Noto Canyon. On a cliff there is no topographic barrier to stop it -
# the ground only falls - so ONLY the envelope can, and the envelope was on the wrong side.
# Approaching the canyon head from the NW (13 -> 14) removes the artefact structurally.
MEGAFLOOD_PATH = [
    (366000, 4096000),  # 0  NW inlet OUTSIDE the grid (start-cap artefacts stay off-map)
    (391200, 4090000),  # 1  enters the frame (~-450 m)
    (408000, 4076000),  # 2  upper trough (~-730 m)
    (427600, 4064000),  # 3  trough axis (~-810 m) - the measured channel starts here
    (427000, 4063400),  # 4  out of the trough onto the Sicilian margin
    (434600, 4060200),  # 5
    (447000, 4055800),  # 6  meander south
    (460200, 4054400),  # 7  the deepest reach of the shelf channel
    (472800, 4057400),  # 8
    (485200, 4060200),  # 9  meander north, tight against the Sicilian slope
    (497600, 4059000),  # 10
    (506800, 4059400),  # 11 the controlling sill (+6 m) at the foot of the escarpment
    (511000, 4066800),  # 12 turns NORTH, WEST of the Malta escarpment crest
    (513000, 4078000),  # 13 still west of the crest - the canyon is approached from the NW
    (521000, 4074500),  # 14 canyon head / convergence, turning E into the canyon
    (532600, 4074000),  # 15 escarpment lip (the waterfall)
    (538200, 4076100),  # 16 plunge down the canyon axis (ENE, ~-2470 m)
    (545000, 4078000),  # 17 plume into the deep Ionian (~-2730 m)
    (552000, 4080000),  # 18 plume runs out into the basin
]

# Corridor HALF-width in km at each waypoint (list same length as MEGAFLOOD_PATH, OR
# a single number for constant width). The flood is clipped to this envelope.
#   waypoints 0-4 = far upstream before/along Sicily (NW inlet -> deep trough);
#   waypoints 5-6 = the flats/bay; 7-9 = approach -> canyon head -> lip (the inflow into
#   the canyon); 10-12 = the plunge + plume down the canyon.
#
# THE CORRIDOR IS A FLOW BOUNDARY, NOT A WATER LEVEL - and that is a deliberate, stated
# scientific framing. Measured on this DEM: the palaeo-topography here is a broad BASIN, not
# a valley. At ANY water level the connected flooded area is an inland sea (38 % of the frame
# even at -31 m). So the terrain CANNOT confine a river, and a corridor with terrain-cut banks
# is not achievable at equilibrium - the honest equilibrium picture is a lake.
#
# The figure instead shows the flood as a SNAPSHOT OF AN ADVANCING TORRENT that has crossed
# the dry basin and not yet filled it. On that reading the corridor edge is the flow front,
# not a shoreline, and the envelope is the right tool. The caption must say so.
#
# Half-widths therefore read as the RIVER's width: wide where it drains the deep trough,
# narrowing as it runs the shelf, pinching hard at the lip.
#
# ONE list for BOTH panels: the regional and hero shots are two views of the SAME flood. The
# old split (tight flats regional / open flats hero) meant the two panels showed two floods.
#
# TRIMMED DOWN (was 22/22/25/28/26/20/18/16/15/13/12/9/8/10/14 on the 15-waypoint route). Two
# reasons, and they are the same reason: the sill is 54 m higher now that the river hugs Sicily
# (see MEGAFLOOD_PATH), so the same half-widths would have inflated the ponded reach back into
# the reservoir. A narrower corridor spends that head on DEPTH instead of AREA - which is what a
# torrent in a channel does - and it keeps the flood reading as a river.
MEGAFLOOD_HALFWIDTH_KM = [16, 16, 15, 14, 11, 10, 9, 9, 9, 8, 7, 6, 5, 4.5, 6, 7, 6, 8, 12]
#                          0   1   2   3   4   5  6  7  8  9 10 11 12  13   14 15 16 17  18
#
# 12-13 (4.5-5 km) IS THE KNOB THAT KEEPS THE ESCARPMENT DRY. That leg runs north at x ~ 511000
# and the Malta escarpment crest is at x ~ 514000-516000, so a half-width above ~5 km puts the
# corridor rim on the cliff - and on a cliff the terrain cannot stop water, only the envelope
# can (see MEGAFLOOD_PATH). Measured: the escarpment face south of the canyon mouth is 381 km2;
# the old envelope wetted 92 km2 of it as a second cascade, this one wets 32 km2 - all of it the
# southern shoulder of the plunge apron itself, none of it the rim the flood used to spill over.
#
# LIP + PLUNGE (now 15-17) STAY PULLED BACK HARD (once 23/25/23, then 9/8/10): a wide falling
# front drapes the cascade sideways along the whole escarpment instead of down the canyon. More
# water in the canyon comes from FILL_DEPTH (a THICKER sheet over the lip), not a wider front.

# Soft edge (km) feathered onto the corridor rim so it reads as a torrent, not a
# canal. Generous: within the feather the SHORELINE comes from the terrain (via the
# sinking surface); beyond halfwidth+feather the flood is guaranteed zero.
MEGAFLOOD_FEATHER_KM = 16.0

# Lower the water surface CONTINUOUSLY beyond the corridor rim, by this many metres
# per km of distance outside the half-width. The sinking surface intersects the
# terrain at some organic contour, so there is never a straight mask cut: flats next
# to the corridor fall dry within a few km, deep source basins to the west keep
# flooding for tens of km, and the falling cascade profile cannot drape sideways
# along the escarpment. 0 = hard envelope clip (weight * waterline).
MEGAFLOOD_EDGE_DROP_PER_KM = 130.0  # back up from the 60 that was tried for a softer shoreline.
                                    # Steep = the sinking surface cuts within ~0.5 km of the rim,
                                    # so the shoreline is set by the TERRAIN (valleys flood,
                                    # ridges stay dry) rather than by a long grazing intersection.
                                    # 20-50 = milky film / km-long glassy levee ramp.

# --- ORGANIC CORRIDOR RIM ---
# Wobble the half-width with a smooth low-frequency noise so the rim is never an exact offset
# of the polyline. This was originally the attempted cure for the ruler-straight shoreline;
# it is NOT - that is fixed structurally, by opening the upstream half-widths until the
# envelope no longer binds (see MEGAFLOOD_HALFWIDTH_KM). What remains here is cosmetic: the
# envelope still binds at the LIP and the PLUNGE (half-widths 8-14 km), and a slight meander
# keeps the falling front and the plume from having machined edges.
# Keep it MODEST: the lip's 9 km half-width is tuned so the cascade cannot drape into the side
# gully ~15 km off-axis, and a large wiggle would eat into that margin.
MEGAFLOOD_EDGE_WIGGLE     = 0.25   # fractional half-width modulation (0 = exact offset). The
                                   # envelope IS binding again now that the corridor is back, so
                                   # this is what keeps the flow front from being a drawn ribbon.
MEGAFLOOD_EDGE_WIGGLE_KM  = 12.0   # wavelength (km) of the meander
                                    # NOTE: this knob was NOT the cause of the straight shoreline
                                    # edge (measured: the long straight segments sit at corridor
                                    # weight w = 1.0, i.e. deep INSIDE the corridor, nowhere near
                                    # the rim). See MEGAFLOOD_HALFWIDTH_KM for what it really was.

# Whitewater band (km) painted upslope of the basin rim where the flood goes over.
# Only used by the legacy (non-confined) mode. The confined mode keys foam to the
# slope of the WATER SURFACE instead (see MEGAFLOOD_FOAM_SURF_SLOPE).
MEGAFLOOD_FOAM_BAND_KM = 4.0

# Confined-mode foam: whitewater where the water surface itself drops steeply (the
# cascade), ramping to full white at 3x this gradient (m/m). 0.05 = 5% surface drop.
# Must stay above EDGE_DROP_M / (FEATHER_KM*1000) or the corridor rim foams too.
MEGAFLOOD_FOAM_SURF_SLOPE = 0.030   # lowered from 0.05: the cascade was reading as blue water
                                    # with a few pale streaks, not as a falling torrent. Foam now
                                    # starts on gentler water-surface slopes, so the whole drop
                                    # into the amphitheatre churns, not just the steepest facets.
# ... plus where a THIN racing sheet (depth below this, in m) skims a steep bed
# (> ZANCLEAN_CASCADE_SLOPE) - catches the lip itself without foaming deep wet slopes.
MEGAFLOOD_FOAM_THIN_DEPTH_M = 40.0

# Lateral feather (km) of the CASCADE section only: the falling jet is confined to
# halfwidth + this, instead of the generous MEGAFLOOD_FEATHER_KM of the lake/flats -
# otherwise the thin falling film paints a curtain along the whole escarpment face.
MEGAFLOOD_FALLS_FEATHER_KM = 3.2   # BACK DOWN from 7.0: the wide feather is exactly what let
                                   # the falling film curtain sideways out of the amphitheatre
                                   # and into the dry side gully. Tight jet again.

# Water-surface slope (m/m) above which the envelope hands over from the generous
# lake/flats feather to the tight falls feather. This replaces the old pooledness
# switch: a broad shallow sheet RACING across the flats keeps the full corridor
# width, only genuinely FALLING water (cascade, escarpment drape) is pinched to
# the canyon. Full handover at 3x this slope.
MEGAFLOOD_FALLS_SLOPE = 0.018   # lowered from 0.025: the handover to the TIGHT falls feather
                                # starts earlier, so water that has begun to drop over the rim
                                # (including the drape toward the side gully) is pinched to the
                                # canyon sooner. The flat racing sheet on the flats is unaffected.

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
MEGAFLOOD_FOAM_MAX         = 0.86  # raised from 0.72 for a whiter, "flutiger" cascade. Cap on
                                   # visible foam whiteness: the 2 km cascade
                                   # face is steep everywhere -> foam saturated to a
                                   # flat white blob. Capping lets the blue water show
                                   # through, so the plunge reads as blue-white streaked
                                   # whitewater (with the broad foam strands) not paint.

# Plunge mist: foam-rich cells may dip this many metres BELOW the palaeo sea level
# before the coverage cutoff removes them -> a white spray zone where the cascade
# hits the sea instead of a hard cut at the waterline.
MEGAFLOOD_PLUNGE_MIST_M  = 250.0

# --- SPRAY / MIST PLUME (3D volume in Blender) ---
# The apron above is FLAT - it lies on the water. A cataract this size throws a spray
# cloud hundreds of metres INTO THE AIR, and that rising plume is the strongest
# remaining "this is really falling" cue. So export a mist DENSITY mask; Blender turns
# it into a bounded Principled Volume box whose density fades with height above the
# local water surface (read back out of flood_surface_16bit.png in the shader).
# NB a *bounded object* volume is fine - it is the WORLD Volume Scatter that turns the
# km-scale scene black.
MEGAFLOOD_MIST_MASK     = True     # export flood_mist.png
MEGAFLOOD_MIST_SPREAD_M = 1200.0   # blur radius (m): how far the spray cloud billows out
                                   # from the churning water that generates it. 1800 smeared
                                   # the plume into a blanket that veiled the whole canyon.
MEGAFLOOD_MIST_CHUTE_GAIN  = 0.35  # weight of the GENERAL churn all down the chute. Keep it
                                   # low: at 1.0 the cascade generates as much spray at its
                                   # head as at its foot, and the mask's centre of mass ends
                                   # up at the canyon head instead of the plunge.
MEGAFLOOD_MIST_PLUNGE_GAIN = 3.5   # weight of the IMPACT zone. THIS is what makes the plume
                                   # a plume rather than an even haze.
MEGAFLOOD_MIST_SEA_BAND_M = 700.0  # height above the palaeo sea level over which "impact
                                   # zone" fades out (1 at sea level, 0 this far above)
MEGAFLOOD_MIST_GAIN     = 1.0      # overall density gain after the blur, before clipping
MEGAFLOOD_MIST_HEIGHT_M = 500.0    # how far the plume rises ABOVE the local water surface (m).
                                   # Density fades to 0 over this height in the Blender volume.
                                   # VE-scaled like everything else, so at VE 5 a 500 m plume
                                   # stands 2.5 km tall in the scene - which is what it takes to
                                   # read next to a 1.8 km drop.

# --- CASCADE RELIEF: REAL 3D GEOMETRY ON THE FALLING WATER ---
# The single reason the cascade never read as a waterfall: the flood sheet is a
# HEIGHTFIELD (one z per cell), so the falling water could only ever be *painted*
# on. Foam and streaks were bump/colour, and at this scale a bump on a smooth sheet
# just returns the sky -> the pale amorphous blobs on the canyon wall.
#
# So bake the cascade's surface structure as an actual signed RELIEF in metres and
# let Blender DISPLACE the sheet with it (second Displace modifier). Two components,
# both keyed to the flow field we already have:
#   rollers  - standing waves TRANSVERSE to the flow (phase = distance along the
#              route): the stacked breaking crests of a cataract.
#   strands  - LIC noise smeared ALONG the flow: the longitudinal water ropes.
# Gated to genuinely FALLING water by the water-surface slope, so the pooled lake
# upstream stays glassy and only the chute churns.
#
# Exported as its own PNG (flood_relief_16bit.png, signed, +-MEGAFLOOD_RELIEF_AMP_M)
# rather than being folded into flood_surface_16bit.png: that lets Blender retune the
# amplitude (CASCADE_RELIEF_SCALE) without re-running the ~10 min merge.
MEGAFLOOD_CASCADE_RELIEF   = True   # False = old flat sheet (colour/bump only)
MEGAFLOOD_CASCADE_GATE_SMOOTH_M = 1500.0  # THE key knob. The gate must key on the LARGE-SCALE
                                    # fall of the water surface, so smooth it this much before
                                    # taking the gradient. Without it (v11-v13) the gate fired on
                                    # the racing thin sheet out on the approach flats: there the
                                    # surface hugs the bed (zw_thin = bed + FILL_DEPTH), so it
                                    # inherits the bed's +-20 m bumps and the LOCAL slope tops
                                    # 0.06 on pure noise - stamping regular rollers across a flat
                                    # bay that should be glassy. At 1.5 km the bumps average out
                                    # and only real fall survives: the chute drops 1.8 km over
                                    # 13 km (~0.14), the flats ~0.
MEGAFLOOD_CASCADE_SLOPE_ON = 0.035  # large-scale surface slope (m/m) where the relief starts
MEGAFLOOD_CASCADE_SLOPE_FULL = 0.120  # ...and reaches full amplitude (the plunging chute)
MEGAFLOOD_ROLLER_LAMBDA_M  = 700.0  # standing-wave wavelength (m). Must stay well above the
                                    # Blender quad size (scene_width / SUBDIVISIONS ~ 38 m at
                                    # 92 km / 2400) or the mesh cannot resolve the crests and
                                    # they alias into noise.
MEGAFLOOD_ROLLER_AMP_M     = 55.0   # crest-to-trough/2 of the rollers (m). Sounds huge, and is:
                                    # the chute is 13 km long and 1.8 km deep, so a 20 m wave is
                                    # invisible. This is the "reads from orbit" exaggeration.
MEGAFLOOD_ROLLER_JITTER    = 0.60   # 0 = perfectly regular corduroy (reads as a machined ramp);
                                    # ~0.6 breaks the phase into irregular natural wave trains
MEGAFLOOD_ROLLER_HARMONIC  = 0.55   # amplitude of a SECOND, shorter roller train (lambda/1.9)
                                    # with its own independent jitter field. One sine wave over
                                    # an EDT-propagated 'along' field reads as nested contour
                                    # lines on a map; a decorrelated second train destroys that
                                    # periodicity. 0 = single train (the old look).
MEGAFLOOD_ROLLER_PATCHINESS = 0.55  # 0 = one even wave amplitude over the whole chute (reads as
                                    # machined). ~0.55 gives zones of stacked standing waves and
                                    # smoother glassy tongues between them, like a real cataract.
MEGAFLOOD_STRAND_AMP_M     = 32.0   # amplitude of the flow-parallel water ropes (m)
MEGAFLOOD_STRAND_LEN_M     = 2600.0 # strand length (m): long ropes down the fall line
MEGAFLOOD_STRAND_LANE_M    = 190.0  # strand WIDTH (m). Narrower than the foam lanes (260) so
                                    # the geometry reads as many ropes, not a few fat tubes.
MEGAFLOOD_RELIEF_DEPTH_FRAC = 0.60  # a trough may cut at most this fraction of the local water
                                    # depth: without it the wave troughs punch the sheet through
                                    # the thin racing film and the bed shows through as holes.

# --- LAKE SWELL: real geometry on the CALM water too ---
# The cascade relief is gated to the chute, which leaves the ponded flood a geometrically FLAT
# plane. At a grazing camera angle a flat plane reflects the uniform sky whatever its bump
# normal says, so the upstream lake rendered as a milky white slab next to a deep-blue Ionian -
# two water bodies, two materials. This is the same lesson the sea already learned (see
# SEA_WAVE_GEO in render_waterfall.py): at this scale you need REAL displacement.
# Rides on the existing relief map, so the depth clip below protects the waterline for free.
MEGAFLOOD_LAKE_SWELL_M     = 9.0   # swell amplitude (m) on ponded/calm flood water. 0 = flat
                                   # (the milky slab). VE-scaled in Blender like everything else.
MEGAFLOOD_LAKE_SWELL_KM    = 2.5   # swell wavelength (km): broad ocean rollers, not chop
MEGAFLOOD_LAKE_SWELL_DEPTH_M = 60.0  # fade the swell out in water shallower than this. Physically
                                   # right (waves shoal and die) and it keeps the swell from
                                   # chewing a ragged waterline into the shallow margins.
MEGAFLOOD_RELIEF_AMP_M     = 120.0  # normalisation half-range of the exported PNG (m). Must
                                    # exceed the real max relief or the crests clip flat.
MEGAFLOOD_CREST_FOAM       = 0.90   # whitewater ON THE ROLLER CRESTS. Without this the cascade
                                    # has the right geometry but the wrong colour: the foam field
                                    # is computed from the water-surface slope and knows nothing
                                    # about the crests, so the chute rendered as BLUE corrugations.
                                    # Keying foam to positive relief whitens the crests and leaves
                                    # the troughs blue - which is exactly what breaking water does.
                                    # 0 = off (foam from slope only).

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
MEGAFLOOD_FILL_DEPTH_M = 70.0   # RAISED from 45: this is the head over the sill NOTCH, and
                                # downstream of the sill it is the SHEET THICKNESS pouring over
                                # the lip - so it, not the corridor width, is the knob for "more
                                # water in the canyon". 45 fed the canyon only in thin threads
                                # (the flood mostly stood on the plateau); 70 sends a solid
                                # sheet over the escarpment. The shoreline stays terrain-cut
                                # because EDGE_DROP is steep again. (Below ~30 the braid bars
                                # dominate the sill crossing and the water reads as vanishing.)

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
MEGAFLOOD_SHEET_TERRAIN_SMOOTH_M = 340.0
# Final light smooth of the water surface itself (m): rounds residual terrace steps so
# foam, colour and the 3D sheet read as one coherent moving body. Keep gentle (100-160).
MEGAFLOOD_SURFACE_SMOOTH_M       = 240.0
MEGAFLOOD_WATERLINE_FEATHER_M  = 10.0   # near-binary: water paints FULL blue close to
                                        # its edge - much larger and wide shallow
                                        # margins render as milky semi-transparent film.

# Smooth the water DEPTH used for the shoreline/extent by this many metres (mask-only:
# the terrain heightmap arr is NOT touched, so the canyon/ridges stay sharp). The
# interpolated MES has contour terraces, so depth = surface - terrain crosses zero in a
# sawtooth -> jagged pixel shorelines and stipple holes. Smoothing the EXTENT (not the
# geometry) gives one clean water body. 0 = raw (jagged). 200-400 m is the useful range.
MEGAFLOOD_EXTENT_SMOOTH_M = 780.0
# Radius (m) up to which dry speckle / small islands inside the flood are closed so the
# body reads solid (not a mottled leopard pattern in the shallow terraced bay). Larger
# also swallows genuine small islands; ~600 keeps big ridges, kills terrace speckle even
# in the wide shallow sill crossing.
MEGAFLOOD_HOLE_FILL_M = 700.0

# Shore->deep colour: shallow water is lighter, deep water tends to the navy. Scale (m)
# over which the flood colour deepens with water depth.
MEGAFLOOD_DEEP_SCALE_M = 220.0    # shore->deep colour scale; higher keeps the turbid
                                  # tone visible over deep water, lower goes navy fast.
                                  # 220: the 20-40 m sill passage reads as solid water
                                  # (at 550 it painted as pale milky film)
# Smooth the water depth used for the shore->deep COLOUR ramp (m). Wide so broad MES
# terraces don't mottle the shallow bay light/dark; the large-scale shore->deep gradient
# survives. Colour only - extent/alpha use the raw depth.
MEGAFLOOD_COLOR_DEPTH_SMOOTH_M = 900.0

# Clean Atlantic -> turbid gradient along the corridor (False = uniform turbid).
MEGAFLOOD_TURBID_RAMP = True

# Also carve the corridor DOWN into the heightmap so the 3D render shows an incised
# scour megachannel. 0 = paint colour only, no geometry change (safe default).
# A value like 150-400 m gives a visible flood-scoured trough. Tapered by corridor weight.
MEGAFLOOD_SCOUR_M = 0.0   # back to 0 (190 was tried). The scour lowers the terrain TAPERED BY
                          # THE CORRIDOR WEIGHT, i.e. it prints the corridor polygon into the
                          # DEM: with 190 m, 78 % of the shoreline ended up sitting on the scour
                          # ramp rather than on real topography. That is a data-falsifying edit
                          # of a terrain that is supposed to BE the data - and it did not even
                          # buy the incised look (removing it changed the shoreline not at all).
                          # If a scoured megachannel is ever wanted, mask it to the CONFINED
                          # canyon reach instead of applying it along the whole route envelope.

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

    # ORGANIC RIM: wobble the half-width with a smooth noise field so the corridor edge is
    # no longer an exact offset of the polyline. Without this, the water surface sinking
    # linearly beyond a perfectly-offset rim cuts FLAT ground along a ruler-straight line -
    # the "Kante" in the regional shot. Modulating the half-width makes the rim meander, so
    # the cut meanders too. Applied to halfw_m, hence to BOTH weight and over_m.
    if MEGAFLOOD_EDGE_WIGGLE > 0:
        wig = _coarse_noise((H, W), MEGAFLOOD_EDGE_WIGGLE_KM * 1000.0, pixel_m,
                            np.random.default_rng(31337))
        halfw_m = np.clip(halfw_m * (1.0 + MEGAFLOOD_EDGE_WIGGLE * wig),
                          0.0, None).astype("float32")
        del wig

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


def _coarse_noise(shape, sigma_m, pixel_m, rng, oversample=4.0):
    """Zero-mean, unit-sd smooth noise field - built COARSE and zoomed up.

    Every noise field we use here (corridor-rim wiggle, roller phase jitter, wave
    patchiness, lake swell) has a correlation length of 1-12 KILOMETRES, yet the grid is
    15-25 m. Generating them at full resolution is not just wasteful, it is fatal: on the
    regional grid each array is 342 MB and the merge died with a MemoryError. Sampling the
    field at ~sigma/oversample and interpolating up is visually identical and ~100x cheaper.
    Falls back to full resolution when the field really is fine-scale.
    """
    step = max(int(sigma_m / max(pixel_m, 1e-6) / oversample), 1)
    if step == 1:
        n = gaussian_filter(rng.random(shape, dtype=np.float32),
                            max(sigma_m / pixel_m, 0.5)).astype("float32")
        return ((n - n.mean()) / (n.std() + 1e-9)).astype("float32")

    small = ((shape[0] + step - 1) // step + 2, (shape[1] + step - 1) // step + 2)
    n = gaussian_filter(rng.random(small, dtype=np.float32),
                        max(sigma_m / (pixel_m * step), 0.5)).astype("float32")
    n = ((n - n.mean()) / (n.std() + 1e-9)).astype("float32")
    from scipy.ndimage import zoom
    n = zoom(n, (shape[0] / (small[0] - 2.0), shape[1] / (small[1] - 2.0)),
             order=1, prefilter=False).astype("float32")
    return np.ascontiguousarray(n[:shape[0], :shape[1]])


def _save_u8_rgb(path, arr01):
    u8 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, u8)
    except Exception:
        from PIL import Image
        Image.fromarray(u8).save(path)


def _save_u8_gray(path, arr01):
    _save_u8_rgb(path, arr01)          # imageio/PIL both take a 2-D uint8 array as L


# filled by write_colormap (the spray plume's bounding box + z-range) and merged into
# heightmap_meta.json by finalize(): Blender needs the box to build the volume in, and
# scanning the 27 Mpx mask from bpy would mean copying ~110 M floats.
_MIST_META = {}


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


def _flow_streaks(zwater, pixel_m, length_m, seed=12345, lane_m=None):
    """Line-integral-convolution style streak texture: white noise smeared along
    the water-surface gradient (the flow direction). 0..1, mean ~0.5.

    lane_m overrides MEGAFLOOD_STREAK_LANE_M: the cascade RELIEF wants narrower
    lanes (many ropes) than the foam COLOUR does (broad current bands).

    COMPUTED ON A DECIMATED GRID and zoomed back up. The lanes are 200-3500 m wide but
    the grid is 15-25 m, so a full-resolution LIC is heavily oversampled - and it is not
    merely wasteful: the routine needs ~8 simultaneous full-size arrays (flow field, two
    coordinate grids, accumulator, map_coordinates temporaries), which on the regional
    grid is ~3 GB and OOM-killed the merge. Decimating so a lane is ~4 px is visually
    identical and an order of magnitude cheaper.
    """
    from scipy.ndimage import map_coordinates, zoom
    lane_full = (lane_m if lane_m else MEGAFLOOD_STREAK_LANE_M)
    dec = max(int(lane_full / max(pixel_m, 1e-6) / 3.0), 1)   # ~3 px per lane is plenty
    Hf, Wf = zwater.shape
    if dec > 1:
        zw = np.ascontiguousarray(zwater[::dec, ::dec])
        px = pixel_m * dec
    else:
        zw = zwater
        px = pixel_m

    H, W = zw.shape
    gy, gx = np.gradient(zw, px)
    mag = np.hypot(gx, gy) + 1e-9
    fx = (gx / mag).astype("float32")
    fy = (gy / mag).astype("float32")
    del gy, gx, mag
    rng = np.random.default_rng(seed)
    # pre-smooth the base noise so the smeared result is BROAD current bands
    # (~lane_m wide) rather than 1-px hairline scratches. Per-pixel white noise
    # smeared along the flow gave pencil-hatching that read as scribbles, not
    # water; broad lanes read as a moving surface. The std-normalisation below
    # restores contrast, so the lanes stay legible - we then apply them at low gain.
    lane_px = max(lane_full / px, 0.7)
    noise = gaussian_filter(rng.random((H, W), dtype=np.float32),
                            lane_px).astype("float32")
    # coordinate grids as BROADCAST 1-D arrays, not a full mgrid: `yy + s*fy` still yields
    # the (H, W) array map_coordinates wants, but the base grids cost O(H+W) instead of
    # two more full-size arrays (684 MB on the regional grid).
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    n_steps = max(int(length_m / px / 2), 1)
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
    del yy, xx, fx, fy, noise
    acc /= cnt
    acc = (acc - acc.mean()) / (acc.std() + 1e-9)
    out = np.clip(0.5 + 0.5 * acc, 0.0, 1.0).astype("float32")
    del acc
    # gentle post-smooth kills any residual per-step aliasing along the strands
    out = gaussian_filter(out, max(lane_px * 0.5, 0.5)).astype("float32")
    if dec > 1:
        out = zoom(out, (Hf / out.shape[0], Wf / out.shape[1]),
                   order=1, prefilter=False).astype("float32")
        out = np.ascontiguousarray(out[:Hf, :Wf])
        if out.shape != (Hf, Wf):        # zoom can land a pixel short
            out = np.pad(out, ((0, Hf - out.shape[0]), (0, Wf - out.shape[1])),
                         mode="edge")
    return out


def megaflood_cascade_relief(zwater, along, depth, pixel_m, route_len_m, gate,
                             seed=4242):
    """Signed 3D relief (metres) for the falling water: transverse standing-wave
    rollers + flow-parallel strands, gated to genuinely FALLING water.

    This is what turns the cascade from a painted blue wall into geometry. Blender
    displaces the flood sheet with it, so the chute has real crests that catch the
    sun and real troughs that go into shadow.

    `gate` (0..1) MUST come from the pre-edge-drop surface (see compute_megaflood):
    keying it to the final surface fires the relief along every shoreline.
    """
    if not MEGAFLOOD_CASCADE_RELIEF or gate is None:
        return None

    # ROLLERS: phase = distance travelled along the route, so the crests stand
    # TRANSVERSE to the flow (which is what a standing wave train in a chute does).
    # 'along' is normalised arc length 0..1, so along * route_len = metres downstream.
    along_m = np.nan_to_num(along, nan=0.0) * float(route_len_m)
    lam = max(MEGAFLOOD_ROLLER_LAMBDA_M, 1e-6)
    rng = np.random.default_rng(seed + 1)

    def _smooth_noise(sigma_m, sd):
        """Zero-mean, unit-sd smooth noise field (coarse-generated; see _coarse_noise)."""
        return _coarse_noise(zwater.shape, sigma_m, pixel_m, rng)

    def _wave(wavelength, jitter_octaves):
        """One roller train at `wavelength`, phase-jittered by its own noise."""
        ph = along_m.copy()
        if MEGAFLOOD_ROLLER_JITTER > 0:
            jit = np.zeros_like(ph)
            for sig_mult, gain in jitter_octaves:
                jit += gain * _smooth_noise(wavelength * sig_mult, None)
            ph += MEGAFLOOD_ROLLER_JITTER * wavelength * jit
            del jit
        return np.sin(2.0 * np.pi * ph / wavelength).astype("float32")

    # TWO harmonics with INDEPENDENT jitter fields. A single sin(along) is a perfectly
    # periodic corduroy, and because 'along' is an EDT-propagated field whose iso-lines
    # fan around the polyline, that corduroy reads as nested contour lines on a map -
    # the single most artificial thing left in the cascade. A second, shorter train that
    # is decorrelated from the first destroys the periodicity: crests merge, cancel and
    # rebuild the way real standing waves do.
    rollers = _wave(lam, ((2.5, 1.0), (0.8, 0.5)))
    rollers += MEGAFLOOD_ROLLER_HARMONIC * _wave(lam / 1.9, ((2.0, 1.0), (0.7, 0.6)))
    rollers /= (1.0 + MEGAFLOOD_ROLLER_HARMONIC)
    del along_m

    # STRANDS: LIC noise smeared along the flow -> longitudinal ropes of water
    strands = (_flow_streaks(zwater, pixel_m, MEGAFLOOD_STRAND_LEN_M, seed=seed,
                             lane_m=MEGAFLOOD_STRAND_LANE_M) - 0.5) * 2.0

    # PATCHINESS: a real cataract is not uniformly rough. It has zones where big standing
    # waves stack up and smooth glassy tongues in between. Without this the chute has one
    # even amplitude everywhere, which is exactly what reads as "machined".
    patch = 0.5 + 0.5 * _smooth_noise(lam * 4.0, None)            # ~0..1, broad
    patch = np.clip(1.0 - MEGAFLOOD_ROLLER_PATCHINESS
                    + MEGAFLOOD_ROLLER_PATCHINESS * 2.0 * patch, 0.0, 1.6)
    # ...and a finer modulation from the strand field, so individual wave trains die out
    roll_amp = MEGAFLOOD_ROLLER_AMP_M * patch * (0.55 + 0.45 * np.abs(strands))
    relief = (gate * (roll_amp * rollers
                      + MEGAFLOOD_STRAND_AMP_M * strands)).astype("float32")
    del rollers, strands, roll_amp, patch

    # LAKE SWELL on the water the cascade gate does NOT cover: without it the ponded flood is
    # a geometrically flat plane, and a flat plane at a grazing angle returns the uniform sky
    # no matter what its bump normal says -> the milky white slab beside a blue Ionian.
    if MEGAFLOOD_LAKE_SWELL_M > 0:
        swell = _smooth_noise(MEGAFLOOD_LAKE_SWELL_KM * 1000.0 * 0.5, None)
        calm = np.clip(1.0 - gate, 0.0, 1.0)                  # only where it is NOT cascading
        shoal = np.clip(depth / max(MEGAFLOOD_LAKE_SWELL_DEPTH_M, 1e-6), 0.0, 1.0)
        relief = (relief + MEGAFLOOD_LAKE_SWELL_M * swell * calm * shoal).astype("float32")
        del swell, calm, shoal

    # a trough must never punch the sheet through its own bed: on the thin racing film
    # the water is only tens of metres deep, and an unclipped 55 m trough would open
    # holes with tan canyon floor showing through the whitewater.
    lim = (MEGAFLOOD_RELIEF_DEPTH_FRAC * depth).astype("float32")
    relief = np.maximum(relief, -lim).astype("float32")
    del lim
    return relief


def megaflood_mist(foam, gate, zwater, coverage, pixel_m):
    """Spray density 0..1 for the 3D mist plume.

    Two sources, because a cataract makes spray in two different ways:
      - the whole churning chute atomises water as it breaks (foam x cascade gate);
      - the IMPACT, where the fall slams into the receiving sea, throws far more of it.
    Blurring the sum gives a cloud that billows out beyond the water that made it.
    """
    if not MEGAFLOOD_MIST_MASK or gate is None:
        return None
    churn = (foam * gate).astype("float32")
    # Impact zone. The naive `churn * near_sea` is ZERO exactly where we want the plume:
    # by the time the cascade reaches the sea its surface is flat on the sea plane, so
    # the cascade gate (and therefore churn) has already fallen to 0 there. The spray at
    # the foot of a fall is not made by the water that is *already* in the plunge pool -
    # it is made by the water arriving from above. So spread the churn spatially FIRST,
    # then ask which of that spread-out churn has landed at sea level.
    churn_near = gaussian_filter(churn, 3000.0 / pixel_m).astype("float32")
    near_sea = np.clip(1.0 - (zwater - PALEO_SEA_LEVEL_M)
                       / max(MEGAFLOOD_MIST_SEA_BAND_M, 1e-6), 0.0, 1.0).astype("float32")
    src = (MEGAFLOOD_MIST_CHUTE_GAIN * churn
           + MEGAFLOOD_MIST_PLUNGE_GAIN * churn_near * near_sea)
    del churn, churn_near, near_sea
    src *= (coverage > 0)

    mist = gaussian_filter(src, MEGAFLOOD_MIST_SPREAD_M / pixel_m).astype("float32")
    del src
    # normalise on a high percentile of the non-zero field, NOT the max: a single hot
    # pixel would otherwise wash the whole plume out to nothing
    nz = mist[mist > 1e-4]
    ref = float(np.percentile(nz, 98)) if nz.size else 0.0
    if ref <= 1e-6:
        return None
    return np.clip(mist / ref * MEGAFLOOD_MIST_GAIN, 0.0, 1.0).astype("float32")


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
        return weight, along, zwater, depth, foam.astype("float32"), None, None, None

    pts = _megaflood_points_px(arr.shape, transform)
    halfw_px = _megaflood_halfwidths_px(pixel_m, len(pts))
    rr, cc, tt, _ = _densify_polyline(pts, halfw_px)
    ri = np.clip(np.round(rr).astype(int), 0, H - 1)
    ci = np.clip(np.round(cc).astype(int), 0, W - 1)
    # physical length of the route: turns the normalised 'along' 0..1 into metres
    # downstream, which is the phase of the cascade's standing-wave rollers
    route_len_m = float(np.hypot(np.diff(rr), np.diff(cc)).sum() * pixel_m)
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

    # CASCADE GATE for the 3D relief - must be taken HERE, from the PRE-edge-drop
    # surface. Beyond the corridor rim the surface is sunk by EDGE_DROP_PER_KM
    # (0.13 m/m), which is far steeper than CASCADE_SLOPE_FULL: a gate keyed to the
    # FINAL surface therefore fires along every shoreline and stamped regular roller
    # bands across the calm upstream lake (the "ripple tank" in hero_v11). Damped by
    # pooledness as well, so only genuinely RACING water churns - a ponded lake behind
    # the sill has a flat surface and must stay glassy.
    cascade_gate = None
    if MEGAFLOOD_CASCADE_RELIEF:
        # LARGE-SCALE fall only: smooth the surface before differentiating, or the
        # thin racing sheet's bed-inherited bumps out on the flats read as "steep"
        # (see MEGAFLOOD_CASCADE_GATE_SMOOTH_M).
        zw_big = gaussian_filter(zwater,
                                 MEGAFLOOD_CASCADE_GATE_SMOOTH_M / pixel_m).astype("float32")
        gyc, gxc = np.gradient(zw_big, pixel_m); del zw_big
        slc = np.hypot(gxc, gyc); del gyc, gxc
        span_c = max(MEGAFLOOD_CASCADE_SLOPE_FULL - MEGAFLOOD_CASCADE_SLOPE_ON, 1e-6)
        cascade_gate = np.clip((slc - MEGAFLOOD_CASCADE_SLOPE_ON) / span_c,
                               0.0, 1.0).astype("float32")
        del slc
        cascade_gate *= np.clip(1.0 - pooledness, 0.0, 1.0).astype("float32")
        cascade_gate = gaussian_filter(cascade_gate,
                                       300.0 / pixel_m).astype("float32")

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
    #
    # SMOOTH THE *SIGNED* DEPTH, NOT THE CLIPPED ONE. This was a real bug, not a tuning
    # value. clip(zwater-arr, 0, None) is one-sided, so blurring it with a 780 m Gaussian
    # BLEEDS the water's depth outward onto dry ground - and the deeper the water at the
    # shore, the further it bleeds. Against a steep bank (the river now runs along the
    # Sicilian slope, where it is 40-60 m deep right at the waterline) that pushed
    # waterline = depth_ext/10 m to a saturated 1.0 for ~1 km UP THE DRY HILLSIDE. The
    # sheet was then drawn coplanar with the terrain there: a translucent, low-roughness
    # water film lying on the slope, which at a grazing sun flashed white - the bright
    # glassy "fold" along the bank.
    #
    # The signed field cancels: dry ground above the surface is strongly negative and pays
    # for the deep water's positive bleed, so the blur moves the zero CONTOUR (which is all
    # it was ever meant to do - de-jag the terraced shoreline) instead of inflating it.
    if MEGAFLOOD_EXTENT_SMOOTH_M and MEGAFLOOD_EXTENT_SMOOTH_M > 0:
        depth_ext = gaussian_filter((zwater - arr).astype("float32"),
                                    MEGAFLOOD_EXTENT_SMOOTH_M / pixel_m).astype("float32")
    else:
        depth_ext = zwater - arr
    waterline = np.clip(depth_ext / feather, 0.0, 1.0)       # soft shoreline (0 at edge -> 1)
    # and never paint water on ground that is genuinely above the water surface
    waterline *= (depth > 0.0)

    # waterline gives the organic terrain shoreline (with the sinking rim surface);
    # the envelope guarantees ZERO flood beyond halfwidth+feather. The racing-sheet
    # sections (non-pooled: cascade, plume) use their own tight feather so the
    # falling jet stays in the canyon instead of curtaining along the escarpment.
    w_falls = np.clip(1.0 - over_m / max(MEGAFLOOD_FALLS_FEATHER_KM * 1000.0, 1e-6),
                      0.0, 1.0).astype("float32")
    envelope = (1.0 - falls01) * weight + falls01 * w_falls
    coverage = (envelope * waterline).astype("float32")
    # MEMORY: on the regional grid every one of these is 342 MB, and by this point ~20 of
    # them are live (~7 GB) - which is what OOM-killed the merge. Everything below only
    # needs coverage, so let the envelope's inputs go.
    del w_falls, envelope, weight, over_m, falls01, waterline
    if depth_ext is not depth:
        del depth_ext

    # cascade foam BEFORE the sea cutoff (the mist grace below needs it): white
    # where the water surface drops steeply (falls) or a thin sheet races over a
    # steep bed (the lip). Wide ramp -> graded whitewater, not a saturated blob.
    # Damped by pooledness: no foam on standing lake margins.
    gyw, gxw = np.gradient(zwater, pixel_m)
    s0 = max(MEGAFLOOD_FOAM_SURF_SLOPE, 1e-6)
    foam_raw = np.clip((np.hypot(gxw, gyw) - s0) / (9.0 * s0), 0.0, 1.0).astype("float32")
    del gyw, gxw
    gy, gx = np.gradient(arr_sheet, pixel_m)    # smoothed bed: no per-terrace foam bands
    foam_thin = np.clip((np.hypot(gx, gy) - ZANCLEAN_CASCADE_SLOPE) /
                        max(ZANCLEAN_CASCADE_SLOPE, 1e-6), 0.0, 1.0) \
        * (depth < MEGAFLOOD_FOAM_THIN_DEPTH_M)
    del gy, gx
    np.maximum(foam_raw, foam_thin, out=foam_raw)
    del foam_thin
    foam_raw *= np.clip(1.0 - pooledness, 0.0, 1.0).astype("float32")

    if arr_sheet is not arr:
        del arr_sheet          # the smoothed bed was only needed for zw_thin + foam_thin

    streaks = None
    if MEGAFLOOD_FOAM_STREAKS:
        # falling-water strands: smear noise along the flow and modulate the foam
        streaks = _flow_streaks(zwater, pixel_m, MEGAFLOOD_STREAK_LEN_M)
        foam_raw *= (MEGAFLOOD_STREAK_MIN
                     + (1.0 - MEGAFLOOD_STREAK_MIN) * streaks).astype("float32")

    # rapids: SPARSE whitecap filaments on racing (non-pooled) water - only the
    # brightest streak lanes turn white. A dense streak**2 wash painted the whole
    # racing section as milky frost instead of white threads on blue water.
    if MEGAFLOOD_RAPIDS_FOAM > 0 and streaks is not None:
        lane = np.clip((streaks - 0.55) / 0.45, 0.0, 1.0) ** 2
        rapids = (MEGAFLOOD_RAPIDS_FOAM * np.clip(1.0 - pooledness, 0.0, 1.0)
                  * lane).astype("float32")
        foam_raw = np.maximum(foam_raw, rapids)
        del lane, rapids
    del pooledness             # last use


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
    # ONLY INWARDS. grey_closing is extensive (cov_solid >= coverage everywhere), so taking
    # its result on the OUTER shore would push the waterline up the bank by the closing
    # radius - the same class of mistake as blurring a one-sided depth (above), just smaller.
    # Restrict it to cells that are genuinely enclosed by water: the interior speckle and the
    # terrace-top islands it exists to close. The outer shoreline is left exactly as the
    # terrain drew it.
    wetb = coverage > 0.15
    filled = binary_fill_holes(wetb) | wetb
    newly = filled & (cov_solid > coverage)
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
    del streaks                      # the foam LIC is consumed; free 342 MB before the next
    body_streaks = None
    if MEGAFLOOD_BODY_STREAK_GAIN > 0:
        body_streaks = _flow_streaks(zwater, pixel_m, MEGAFLOOD_BODY_STREAK_LEN_M,
                                     seed=777)

    # REAL cascade geometry (see MEGAFLOOD_CASCADE_RELIEF). The flow field and the
    # trough clip come from the FINAL surface/depth (what Blender actually displaces),
    # but the GATE is the pre-edge-drop one built above - see its comment.
    relief = megaflood_cascade_relief(zwater, along, depth, pixel_m, route_len_m,
                                      cascade_gate)
    if relief is not None:
        relief = (relief * (coverage > 0)).astype("float32")
        # WHITEWATER ON THE CRESTS. foam above is keyed to the water-surface slope and
        # knows nothing about the rollers, so the chute came out as blue corrugations.
        # Breaking water is white where it crests and blue in the trough - so key the
        # foam to the positive relief. Uncapped by FOAM_MAX on purpose: a crest IS white.
        if MEGAFLOOD_CREST_FOAM > 0:
            crest = np.clip(relief / max(MEGAFLOOD_ROLLER_AMP_M, 1e-6), 0.0, 1.0)
            foam = np.maximum(foam, (MEGAFLOOD_CREST_FOAM * crest
                                     * (coverage > 0)).astype("float32"))
            del crest

    # spray plume (uses the FINAL foam, so the crest whitewater feeds it)
    mist = megaflood_mist(foam, cascade_gate, zwater, coverage, pixel_m)
    return coverage, along, zwater, depth, foam, body_streaks, relief, mist


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
        coverage, along, zwater, depth, foam_w, body_streaks, relief, mist = \
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

        # composite IN PLACE and in float32: the regional grid is 8000x11200, so a float64
        # RGB temporary here is 2 GB and blew the merge up. rgb += w*(cor-rgb) is the same
        # lerp with one temporary instead of three.
        cor_rgb = cor_rgb.astype("float32", copy=False)
        rgb = rgb.astype("float32", copy=False)
        w3 = coverage[..., None].astype("float32", copy=False)
        cor_rgb -= rgb
        cor_rgb *= w3
        rgb += cor_rgb
        del cor_rgb, w3, watercol

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

            # cascade relief: SIGNED metres, normalised over +-MEGAFLOOD_RELIEF_AMP_M so
            # 0 m sits at 0.5. Blender adds it with a second Displace (mid_level 0.5), so
            # the amplitude can be retuned there without re-running the merge. Kept OUT of
            # the surface PNG above on purpose - the surface must stay the smooth water
            # LEVEL that the colour/extent logic agrees with.
            if relief is not None:
                amp = max(float(MEGAFLOOD_RELIEF_AMP_M), 1e-6)
                rmax = float(np.nanmax(np.abs(relief))) if relief.size else 0.0
                if rmax > amp:
                    print(f"  ! cascade relief {rmax:.0f} m exceeds "
                          f"MEGAFLOOD_RELIEF_AMP_M={amp:.0f} m - crests will clip flat")
                _save_u16_norm(os.path.join(OUT_DIR, OUT_FLOOD_RELIEF_PNG),
                               relief, -amp, amp)
                print(f"  cascade relief: max {rmax:.0f} m "
                      f"(rollers {MEGAFLOOD_ROLLER_AMP_M:.0f} m / "
                      f"strands {MEGAFLOOD_STRAND_AMP_M:.0f} m)")

            # spray plume: the mask plus the BOX Blender should build the volume in.
            # Blender cannot scan a 27 Mpx mask itself (bpy image pixel access would
            # copy ~110 M floats), so hand it the bounding box and the z-range here.
            if mist is not None:
                # DOWNSAMPLE the mask before writing. Blender expands every image to float
                # RGBA internally - at 6134x4533 that is ~445 MB per image, and the scene
                # already carries five of them. The mist is a field blurred with a ~1.8 km
                # sigma at 15 m pixels, i.e. wildly oversampled, so an 8x decimation is
                # visually free and saves ~430 MB. (Full-res here OOM-killed the render.)
                step = max(int(MEGAFLOOD_MIST_SPREAD_M / pixel_m / 4), 1)
                _save_u8_gray(os.path.join(OUT_DIR, OUT_FLOOD_MIST_PNG),
                              mist[::step, ::step])
                rows, cols = np.nonzero(mist > 0.05)
                if rows.size:
                    H, W = mist.shape
                    r0, r1 = int(rows.min()), int(rows.max()) + 1
                    c0, c1 = int(cols.min()), int(cols.max()) + 1
                    # z-range from the EXPORTED surface, not raw zwater: the blur carries
                    # the mist beyond the flood coverage, and out there raw zwater is the
                    # edge-dropped profile, which runs hundreds of metres below the terrain
                    # (it reported a -3950 m plume base). `surf` is what the shader samples.
                    zw_m = surf[r0:r1, c0:c1][mist[r0:r1, c0:c1] > 0.05]
                    _MIST_META.update(
                        flood_mist_png=OUT_FLOOD_MIST_PNG,
                        # bbox as fractions of the grid: u = col/ncols, v = row/nrows
                        # (v measured from the NORTH edge, like the raster)
                        mist_bbox_uv=[c0 / W, r0 / H, c1 / W, r1 / H],
                        mist_z_base_m=float(zw_m.min()),
                        mist_z_top_m=float(zw_m.max()) + float(MEGAFLOOD_MIST_HEIGHT_M),
                        mist_height_m=float(MEGAFLOOD_MIST_HEIGHT_M),
                    )
                    print(f"  spray plume: {100.0 * (mist > 0.05).mean():.1f} % of grid, "
                          f"water {zw_m.min():.0f}..{zw_m.max():.0f} m, "
                          f"rises {MEGAFLOOD_MIST_HEIGHT_M:.0f} m")

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
        if MEGAFLOOD_CASCADE_RELIEF:
            meta["flood_relief_png"] = OUT_FLOOD_RELIEF_PNG
            # half-range of the signed relief PNG: elev_m = (u16/65535 - 0.5) * 2 * amp
            meta["flood_relief_amp_m"] = float(MEGAFLOOD_RELIEF_AMP_M)
        meta.update(_MIST_META)        # spray plume mask + its bounding box (may be empty)
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
