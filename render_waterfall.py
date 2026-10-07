#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_waterfall.py  --  Blender scene builder for the palaeo-DEM
=================================================================
Builds a 3D perspective render of the merged palaeo-DEM produced by
merge_paleo_dem.py. Loads the 16-bit heightmap as a displaced grid,
adds elevation-based colour, a desiccated-sea-level water plane, a sun,
an oblique camera, and renders a still.

HOW TO RUN
----------
A) In Blender (GUI):
   1. Open Blender (portable build is fine - no admin needed:
      download the "Portable (.zip)" from blender.org/download, unzip
      anywhere in your user folder, run blender.exe).
   2. Set DATA_DIR below to the folder that contains
      heightmap_16bit.png and heightmap_meta.json.
   3. Switch to the "Scripting" workspace (top tab), open this file
      (Text > Open) or paste it, and click "Run Script" (the ▶ button).
   4. The render appears; it is also saved to OUTPUT_PNG.

B) Headless (e.g. batch / cluster), no GUI:
       blender -b -P render_waterfall.py
   (builds the scene from scratch and writes OUTPUT_PNG)

Tested against Blender 4.x. Engine/material attribute differences between
4.0-4.1 and 4.2+ (Eevee Next) are handled with fallbacks.
"""

import bpy
import os
import sys
import json
import math
import warnings
from mathutils import Vector, Matrix

# =====================================================================
# CONFIG
# =====================================================================
DATA_DIR   = "output_overview"
OUTPUT_PNG = "output_final/render_waterfall.png"
                                     # the submitted Fig-3b panel. The Extended-Data twin is
                                     # the same run with SHOW_WATER / SHOW_FLOOD_SHEET /
                                     # PLUNGE_MIST off and colormap_dry_png as the ground.

# --- console ---
QUIET = True                         # clean output: one progress bar + summary. False = verbose debug logs

# --- terrain ---
SUBDIVISIONS = 2400                  # grid resolution (cuts per side). 600-1000 for drafts;
                                     # raise toward 3200 for finals if the cascade rollers
                                     # (MEGAFLOOD_ROLLER_LAMBDA_M) round into puffy pillows -
                                     # the quad size must resolve their wavelength
VE           = 3                     # regional overview
# VE         = None                  # None = ve_hint from metadata (hero merge writes 5)

# --- camera (oblique "fly-over" view) ---
CAM_AZIMUTH_DEG  = 300.0           # regional overview: oblique from the SE, swung ~10 deg
                                   # more from the E (viewer's right) so the DEM cut-edge
                                   # void behind Etna leaves the frame
CAM_ALTITUDE_DEG = 24.0
CAM_DISTANCE_KM  = None            # None = auto (~1.6 x scene width) = 448 km here
#CAM_AZIMUTH_DEG  = 250.0          # hero: from over the Ionian, looking WSW straight
                                    # at the escarpment wall + the falls in the notch
#CAM_ALTITUDE_DEG = 24.0           # a waterfall is a VERTICAL event: too high and the camera
                                    # looks down ON the chute, foreshortening the 1.8 km drop
                                    # into a blue patch on a slope; too low and the camera sees
                                    # past the DEM's edge into the cut-edge void. 24 keeps the
                                    # drop steep and the data in frame.
#CAM_DISTANCE_KM  = 80.0           # hero: pulled back so the WHOLE canyon fits
CAM_LENS_MM      = 160.0            # regional overview: zoomed into the escarpment/canyon so
                                    # the waterfall reads and the far void goes out of frame

#CAM_LENS_MM    = 88.0            # hero: TIGHT on the cascade. A wider lens fits the whole
                                    # canyon but spends half the frame on the distant approach
                                    # lake - a flat plate that at a low camera angle reads as a
                                    # blown-out white slab with a straight DEM-edge cut. The
                                    # figure is about the FALL, so crop to it. (Zoom by lens, not
                                    # by CAM_DISTANCE_KM: it keeps the perspective compression.)

# --- sun ---
SUN_AZIMUTH_DEG  = 130.0           # regional overview: ESE (roughly BEHIND the camera) so
                                   # the sun's reflection glitters the foreground Ionian
#SUN_AZIMUTH_DEG   = 85.0           # hero: ENE, ~behind the WSW-facing camera -> lights
                                    # INTO the amphitheatre AND glitters the foreground
                                    # Ionian (SE light let the rim shadow cut the falls)
SUN_ALTITUDE_DEG  = 22.0           # low: a long glitter path across the sea
SUN_STRENGTH      = 6.0             # our lamp is the only sun now (sky sun disc off);
                                    # tamed from 8 to curb specular blowout on the glitter

# --- colour rendering ---
VIEW_TRANSFORM = "AGX"               # photoreal: AGX/Filmic give cinematic highlight rolloff; "Standard" = flat/literal
# Use the baked colormap.png from the merge script as the surface colour (shadeless
# emission). This renders exactly as the merge-step preview - the reliable path.
# If False, colour is computed in-Blender from the heightmap via a ColorRamp.
USE_COLORMAP = True

# --- terrain colour ---
# "desiccated": exposed seafloor reads as dry land (tan/brown) - matches the
#               Euphrates Fig.4 desiccated look. "bathymetric": blue sea + green
#               land (classic hypsometric). Both use absolute-elevation stops.
COLOR_SCHEME = "desiccated"

# --- water plane (desiccated lowstand) ---
SHOW_WATER   = True
WATER_COLOR  = (0.03, 0.14, 0.35)   # dark turbid flood blue
WATER_ALPHA  = 0.92                  # nearly opaque – the flood is sediment-laden, not a clear pool
WATER_MARGIN = 2.5                   # plane size vs terrain. LARGE: the sea runs far past
                                     # the terrain edge, so oblique views show water in the
                                     # frame corners instead of the black void

# --- render ---
RENDER_ON_RUN = True
RES_X, RES_Y  = 4800, 3200
SAMPLES       = 256
ENGINE_PREF   = "CYCLES"            # "EEVEE" (fast) or "CYCLES" (slower, photoreal)
BG_COLOR      = (0.66, 0.74, 0.84)  # flat sky colour (used when SKY_TYPE="flat")
BG_STRENGTH   = 0.35               # ambient fill for the flat/emission look

# --- photorealism ---
PHOTOREAL   = True                  # True: real sun + shadows + sky + reflective water (vs flat shadeless map)
SKY_TYPE    = "nishita"             # "nishita" (realistic sun-sky, also lights the scene) | "flat"
SKY_STRENGTH = 0.45                 # sky brightness: ambient fill + water reflection
                                    # (keep LOW - a strong blue sky ambient washes the
                                    # warm terrain tones grey)
EXPOSURE     = -1.0                 # global exposure in stops; lower if washed out, raise toward 0 if too dark
SUN_ANGLE_SOFT_DEG = 1.5            # sun angular size; >0 softens shadow edges (realism)
TERRAIN_ROUGHNESS  = 0.95           # matte ground

# --- PHOTOREAL detail: break the flat "clay model" look with procedural surface texture ---
TERRAIN_DETAIL        = True        # procedural sediment colour variation + micro rock bump
TERRAIN_COLOR_VAR     = 0.36        # +-fraction albedo brightness variation (sediment patches)
TERRAIN_COLOR_SCALE   = 17.0        # patch scale (bigger number = smaller patches)
TERRAIN_BUMP_STRENGTH = 0.55        # micro rock bump (0 = smooth clay); keep subtle
TERRAIN_BUMP_FEATURE_M = 130.0      # GRAIN SIZE IN METRES, fed through Object coordinates (km
                                    # scene units) rather than Generated coordinates - a
                                    # Generated-coordinate frequency normalises over the
                                    # object's bounding box, so the SAME setting gives a
                                    # different physical grain in each panel (930 m regional vs
                                    # 310 m hero) and is anisotropic on a non-square DEM. Those
                                    # are LANDFORM-scale artefacts, not grain: with the rock bump
                                    # multiplier on top they render the escarpment as blisters.
TERRAIN_BUMP_FINE     = 0.45        # weight of a 2nd, 5x finer bump octave. One octave of noise
                                    # has one grain size, and a single grain size IS the plastic
                                    # look; real ground is self-similar. Cheap, big payoff.

# --- SLOPE-KEYED ROCK: the strongest single cure for the clay/plasticine look -------------
# The baked colormap is keyed to ELEVATION ALONE, so the 2 km canyon wall and the flat plain
# 500 m away get the exact same tan. Nothing on the cliff says "rock", so it renders as a
# smooth moulded ramp of the same putty. Real escarpments are stripped to bare rock: darker,
# greyer, glossier when wet-looking, and far more textured than the sediment flats.
#
# The key is the surface NORMAL, i.e. it is derived from the DEM's own geometry - this invents
# no relief, it only colours the relief that the data already has. (It reads the VE-exaggerated
# normal, which is the right choice: it must match what the viewer SEES, not the true gradient.)
TERRAIN_ROCK          = True
TERRAIN_ROCK_COLOR    = (0.30, 0.27, 0.24)   # dry exposed rock: dark warm grey
TERRAIN_ROCK_STRENGTH = 0.62        # max mix toward the rock colour on a vertical face
TERRAIN_ROCK_SLOPE_ON = 0.97        # cos(slope) where rock starts (0.97 = ~14 deg)
TERRAIN_ROCK_SLOPE_FULL = 0.40      # cos(slope) of full rock  (0.40 = ~66 deg)
                                    # WIDE ramp on purpose. A narrow one (0.90 -> 0.55) turns
                                    # the escarpment's own km-scale gullies and spurs into a
                                    # hard threshold, and the cliff comes out as SCALES.
TERRAIN_ROCK_BUMP     = 1.0         # DO NOT BOOST THE BUMP ON ROCK (1.9, then 1.5, both failed).
                                    # ShaderNodeBump differentiates in screen space, and on the
                                    # escarpment the VE-exaggerated quads are nearly vertical and
                                    # hugely stretched - exactly where that derivative is least
                                    # reliable. Multiplying the bump there amplifies the artefact
                                    # at its weakest point: the cliff rendered as blistered
                                    # scales with dark rims. Real cliff texture has to come from
                                    # the DEM, not from a bump on a stretched quad.
TERRAIN_ROCK_ROUGH    = 0.88        # rock roughness, mildly below TERRAIN_ROUGHNESS -> cliffs
                                    # catch a faint sheen the matte flats do not. Keep it MILD:
                                    # a big roughness swing keyed to a noisy normal is itself a
                                    # blotch generator (0.72 scaled the escarpment).

# --- PHOTOREAL atmosphere: distance haze so far terrain fades to sky (depth + scale cue) ---
# Applied IN THE SHADER: each surface is mixed toward a haze colour by its camera distance
# (a MixShader driven by Camera Data -> View Distance). This is the only method that works
# here - the Mist compositor pass doesn't apply headless, and a world Volume Scatter turns
# the km-scale scene black (the sky at infinity is fully extinguished). Distance-correct,
# so the same settings suit both the wide overview and the close hero.
ATMOSPHERE       = True             # the single biggest realism win for a wide landscape
HAZE_COLOR       = (0.62, 0.72, 0.86)   # haze tint - match the sky horizon (blue-grey)
HAZE_STRENGTH    = 0.55             # max haze mix on the farthest terrain (0..1). Keep it low
                                    # enough that the facies texture and the dark flood belt
                                    # stay visible - too strong washes the whole frame grey.
HAZE_EMIT_STRENGTH = 1.1            # haze brightness (match the horizon sky so terrain blends in)
# ABSOLUTE camera-distance thresholds (km), NOT relative to the shot: the same values keep
# the close hero (terrain ~35-105 km) fully crisp while the wide overview (terrain ~310-590
# km) keeps its NEAR ground crisp and only the FAR mountains recede into haze. START must
# sit near the overview's near edge (~310 km) or the whole overview washes out; the hero
# is entirely nearer than START so it stays haze-free. Lower START to push haze closer.
HAZE_START_KM    = 430.0           # REGIONAL overview: no haze nearer than this (near-ground
                                   # crisp) - must sit beyond where the overview terrain starts
                                   # (~380 km from camera) or the foreground hazes too
HAZE_END_KM      = 800.0           # full HAZE_STRENGTH at/beyond this distance (far mountains)
# HERO: the camera sits ~80 km out, so terrain runs from ~40 km (foreground escarpment) to
# ~140 km (far horizon). Haze from 85 km keeps the falls and the near canyon crisp while the
# far edge of the DEM - the straight cut border with its lit vertical face - dissolves into
# the sky instead of announcing itself as a cliff. Swap the comments for the overview.
# NOTE: leaving the HERO pair active on the 448 km overview camera puts the WHOLE terrain
# beyond HAZE_END -> full haze mix everywhere, i.e. a flat grey wash across the frame.
#HAZE_START_KM    = 85.0
#HAZE_END_KM      = 150.0

# --- cloud shadows (PHOTOREAL only) ---
# A shadow-only plane high above the terrain with a procedural cloud pattern: the
# clouds themselves are invisible (camera/glossy/diffuse rays skip the plane), only
# their shadows land on the ground - the classic aerial/satellite-photo cue. Clear
# zones keep the canyon, the cascade and the flood belt in full sun: mottled light on
# the DATA would read as artefacts, so the clouds only dapple the supporting cast
# (Sicily interior, open Ionian). Zone centres are auto-offset toward the sun so the
# SHADOW (not the cloud) avoids the protected ground.
CLOUD_SHADOWS    = True
CLOUD_ALT_KM     = 12.0             # plane altitude in world km (above VE-scaled peaks)
CLOUD_SCALE_KM   = 38.0             # cloud feature size (km)
CLOUD_COVER      = 0.45             # noise-gate threshold = 1-COVER. The noise Fac is roughly
                                    # Gaussian around 0.5, so thresholds much above 0.6 produce
                                    # near-zero actual cover; ~0.45 (thr 0.55) gives a real
                                    # scattered field.
CLOUD_DENSITY    = 0.65             # max shadow opacity: clouds are translucent, not plates
CLOUD_EDGE_SOFT  = 0.14             # gate ramp half-width: bigger = softer cloud edges
# Clear zones: keep them TIGHT. The 160 mm overview frame is only ~100 km wide, so a
# generous radius/feather blankets the entire visible frame and no shadow survives
# anywhere. Protect only the cascade/bay and two spots of the belt.
CLOUD_CLEAR_KM   = [(13.0, -6.0, 26.0),   # (x, y, radius) world km: canyon head + bay + cascade
                    (-24.0, -15.0, 18.0), # lower belt reach
                    (-58.0, -13.0, 18.0)] # mid belt reach
CLOUD_CLEAR_FEATHER_KM = 14.0       # soft rim of the clear zones

# --- scale bars: dashed height/width reference sticks, ONLY near the waterfall. The ONLY
# annotation baked into the render itself - geological place-names
# and the north arrow stay flat 2D overlays (see below), because those don't need true 3D
# placement. The scale bars do: they must read correctly in the render's oblique perspective,
# which a flat line drawn on top afterwards cannot fake.
#
# Elevations are MEASURED off merged_paleo_dem.tif, not eyeballed: rim crest = MEGAFLOOD_PATH
# waypoint 14, base of the fall = waypoint 15. Drop = 1973.6 m, consistent with the
# manuscript's 1.7-2.0 km corrected palaeo-waterfall height window.
SCALE_BARS = True
SCALE_VERTICAL_XY_M    = (532600.0, 4075000.0)             # beside the base of the fall (wp15 x,y)
SCALE_VERTICAL_TOP_M   = (*SCALE_VERTICAL_XY_M, -221.8)     # rim elevation (wp14's z)
SCALE_VERTICAL_BASE_M  = (*SCALE_VERTICAL_XY_M, -2195.4)    # base elevation (wp15's z)
SCALE_VERTICAL_OFFSET_KM = 2.0      # sideways (north, +Y) shift off the corridor centreline so
                                     # the bar sits beside the cascade, not inside the flood sheet
# Amphitheatre falling-front width (measured: flood_corridor R>0.5, -180>elev>-2200 m):
# N-S span ~17.5 km, consistent with the manuscript's ">20 km" measured along the curved rim
# (a straight chord under-reads the curve on purpose - it's what a scale bar draws).
SCALE_HORIZONTAL_X_M   = 519500.0   # near the rim crest, just upstream of the canyon head
SCALE_HORIZONTAL_Y_RANGE_M = (4068800.0, 4086300.0)
SCALE_HORIZONTAL_Z_M   = -221.8     # rim elevation - reads the width at the TOP of the fall
# Sizes are set for the PRINTED figure, not for the 4800 px master. At 180 mm double-column
# width the master is ~27 px/mm, so the old 2 km label came out ~1.4 mm cap height - about
# 4 pt, below the 5-7 pt floor most journals set, and visibly smaller than panel (a)'s
# annotations sitting right above it. Everything here is ~2.5x the first pass.
SCALE_DASH_KM       = 1.6
SCALE_GAP_KM        = 0.8
SCALE_THICKNESS_KM  = 0.32
SCALE_TICK_KM       = 3.0           # end-cap tick length (perpendicular to the bar)
# ...but both of the above are the values for a LONG bar. The two bars differ threefold - the
# amphitheatre is 17.5 km wide, the drop is 2.0 km which at VE 3 is ~6 km of scene - and sized
# for the long one the short one came out as three stubs between two ticks half its own length.
# So each bar shrinks its own dashes to show at least SCALE_DASH_MIN_COUNT of them, and caps
# its ticks at a fraction of its length. Both bars keep the same line WEIGHT (thickness), which
# is what makes them read as one instrument.
SCALE_DASH_MIN_COUNT = 4.0          # fewest dashes a bar may show before they shrink to fit.
                                     # Keep it LOW: each dash carries its own halo outline, so
                                     # squeezing more of them onto the short vertical bar turns
                                     # it into a string of beads instead of a dashed line.
SCALE_TICK_MAX_FRAC  = 0.30         # end tick never exceeds this fraction of the bar length
SCALE_CAMERA_PULL_KM = 3.0          # pull the bars toward the camera along the actual view ray
                                     # so they float clear of intervening terrain (needs
                                     # make_camera() to have already run - see main())
SCALE_COLOR         = (0.03, 0.028, 0.025, 1.0)  # near-black ink. It reads against the foam,
                                     # the blue water and the tan sill - but NOT against the
                                     # shaded escarpment face, which is where the vertical bar
                                     # necessarily stands. That is what ANNOT_HALO is for.

# --- annotation halo: the one thing that makes dark ink survive a dark background -----------
# The vertical bar has to stand at the cascade, and the cascade is a shaded rock face, so on
# the desiccated (Extended Data) render it was dark grey ink on a dark grey slope and all but
# vanished. Making it bigger and blacker does not help: the problem is contrast, not size.
# So every annotation gets a pale outline drawn slightly BEHIND it along the view ray - the
# standard cartographic halo. Text is fattened with the font curve's own outline offset (an
# exact glyph outset, not a scaled copy); bars are rebuilt a little thicker and longer.
ANNOT_HALO          = True
ANNOT_HALO_COLOR    = (0.96, 0.955, 0.94, 1.0)   # warm off-white, not pure white: pure white
                                     # blooms against the AGX highlight rolloff on the sill
# The outsets are FRACTIONS of the thing being outlined, never absolute km. An absolute value
# cannot work here because the annotation layer spans two wildly different scales: the place
# names are kilometres tall while the north arrow's "N" is ~25 m tall on a HUD 5 km from the
# lens. A single 0.14 km outset drew a proportionate halo on the labels and turned the "N"
# into a solid white box and the arrow into a double-line outline drawing.
ANNOT_HALO_TEXT_FRAC = 0.028        # glyph outset as a fraction of the text height. Much more
                                     # and the counters of "~2.0" close up into blobs.
ANNOT_HALO_BAR_FRAC  = 0.5          # bar/arrow outset as a fraction of its own thickness
ANNOT_HALO_DEPTH_FRAC = 0.0015      # push behind the ink, as a fraction of the annotation's
                                     # distance from the camera - so it scales for the 448 km
                                     # scale bars and the 5 km HUD alike. Purely to avoid
                                     # z-fighting; the outset above is what you actually see.
                                     # It must stay well BELOW ANNOT_HALO_TEXT_FRAC: pushing
                                     # the halo back also shrinks it in perspective by the same
                                     # fraction, so a depth larger than the outset makes the
                                     # halo smaller than the ink and it vanishes behind it
                                     # entirely. At 0.0015 the shrink is 0.15 % against a 2.8 %
                                     # outset, while still leaving 0.7 km of clearance at the
                                     # scale bars and ~8 m at the HUD compass.

# --- scale bar km labels: a bare dashed stick reads as decoration without a number next to
# it. Values are NOT re-measured here - they reuse the exact drop_m/width_m this function
# already computes from the measured waypoints above, so the label can never drift out of
# sync with the geometry it labels.
SCALE_LABELS         = True
SCALE_LABEL_HEIGHT_KM = 4.8         # see the print-size note above the dash settings - a
                                     # smaller value prints below the journal's font-size floor
SCALE_LABEL_GAP_KM   = 1.4          # a TRUE gap between the tick end and the nearest ink of
                                     # the text. The label is centred and then slid clear by
                                     # its own measured half-width (see _clear_label), so this
                                     # no longer has to be guessed against the string length -
                                     # which is how the bar previously ran through its own
                                     # number when the text grew.
SCALE_VERTICAL_LABEL   = None       # None = auto "~{drop_m/1000:.1f} km"; string overrides
SCALE_HORIZONTAL_LABEL = "~20 km"   # the manuscript's own wording ("more than 20 km wide"),
                                     # not the bar's own ~17.5 km straight-chord length -
                                     # this only overrides what's PRINTED, not the geometry

# --- geological / topographic place-names: flat billboarded text (always faces the camera),
# same dark ink as the scale bars, only THREE of them. Positions are PROBED against
# merged_paleo_dem.tif and confirmed in-frame via world_to_camera_view, not eyeballed.
# Terms and siting follow the manuscript directly:
#   - "Sicily Sill": the broad near-planar shelf landward of the canyon rim, on dry
#     (desiccated) ground - the whole sill was subaerially exposed during the MSC.
#   - "Hyblean Plateau": the uplifted foreland domain forming Sicily's southern margin -
#     sited on clearly elevated (+779 m) dry terrain near the frame's upper-left edge.
#   - "Ionian Basin": the receiving abyssal plain the flood empties into - sited in open
#     deep water SE of the plunge, clear of the corridor/plume. Excluded from glossy rays
#     (GEO_LABEL_NO_REFLECTION) so it doesn't mirror in the sea below it.
# Deliberately NOT labelled: "Noto Canyon"/"Malta Escarpment". The canyon is the render's own
# subject (a label would compete with the scale bars sitting right on it), and the manuscript
# is explicit that only the Noto Canyon - not the Malta Escarpment generally - carries
# megaflood evidence, so naming the escarpment risks implying the wrong thing.
GEO_LABELS = True
GEO_LABEL_HEIGHT_KM      = 3.6      # kept a step BELOW the scale labels: the place-names are
                                     # orientation, the scale bars carry the measurement, and
                                     # the figure should say which of the two matters more
GEO_LABEL_LIFT_KM        = 0.6      # float above the terrain surface (avoid z-fighting)
GEO_LABEL_CAMERA_PULL_KM = 3.0      # same trick as the scale bars - guarantees the label
                                     # floats clear of intervening terrain on this sightline
GEO_LABEL_NO_REFLECTION  = True     # hide these from glossy/reflection rays; camera rays
                                     # still see the label normally
GEO_LABELS_LIST = [
    # (display text, x_m, y_m, elev_m) - UTM33N, elev_m read from merged_paleo_dem.tif
    ("Sicily Sill",     506347.5, 4043169.0,   -70.5),
    ("Hyblean Plateau", 475000.0, 4095000.0,   779.4),
    ("Ionian Basin",    555000.0, 4060000.0, -2907.0),
]

# --- north arrow ---------------------------------------------------------------------------
# A flat HUD compass, NOT a 3D object lying on the terrain: at CAM_ALTITUDE_DEG=24 a
# ground-plane arrow would foreshorten into a squashed sliver. Instead it's built a few km
# directly in front of the camera lens (NORTH_ARROW_DIST_KM), nearer than any terrain in the
# scene so it can never be occluded, and oriented from the camera's own right/up basis - i.e.
# glued to the image plane, like a corner compass icon in a GIS layout, but built as real
# Cycles geometry. Its bearing is the projection of true world north (+Y, per CAM_AZIMUTH_DEG's
# compass convention: az 0 = north = scene +Y) onto the camera's image-plane basis.
NORTH_ARROW = True
NORTH_ARROW_DIST_KM       = 5.0     # HUD distance in front of the lens (km)
NORTH_ARROW_MARGIN_X_FRAC = 0.90    # corner position, as a fraction of the half-frame WIDTH
NORTH_ARROW_MARGIN_Y_FRAC = 0.97    # ...and HEIGHT, at that distance
NORTH_ARROW_LEN_FRAC    = 0.20      # total shaft+head length, as a fraction of the HUD
                                     # half-frame HEIGHT - reads as a small corner icon. Still
                                     # an icon, but the 0.14 version printed at ~3 mm and its
                                     # "N" was the smallest mark in the whole figure.
NORTH_ARROW_HEAD_FRAC   = 0.32      # arrowhead length as a fraction of the total length
NORTH_ARROW_THICK_FRAC  = 0.10      # shaft thickness as a fraction of the total length
NORTH_ARROW_LABEL_FRAC  = 0.34      # "N" cap height as a fraction of the total length

# --- realistic water (used when PHOTOREAL) ---
WATER_ROUGHNESS     = 0.16          # low enough that the sun stays a sharp highlight the
                                    # fine sparkle wavelets shatter into a glitter path, but
                                    # spread a touch to curb the blown-white hotspot
WATER_IOR           = 1.33
WATER_TRANSMISSION  = 0.65          # < 1.0 → base colour visible through water (less transparent)
WATER_WAVE_STRENGTH = 0.19          # medium chop bump — high values cause white specular blowout
WATER_WAVE_SCALE    = 40.0          # medium-scale waves
# --- sea realism: a large rolling SWELL layered on the chop + broad COLOUR patches, so
# the Ionian basin reads as a living ocean, not a flat blue slab. All subtle - the sea
# is the calm receiving basin, not the torrent.
SEA_SWELL_STRENGTH   = 0.45         # large slow swell bump added under the chop (0 = off)
SEA_SWELL_SCALE      = 2.6          # swell size (SMALL number = big rolling waves) - at the
                                    # 448 km overview distance a larger feature size goes
                                    # sub-pixel and the sea renders as a blank gradient
SEA_COLOR_LIGHT      = (0.10, 0.33, 0.60)   # lighter blue for the varying patches
SEA_COLOR_VARIATION  = 0.95         # 0 = uniform WATER_COLOR; ~0.8 = strong patchiness
SEA_COLOR_SCALE      = 0.7          # colour-patch size (small = broad ocean-scale patches)
# --- SUN GLITTER: a fine high-frequency normal layer breaks the mirror into countless
# wavelets so the sun reflection shatters into a glittering path (the single biggest
# "real ocean" cue). Needs a moderate-low WATER_ROUGHNESS so the sun stays a sharp
# highlight for the wavelets to chop up (too rough = the glitter smears into a dull sheen).
SEA_SPARKLE_STRENGTH = 0.28         # fine wavelet normal strength (0 = off; too high reads noisy)
SEA_SPARKLE_SCALE    = 480.0        # wavelet frequency (BIG = fine sparkle). Too low and the
                                    # feature goes sub-pixel at overview distance, and the
                                    # glitter path reads as one flat white slab
SEA_SPARKLE_DETAIL   = 3.0          # fractal octaves on the sparkle (crisper glints)
# --- REAL WAVE GEOMETRY: bump alone can't show waves at this scale (a flat plane reflects
# the uniform sky no matter the normal). So DISPLACE actual rolling swell into a subdivided
# sea plane -> the surface genuinely undulates, the horizon ripples, and the sun glitter
# breaks into a wavy field. Amplitude is VE-scaled and deliberately a bit exaggerated so it
# reads from altitude (real 1-3 m waves would be sub-pixel).
SEA_WAVE_GEO            = True       # False = flat plane (old behaviour)
SEA_WAVE_GEO_STRENGTH_M = 30.0      # swell amplitude in metres (VE-scaled). 0 = flat
SEA_WAVE_GEO_SCALE      = 2.4       # swell feature size in km (world/GLOBAL coords) - a
                                    # larger feature is only a few px at overview distance
                                    # and reads as invisible
SEA_WAVE_GEO_SUBDIV     = 500       # plane subdivisions (must resolve the swell wavelength)

# --- 3D megaflood sheet (real water in the corridor; needs flood_corridor.png) ---
# A water sheet that sits at the flood WATER SURFACE (baked by the merge, sloping down
# the corridor to the sea) and only shows inside the corridor. It fills low ground,
# lets ridges/hills poke through, and drops where the land drops -> a real 3D cascade
# into the sea plane. Needs MEGAFLOOD_WRITE_MASK=True in merge_paleo_dem.py.
SHOW_FLOOD_SHEET     = True
FLOOD_SHEET_ZBIAS_M  = 2.0           # tiny lift to avoid z-fighting at the waterline
                                     # (the real water depth is baked into the surface)
# --- WHERE THE CASCADE MEETS THE SEA -----------------------------------------------------
# The flood sheet and the sea are two different materials (the sheet is darker, rougher, and
# carries the corridor's foam streaks; the sea is lighter, glossier and colour-patched). Out
# over the deep Ionian the corridor still covers the sea, so the two lie COPLANAR and the
# sheet's coverage edge cuts a hard line across open water - the "abrupt transition".
#
# It is not a shading bug, it is a category error: once the torrent has entered the sea it IS
# the sea. So the sheet is faded out over the last FLOOD_SEA_MERGE_M above the sea plane
# (driven by the geometry's own world Z, which is exactly the water surface), and what the
# flood leaves behind in the sea is painted where it belongs - as a TURBIDITY PLUME on the
# sea itself. A megaflood delivering a basin's worth of sediment does not enter a blue sea
# invisibly; the plume is the physically expected signature and it dissolves the seam.
FLOOD_SEA_MERGE_M    = 140.0        # height band (m) over which the sheet dissolves into the sea
SEA_PLUME            = True         # turbid outflow plume on the sea, from the corridor mask
SEA_PLUME_COLOR      = (0.30, 0.40, 0.42)   # sediment-laden water: pale desaturated blue-grey
SEA_PLUME_STRENGTH   = 0.85         # max mix toward the plume colour inside the corridor
SEA_PLUME_ROUGH      = 0.55         # the plume is churned + turbid: it kills the sea's gloss,
                                    # which is what makes it read as suspended load and not paint

FLOOD_SHEET_COLOR    = (0.03, 0.12, 0.28)   # turbid flood blue (dark - most colour comes
                                            # from the painted terrain showing through)
FLOOD_FOAM_COLOR     = (0.90, 0.94, 0.97)   # whitewater tint at the cascade (mask B channel)
FLOOD_SHEET_ALPHA    = 0.95          # opacity of WHITEWATER (foam B >= ~0.35): the cascade
                                     # must stay a solid tumbling body
FLOOD_SHEET_ALPHA_OPEN = 0.50        # opacity of open (foam-free) water. The braided-channel
                                     # texture lives in the painted BED, and only untinted ALPHA
                                     # lets it read through as painted braids + a dark-blue veil
FLOOD_SHEET_IOR      = 1.0           # 1.0 = NO Fresnel reflection. The sediment torrent must
                                     # read dark like the reference's channels, but at the
                                     # overview camera's grazing incidence (75-85 deg) any
                                     # refractive water's Fresnel goes to ~1 regardless of IOR
                                     # or roughness. Only a unity IOR kills the mirror: the
                                     # sheet becomes matte sediment water, foam stays white as
                                     # albedo. The SEA keeps WATER_IOR 1.33 and its sun glitter.
FLOOD_SHEET_ROUGHNESS = 0.18         # kept low because MEGAFLOOD_LAKE_SWELL_M puts real swell
                                     # geometry on the calm water (same lesson as SEA_WAVE_GEO):
                                     # the geometry itself breaks the sky reflection, so
                                     # roughness no longer needs to blur it - a high roughness
                                     # scatters the sky diffusely and reads as a milky white
                                     # slab instead of blue water. Foam still drives roughness
                                     # up to 0.9 at the cascade via the mask's B channel.
FLOOD_SHEET_TRANSMISSION = 0.38      # light passes through to the painted flooded ground;
                                     # lower = the sheet's own dark blue asserts more. Must stay
                                     # high enough that the baked BRAIDED channel threads
                                     # (BRAID_THREADS) read through the sheet.
FLOOD_WAVE_STRENGTH  = 0.09          # ripple bump on the flood sheet (a torrent, not glass) -
                                     # too low and the approach reads as a glassy mirror, too
                                     # high (>~0.2) and the sky speckles unrealistically bright
FLOOD_WAVE_SCALE     = 90.0          # ripple scale (bigger = finer chop at scene scale)
# broad rolling SWELL under the fine ripple: gives the flood sheet undulating relief so it
# reads as a moving water body instead of a flat glassy mirror (the fine ripple alone left
# it looking like glass). Layered like the sea's swell+chop.
FLOOD_SWELL_STRENGTH = 0.20          # rolling swell bump (0 = off)
FLOOD_SWELL_SCALE    = 32.0          # swell wavelength (smaller number = longer rollers)
FLOOD_FRONT_SOFTNESS = 0.08          # softness of the advancing flood front (fraction of along)
FLOOD_CORRIDOR_PNG   = None          # None = read 'flood_corridor_png' from meta

# --- CASCADE RELIEF: the falling water as REAL GEOMETRY ---
# The flood sheet is a heightfield, so painting foam on it could never make the chute
# read as a waterfall - a bump on a smooth sheet just returns the sky, which is what
# produced the pale amorphous blobs on the canyon wall. The merge now bakes the
# cascade's standing-wave rollers + flow-parallel strands as a SIGNED relief in metres
# (flood_relief_16bit.png); this displaces the sheet with it, so the crests are real
# geometry that catches the sun and casts into its own troughs.
# Needs MEGAFLOOD_CASCADE_RELIEF=True in merge_paleo_dem.py.
CASCADE_RELIEF       = True
CASCADE_RELIEF_SCALE = 1.4          # multiplier on the baked amplitude - taller crests catch
                                     # more sun and self-shadow their troughs for whitewater
                                     # plasticity. The whole point of exporting the relief as
                                     # its own map is that this retunes WITHOUT re-running the
                                     # ~10 min merge. >1 = wilder whitewater; 0 = flat sheet.

# --- SPRAY PLUME (3D volume where the cascade lands) ---
# The merge's PLUNGE_MIST_M apron is FLAT - it lies on the water. This is the real thing:
# a bounded Principled Volume box over the cascade whose density comes from flood_mist.png
# and fades out with height above the LOCAL water surface. Needs MEGAFLOOD_MIST_MASK=True.
# (A bounded OBJECT volume is safe; a world Volume Scatter is what turns this scene black.)
PLUNGE_MIST      = True
MIST_DENSITY     = 1.4               # volume density multiplier - the single knob for "how
                                     # much spray", retunable WITHOUT re-running the merge.
                                     # ~2.2 buries the canyon in fog, ~0.25 is a thin haze, 1.0
                                     # is a spray veil you can still see the cascade through.
                                     # Scales with how many km of volume a camera ray crosses,
                                     # so a tighter crop needs a bit more.
MIST_COLOR       = (0.92, 0.95, 0.98)  # spray is white with a cold cast
MIST_ANISOTROPY  = 0.35              # forward scattering: the plume GLOWS when the sun is
                                     # behind it, like real backlit spray. 0 = flat fog.
MIST_NOISE_SCALE = 12.0              # billow size (bigger number = finer billows)
MIST_NOISE_GAIN  = 0.55              # 0 = smooth airbrushed blob; ~0.5-0.7 = billowing cloud
MIST_STEP_RATE   = 0.5               # Cycles volume step rate. <1 = finer steps = less noise
                                     # but slower. Raise toward 1-2 for fast drafts.
FLOOD_ADVANCE        = 1.0           # still: reveal fraction 0..1 (source->lip). 1 = full.

# --- zoom / framing ---
CAM_TARGET_OFFSET_KM = (14.0, -9.0)  # regional overview: recentred onto the escarpment
                                     # lip / canyon for the zoomed-in (lens 160) framing.
                                     # (The plunge itself sits at scene (6, -3) km - read off
                                     # the merge's mist_bbox_uv - so this looks just SE of it,
                                     # which puts the canyon centre-frame at az 300.)
#CAM_TARGET_OFFSET_KM = (6.5, 7.0)  # hero crop: same look-at point as the old (0.5, 7.0), just
                                    # re-expressed for the WIDER crop box - the box centre moved
                                    # 6 km west, so the offset has to move 6 km east to keep the
                                    # falls framed where they were.
# to zoom: raise CAM_LENS_MM (telephoto, keeps perspective) or lower CAM_DISTANCE_KM (move in)

# =====================================================================
# implementation
# =====================================================================

def log(m):
    if not QUIET:                    # verbose debug only; quiet mode uses the progress bar
        print(f"[blender] {m}")


class _Bar:
    """Minimal tqdm-style progress bar for the build phase (no external deps)."""
    def __init__(self, total, label="building scene"):
        self.total = max(total, 1); self.n = 0; self.label = label
    def step(self, desc=""):
        self.n += 1
        if not QUIET:
            return
        w = 30
        f = int(w * self.n / self.total)
        bar = "\u2588" * f + "\u2591" * (w - f)
        tail = "\n" if self.n >= self.total else ""
        sys.stdout.write(f"\r{self.label} |{bar}| {self.n}/{self.total}  {desc:<14}{tail}")
        sys.stdout.flush()


def load_meta():
    with open(os.path.join(DATA_DIR, "heightmap_meta.json")) as f:
        return json.load(f)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                 bpy.data.images, bpy.data.lights, bpy.data.cameras):
        for d in list(coll):
            try:
                coll.remove(d)
            except Exception:
                pass


def set_engine():
    scene = bpy.context.scene
    candidates = []
    if ENGINE_PREF.upper() == "CYCLES":
        candidates = ["CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
    else:
        candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"]
    for eng in candidates:
        try:
            scene.render.engine = eng
            log(f"render engine: {eng}")
            break
        except (TypeError, ValueError):
            continue
    # sampling (attribute name differs by engine/version)
    try:
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = SAMPLES
        else:
            scene.eevee.taa_render_samples = SAMPLES
    except Exception:
        pass
    # view transform: AGX (4.x default) heavily desaturates -> "Standard" gives
    # literal, saturated colour, which is what a hypsometric map wants.
    try:
        scene.view_settings.view_transform = VIEW_TRANSFORM
        log(f"view transform: {VIEW_TRANSFORM}")
    except (TypeError, AttributeError):
        pass
    if PHOTOREAL:
        try:
            scene.view_settings.exposure = EXPOSURE
            log(f"exposure: {EXPOSURE:+.1f} stops")
        except (TypeError, AttributeError):
            pass

    if PHOTOREAL:
        # turn on shadows + reflections/refraction (attribute names vary EEVEE/EEVEE-Next)
        ee = getattr(scene, "eevee", None)
        for attr, val in (("use_ssr", True), ("use_ssr_refraction", True),
                          ("use_gtao", True), ("use_raytracing", True),
                          ("use_shadows", True), ("use_soft_shadows", True)):
            try:
                setattr(ee, attr, val)
            except (AttributeError, TypeError):
                pass
        log("photoreal: shadows + reflections enabled")


def make_terrain(meta):
    ve = float(VE) if VE is not None else float(meta.get("ve_hint", 3.0))
    zmin, zmax = meta["zmin_m"], meta["zmax_m"]
    width_km = meta["width_m"] / 1000.0
    height_km = meta["height_m"] / 1000.0

    # grid: 'size' = width in km; we scale Y afterwards for correct aspect
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=SUBDIVISIONS, y_subdivisions=SUBDIVISIONS,
        size=width_km, location=(0, 0, 0))
    terrain = bpy.context.active_object
    terrain.name = "palaeo_terrain"
    terrain.scale = (1.0, height_km / width_km, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # load heightmap as NON-COLOR data (so values aren't gamma-shifted)
    img_path = os.path.join(DATA_DIR, meta.get("heightmap_png", "heightmap_16bit.png"))
    img = bpy.data.images.load(img_path)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass

    # displacement: final Z(km) = VE/1000 * elevation_m
    tex = bpy.data.textures.new("heightmap_tex", type="IMAGE")
    tex.image = img
    tex.extension = "EXTEND"
    mod = terrain.modifiers.new("displace", type="DISPLACE")
    mod.texture = tex
    mod.texture_coords = "UV"
    mod.direction = "Z"
    mod.mid_level = 0.0
    mod.strength = ve * (zmax - zmin) / 1000.0
    terrain.location.z = ve * zmin / 1000.0

    # smooth normals: without this every displaced quad renders flat-shaded and
    # steep cliff faces break into visible facets
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass

    cmap_png = meta.get("colormap_png")
    has_cmap = cmap_png and os.path.isfile(os.path.join(DATA_DIR, cmap_png))
    _cam_dist = CAM_DISTANCE_KM if CAM_DISTANCE_KM else 1.6 * max(width_km, height_km)
    if PHOTOREAL and has_cmap:
        _terrain_material_albedo(terrain, os.path.join(DATA_DIR, cmap_png), _cam_dist)
    elif USE_COLORMAP and has_cmap:
        _terrain_material_colormap(terrain, os.path.join(DATA_DIR, cmap_png))
    else:
        _terrain_material(terrain, img, meta)
    return terrain, ve, width_km, height_km


def _terrain_material_albedo(obj, colormap_path, cam_dist_km=100.0):
    """Photoreal: baked colormap as diffuse albedo on a Principled BSDF, lit by the
    real sun so canyon walls cast shadows. (Bake the colormap flatter - lower the
    merge hillshade - if the combined shading looks too strong.) Optional sediment
    colour variation + micro rock bump + distance haze for a natural-landscape look."""
    img = bpy.data.images.load(colormap_path)        # sRGB colour
    mat = bpy.data.materials.new("terrain_albedo")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    try:
        bsdf.inputs["Specular IOR Level"].default_value = 0.1
    except KeyError:
        pass

    # ---- how ROCKY is this point? (0 = flat sediment, 1 = bare cliff) -------------------
    # cos(slope) = world normal . Z. Drives colour, roughness AND bump together - which is
    # what makes it read as a change of material rather than a painted-on tint.
    rock_out = None
    if TERRAIN_ROCK and TERRAIN_ROCK_STRENGTH > 0:
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = (0.0, 0.0, 1.0)
        nt.links.new(geo.outputs["Normal"], dot.inputs[0])
        rmap = nt.nodes.new("ShaderNodeMapRange")
        rmap.inputs["From Min"].default_value = float(TERRAIN_ROCK_SLOPE_ON)    # flat
        rmap.inputs["From Max"].default_value = float(TERRAIN_ROCK_SLOPE_FULL)  # vertical
        rmap.inputs["To Min"].default_value = 0.0
        rmap.inputs["To Max"].default_value = float(TERRAIN_ROCK_STRENGTH)
        try:
            rmap.clamp = True
        except AttributeError:
            pass
        nt.links.new(dot.outputs["Value"], rmap.inputs["Value"])
        rock_out = rmap.outputs["Result"]

    # sediment patches: a broad noise brightens/darkens the baked albedo by +-VAR, so the
    # ground isn't one flat tan slab. albedo * grey(map(noise)) via a MULTIPLY mix.
    base_out = tex.outputs["Color"]
    if TERRAIN_DETAIL and TERRAIN_COLOR_VAR > 0:
        cnoise = nt.nodes.new("ShaderNodeTexNoise")
        cnoise.inputs["Scale"].default_value = TERRAIN_COLOR_SCALE
        try:
            cnoise.inputs["Detail"].default_value = 5.0
        except KeyError:
            pass
        cmap = nt.nodes.new("ShaderNodeMapRange")
        cmap.inputs["To Min"].default_value = 1.0 - TERRAIN_COLOR_VAR
        cmap.inputs["To Max"].default_value = 1.0 + TERRAIN_COLOR_VAR
        nt.links.new(cnoise.outputs["Fac"], cmap.inputs["Value"])
        cmix = nt.nodes.new("ShaderNodeMixRGB")
        cmix.blend_type = "MULTIPLY"
        cmix.inputs["Fac"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], cmix.inputs["Color1"])
        nt.links.new(cmap.outputs["Result"], cmix.inputs["Color2"])   # value -> grey broadcast
        base_out = cmix.outputs["Color"]

    # ---- strip the steep faces to bare rock --------------------------------------------
    if rock_out is not None:
        rmix = nt.nodes.new("ShaderNodeMixRGB")
        rmix.blend_type = "MIX"
        rmix.inputs["Color2"].default_value = (*TERRAIN_ROCK_COLOR, 1.0)
        nt.links.new(rock_out, rmix.inputs["Fac"])
        nt.links.new(base_out, rmix.inputs["Color1"])
        base_out = rmix.outputs["Color"]

        rrough = nt.nodes.new("ShaderNodeMapRange")
        rrough.inputs["From Min"].default_value = 0.0
        rrough.inputs["From Max"].default_value = float(TERRAIN_ROCK_STRENGTH)
        rrough.inputs["To Min"].default_value = float(TERRAIN_ROUGHNESS)
        rrough.inputs["To Max"].default_value = float(TERRAIN_ROCK_ROUGH)
        try:
            rrough.clamp = True
        except AttributeError:
            pass
        nt.links.new(rock_out, rrough.inputs["Value"])
        nt.links.new(rrough.outputs["Result"], bsdf.inputs["Roughness"])
    else:
        try:
            bsdf.inputs["Roughness"].default_value = TERRAIN_ROUGHNESS
        except KeyError:
            pass
    nt.links.new(base_out, bsdf.inputs["Base Color"])

    # micro rock bump: fine procedural relief so the smooth DEM surface reads as rock/
    # sediment, not moulded clay. Subtle - a strong bump looks like sandpaper. TWO octaves:
    # one grain size is what the eye reads as plastic, real ground is self-similar.
    if TERRAIN_DETAIL and TERRAIN_BUMP_STRENGTH > 0:
        # OBJECT coordinates, so the noise Scale is a real length: scene units are km, hence
        # scale = 1000 / feature_m. Generated coords would instead normalise over the object's
        # bounding box - a different physical grain in each panel, and anisotropic to boot
        # (the terrain is 280 x 200 km, so X and Y would be stretched 1.4x against each other).
        tc = nt.nodes.new("ShaderNodeTexCoord")

        def _noise(feature_m, detail):
            n = nt.nodes.new("ShaderNodeTexNoise")
            n.inputs["Scale"].default_value = 1000.0 / max(feature_m, 1e-3)
            try:
                n.inputs["Detail"].default_value = detail
                n.inputs["Roughness"].default_value = 0.7
            except KeyError:
                pass
            nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
            return n

        height = _noise(TERRAIN_BUMP_FEATURE_M, 8.0).outputs["Fac"]
        if TERRAIN_BUMP_FINE > 0:
            add = nt.nodes.new("ShaderNodeMixRGB")
            add.blend_type = "ADD"
            add.inputs["Fac"].default_value = float(TERRAIN_BUMP_FINE)
            nt.links.new(height, add.inputs["Color1"])
            nt.links.new(_noise(TERRAIN_BUMP_FEATURE_M / 5.0, 6.0).outputs["Fac"],
                         add.inputs["Color2"])
            height = add.outputs["Color"]

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = TERRAIN_BUMP_STRENGTH
        if rock_out is not None:
            # cliffs are gullied and blocky; the flats are not
            bstr = nt.nodes.new("ShaderNodeMapRange")
            bstr.inputs["From Min"].default_value = 0.0
            bstr.inputs["From Max"].default_value = float(TERRAIN_ROCK_STRENGTH)
            bstr.inputs["To Min"].default_value = float(TERRAIN_BUMP_STRENGTH)
            bstr.inputs["To Max"].default_value = float(TERRAIN_BUMP_STRENGTH * TERRAIN_ROCK_BUMP)
            try:
                bstr.clamp = True
            except AttributeError:
                pass
            nt.links.new(rock_out, bstr.inputs["Value"])
            nt.links.new(bstr.outputs["Result"], bump.inputs["Strength"])
        nt.links.new(height, bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # distance haze: mix the surface toward a haze emission by camera distance, so far
    # terrain fades into the sky (atmospheric perspective). Reliable in headless Cycles.
    surf_out = bsdf.outputs["BSDF"]
    if PHOTOREAL and ATMOSPHERE and HAZE_STRENGTH > 0:
        camd = nt.nodes.new("ShaderNodeCameraData")
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = float(HAZE_START_KM)
        mr.inputs["From Max"].default_value = float(HAZE_END_KM)
        mr.inputs["To Min"].default_value = 0.0
        mr.inputs["To Max"].default_value = float(HAZE_STRENGTH)
        try:
            mr.clamp = True
        except AttributeError:
            pass
        nt.links.new(camd.outputs["View Distance"], mr.inputs["Value"])
        haze = nt.nodes.new("ShaderNodeEmission")
        haze.inputs["Color"].default_value = (*HAZE_COLOR, 1)
        haze.inputs["Strength"].default_value = float(HAZE_EMIT_STRENGTH)
        mixsh = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mr.outputs["Result"], mixsh.inputs["Fac"])
        nt.links.new(bsdf.outputs["BSDF"], mixsh.inputs[1])
        nt.links.new(haze.outputs["Emission"], mixsh.inputs[2])
        surf_out = mixsh.outputs["Shader"]

    nt.links.new(surf_out, out.inputs["Surface"])
    obj.data.materials.append(mat)


def _terrain_material_colormap(obj, colormap_path):
    """Shadeless emission from the baked colormap PNG: renders exactly as the preview."""
    img = bpy.data.images.load(colormap_path)        # sRGB colour (leave default colourspace)
    mat = bpy.data.materials.new("terrain_colormap")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)


def _terrain_material(obj, img, meta):
    zmin, zmax = meta["zmin_m"], meta["zmax_m"]
    rng = (zmax - zmin) if (zmax > zmin) else 1.0

    mat = bpy.data.materials.new("terrain_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    try:
        tex.image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    ramp = nt.nodes.new("ShaderNodeValToRGB")

    def t(elev_m):
        return min(max((elev_m - zmin) / rng, 0.0), 1.0)

    # (elevation_m, RGB) stops — absolute elevations so colour == height consistently.
    if COLOR_SCHEME == "bathymetric":
        stops = [(-4000, (0.04, 0.10, 0.30)), (-2000, (0.07, 0.22, 0.48)),
                 (-400,  (0.20, 0.55, 0.62)), (0,    (0.78, 0.70, 0.48)),
                 (600,   (0.36, 0.52, 0.26)), (1600, (0.45, 0.36, 0.24)),
                 (3000,  (0.95, 0.95, 0.96))]
    else:  # "desiccated": exposed floor/land reads dry
        stops = [(-4000, (0.28, 0.24, 0.22)), (-2000, (0.42, 0.34, 0.27)),
                 (-600,  (0.60, 0.49, 0.35)), (0,    (0.72, 0.60, 0.42)),
                 (700,   (0.50, 0.46, 0.30)), (1700, (0.40, 0.31, 0.22)),
                 (3000,  (0.93, 0.92, 0.92))]

    cr = ramp.color_ramp
    # keep only strictly-increasing stop positions within [0,1] (clamping at the
    # data range can collapse several stops onto 1.0 - drop those rather than stack)
    pcs, last = [], -1.0
    for elev, rgb in stops:
        pos = t(elev)
        if pos > last + 1e-4:
            pcs.append((pos, rgb)); last = pos
    if len(pcs) < 2:
        pcs = [(0.0, stops[0][1]), (1.0, stops[-1][1])]
    cr.elements[0].position = pcs[0][0]; cr.elements[0].color = (*pcs[0][1], 1)
    cr.elements[1].position = pcs[1][0]; cr.elements[1].color = (*pcs[1][1], 1)
    for pos, rgb in pcs[2:]:
        e = cr.elements.new(pos); e.color = (*rgb, 1)

    try:
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    except KeyError:
        pass

    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Red"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.append(mat)


def make_water(meta, ve, width_km, height_km):
    if not SHOW_WATER:
        return None
    z = ve * meta["paleo_sea_level_m"] / 1000.0
    waves = PHOTOREAL and SEA_WAVE_GEO and SEA_WAVE_GEO_STRENGTH_M > 0
    if waves:
        # subdivided grid so we can DISPLACE real rolling swell into it (a single quad
        # can only carry bump, which doesn't read at this scale). Real geometry gives the
        # surface genuine undulation + a rippled sun glitter.
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=SEA_WAVE_GEO_SUBDIV,
                                        y_subdivisions=SEA_WAVE_GEO_SUBDIV,
                                        size=1.0, location=(0, 0, z))
    else:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, z))
    water = bpy.context.active_object
    water.name = "palaeo_water"
    water.scale = (width_km * WATER_MARGIN, height_km * WATER_MARGIN, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    if waves:
        # world-scale rolling swell via a Displace modifier + a legacy CLOUDS texture.
        # GLOBAL coords -> the wave size is in scene units (km), independent of the plane.
        wtex = bpy.data.textures.new("sea_wave_tex", type="CLOUDS")
        try:
            wtex.noise_scale = SEA_WAVE_GEO_SCALE       # feature size in km
            wtex.noise_depth = 2
        except AttributeError:
            pass
        mod = water.modifiers.new("sea_swell", type="DISPLACE")
        mod.texture = wtex
        mod.texture_coords = "GLOBAL"
        mod.strength = ve * SEA_WAVE_GEO_STRENGTH_M / 1000.0
        mod.mid_level = 0.5
        try:
            bpy.ops.object.shade_smooth()               # smooth the swell (no facets)
        except Exception:
            pass

    if PHOTOREAL:
        _water_material_real(water, meta)
    else:
        _water_material_flat(water)
    return water


def _set_input(bsdf, names, value):
    for n in names:
        try:
            bsdf.inputs[n].default_value = value
            return True
        except (KeyError, TypeError):
            continue
    return False


def _water_material_flat(water):
    mat = bpy.data.materials.new("water_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*WATER_COLOR, 1)
        bsdf.inputs["Roughness"].default_value = 0.08
        _set_input(bsdf, ["Alpha"], WATER_ALPHA)
    for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    water.data.materials.append(mat)


def _water_material_real(water, meta=None):
    """Reflective/refractive sea with layered swell + chop and broad colour patches, so
    the receiving basin reads as a living ocean instead of a flat blue slab. Reflects the
    sky (use a sky). Plus the megaflood's turbidity plume where the corridor reaches it."""
    mat = bpy.data.materials.new("water_real")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = WATER_ROUGHNESS
    _set_input(bsdf, ["Transmission Weight", "Transmission"], WATER_TRANSMISSION)
    _set_input(bsdf, ["IOR"], WATER_IOR)

    # base colour: broad low-frequency noise mixes deep <-> lighter blue -> ocean patches
    base_out = None
    if SEA_COLOR_VARIATION > 0:
        cnoise = nt.nodes.new("ShaderNodeTexNoise")
        cnoise.inputs["Scale"].default_value = SEA_COLOR_SCALE
        try:
            cnoise.inputs["Detail"].default_value = 2.0
        except KeyError:
            pass
        cfac = nt.nodes.new("ShaderNodeMath")
        cfac.operation = "MULTIPLY"; cfac.use_clamp = True
        cfac.inputs[1].default_value = float(SEA_COLOR_VARIATION)
        nt.links.new(cnoise.outputs["Fac"], cfac.inputs[0])
        cmix = nt.nodes.new("ShaderNodeMixRGB")
        try:
            cmix.inputs["Color1"].default_value = (*WATER_COLOR, 1)
            cmix.inputs["Color2"].default_value = (*SEA_COLOR_LIGHT, 1)
            nt.links.new(cfac.outputs["Value"], cmix.inputs["Fac"])
            base_out = cmix.outputs["Color"]
        except (KeyError, TypeError):
            bsdf.inputs["Base Color"].default_value = (*WATER_COLOR, 1)
    else:
        bsdf.inputs["Base Color"].default_value = (*WATER_COLOR, 1)

    # ---- the megaflood's TURBIDITY PLUME (see FLOOD_SEA_MERGE_M) ------------------------
    # The corridor mask's R channel already knows exactly where the flood reaches the sea.
    # The sea plane is WATER_MARGIN times the terrain, so its UV has to be shrunk back onto
    # the DEM footprint before the mask lines up.
    corr = meta.get("flood_corridor_png") if meta else None
    if SEA_PLUME and SEA_PLUME_STRENGTH > 0 and corr and base_out is not None:
        path = os.path.join(DATA_DIR, corr)
        if os.path.exists(path):
            cimg = bpy.data.images.load(path)
            try:
                cimg.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            uv = nt.nodes.new("ShaderNodeTexCoord")
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.inputs["Location"].default_value = (0.5 - 0.5 * WATER_MARGIN,
                                                   0.5 - 0.5 * WATER_MARGIN, 0.0)
            mp.inputs["Scale"].default_value = (WATER_MARGIN, WATER_MARGIN, 1.0)
            nt.links.new(uv.outputs["UV"], mp.inputs["Vector"])
            ctex = nt.nodes.new("ShaderNodeTexImage")
            ctex.image = cimg
            # CLIP, not the EXTEND used everywhere else: the sea plane is WATER_MARGIN times
            # the DEM, so most of it lies OFF the mask. EXTEND would smear the border pixels
            # (the corridor touches the west edge, where it enters the frame) across the open
            # ocean as a plume stripe. Here the mask is only ever used as a 0..1 FACTOR, so
            # CLIP's black-outside is exactly right: no mask -> no plume.
            ctex.extension = "CLIP"
            nt.links.new(mp.outputs["Vector"], ctex.inputs["Vector"])
            csep = nt.nodes.new("ShaderNodeSeparateColor")
            nt.links.new(ctex.outputs["Color"], csep.inputs["Color"])

            pfac = nt.nodes.new("ShaderNodeMath")
            pfac.operation = "MULTIPLY"; pfac.use_clamp = True
            pfac.inputs[1].default_value = float(SEA_PLUME_STRENGTH)
            nt.links.new(csep.outputs["Red"], pfac.inputs[0])

            pmix = nt.nodes.new("ShaderNodeMixRGB")
            pmix.inputs["Color2"].default_value = (*SEA_PLUME_COLOR, 1)
            nt.links.new(pfac.outputs["Value"], pmix.inputs["Fac"])
            nt.links.new(base_out, pmix.inputs["Color1"])
            base_out = pmix.outputs["Color"]

            prgh = nt.nodes.new("ShaderNodeMapRange")
            prgh.inputs["From Min"].default_value = 0.0
            prgh.inputs["From Max"].default_value = float(SEA_PLUME_STRENGTH)
            prgh.inputs["To Min"].default_value = float(WATER_ROUGHNESS)
            prgh.inputs["To Max"].default_value = float(SEA_PLUME_ROUGH)
            try:
                prgh.clamp = True
            except AttributeError:
                pass
            nt.links.new(pfac.outputs["Value"], prgh.inputs["Value"])
            nt.links.new(prgh.outputs["Result"], bsdf.inputs["Roughness"])
            log("sea: turbidity plume from %s" % corr)

    if base_out is not None:
        nt.links.new(base_out, bsdf.inputs["Base Color"])

    # waves: layer a large rolling SWELL under the medium CHOP by chaining two bump nodes
    prev_normal = None
    if WATER_WAVE_STRENGTH > 0:
        chop = nt.nodes.new("ShaderNodeTexNoise")
        chop.inputs["Scale"].default_value = WATER_WAVE_SCALE
        try:
            chop.inputs["Detail"].default_value = 4.0
        except KeyError:
            pass
        bump1 = nt.nodes.new("ShaderNodeBump")
        bump1.inputs["Strength"].default_value = WATER_WAVE_STRENGTH
        nt.links.new(chop.outputs["Fac"], bump1.inputs["Height"])
        prev_normal = bump1.outputs["Normal"]
    if SEA_SWELL_STRENGTH > 0:
        swell = nt.nodes.new("ShaderNodeTexNoise")
        swell.inputs["Scale"].default_value = SEA_SWELL_SCALE
        try:
            swell.inputs["Detail"].default_value = 2.0
        except KeyError:
            pass
        bump2 = nt.nodes.new("ShaderNodeBump")
        bump2.inputs["Strength"].default_value = SEA_SWELL_STRENGTH
        nt.links.new(swell.outputs["Fac"], bump2.inputs["Height"])
        if prev_normal is not None:
            nt.links.new(prev_normal, bump2.inputs["Normal"])   # swell rides under the chop
        prev_normal = bump2.outputs["Normal"]
    # fine SPARKLE wavelets on top: high-frequency normal detail that shatters the sun
    # reflection into a glitter path (the strongest "real ocean" cue at this scale)
    if SEA_SPARKLE_STRENGTH > 0:
        spark = nt.nodes.new("ShaderNodeTexNoise")
        spark.inputs["Scale"].default_value = SEA_SPARKLE_SCALE
        try:
            spark.inputs["Detail"].default_value = SEA_SPARKLE_DETAIL
        except KeyError:
            pass
        bump3 = nt.nodes.new("ShaderNodeBump")
        bump3.inputs["Strength"].default_value = SEA_SPARKLE_STRENGTH
        nt.links.new(spark.outputs["Fac"], bump3.inputs["Height"])
        if prev_normal is not None:
            nt.links.new(prev_normal, bump3.inputs["Normal"])   # sparkle rides on the swell/chop
        prev_normal = bump3.outputs["Normal"]
    if prev_normal is not None:
        nt.links.new(prev_normal, bsdf.inputs["Normal"])

    # enable refraction in EEVEE (legacy + Next); Cycles needs nothing
    for attr, val in (("use_screen_refraction", True), ("use_raytrace_refraction", True)):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    water.data.materials.append(mat)


def make_flood_sheet(meta, ve, width_km, height_km):
    """Water sheet at the flood water surface, clipped to the megaflood corridor.

    Displaces the flood water-surface heightmap (flood_surface_16bit.png) so the sheet
    sits at the flood level - filling lows, letting highs poke through, dropping where
    the land drops. The corridor mask (flood_corridor.png) drives the alpha: R=coverage
    (topographically confined), G=along (advance reveal), B=foam (whitewater tint).
    Returns the sheet object (or None if no mask / disabled).
    """
    if not SHOW_FLOOD_SHEET:
        return None
    corr_png = FLOOD_CORRIDOR_PNG or meta.get("flood_corridor_png")
    if not corr_png:
        log("flood sheet: no flood_corridor_png in meta "
            "(set MEGAFLOOD_WRITE_MASK=True in the merge) - skipping")
        return None
    corr_path = os.path.join(DATA_DIR, corr_png)
    if not os.path.isfile(corr_path):
        log(f"flood sheet: corridor mask not found ({corr_path}) - skipping")
        return None

    zmin, zmax = meta["zmin_m"], meta["zmax_m"]
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=SUBDIVISIONS, y_subdivisions=SUBDIVISIONS,
        size=width_km, location=(0, 0, 0))
    sheet = bpy.context.active_object
    sheet.name = "flood_sheet"
    sheet.scale = (1.0, height_km / width_km, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # displace from the FLOOD WATER-SURFACE heightmap (not the terrain): the sheet sits
    # at the flood water level, so it fills lows and lets highs poke through instead of
    # draping over the ground like a tarp. Falls back to the terrain heightmap if the
    # merge didn't export a surface (older run) - that reproduces the old drape.
    surf_png = meta.get("flood_surface_png") or meta.get("heightmap_png", "heightmap_16bit.png")
    if not meta.get("flood_surface_png"):
        log("flood sheet: no flood_surface_png in meta - draping over terrain "
            "(re-run the merge to get the topographically-confined water surface)")
    img = bpy.data.images.load(os.path.join(DATA_DIR, surf_png))
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    tex = bpy.data.textures.new("flood_hm_tex", type="IMAGE")
    tex.image = img
    tex.extension = "EXTEND"
    mod = sheet.modifiers.new("displace", type="DISPLACE")
    mod.texture = tex
    mod.texture_coords = "UV"
    mod.direction = "Z"
    mod.mid_level = 0.0
    mod.strength = ve * (zmax - zmin) / 1000.0
    sheet.location.z = ve * zmin / 1000.0 + ve * FLOOD_SHEET_ZBIAS_M / 1000.0

    # SECOND displace: the cascade's own relief (rollers + strands), signed metres.
    # The PNG stores relief/(2*amp) + 0.5, so with mid_level 0.5 a strength of
    # ve*2*amp/1000 puts the vertex at exactly ve * relief_m / 1000 km. This is the
    # step that makes the falling water geometry instead of paint.
    relief_png = meta.get("flood_relief_png") if CASCADE_RELIEF else None
    if relief_png and CASCADE_RELIEF_SCALE > 0:
        rpath = os.path.join(DATA_DIR, relief_png)
        if os.path.isfile(rpath):
            amp = float(meta.get("flood_relief_amp_m", 100.0))
            rimg = bpy.data.images.load(rpath)
            try:
                rimg.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            rtex = bpy.data.textures.new("cascade_relief_tex", type="IMAGE")
            rtex.image = rimg
            rtex.extension = "EXTEND"
            rmod = sheet.modifiers.new("cascade_relief", type="DISPLACE")
            rmod.texture = rtex
            rmod.texture_coords = "UV"
            rmod.direction = "Z"
            rmod.mid_level = 0.5
            rmod.strength = ve * 2.0 * amp / 1000.0 * CASCADE_RELIEF_SCALE
            log(f"flood sheet: cascade relief +-{amp:.0f} m "
                f"x{CASCADE_RELIEF_SCALE} ({relief_png})")
        else:
            log(f"flood sheet: cascade relief not found ({rpath}) - flat sheet")
    elif CASCADE_RELIEF:
        log("flood sheet: no flood_relief_png in meta - flat sheet "
            "(set MEGAFLOOD_CASCADE_RELIEF=True and re-run the merge)")

    try:
        bpy.ops.object.shade_smooth()          # smooth normals (see terrain)
    except Exception:
        pass

    _flood_sheet_material(sheet, corr_path,
                          sea_z_km=ve * meta["paleo_sea_level_m"] / 1000.0,
                          merge_km=ve * FLOOD_SEA_MERGE_M / 1000.0)
    # the thin water skin must not cast shadows onto the terrain below - in Cycles
    # that painted a hard dark band across the flood (mat.shadow_method only covers
    # legacy EEVEE)
    try:
        sheet.visible_shadow = False
    except AttributeError:
        pass
    try:
        sheet.cycles_visibility.shadow = False
    except AttributeError:
        pass
    log("flood sheet: corridor water sheet built (%s)" % corr_png)
    return sheet


def _flood_sheet_material(obj, corr_path, sea_z_km=None, merge_km=0.0):
    """Water material clipped by the corridor mask; adds a 'flood_advance_value' node
    that gates the reveal (source->lip), driven by FLOOD_ADVANCE.

    sea_z_km/merge_km fade the sheet out as its surface reaches the sea plane, so the
    torrent dissolves into the sea instead of ending at a material seam (FLOOD_SEA_MERGE_M).
    """
    img = bpy.data.images.load(corr_path)
    try:
        img.colorspace_settings.name = "Non-Color"       # mask channels are data, not colour
    except Exception:
        pass
    mat = bpy.data.materials.new("flood_sheet_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])   # R,G,B = coverage,along,foam

    # base colour: turbid water -> foam by the B (foam) channel
    mix = nt.nodes.new("ShaderNodeMixRGB")
    try:
        mix.inputs["Color1"].default_value = (*FLOOD_SHEET_COLOR, 1)
        mix.inputs["Color2"].default_value = (*FLOOD_FOAM_COLOR, 1)
        nt.links.new(sep.outputs["Blue"], mix.inputs["Fac"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    except (KeyError, TypeError):
        bsdf.inputs["Base Color"].default_value = (*FLOOD_SHEET_COLOR, 1)

    # foam is rough/diffuse whitewater, open water stays glossier: drive the
    # roughness from the foam (B) channel so the cascade reads as churned water
    bsdf.inputs["Roughness"].default_value = FLOOD_SHEET_ROUGHNESS
    try:
        rough = nt.nodes.new("ShaderNodeMapRange")
        rough.inputs["To Min"].default_value = FLOOD_SHEET_ROUGHNESS
        rough.inputs["To Max"].default_value = 0.9
        nt.links.new(sep.outputs["Blue"], rough.inputs["Value"])
        nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    except (KeyError, TypeError):
        pass
    _set_input(bsdf, ["Transmission Weight", "Transmission"], FLOOD_SHEET_TRANSMISSION)
    _set_input(bsdf, ["IOR"], FLOOD_SHEET_IOR)   # NOT WATER_IOR - see FLOOD_SHEET_IOR

    # ripple bump: the megaflood is a torrent, not a mirror - breaks up the flat
    # sky reflection that made the sheet render as a featureless pale wedge.
    # The foam (B) channel carries flow-aligned streaks from the merge; adding it
    # to the bump height turns them into directional surface texture (current lanes).
    prev_normal = None
    # broad rolling SWELL first: undulating relief that breaks the flat sky mirror so the
    # sheet reads as a moving water body, not glass.
    if FLOOD_SWELL_STRENGTH > 0:
        swell = nt.nodes.new("ShaderNodeTexNoise")
        swell.inputs["Scale"].default_value = FLOOD_SWELL_SCALE
        try:
            swell.inputs["Detail"].default_value = 2.0
        except KeyError:
            pass
        bump_s = nt.nodes.new("ShaderNodeBump")
        bump_s.inputs["Strength"].default_value = FLOOD_SWELL_STRENGTH
        nt.links.new(swell.outputs["Fac"], bump_s.inputs["Height"])
        prev_normal = bump_s.outputs["Normal"]
    # fine ripple rides on top of the swell (foam channel adds flow-aligned streaks)
    if FLOOD_WAVE_STRENGTH > 0:
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = FLOOD_WAVE_SCALE
        try:
            noise.inputs["Detail"].default_value = 6.0
        except KeyError:
            pass
        hadd = nt.nodes.new("ShaderNodeMath")
        hadd.operation = "MULTIPLY_ADD"          # height = foam * 0.25 + noise
        hadd.inputs[1].default_value = 0.25      # kept low: a stronger foam-driven ridge
                                                 # catches the sky as hard white scratches
        nt.links.new(sep.outputs["Blue"], hadd.inputs[0])
        nt.links.new(noise.outputs["Fac"], hadd.inputs[2])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = FLOOD_WAVE_STRENGTH
        nt.links.new(hadd.outputs["Value"], bump.inputs["Height"])
        if prev_normal is not None:
            nt.links.new(prev_normal, bump.inputs["Normal"])   # ripple rides on the swell
        prev_normal = bump.outputs["Normal"]
    if prev_normal is not None:
        nt.links.new(prev_normal, bsdf.inputs["Normal"])

    # reveal front: clamp((advance - along) / softness, 0, 1)  -> soft advancing edge
    adv = nt.nodes.new("ShaderNodeValue")
    adv.name = "flood_advance_value"
    adv.label = "flood_advance"
    adv.outputs[0].default_value = float(FLOOD_ADVANCE)
    sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"
    nt.links.new(adv.outputs[0], sub.inputs[0])
    nt.links.new(sep.outputs["Green"], sub.inputs[1])
    rev = nt.nodes.new("ShaderNodeMath"); rev.operation = "MULTIPLY"; rev.use_clamp = True
    rev.inputs[1].default_value = 1.0 / max(FLOOD_FRONT_SOFTNESS, 1e-3)
    nt.links.new(sub.outputs["Value"], rev.inputs[0])

    # alpha = coverage(R) * reveal * sea-dissolve * global alpha
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Red"], m1.inputs[0])
    nt.links.new(rev.outputs["Value"], m1.inputs[1])
    alpha_out = m1.outputs["Value"]

    # dissolve into the sea: the sheet's world Z *is* its water surface, so fade it out over
    # the last merge_km above the sea plane. Below that the sea plane carries the water (and
    # SEA_PLUME paints what the flood delivers into it).
    if sea_z_km is not None and merge_km > 0:
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        pos = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], pos.inputs["Vector"])
        fade = nt.nodes.new("ShaderNodeMapRange")
        fade.inputs["From Min"].default_value = float(sea_z_km)
        fade.inputs["From Max"].default_value = float(sea_z_km + merge_km)
        fade.inputs["To Min"].default_value = 0.0
        fade.inputs["To Max"].default_value = 1.0
        try:
            fade.clamp = True
        except AttributeError:
            pass
        nt.links.new(pos.outputs["Z"], fade.inputs["Value"])
        msea = nt.nodes.new("ShaderNodeMath"); msea.operation = "MULTIPLY"
        nt.links.new(alpha_out, msea.inputs[0])
        nt.links.new(fade.outputs["Result"], msea.inputs[1])
        alpha_out = msea.outputs["Value"]

    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
    m2.inputs[1].default_value = float(FLOOD_SHEET_ALPHA)
    try:
        # foam-driven opacity (see FLOOD_SHEET_ALPHA_OPEN): whitewater stays a
        # solid body, open water thins so the painted bed's braided channels
        # read through an UNTINTED window (the transmission lobe cannot do this)
        arange = nt.nodes.new("ShaderNodeMapRange")
        arange.inputs["From Min"].default_value = 0.0
        arange.inputs["From Max"].default_value = 0.35
        arange.inputs["To Min"].default_value = float(FLOOD_SHEET_ALPHA_OPEN)
        arange.inputs["To Max"].default_value = float(FLOOD_SHEET_ALPHA)
        try:
            arange.clamp = True
        except AttributeError:
            pass
        nt.links.new(sep.outputs["Blue"], arange.inputs["Value"])
        nt.links.new(arange.outputs["Result"], m2.inputs[1])
    except (KeyError, TypeError, NameError):
        pass                            # no corridor image: constant alpha
    nt.links.new(alpha_out, m2.inputs[0])
    _set_input(bsdf, ["Alpha"], 1.0)
    nt.links.new(m2.outputs["Value"], bsdf.inputs["Alpha"])

    for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED"),
                      ("use_screen_refraction", True), ("use_raytrace_refraction", True)):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    try:
        mat.shadow_method = "NONE"          # thin water skin shouldn't cast hard shadows
    except (AttributeError, TypeError):
        pass

    obj.data.materials.append(mat)


def make_plunge_mist(meta, ve, width_km, height_km):
    """3D spray plume: a bounded Principled Volume box over the cascade.

    Density = mist mask (flood_mist.png, XY) x a vertical falloff above the LOCAL water
    surface x noise. The water surface is read back out of flood_surface_16bit.png inside
    the shader, so the plume hugs the cascade all the way down the chute instead of
    sitting at one flat altitude.

    A bounded object volume is safe here - it is the WORLD Volume Scatter that turns a
    km-scale scene black by extinguishing the sky at infinity.
    """
    if not PLUNGE_MIST:
        return None
    mist_png = meta.get("flood_mist_png")
    bbox = meta.get("mist_bbox_uv")
    if not mist_png or not bbox:
        log("mist: no flood_mist_png/mist_bbox_uv in meta - skipping "
            "(set MEGAFLOOD_MIST_MASK=True and re-run the merge)")
        return None
    mist_path = os.path.join(DATA_DIR, mist_png)
    surf_png = meta.get("flood_surface_png")
    if not os.path.isfile(mist_path) or not surf_png:
        log(f"mist: mask or flood surface missing - skipping")
        return None

    zmin, zmax = meta["zmin_m"], meta["zmax_m"]
    u0, v0, u1, v1 = [float(t) for t in bbox]           # v measured from the NORTH edge
    # raster (u, v) -> scene km. The terrain plane spans [-w/2, +w/2] x [-h/2, +h/2],
    # with raster row 0 (v=0) at the NORTH edge, i.e. +y.
    x0, x1 = (u0 - 0.5) * width_km, (u1 - 0.5) * width_km
    y0, y1 = (0.5 - v1) * height_km, (0.5 - v0) * height_km      # note the swap
    zb = ve * float(meta.get("mist_z_base_m", zmin)) / 1000.0
    zt = ve * float(meta.get("mist_z_top_m", zmax)) / 1000.0
    if x1 <= x0 or y1 <= y0 or zt <= zb:
        log("mist: degenerate bounding box - skipping")
        return None

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    box = bpy.context.active_object
    box.name = "plunge_mist"
    box.location = ((x0 + x1) / 2, (y0 + y1) / 2, (zb + zt) / 2)
    box.scale = (x1 - x0, y1 - y0, zt - zb)
    bpy.ops.object.transform_apply(scale=True)

    mat = bpy.data.materials.new("plunge_mist_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):                      # drop the default surface BSDF
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = nt.nodes["Material Output"]

    # Generated coords of the box run 0..1 over the box; map them back onto the FULL
    # grid's UV so the two images line up with the terrain.
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.vector_type = "POINT"
    # uv = location + scale * generated   (Mapping applies scale then location)
    mp.inputs["Location"].default_value = (x0 / width_km + 0.5,
                                           y0 / height_km + 0.5, 0.0)
    mp.inputs["Scale"].default_value = ((x1 - x0) / width_km,
                                        (y1 - y0) / height_km, 1.0)
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])

    def _img(path):
        # REUSE an already-loaded datablock if there is one. Blender expands every image
        # to float RGBA, so at 6134x4533 a second copy of the flood surface costs ~445 MB
        # on top of the five maps the scene already holds - which on a 16 GB machine
        # OOM-killed the render outright.
        im = bpy.data.images.get(os.path.basename(path))
        if im is None or bpy.path.abspath(im.filepath) != path:
            im = bpy.data.images.load(path)
        try:
            im.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = im
        node.extension = "EXTEND"
        node.interpolation = "Linear"
        nt.links.new(mp.outputs["Vector"], node.inputs["Vector"])
        return node

    mist_tex = _img(mist_path)
    surf_tex = _img(os.path.join(DATA_DIR, surf_png))

    # water-surface elevation (km, VE-scaled) at this XY, decoded from the 16-bit PNG
    wz = nt.nodes.new("ShaderNodeMapRange")
    wz.inputs["From Min"].default_value = 0.0
    wz.inputs["From Max"].default_value = 1.0
    wz.inputs["To Min"].default_value = ve * zmin / 1000.0
    wz.inputs["To Max"].default_value = ve * zmax / 1000.0
    nt.links.new(surf_tex.outputs["Color"], wz.inputs["Value"])

    # height of THIS volume sample above the water surface below it
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs["Vector"])
    dz = nt.nodes.new("ShaderNodeMath"); dz.operation = "SUBTRACT"
    nt.links.new(sep.outputs["Z"], dz.inputs[0])
    nt.links.new(wz.outputs["Result"], dz.inputs[1])

    mist_h_km = ve * float(meta.get("mist_height_m", 500.0)) / 1000.0
    # fade out with height (dense at the water, gone at the plume top) ...
    up = nt.nodes.new("ShaderNodeMapRange")
    up.clamp = True
    up.inputs["From Min"].default_value = 0.0
    up.inputs["From Max"].default_value = mist_h_km
    up.inputs["To Min"].default_value = 1.0
    up.inputs["To Max"].default_value = 0.0
    nt.links.new(dz.outputs["Value"], up.inputs["Value"])
    # soften the ramp with a POWER node rather than MapRange's SMOOTHSTEP: a SMOOTHSTEP
    # MapRange together with a live noise node in the SAME volume shader hard-crashes
    # Cycles 5.1 (EXCEPTION_ACCESS_VIOLATION during render; bisected - either one alone
    # is fine). Squaring the linear ramp gives the same dense-at-the-water, wispy-at-the-
    # top profile.
    upp = nt.nodes.new("ShaderNodeMath")
    upp.operation = "POWER"
    upp.inputs[1].default_value = 1.8
    nt.links.new(up.outputs["Result"], upp.inputs[0])
    # ... and kill anything BELOW the water surface: MapRange clamps to 1 for negative
    # heights, which would fill the water body (and the rock under it) with fog.
    lo = nt.nodes.new("ShaderNodeMapRange")
    lo.clamp = True
    lo.inputs["From Min"].default_value = -0.25 * mist_h_km
    lo.inputs["From Max"].default_value = 0.0
    lo.inputs["To Min"].default_value = 0.0
    lo.inputs["To Max"].default_value = 1.0
    nt.links.new(dz.outputs["Value"], lo.inputs["Value"])

    band = nt.nodes.new("ShaderNodeMath"); band.operation = "MULTIPLY"
    nt.links.new(upp.outputs["Value"], band.inputs[0])
    nt.links.new(lo.outputs["Result"], band.inputs[1])

    # billowing: without this the plume is a smooth airbrushed blob
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = MIST_NOISE_SCALE
    try:
        noise.inputs["Detail"].default_value = 4.0
    except KeyError:
        pass
    nmix = nt.nodes.new("ShaderNodeMath"); nmix.operation = "MULTIPLY_ADD"
    nt.links.new(noise.outputs["Fac"], nmix.inputs[0])
    nmix.inputs[1].default_value = MIST_NOISE_GAIN            # noise * gain + base
    nmix.inputs[2].default_value = 1.0 - MIST_NOISE_GAIN

    d1 = nt.nodes.new("ShaderNodeMath"); d1.operation = "MULTIPLY"
    nt.links.new(mist_tex.outputs["Color"], d1.inputs[0])
    nt.links.new(band.outputs["Value"], d1.inputs[1])
    d2 = nt.nodes.new("ShaderNodeMath"); d2.operation = "MULTIPLY"
    nt.links.new(d1.outputs["Value"], d2.inputs[0])
    nt.links.new(nmix.outputs["Value"], d2.inputs[1])
    d3 = nt.nodes.new("ShaderNodeMath"); d3.operation = "MULTIPLY"
    nt.links.new(d2.outputs["Value"], d3.inputs[0])
    d3.inputs[1].default_value = float(MIST_DENSITY)

    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (*MIST_COLOR, 1)
    nt.links.new(d3.outputs["Value"], vol.inputs["Density"])
    try:
        vol.inputs["Anisotropy"].default_value = MIST_ANISOTROPY
    except KeyError:
        pass
    try:
        vol.inputs["Emission Strength"].default_value = 0.0
    except KeyError:
        pass
    nt.links.new(vol.outputs["Volume"], out.inputs["Volume"])

    box.data.materials.append(mat)

    # TAKE THE VOLUME OUT OF THE SHADOW PATH. Cycles 5.1 hard-crashes
    # (EXCEPTION_ACCESS_VIOLATION) when it evaluates an IMAGE TEXTURE inside a volume
    # shader on a shadow ray - the crash log lands in
    #   kernel_image_interp <- svm_node_tex_image <- svm_eval_nodes <- integrator_shade_shadow
    # and it is size-dependent, because it only fires once enough shadow rays actually
    # traverse the box (a 900x600/16 spp draft survived; 1800x1200/64 spp did not).
    # Our density is driven by two image textures, so the only way to keep them is to
    # never let a shadow ray shade this object. The flood sheet is excluded for the same
    # reason (a thin water skin should not cast a hard band anyway), and a spray plume
    # that does not darken the canyon below it is a cheap price for the plume existing.
    for attr, val in (("visible_shadow", False), ("is_shadow_catcher", False)):
        try:
            setattr(box, attr, val)
        except AttributeError:
            pass
    try:
        box.cycles_visibility.shadow = False
    except AttributeError:
        pass

    log(f"mist: volume {x1 - x0:.0f}x{y1 - y0:.0f} km, z {zb:.1f}..{zt:.1f} "
        f"(plume {mist_h_km:.2f} km), density {MIST_DENSITY}")
    return box


def make_cloud_shadows(ve, width_km, height_km):
    """Shadow-only cloud plane: dappled sunlight without visible clouds.

    The plane is skipped by camera/diffuse/glossy/transmission rays and seen ONLY by
    shadow rays, so the pattern never occludes the terrain or shows up in the water's
    reflection. Cloud opacity = noise gate x clear-zone masks; the clear zones are
    shifted toward the sun so the SHADOWS (cast slantwise at SUN_ALTITUDE_DEG) miss
    the protected ground, not merely the clouds above it.
    """
    if not (PHOTOREAL and CLOUD_SHADOWS):
        return None
    size = max(width_km, height_km) * (2 * WATER_MARGIN)   # cover the sea plane too
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, CLOUD_ALT_KM))
    plane = bpy.context.active_object
    plane.name = "cloud_shadows"

    mat = bpy.data.materials.new("cloud_shadow_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = nt.nodes["Material Output"]

    tc = nt.nodes.new("ShaderNodeTexCoord")     # Object coords = world km (origin at 0,0,alt)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0 / CLOUD_SCALE_KM
    noise.inputs["Detail"].default_value = 6.0
    try:
        noise.inputs["Roughness"].default_value = 0.62
    except KeyError:
        pass
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])

    # gate the noise so ~CLOUD_COVER of the sky carries cloud, with soft edges
    gate = nt.nodes.new("ShaderNodeMapRange")
    thr = 1.0 - CLOUD_COVER
    gate.inputs["From Min"].default_value = thr - CLOUD_EDGE_SOFT
    gate.inputs["From Max"].default_value = thr + CLOUD_EDGE_SOFT
    nt.links.new(noise.outputs["Fac"], gate.inputs["Value"])

    # clear zones: fac *= ramp(distance from shifted centre), one per protected spot
    fac_out = gate.outputs["Result"]
    dxy = CLOUD_ALT_KM / max(math.tan(math.radians(SUN_ALTITUDE_DEG)), 1e-3)
    off = (dxy * math.sin(math.radians(SUN_AZIMUTH_DEG)),
           dxy * math.cos(math.radians(SUN_AZIMUTH_DEG)))
    for cx, cy, rad in CLOUD_CLEAR_KM:
        sub = nt.nodes.new("ShaderNodeVectorMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = (cx + off[0], cy + off[1], CLOUD_ALT_KM)
        nt.links.new(tc.outputs["Object"], sub.inputs[0])
        # measure distance in XY only (kill Z before LENGTH)
        flat = nt.nodes.new("ShaderNodeVectorMath")
        flat.operation = "MULTIPLY"
        flat.inputs[1].default_value = (1.0, 1.0, 0.0)
        nt.links.new(sub.outputs["Vector"], flat.inputs[0])
        dist = nt.nodes.new("ShaderNodeVectorMath")
        dist.operation = "LENGTH"
        nt.links.new(flat.outputs["Vector"], dist.inputs[0])
        ramp = nt.nodes.new("ShaderNodeMapRange")   # 0 inside radius -> 1 outside rim
        ramp.inputs["From Min"].default_value = rad
        ramp.inputs["From Max"].default_value = rad + CLOUD_CLEAR_FEATHER_KM
        nt.links.new(dist.outputs["Value"], ramp.inputs["Value"])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(fac_out, mul.inputs[0])
        nt.links.new(ramp.outputs["Result"], mul.inputs[1])
        fac_out = mul.outputs["Value"]

    dens = nt.nodes.new("ShaderNodeMath")
    dens.operation = "MULTIPLY"
    dens.inputs[1].default_value = CLOUD_DENSITY
    nt.links.new(fac_out, dens.inputs[0])

    # opaque-to-shadow where cloudy, transparent elsewhere
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    solid = nt.nodes.new("ShaderNodeBsdfDiffuse")
    solid.inputs["Color"].default_value = (0, 0, 0, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(dens.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(solid.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    plane.data.materials.append(mat)

    # shadow rays only: invisible to camera, reflections and bounce light
    for attr in ("visible_camera", "visible_diffuse", "visible_glossy",
                 "visible_transmission", "visible_volume_scatter"):
        try:
            setattr(plane, attr, False)
        except AttributeError:
            pass
    try:
        plane.visible_shadow = True
    except AttributeError:
        pass
    log("cloud shadows: cover %.0f%%, %d clear zones, alt %.0f km"
        % (CLOUD_COVER * 100, len(CLOUD_CLEAR_KM), CLOUD_ALT_KM))
    return plane


def _crs_to_scene(meta, ve, width_km, height_km, x_m, y_m, elev_m):
    """Real-world CRS metres -> scene km, using crs_bbox_m from heightmap_meta.json
    (written by merge_paleo_dem.py's finalize(); patch the json by hand if it predates that
    field). Verified against CLOUD_CLEAR_KM's hand-tuned canyon-head/lip values - no axis flip
    either way: scene_x = (real_x-xmin)/1000 - width_km/2, same for y; scene_z = ve*elev_m/1000,
    matching make_terrain()'s displacement."""
    xmin, ymin, xmax, ymax = meta["crs_bbox_m"]
    sx = (x_m - xmin) / 1000.0 - width_km / 2.0
    sy = (y_m - ymin) / 1000.0 - height_km / 2.0
    sz = ve * elev_m / 1000.0
    return sx, sy, sz


def make_annotation_material():
    """Shared 'ink' material for every baked-in annotation - scale bars, their km labels,
    geological place-names, the north arrow - so the whole annotation layer reads as ONE
    consistent overlay rather than a patchwork of separately-tuned materials."""
    mat = bpy.data.materials.new("annotation_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        _set_input(bsdf, ["Base Color"], SCALE_COLOR)
        _set_input(bsdf, ["Roughness"], 0.55)
        _set_input(bsdf, ["Metallic"], 0.0)
        _set_input(bsdf, ["Specular", "Specular IOR Level"], 0.1)
        # a little self-emission of the SAME dark colour - not a glow, just insurance against
        # the ink going fully black-in-shadow (indistinguishable from a cast shadow) on the
        # side of the terrain away from the sun
        _set_input(bsdf, ["Emission Color"], SCALE_COLOR)
        _set_input(bsdf, ["Emission Strength"], 0.6)
    return mat


def make_halo_material():
    """Pale outline material drawn behind the ink - see ANNOT_HALO in the CONFIG block.

    Mostly emissive on purpose. A halo that is merely a light *diffuse* surface still goes
    dark wherever the scene's own light does, which is precisely the shaded escarpment face
    the halo exists to survive; it would fade out exactly where it is needed. Emission makes
    it independent of the sun, so the outline is as bright on the shadowed cliff as on the
    sunlit sill.
    """
    mat = bpy.data.materials.new("annotation_halo_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        _set_input(bsdf, ["Base Color"], ANNOT_HALO_COLOR)
        _set_input(bsdf, ["Roughness"], 0.9)
        _set_input(bsdf, ["Metallic"], 0.0)
        _set_input(bsdf, ["Specular", "Specular IOR Level"], 0.0)
        _set_input(bsdf, ["Emission Color"], ANNOT_HALO_COLOR)
        _set_input(bsdf, ["Emission Strength"], 1.0)
    return mat


def _push_from_cam(obj, cam, frac=None):
    """Slide an object away from the camera along its own view ray, so a halo sits BEHIND the
    ink it outlines instead of z-fighting with it.

    The shove is a fraction of the object's OWN distance from the camera, which is what makes
    one setting work for both the 448 km scale bars and the 5 km HUD compass. An absolute
    distance that merely separates the scale bars would drive the arrow's halo most of the way
    to the lens.
    """
    if cam is None:
        return
    f = ANNOT_HALO_DEPTH_FRAC if frac is None else frac
    if f == 0:
        return
    # Ray from the camera to where the object actually IS - its geometry's world centre, not
    # its origin. _flat_mesh builds the north arrow from literal world-space vertices and
    # leaves the origin at (0,0,0), so an origin-based ray aims at the world origin instead
    # and shoves the halo hundreds of metres sideways, off its own ink.
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    centre = sum(corners, Vector((0.0, 0.0, 0.0))) / len(corners)
    d = centre - cam.location
    if d.length < 1e-6:
        return
    obj.matrix_world.translation = obj.matrix_world.translation + d * f


def _no_reflection(*objs):
    """Keep an annotation out of glossy rays. A bright halo made the scale label show up as a
    legible MIRRORED copy of itself in the Ionian - the dark ink alone had always reflected
    too, it was just too faint to notice."""
    for o in objs:
        if o is None:
            continue
        try:
            o.visible_glossy = False
        except AttributeError:
            pass


def _camera_basis(cam):
    """right, up, to_cam unit vectors (world space) from the camera's OWN local axes.
    Needs bpy.context.view_layer.update() to have run since make_camera() - the TRACK_TO
    constraint's rotation is only baked into matrix_world after the depsgraph evaluates it."""
    m = cam.matrix_world.to_3x3()
    right = m.col[0].normalized()
    up = m.col[1].normalized()
    # camera looks down its own local -Z, so local +Z points AWAY from what it's looking at -
    # i.e. "toward the camera" for anything positioned in front of the lens
    to_cam = m.col[2].normalized()
    return right, up, to_cam


def _billboard_matrix(loc, cam):
    """World matrix for a flat object at `loc` (km) that faces the camera: local +X = camera
    right, +Y = camera up, +Z = toward the camera - so a FONT curve built in its own local XY
    plane reads right-side-up and un-mirrored from the camera's point of view, at any azimuth
    or altitude, without a Track-To constraint (simpler, and guaranteed evaluated by render
    time since it's a plain matrix write, not something the depsgraph has to catch up on)."""
    right, up, to_cam = _camera_basis(cam)
    rot = Matrix((tuple(right), tuple(up), tuple(to_cam))).transposed().to_4x4()
    rot.translation = Vector(loc)
    return rot


def _make_text(name, text, loc, cam, height_km, mat, halo_mat=None, outset_km=0.0):
    """A flat, camera-facing text label (see _billboard_matrix). Used for both the scale-bar
    km numbers and the geological place-names - same billboarding, same ink.

    `outset_km` drives the halo through the font curve's own outline offset, which grows the
    glyph outlines outward in every direction. That is a genuine outline, unlike scaling a
    copy of the text: a scaled copy also moves the letters apart, so it drifts out of
    register with the ink it is supposed to be sitting behind.
    """
    def _build(obj_name, size, offset, material):
        curve = bpy.data.curves.new(obj_name, type="FONT")
        curve.body = text
        curve.size = size
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.extrude = 0.0    # flat: any extrusion would run along local Z (toward the
                                # camera), invisible face-on and pure risk at the silhouette
        try:
            curve.offset = offset
        except AttributeError:
            pass
        o = bpy.data.objects.new(obj_name, curve)
        bpy.context.collection.objects.link(o)
        o.matrix_world = _billboard_matrix(loc, cam)
        o.data.materials.append(material)
        try:
            o.visible_shadow = False
        except AttributeError:
            pass
        return o

    obj = _build(name, height_km, 0.0, mat)
    if halo_mat is not None and outset_km > 0:
        halo = _build(name + "_halo", height_km, outset_km, halo_mat)
        _push_from_cam(halo, cam)
        obj["halo_obj"] = halo.name        # so callers can move the pair together
    return obj


def _text_halo_outset(height_km):
    """Glyph outset for a label of this height - always a fraction of the glyph, never a
    fixed length. See ANNOT_HALO_TEXT_FRAC."""
    return height_km * ANNOT_HALO_TEXT_FRAC if ANNOT_HALO else 0.0


def _label_halo(obj):
    """The halo object belonging to a label, if it has one."""
    name = obj.get("halo_obj") if hasattr(obj, "get") else None
    return bpy.data.objects.get(name) if name else None


def _clear_label(obj, cam, world_dir, gap_km):
    """Slide a centred billboard label clear of the thing it labels, by its OWN measured size.

    The gap has to be resolved in SCREEN space. The label is offset along a world axis (the
    tick direction, so it reads as a continuation of the ruler) but drawn facing the camera,
    so how much of it still lies over the bar depends on how that world axis projects onto the
    camera's right/up vectors. Centring alone therefore leaves half the text on the bar - which
    is exactly how the dashed bar came to run through its own number - and guessing a constant
    gap breaks again the moment the string or the font size changes.
    """
    if cam is None:
        return
    bpy.context.view_layer.update()     # font geometry (and so .dimensions) is depsgraph-built
    right, up, _ = _camera_basis(cam)
    d = Vector(world_dir).normalized()
    sx, sy = d.dot(right), d.dot(up)
    if abs(sx) >= abs(sy):
        shift = right * math.copysign(obj.dimensions.x / 2.0 + gap_km, sx or 1.0)
    else:
        shift = up * math.copysign(obj.dimensions.y / 2.0 + gap_km, sy or 1.0)
    for o in (obj, _label_halo(obj)):
        if o is not None:
            o.matrix_world.translation = o.matrix_world.translation + shift


def _dashed_bar(name, p0, p1, tick_dir, mat, grow_km=0.0):
    """One dashed straight bar p0->p1 (scene km, Vector) with a short perpendicular
    tick cap at each end. Returns the joined object. Segment length is SCALE_DASH_KM so it
    doesn't machine-look like a solid rod; the tick caps read as ruler end-marks.

    `grow_km` fattens every element for the halo pass. It grows the bar SIDEWAYS and extends
    each dash and tick lengthwise, but never changes where the dashes start or how long the
    bar is overall - the bar depicts a measurement, so a halo that scaled it would quietly
    redraw the quantity it is annotating.
    """
    v0, v1 = Vector(p0), Vector(p1)
    length = (v1 - v0).length
    direction = (v1 - v0).normalized() if length > 1e-6 else Vector((0, 0, 1))
    rot = Vector((0, 0, 1)).rotation_difference(direction)
    tick_rot = Vector((0, 0, 1)).rotation_difference(tick_dir.normalized())
    thick = SCALE_THICKNESS_KM + 2.0 * grow_km

    # Proportion the dashes and the end ticks to THIS bar's own length. The two bars differ by
    # a factor of three - the amphitheatre is 17.5 km across while the drop is 2.0 km, which at
    # VE 3 is only ~6 km of scene - so one absolute dash/tick size cannot serve both. Sized for
    # the long bar, the short one came out as three stubs between two ticks half its own
    # length: it stopped reading as a ruler at all.
    period = SCALE_DASH_KM + SCALE_GAP_KM
    shrink = min(1.0, length / (SCALE_DASH_MIN_COUNT * period)) if period > 0 else 1.0
    dash_km, gap_km = SCALE_DASH_KM * shrink, SCALE_GAP_KM * shrink
    tick_km = min(SCALE_TICK_KM, SCALE_TICK_MAX_FRAC * length)

    parts = []
    d = 0.0
    while d < length:
        seg_len = min(dash_km, length - d)
        centre = v0 + direction * (d + seg_len / 2.0)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(centre))
        cube = bpy.context.active_object
        cube.rotation_mode = "QUATERNION"
        cube.rotation_quaternion = rot
        cube.scale = (thick, thick, seg_len / 2.0 + grow_km)
        parts.append(cube)
        d += dash_km + gap_km

    for end in (v0, v1):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=tuple(end))
        cap = bpy.context.active_object
        cap.rotation_mode = "QUATERNION"
        cap.rotation_quaternion = tick_rot
        cap.scale = (thick, thick, tick_km / 2.0 + grow_km)
        parts.append(cap)

    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    bar = bpy.context.active_object
    bar.name = name
    bar.data.materials.append(mat)
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    return bar


def make_scale_bars(meta, ve, width_km, height_km, cam, mat, halo_mat=None):
    """Dashed height/width reference sticks planted right at the cascade, each with a km
    label - see the CONFIG block above for why this is baked into the 3D scene instead of
    drawn in Illustrator, and for the manuscript cross-check behind the two label values.

    Called AFTER make_camera() on purpose: at a 448 km camera distance the bar's own
    sideways offset (meant to clear the flow) isn't enough to also clear the terrain
    along this specific sightline, which can otherwise occlude the bars behind the
    canyon wall regardless of colour. Pulling each endpoint a few
    km TOWARD the camera (SCALE_CAMERA_PULL_KM) guarantees it floats in front of
    whatever terrain would otherwise hide it, the same trick used for any always-visible
    3D annotation/gizmo."""
    if not SCALE_BARS:
        return None
    if "crs_bbox_m" not in meta:
        log("scale bars: heightmap_meta.json has no crs_bbox_m (re-merge, or patch the json "
            "by hand) - skipped")
        return None

    def pull_toward_cam(pt, pull_km=SCALE_CAMERA_PULL_KM):
        if cam is None:
            return pt
        v = Vector(pt)
        d = (cam.location - v)
        if d.length < 1e-6:
            return pt
        return tuple(v + d.normalized() * pull_km)

    vx, vy, vz_top = _crs_to_scene(meta, ve, width_km, height_km, *SCALE_VERTICAL_TOP_M)
    bx, by, bz_base = _crs_to_scene(meta, ve, width_km, height_km, *SCALE_VERTICAL_BASE_M)
    off = SCALE_VERTICAL_OFFSET_KM
    top = pull_toward_cam((vx, vy + off, vz_top))
    base = pull_toward_cam((bx, by + off, bz_base))
    halo_on = bool(ANNOT_HALO and halo_mat is not None)
    halos = []
    if halo_on:
        h = _dashed_bar("scale_bar_vertical_halo", base, top,
                         Vector((1, 0, 0)), halo_mat,
                         grow_km=SCALE_THICKNESS_KM * ANNOT_HALO_BAR_FRAC)
        _push_from_cam(h, cam)
        halos.append(h)
    vertical = _dashed_bar("scale_bar_vertical", base, top,
                            Vector((1, 0, 0)), mat)

    y0_m, y1_m = SCALE_HORIZONTAL_Y_RANGE_M
    left = pull_toward_cam(_crs_to_scene(meta, ve, width_km, height_km,
                                          SCALE_HORIZONTAL_X_M, y0_m, SCALE_HORIZONTAL_Z_M))
    right = pull_toward_cam(_crs_to_scene(meta, ve, width_km, height_km,
                                           SCALE_HORIZONTAL_X_M, y1_m, SCALE_HORIZONTAL_Z_M))
    if halo_on:
        h = _dashed_bar("scale_bar_horizontal_halo", left, right,
                         Vector((0, 0, 1)), halo_mat,
                         grow_km=SCALE_THICKNESS_KM * ANNOT_HALO_BAR_FRAC)
        _push_from_cam(h, cam)
        halos.append(h)
    horizontal = _dashed_bar("scale_bar_horizontal", left, right,
                              Vector((0, 0, 1)), mat)

    for obj in (vertical, horizontal, *halos):
        for attr in ("visible_shadow",):
            try:
                setattr(obj, attr, False)   # a floating annotation shouldn't cast a shadow
            except AttributeError:          # onto the terrain it's been pulled in front of
                pass
        _no_reflection(obj)

    drop_m = SCALE_VERTICAL_TOP_M[2] - SCALE_VERTICAL_BASE_M[2]
    width_m = y1_m - y0_m

    labels = []
    if SCALE_LABELS:
        vert_text = SCALE_VERTICAL_LABEL or f"~{drop_m / 1000.0:.1f} km"
        horiz_text = SCALE_HORIZONTAL_LABEL or f"~{width_m / 1000.0:.1f} km"
        # place beyond each bar's own tick, along the SAME direction the tick already points
        # (vertical bar's ticks run in X, horizontal bar's ticks run in Z) - reads as a
        # continuation of the ruler rather than a random floating caption
        hk = _text_halo_outset(SCALE_LABEL_HEIGHT_KM) if halo_on else 0.0
        vdir = Vector((1, 0, 0))
        vmid = (Vector(base) + Vector(top)) / 2.0
        vlabel_pt = pull_toward_cam(tuple(vmid + vdir * (SCALE_TICK_KM / 2.0)), pull_km=1.0)
        vlab = _make_text("scale_label_vertical", vert_text, vlabel_pt, cam,
                           SCALE_LABEL_HEIGHT_KM, mat, halo_mat if halo_on else None, hk)
        _clear_label(vlab, cam, vdir, SCALE_LABEL_GAP_KM)
        labels.append(vlab)

        hdir = Vector((0, 0, 1))
        hmid = (Vector(left) + Vector(right)) / 2.0
        hlabel_pt = pull_toward_cam(tuple(hmid + hdir * (SCALE_TICK_KM / 2.0)), pull_km=1.0)
        hlab = _make_text("scale_label_horizontal", horiz_text, hlabel_pt, cam,
                           SCALE_LABEL_HEIGHT_KM, mat, halo_mat if halo_on else None, hk)
        _clear_label(hlab, cam, hdir, SCALE_LABEL_GAP_KM)
        labels.append(hlab)
        # The horizontal label stands over the flooded corridor and the sea beyond it, and its
        # pale halo made it show up as a fully legible MIRRORED copy of itself out in the
        # Ionian. The dark ink had always reflected there too - it was just too faint to spot.
        for lab in labels:
            _no_reflection(lab, _label_halo(lab))
        log(f"scale labels: '{vert_text}' / '{horiz_text}' "
            f"(height {SCALE_LABEL_HEIGHT_KM} km, halo {'on' if halo_on else 'off'})")

    log("scale bars: vertical drop %.0f m (rim %.0f m -> base %.0f m), horizontal front %.1f km, "
        "pulled %.1f km toward camera"
        % (drop_m, SCALE_VERTICAL_TOP_M[2], SCALE_VERTICAL_BASE_M[2], width_m / 1000.0,
           SCALE_CAMERA_PULL_KM))
    return vertical, horizontal, labels


def make_geo_labels(meta, ve, width_km, height_km, cam, mat, halo_mat=None):
    """Three billboarded place-name labels - see the CONFIG block for the manuscript-checked
    siting of each. Camera-pulled exactly like the scale bars, for the same occlusion reason."""
    if not GEO_LABELS or cam is None:
        return []
    sea_z = ve * meta["paleo_sea_level_m"] / 1000.0 if SHOW_WATER else None
    objs = []
    for text, x_m, y_m, elev_m in GEO_LABELS_LIST:
        sx, sy, sz = _crs_to_scene(meta, ve, width_km, height_km, x_m, y_m, elev_m)
        # "Ionian Basin" sits on the sea FLOOR (elev far below sea level) - floating it above
        # the terrain still buries it under the water surface, since the wavy displaced sea
        # mesh (SEA_WAVE_GEO) eats ragged holes through letters where crests poke in front of
        # them. Anything below the waterline floats from the WATER surface instead.
        base_z = max(sz, sea_z) if sea_z is not None else sz
        lifted = Vector((sx, sy, base_z + GEO_LABEL_LIFT_KM))
        d = cam.location - lifted
        loc = lifted + d.normalized() * GEO_LABEL_CAMERA_PULL_KM if d.length > 1e-6 else lifted
        halo_on = bool(ANNOT_HALO and halo_mat is not None)
        obj = _make_text(f"geo_label_{text.replace(' ', '_')}", text, tuple(loc), cam,
                          GEO_LABEL_HEIGHT_KM, mat,
                          halo_mat if halo_on else None,
                          _text_halo_outset(GEO_LABEL_HEIGHT_KM) if halo_on else 0.0)
        if GEO_LABEL_NO_REFLECTION:
            # "Ionian Basin" over open water was picking up a soft mirror double in the
            # glossy sea below it (SEA_WAVE_GEO + a fairly low WATER_ROUGHNESS) - the label
            # still renders normally to camera rays, it just stops appearing IN reflections.
            # The halo has to be excluded too, or the reflection shows a ghost outline of a
            # label that is no longer there.
            _no_reflection(obj, _label_halo(obj))
        objs.append(obj)
    log("geo labels: " + ", ".join(t for t, *_ in GEO_LABELS_LIST))
    return objs


def _flat_mesh(name, verts, faces, mat):
    """A mesh built from literal world-space vertex positions - no rotation step, so (unlike
    a cube/cone rotated only by rotation_difference, which leaves an undetermined 'roll'
    around its own axis) there is no ambiguity about which way it faces the camera. Used for
    the north arrow so its silhouette is a crisp flat triangle/rectangle, not a foreshortened
    3D primitive (a rotated cone arrowhead renders as a blobby paper-aeroplane shape for
    exactly this reason)."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    try:
        obj.visible_shadow = False
    except AttributeError:
        pass
    return obj


def make_north_arrow(cam, mat, halo_mat=None):
    """A flat HUD compass glued a few km in front of the lens - see the CONFIG block for why
    this, not a ground-plane arrow or an Illustrator overlay. Bearing = true north (+Y)
    projected onto the camera's own right/up basis; a direction needs no perspective divide,
    so this is exact for a flat icon regardless of camera azimuth/altitude/lens. Built as flat
    world-space triangles/rectangles in the camera's own image plane (right/up), not rotated
    3D primitives - see _flat_mesh for why."""
    if not NORTH_ARROW or cam is None:
        return None
    right, up, to_cam = _camera_basis(cam)

    cd = cam.data
    sensor_w = getattr(cd, "sensor_width", 36.0)
    hfov = 2.0 * math.atan((sensor_w / 2.0) / cd.lens)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (RES_Y / RES_X))

    d = NORTH_ARROW_DIST_KM
    half_w = d * math.tan(hfov / 2.0)
    half_h = d * math.tan(vfov / 2.0)
    centre = Vector(cam.location) - to_cam * d
    corner = centre - right * (half_w * NORTH_ARROW_MARGIN_X_FRAC) \
                     - up * (half_h * NORTH_ARROW_MARGIN_Y_FRAC)

    north = Vector((0.0, 1.0, 0.0))
    theta = math.atan2(north.dot(up), north.dot(right))
    dir_world = (right * math.cos(theta) + up * math.sin(theta)).normalized()
    perp = (right * -math.sin(theta) + up * math.cos(theta)).normalized()

    # sized as a fraction of the HUD frame's own half-height, so it scales sensibly with
    # NORTH_ARROW_DIST_KM/lens/aspect - see the CONFIG block for target proportions (a small
    # corner icon, ~7% of frame height, not the ~50%-of-frame first draft)
    total_len = NORTH_ARROW_LEN_FRAC * half_h
    head_len = NORTH_ARROW_HEAD_FRAC * total_len
    shaft_len = total_len - head_len
    thick = NORTH_ARROW_THICK_FRAC * total_len
    head_half_w = thick * 2.2

    tail = corner
    shaft_tip = corner + dir_world * shaft_len
    tip = corner + dir_world * total_len

    def _build_arrow(suffix, material, g=0.0):
        """The arrow, optionally outset by `g` for the halo. The outset grows the silhouette
        along both the shaft axis and the perpendicular, which is a real outline - a uniformly
        scaled copy would also slide the whole icon along its own axis, because the arrow is
        anchored at its tail rather than at its centre."""
        shaft = _flat_mesh(
            "north_arrow_shaft" + suffix,
            [tail - dir_world * g + perp * (thick / 2.0 + g),
             tail - dir_world * g - perp * (thick / 2.0 + g),
             shaft_tip - perp * (thick / 2.0 + g),
             shaft_tip + perp * (thick / 2.0 + g)],
            [0, 1, 2, 3], material)
        head = _flat_mesh(
            "north_arrow_head" + suffix,
            [tip + dir_world * g,
             shaft_tip - perp * (head_half_w + g),
             shaft_tip + perp * (head_half_w + g)],
            [0, 1, 2], material)
        bpy.ops.object.select_all(action="DESELECT")
        shaft.select_set(True)
        head.select_set(True)
        bpy.context.view_layer.objects.active = shaft
        bpy.ops.object.join()
        obj = bpy.context.active_object
        obj.name = "north_arrow" + suffix
        return obj

    halo_on = bool(ANNOT_HALO and halo_mat is not None)
    if halo_on:
        # the HUD sits a few km in front of the lens, so the halo's push-back is scaled down
        # to match - a 0.3 km shove here would be a large fraction of NORTH_ARROW_DIST_KM
        _push_from_cam(_build_arrow("_halo", halo_mat, thick * ANNOT_HALO_BAR_FRAC), cam)
    arrow = _build_arrow("", mat)

    label_pt = tip + dir_world * (NORTH_ARROW_LABEL_FRAC * total_len * 1.3)
    label = _make_text("north_arrow_N", "N", tuple(label_pt), cam,
                        NORTH_ARROW_LABEL_FRAC * total_len, mat,
                        halo_mat if halo_on else None,
                        _text_halo_outset(NORTH_ARROW_LABEL_FRAC * total_len)
                        if halo_on else 0.0)

    log(f"north arrow: bearing {math.degrees(theta):.1f} deg on screen "
        f"(HUD {d:.1f} km in front of lens, length {total_len:.3f} km)")
    return arrow, label


def make_sun():
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
    sun = bpy.context.active_object
    sun.data.energy = SUN_STRENGTH
    if PHOTOREAL:
        try:
            sun.data.angle = math.radians(SUN_ANGLE_SOFT_DEG)   # soft shadow edges
        except AttributeError:
            pass
    az = math.radians(SUN_AZIMUTH_DEG)
    alt = math.radians(SUN_ALTITUDE_DEG)
    # point the sun: rotate so it shines from the given az/alt
    sun.rotation_euler = (math.radians(90) - alt, 0, -az)
    return sun


def make_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")

    if PHOTOREAL and SKY_TYPE == "nishita":
        sky = nt.nodes.new("ShaderNodeTexSky")
        # set each attribute in its own try: Blender 5.x renamed the sky model to
        # MULTIPLE_SCATTERING, and a single try-block silently skipped elevation/
        # rotation too - leaving the sky's own sun DISC at default az 0/elev 15
        # as the dominant light, ignoring SUN_* entirely.
        for attr, val in (("sky_type", "NISHITA"),
                          ("sun_disc", False),   # our sun lamp is the sun
                          ("sun_elevation", math.radians(SUN_ALTITUDE_DEG)),
                          ("sun_rotation", math.radians(-SUN_AZIMUTH_DEG))):
            try:
                setattr(sky, attr, val)
            except (AttributeError, TypeError):
                pass
        if bg:
            nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
            bg.inputs["Strength"].default_value = SKY_STRENGTH
        log("sky: Nishita (sun el %.0f / az %.0f deg, strength %.2f)"
            % (SUN_ALTITUDE_DEG, SUN_AZIMUTH_DEG, SKY_STRENGTH))
    elif bg:
        bg.inputs["Color"].default_value = (*BG_COLOR, 1)
        bg.inputs["Strength"].default_value = BG_STRENGTH


def make_camera(meta, ve, width_km, height_km):
    # target at scene centre (+ optional offset to frame the canyon), at mid terrain height
    tz = ve * ((meta["zmin_m"] + meta["zmax_m"]) / 2.0) / 1000.0
    target = bpy.data.objects.new("cam_target", None)
    bpy.context.collection.objects.link(target)
    target.location = (CAM_TARGET_OFFSET_KM[0], CAM_TARGET_OFFSET_KM[1], tz)

    dist = CAM_DISTANCE_KM if CAM_DISTANCE_KM else 1.6 * max(width_km, height_km)
    az = math.radians(CAM_AZIMUTH_DEG)
    alt = math.radians(CAM_ALTITUDE_DEG)
    # look direction (toward az); place camera opposite, elevated, relative to target
    look = Vector((math.sin(az), math.cos(az), 0))
    cam_xy = Vector((target.location.x, target.location.y)) - dist * math.cos(alt) * Vector((look.x, look.y))
    cam_z = tz + dist * math.sin(alt)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = CAM_LENS_MM
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (cam_xy.x, cam_xy.y, cam_z)

    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    bpy.context.scene.camera = cam
    return cam, target


def setup_render():
    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = OUTPUT_PNG
    # volume stepping for the spray plume: the default step is tuned for room-scale
    # scenes and under-samples a km-scale volume into blotchy noise
    if PLUNGE_MIST:
        try:
            scene.cycles.volume_step_rate = MIST_STEP_RATE
            scene.cycles.volume_preview_step_rate = MIST_STEP_RATE
            scene.cycles.volume_max_steps = 256
        except AttributeError:
            pass


def main():
    global DATA_DIR, OUTPUT_PNG
    # Blender resolves relative image paths against the .blend, not the working directory
    DATA_DIR, OUTPUT_PNG = os.path.abspath(DATA_DIR), os.path.abspath(OUTPUT_PNG)
    if QUIET:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR} (edit the CONFIG block)")
    meta = load_meta()

    bar = _Bar(10)
    clear_scene();                                  bar.step("scene")
    set_engine();                                   bar.step("engine")
    terrain, ve, wkm, hkm = make_terrain(meta);     bar.step("terrain")
    water = make_water(meta, ve, wkm, hkm);         bar.step("water")
    flood_sheet = make_flood_sheet(meta, ve, wkm, hkm); bar.step("flood")
    make_plunge_mist(meta, ve, wkm, hkm);           bar.step("mist")
    make_sun();                                     bar.step("sun")
    make_cloud_shadows(ve, wkm, hkm)
    make_world();                                   bar.step("world")
    cam, target = make_camera(meta, ve, wkm, hkm);  bar.step("camera")
    # depsgraph must catch up on the camera's TRACK_TO constraint before matrix_world (and
    # therefore world_to_camera_view / the annotation billboarding below) is trustworthy.
    bpy.context.view_layer.update()
    annot_mat = make_annotation_material()
    halo_mat = make_halo_material() if ANNOT_HALO else None
    make_scale_bars(meta, ve, wkm, hkm, cam, annot_mat, halo_mat)  # after the camera - see docstring
    make_geo_labels(meta, ve, wkm, hkm, cam, annot_mat, halo_mat)
    make_north_arrow(cam, annot_mat, halo_mat)
    setup_render();                                 bar.step("ready")

    eng = bpy.context.scene.render.engine.replace("BLENDER_", "")
    print(f"  scene {wkm:.0f}×{hkm:.0f} km  VE {ve}  |  {RES_X}×{RES_Y}  {eng}  {SAMPLES} spp  exp {EXPOSURE:+.0f}")

    if not RENDER_ON_RUN:
        print("  scene built (RENDER_ON_RUN=False). Press F12 to render.")
        return

    print(f"  rendering still  [{SAMPLES} spp] …")
    bpy.ops.render.render(write_still=True)
    print(f"  done  →  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
