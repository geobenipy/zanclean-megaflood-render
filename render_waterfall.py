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
import time
import warnings
from mathutils import Vector, Matrix

# =====================================================================
# CONFIG
# =====================================================================
DATA_DIR   = "output_overview"
                                     # ^ older merge, kept as a fallback. The active line adds
                                     # MEGAFLOOD_SEA_GRACE_M so the corridor/flood-sheet mask
                                     # reaches the plunge->basin waypoints instead of being
                                     # zeroed at the sea cutoff - see MEGAFLOOD_SEA_GRACE_M in
                                     # merge_paleo_dem.py.
OUTPUT_PNG = "output_final/render_waterfall.png"
                                     # the submitted Fig-3b panel. The Extended-Data twin is
                                     # the same run with SHOW_WATER / SHOW_FLOOD_SHEET /
                                     # PLUNGE_MIST off and colormap_dry_png as the ground.

# --- console ---
QUIET = True                         # clean output: one progress bar + summary. False = verbose debug logs

# --- terrain ---
SUBDIVISIONS = 1600                  # grid resolution (cuts per side). 600-1000 for drafts;
                                     # capped at 1600 here, not raised toward 2400-3200 for
                                     # crisper cascade rollers (MEGAFLOOD_ROLLER_LAMBDA_M),
                                     # because this machine only has 16 GB RAM with its
                                     # pagefile on a slow HDD (no admin rights to relocate it).
                                     # At 2400 a full 4800x3200/256spp Cycles render reliably
                                     # exhausted memory and took the whole OS down mid-render -
                                     # 3 unexpected-shutdown events (Kernel-Power ID 41) in two
                                     # days, not just a Blender crash. 1600 is the highest value
                                     # verified end-to-end without one. Raise it again only on a
                                     # machine with more RAM or a faster pagefile disk.
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
# beyond HAZE_END -> full haze mix everywhere, i.e. a flat grey wash.
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
# Clear zones: keep them TIGHT. The 160 mm overview frame is only ~100 km wide, so the
# first attempt (r 45 + 22 feather, two zones) blanketed the entire visible frame and
# no shadow survived anywhere. Protect only the cascade/bay and two spots of the belt.
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
SCALE_LABEL_HEIGHT_KM = 4.8         # was 2, which printed at roughly 4 pt - see the size note
                                     # above the dash settings
SCALE_LABEL_GAP_KM   = 1.4          # a TRUE gap between the tick end and the nearest ink of
                                     # the text. The label is centred and then slid clear by
                                     # its own measured half-width (see _clear_label), so this
                                     # no longer has to be guessed against the string length -
                                     # which is how the bar previously ran through its own
                                     # number when the text grew.
SCALE_VERTICAL_LABEL   = "~2 km"    # explicit string so the label has no decimal point.
                                     # None = auto "~{drop_m/1000:.1f} km"; string overrides
SCALE_HORIZONTAL_LABEL = "~20 km"   # the manuscript's own wording ("more than 20 km wide"),
                                     # not the bar's own ~17.5 km straight-chord length -
                                     # this only overrides what's PRINTED, not the geometry

# --- geological / topographic place-names: flat billboarded text (always faces the camera),
# same dark ink as the scale bars. Positions are PROBED against merged_paleo_dem.tif and
# confirmed in-frame via world_to_camera_view, not eyeballed. Terms and siting follow the
# manuscript directly:
#   - "Sicily Sill": the broad near-planar shelf landward of the canyon rim, on dry
#     (desiccated) ground - the whole sill was subaerially exposed during the MSC.
#   - "Hyblean Plateau": the uplifted foreland domain forming Sicily's southern margin -
#     sited on clearly elevated (+779 m) dry terrain near the frame's upper-left edge. This
#     anchor is right for the 3:2 STILL and is deliberately left alone; the 16:9 animation
#     crops it off the top and moves it via ANIM_GEO_LABEL_MOVES.
#   - "Ionian Basin": the receiving abyssal plain the flood empties into - sited in open
#     deep water SE of the plunge, clear of the corridor/plume. Excluded from glossy rays
#     (GEO_LABEL_NO_REFLECTION) so it doesn't mirror in the sea below it.
#   - "Noto Canyon" / "Malta Escarpment": sited directly on the visible rock, not the
#     real-world crest coordinates, so each label sits on the feature it names.
#   - "Zanclean Flood Corridor": labels the flood route itself, so a reader can tell it's
#     the (hypothetical) megaflood path rather than a modern strait.
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
    # (display text, x_m, y_m, elev_m) - UTM33N, elev_m read from merged_paleo_dem.tif.
    # Each anchor is placed by projecting a candidate point through the actual camera
    # (world_to_camera_view), rendering it, and checking the measured on-screen text
    # position - not the raw projection, which lands ~50-90 px off from the glyphs
    # themselves - against the scale-bar ink and the neighbouring labels. A clear
    # sightline to the ANCHOR is not enough: occlusion is checked across the full span
    # of the rendered text (a single-point or endpoint-only check can miss a nearer
    # ridge under the middle of a long label), and the halo (a separate object pushed
    # further back by ANNOT_HALO_DEPTH_FRAC) needs its own clearance check since a
    # margin that clears the ink can still sit behind a more distant halo.
    ("Sicily Sill",     506347.5, 4043169.0,   -70.5),
    ("Hyblean Plateau", 475000.0, 4095000.0,   779.4),
    ("Ionian Basin",    555000.0, 4060000.0, -2907.0),
    # Sits just above the vertical scale bar, labelling the canyon itself.
    ("Noto Canyon",     531911.0, 4091826.0,   -376.4),
    # On the escarpment's own visible cliff face (the dark rock-material wall dropping
    # from the Hyblean Plateau into the Ionian, right of the canyon), not the crest
    # coordinates further west - siting it on the visible feature rather than the
    # real-world crest keeps the label from floating off the rendered geometry.
    ("Malta Escarpment", 533776.0, 4047458.0,  -304.0),
    # Labels the blue corridor itself, since without this a reader can't tell it's the
    # (hypothetical) Zanclean flood route rather than a modern strait. Sited upstream,
    # still on-corridor water (colour (9,18,36), navy).
    ("Zanclean Flood Corridor", 482260.0, 4062825.0,   13.0),
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
SEA_SWELL_STRENGTH   = 0.45         # large slow swell bump added under the chop (0 = off); was 0.30
SEA_SWELL_SCALE      = 2.6          # swell size (SMALL number = big rolling waves); was 4.0 -
                                    # at the 448 km overview distance the 4.0 features were
                                    # sub-pixel and the sea rendered as a blank gradient (v9)
SEA_COLOR_LIGHT      = (0.10, 0.33, 0.60)   # lighter blue for the varying patches
SEA_COLOR_VARIATION  = 0.95         # 0 = uniform WATER_COLOR; ~0.8 = strong patchiness
SEA_COLOR_SCALE      = 0.7          # colour-patch size (small = broad ocean-scale patches); was 1.1
# --- SUN GLITTER: a fine high-frequency normal layer breaks the mirror into countless
# wavelets so the sun reflection shatters into a glittering path (the single biggest
# "real ocean" cue). Needs a moderate-low WATER_ROUGHNESS so the sun stays a sharp
# highlight for the wavelets to chop up (too rough = the glitter smears into a dull sheen).
SEA_SPARKLE_STRENGTH = 0.28         # fine wavelet normal strength (0 = off). Too high = noisy; was 0.18
SEA_SPARKLE_SCALE    = 480.0        # wavelet frequency (BIG = fine sparkle). Was 900: sub-pixel
                                    # at overview distance, the glitter path read as one white slab
SEA_SPARKLE_DETAIL   = 3.0          # fractal octaves on the sparkle (crisper glints)
# --- REAL WAVE GEOMETRY: bump alone can't show waves at this scale (a flat plane reflects
# the uniform sky no matter the normal). So DISPLACE actual rolling swell into a subdivided
# sea plane -> the surface genuinely undulates, the horizon ripples, and the sun glitter
# breaks into a wavy field. Amplitude is VE-scaled and deliberately a bit exaggerated so it
# reads from altitude (real 1-3 m waves would be sub-pixel).
SEA_WAVE_GEO            = True       # False = flat plane (old behaviour)
SEA_WAVE_GEO_STRENGTH_M = 30.0      # swell amplitude in metres (VE-scaled). 0 = flat; was 24
SEA_WAVE_GEO_SCALE      = 2.4       # swell feature size in km (world/GLOBAL coords); was 3.5
                                    # (barely 4 px at the overview distance - invisible)
SEA_WAVE_GEO_SUBDIV     = 500       # plane subdivisions (must resolve the swell wavelength)

# --- 3D megaflood sheet (real water in the corridor; needs flood_corridor.png) ---
# A water sheet that sits at the flood WATER SURFACE (baked by the merge, sloping down
# the corridor to the sea) and only shows inside the corridor. It fills low ground,
# lets ridges/hills poke through, and drops where the land drops -> a real 3D cascade
# into the sea plane. Needs MEGAFLOOD_WRITE_MASK=True in merge_paleo_dem.py.
SHOW_FLOOD_SHEET     = True
FLOOD_SHEET_ZBIAS_M  = 2.0           # tiny lift to avoid z-fighting at the waterline
                                     # (the real water depth is baked into the surface)
# The sheet mesh is CROPPED to the wet corridor's own bounding box (flood_sheet_bbox_uv,
# written by the merge - see _SHEET_META in merge_paleo_dem.py) instead of spanning the
# whole regional frame like the terrain does. The corridor is a small fraction of a
# 280x200 km overview, so the old full-extent grid spent the same SUBDIVISIONS cuts on
# invisible dry land as on the water - which is what forced a choice on this 16 GB machine
# between a crash (SUBDIVISIONS high enough to resolve MEGAFLOOD_LAKE_SWELL_M/
# ROLLER_LAMBDA_M) and an under-resolved, milky calm-water surface (SUBDIVISIONS low
# enough to fit in RAM). Cropping decouples the two: the same vertex budget now buys a far
# finer quad size where the water actually is, independent of the terrain's own
# SUBDIVISIONS. Falls back to the old full-extent behaviour if an older merge run has no
# flood_sheet_bbox_uv in its metadata.
FLOOD_SHEET_TARGET_QUAD_M = 60.0     # PREFERRED mesh quad size (m) inside the crop - clears
                                     # MEGAFLOOD_ROLLER_LAMBDA_M (950 m, ~16 quads/wavelength)
                                     # and MEGAFLOOD_LAKE_SWELL_KM (2.5 km) comfortably
FLOOD_SHEET_MIN_SUBDIV  = 400        # floor, in case the corridor bbox is tiny (hero crop)
FLOOD_SHEET_MAX_VERTS   = 2_600_000  # hard cap on cuts_x*cuts_y, not a per-axis cut count -
                                     # a per-axis cap alone doesn't bound total memory when
                                     # the bbox is a very different shape than the frame (the
                                     # video merge's MEGAFLOOD_SEA_GRACE_M reaches deep into
                                     # the basin, so its bbox is far bigger than the still's).
                                     # 2.6 M ~= a full-frame 1600x1600 grid, a flood-sheet vertex
                                     # count already proven to render without crashing - so
                                     # cropping only ever trades that SAME budget for a finer
                                     # quad, never spends more of it. If the target quad would exceed the
                                     # budget, both axes are scaled down together so quads
                                     # stay square instead of clamping one axis independently.
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
SEA_PLUME_CORRIDOR_PNG = "output_overview_old/output_overview/flood_corridor.png"
                                    # None = use the same corridor mask as the flood sheet
                                    # (meta['flood_corridor_png'] in DATA_DIR). Pinned here to
                                    # the PRE-MEGAFLOOD_SEA_GRACE_M mask on purpose: that grace
                                    # value now lets the corridor's R channel (coverage) stay
                                    # 255 far into the deep basin so flood_advance has an
                                    # outflow to reveal (see merge_paleo_dem.py) - but this
                                    # same R channel ALSO drives the sea's OWN turbidity tint,
                                    # and the tint was only ever tuned for the OLD, narrow
                                    # ~19 km2 near-shore extent. Painting it across the whole
                                    # newly-reachable basin ramped SEA_PLUME_ROUGH (0.55) over
                                    # a huge area, and a rough surface under a bright sky at a
                                    # grazing angle reads as a flat, uniformly-lit white slab -
                                    # the same failure mode this file already documents for
                                    # under-geometried water elsewhere. Pointing the PLUME
                                    # specifically at the old mask keeps its footprint exactly
                                    # what it was in the last clean clip, independent of how
                                    # far the flood sheet's own mask now reaches.
SEA_PLUME_COLOR      = (0.30, 0.40, 0.42)   # sediment-laden water: pale desaturated blue-grey
SEA_PLUME_STRENGTH   = 0.85         # max mix toward the plume colour inside the corridor
SEA_PLUME_ROUGH      = 0.55         # the plume is churned + turbid: it kills the sea's gloss,
                                    # which is what makes it read as suspended load and not paint
SEA_PLUME_ALONG_ON   = 0.80         # along-route fraction (mask G channel) where the plume
SEA_PLUME_ALONG_FULL = 0.88         # starts fading in / reaches full strength. The corridor's
                                    # R (coverage) channel is 255 the whole route, including the
                                    # calm Sicily-hugging approach belt (measured: along~0.72,
                                    # zero foam there) - gating by coverage alone painted a
                                    # second turbid "plume" patch at that coastal bend, far from
                                    # any real outflow. The true plunge sits at along~0.906
                                    # (LIP_ALONG); ON/FULL sit just below it so the plume is
                                    # already at full strength by the lip, not still ramping in.

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
FLOOD_WAVE_STRENGTH  = 0.09          # ripple bump on the flood sheet (a torrent, not glass);
                                     # raised from 0.035 -> the approach was reading as a
                                     # glassy mirror; some break-up makes it read as moving
                                     # water. (0.20 was too much -> bright sky speckle.)
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
                                     # the 'flood_advance' animation sweeps this 0 -> 1.

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

# --- animation (the supplementary video) ---------------------------------------------------
# The still above IS the publication figure and this block must never change it: every
# animation setting is read only when ANIMATE is True, and the overrides below are applied
# to the module globals at the top of main(). Flip ANIMATE and run the script exactly as for
# a still - nothing else needs editing.
#
# The one thing that makes or breaks the clip: a 'flood_advance' that only sweeps the corridor
# mask reveals a FROZEN PHOTOGRAPH of water. Alpha rises, but every ripple, every foam streak
# and every wisp of spray stands perfectly still - so it reads as a wipe transition over a
# still image, not as a flood. ANIM_WATER_MOTION is the fix: the sheet's ripple/swell bump,
# the sea's chop and glitter, the spray plume's billows and the cloud shadows are all advected
# per frame. (The CASCADE RELIEF is deliberately NOT advected: those are STANDING waves in a
# chute and standing waves do not travel with the current. Sliding that map would also drag
# the rollers off the chute and onto the flats the merge's gate exists to keep smooth.)
ANIMATE     = True                  # render an image sequence instead of a single still. ON on
                                    # this branch, OFF on main: that is the whole difference
                                    # between the two, so `blender -b -P render_waterfall.py`
                                    # does the right thing on either without an edit first.
ANIM_TYPE   = "flood_advance"       # "flood_advance" | "flood_rise" | "orbit" | "flood_and_orbit"
                                    # flood_advance = the corridor reveals source->lip: the
                                    # megaflood visibly rolls in and hits the falls (this is the
                                    # hero anim; flood_rise only moves the distant sea plane a
                                    # little). "orbit" swings the camera a full 360 deg, which
                                    # walks it through the DEM's cut edges and invalidates the
                                    # sun, the haze ramp, the CLOUD_CLEAR_KM zones and every
                                    # annotation placement - drafts only, never a submission.
ANIM_FPS     = 25                   # 25 fps is the usual rate for a journal SI video
ANIM_SECONDS = 30.0                 # clip length; ANIM_FRAMES follows from it. Longer is NOT
                                    # automatically better even with time to spare: Nature caps
                                    # a supplementary video at 30 MB, so every extra second is
                                    # paid for in bitrate by every other second. 30 s at 30 MB
                                    # is ~8 Mbit/s, which holds up on moving water; a 60 s clip
                                    # of the same file size would visibly break up.
ANIM_FRAMES  = int(round(ANIM_SECONDS * ANIM_FPS))
ANIM_FORMAT  = "PNG"                # "PNG" (frame sequence -> video_gen.py) | "MP4" (Blender's
                                    # own FFmpeg). Use PNG: a multi-hour run that dies at frame
                                    # 400 loses the whole MP4 but keeps every PNG (see
                                    # ANIM_RESUME), and PNG frames can be re-graded/re-encoded
                                    # for whatever format the journal asks for.
ANIM_DIR     = "output_final/anim_16x9_v4"
                                    # Explicit, versioned folder rather than a default: ANIM_RESUME
                                    # matches frames by NUMBER only, so pointing a run with new
                                    # settings (a route change, a label fix, a mist fix, an aspect
                                    # ratio change...) at a folder holding an old frame set would
                                    # silently "resume" and keep the stale frames instead of
                                    # re-rendering them. A fresh folder per settings change avoids
                                    # that entirely; keep the previous one as a fallback until the
                                    # new run is verified end to end. (_check_resume_frames also
                                    # warns if a folder's frames were rendered at a different
                                    # resolution.)
FLOOD_RISE_FROM_M = -3500.0         # flood_rise: start level (deep) -> rises to paleo_sea_level_m

# Per-frame quality. The still is 4800x3200 / 256 spp and takes ~2 h; multiplied by several
# hundred frames that is a run measured in weeks. These override RES_X/RES_Y/SAMPLES for the
# animation only. SUBDIVISIONS is deliberately NOT reduced: the mesh is built ONCE and then
# reused by every frame, so its cost is a one-off, while dropping it would round the cascade
# rollers back into pillows - which is the whole point of CASCADE_RELIEF.
ANIM_RES_X, ANIM_RES_Y = 1920, 1080  # 16:9, which is what Nature asks for (4:3 second best).
                                     # With sensor_fit AUTO and a landscape frame the sensor is
                                     # fitted to the WIDTH, so the horizontal FOV depends on the
                                     # lens alone: dropping Y from 1280 (3:2) to 1080 is exactly a
                                     # symmetric vertical crop and leaves the still's framing
                                     # horizontally untouched. What it does cost is 100 px top and
                                     # bottom - which is why the "Hyblean Plateau" label had to be
                                     # re-sited (see GEO_LABELS_LIST); the north arrow is a HUD
                                     # placed at a screen fraction and re-seats itself.
                                     # Do NOT raise this to spend a bigger time budget. Nature
                                     # caps a supplementary video at 30 MB, so more pixels are
                                     # spent at a LOWER bitrate each, and the H.264 blocking that
                                     # buys is far more visible than the sharpness gained. It is
                                     # not cheap either: 2304x1536 measured ~1.45x the cost per
                                     # frame, i.e. exactly its pixel ratio.
ANIM_SAMPLES        = 96             # A render at 96 spp and one at 64 spp are indistinguishable
                                     # 1:1 at the cascade, where the spray volume makes noise most
                                     # likely - the denoiser carries it. This sits one step above
                                     # that point rather than exactly on it, since the time budget
                                     # allows it. Below ~48 spp risks frame-to-frame denoiser
                                     # flicker, far more objectionable in motion than static noise.
                                     # Above ~96 there is nothing left to buy - see ANIM_RES_* for
                                     # why the spare time should not go into pixels either.
ANIM_DENOISE        = True
ANIM_MIST_STEP_RATE = 0.5            # the still's value - fine volume steps. The spray box is the
                                     # most expensive thing in the scene, so raising this (coarser
                                     # steps, less time) is the first lever if the render budget is
                                     # tight, but at 96 spp / 1920x1280 the fine steps already cost
                                     # ~6.3 min/frame all-in, which is affordable for this clip.
ANIM_PERSISTENT_DATA = False         # keep the synced scene + BVH between frames instead of
                                     # rebuilding them. Looks like an obvious win since nothing in
                                     # this animation moves in geometry, but measured per-frame
                                     # cost with it on vs off is within noise of each other -
                                     # essentially all the cost is path tracing, not scene sync -
                                     # while it holds the memory for the whole run for nothing, so
                                     # it stays off. Cut ANIM_SAMPLES, ANIM_RES_* or ANIM_SECONDS
                                     # instead for a speed-up.

# A weekend-long unattended run WILL be interrupted (crash, reboot, full disk). With this on,
# re-launching the very same command skips every frame already written and carries on where it
# stopped instead of starting from scratch.
ANIM_RESUME      = True
ANIM_FRAME_START = 1                 # or set by hand to restart at a known-bad frame

# Dramaturgy. The old setup ramped the reveal linearly across the WHOLE clip, so the flood
# arrived exactly on the last frame and the cascade at full discharge - the subject of the
# figure - was never actually seen running. Three phases instead:
#   [0 .. HOLD]                 desiccated basin, calm sea: establishes the "before"
#   [HOLD .. HOLD+ADVANCE]      the torrent crosses the basin and goes over the lip
#   [.. end]                    full discharge: the cascade runs and the spray plume builds
ANIM_HOLD_S       = 3.0
ANIM_ADVANCE_S    = 12.0             # the remainder of the clip is the cascade at full flow -
                                     # 15 s of it at ANIM_SECONDS 30, which is where the extra
                                     # length went. The reveal did not need to be slower; the
                                     # subject needed longer on screen.
ANIM_ADVANCE_EASE = True             # smooth start/stop instead of a mechanical linear ramp
ANIM_ADVANCE_FROM = 0.32             # the reveal starts HERE, not at 0. The corridor's
                                     # along-path coordinate is normalised over the WHOLE
                                     # route, most of which is off to the west behind the
                                     # camera, so a reveal from 0 spends its first seconds
                                     # flooding ground nobody can see. `along` is cumulative-
                                     # distance/total-length, so every waypoint's along-value
                                     # shifts whenever the route's total length changes (e.g. a
                                     # rerouted outflow tail) - re-measure this whenever
                                     # MEGAFLOOD_PATH or the camera changes, by scaling the old
                                     # value by the LIP_ALONG ratio and confirming against a
                                     # true frame-1 render through the real ANIMATE=True path
                                     # (see LIP_ALONG's comment on why ANIMATE=False can't be
                                     # used for this measurement).

# --- the Ionian starts DESICCATED and fills ------------------------------------------------
# The still shows the sea already standing at PALEO_SEA_LEVEL_M (-2200 m), which is a lowstand
# remnant, not a modern sea. In a clip that is a contradiction: the flood arrives at a basin
# that is already full, so the cascade pours into water it is supposed to be delivering. With
# the fill on, frame 1 is a bone-dry Mediterranean and the sea rises only once the torrent goes
# over the lip - which is the event the figure is about.
#
# This costs nothing in the merge: the sea floor is already painted as DRY ground in both
# colormaps (identical RGB in colormap.png and colormap_dry.png), and the flood sheet's own
# reach past the shoreline stays inside its FLOOD_SEA_MERGE_M dissolve band - so taking the sea
# away leaves no water slab hanging over the empty basin.
ANIM_SEA_FILL        = True
ANIM_SEA_FILL_FROM_M = None          # start level; None = the frame's own zmin_m, i.e. the plane
                                     # sits exactly at the lowest ground so there is no water
                                     # anywhere and every metre of the ramp is on screen. A
                                     # deeper start just wastes the first part of the fill.
ANIM_SEA_FILL_LAG_S  = 0.5           # after the front reaches the lip. Small but not zero: the
                                     # basin must be seen to be dry at the moment the water
                                     # arrives, otherwise the fill reads as a level that was
                                     # rising anyway.
ANIM_SEA_FILL_S      = 13.0          # ramp duration. It ends before the last frame on purpose -
                                     # the clip then holds a few seconds at exactly
                                     # PALEO_SEA_LEVEL_M, so its final frame IS the published
                                     # figure. Note this is not a discharge rate: filling
                                     # ~12600 km3 would take minutes at any plausible discharge,
                                     # and the advance across the basin is compressed just as
                                     # hard. The clip is a schematic of the sequence, not a
                                     # timeline, and the caption should say so.

ANIM_WATER_MOTION   = True
ANIM_FLOW_DIR_DEG   = 100.0          # compass bearing the torrent FLOWS TOWARD at the cascade
                                     # (the route runs W->E and plunges ESE). Only the surface
                                     # TEXTURE is advected along it, so this is one mean
                                     # direction, not a flow field - which is all that reads at
                                     # 448 km with the chute occupying ~13 km of frame.
# Speeds. NOT physical velocities - they are chosen so the motion READS at 448 km. The hard
# ceiling is aliasing: a pattern strobes once it moves more than about half its own wavelength
# per frame. Those wavelengths are known, so the ceiling is arithmetic rather than taste:
#
#   flood sheet   FLOOD_WAVE_SCALE 90 over 280 km  -> lambda 3.1 km  -> ceiling ~39 km/s
#   sea           SEA_SPARKLE_SCALE 480 over 700 km -> lambda 1.5 km -> ceiling ~18 km/s
#   (Generated coordinates span the object, hence the division by its width; the sea plane is
#    WATER_MARGIN = 2.5 times the terrain.)
#
# A speed using only 2-8% of the ceiling reads as no motion at all (indistinguishable
# frame-to-frame from static terrain); these sit at 15-25%, visible motion with a 4x margin
# against strobing.
ANIM_FLOW_KM_S      = 10.0           # ~26 % of the sheet's aliasing ceiling
ANIM_SEA_KM_S       = 2.5            # the Ionian is the calm receiving basin - it should still
                                     # only crawl, but slow enough to be imperceptible defeats
                                     # the point
# Label anchors that only the 16:9 crop needs. Keyed by the label's text; the value replaces
# that entry's (x_m, y_m, elev_m) in GEO_LABELS_LIST for the animation only, so the printed
# figure keeps the anchors it was published with.
# "Hyblean Plateau" has to move because its still anchor projects to y = -0.03 in 16:9, i.e.
# just off the top edge. The replacement was not nudged by eye: every DEM point above 300 m
# was projected through the actual camera at the TIGHTEST lens (ANIM_PUSH_LENS_MM, where the
# frame is smallest and a label is most likely to fall out), and this is the HIGHEST ground
# that lands inside it - 20 km east, on the same plateau, +643 m. Re-derive the same way if
# the crop, the lens or the camera changes: read elevations off the heightmap, run each
# candidate through _crs_to_scene + the lift/camera-pull make_geo_labels applies, and project
# with bpy_extras world_to_camera_view. Do not nudge it by eye - the label has to sit on
# ground that IS the named landform.
ANIM_GEO_LABEL_MOVES = {
    "Hyblean Plateau": (495000.0, 4095000.0, 642.7),
}

ANIM_FLOOD_WAVE_BOOST = 2.5          # multiplies FLOOD_WAVE_STRENGTH / FLOOD_SWELL_STRENGTH for
                                     # the ANIMATION ONLY - the still on main is untouched.
                                     # Motion on the flood sheet is inherently weaker than on the
                                     # sea, and not because the drift speeds are too low: they sit
                                     # at ~26 % of the aliasing ceiling. FLOOD_SHEET_IOR = 1.0
                                     # deliberately kills the sheet's Fresnel reflection so the
                                     # torrent reads as matte sediment water, and a bump layer can
                                     # only modulate what a surface reflects - with no reflection
                                     # there is almost nothing for a moving normal to change. The
                                     # amplitude is the limit, so the amplitude is what is raised.
                                     # Raising IOR instead would work far better and was rejected:
                                     # it costs the matte look that separates the torrent from the
                                     # blue Ionian, which is a figure decision, not a video one.
                                     # 1.0 = exactly the still's water.
ANIM_MIST_MOTION    = True           # billow + lift the spray plume. Separate switch because it
                                     # touches the volume shader, and Cycles 5.1 has a history
                                     # of crashing on those (see make_plunge_mist) - if a test
                                     # frame dies with EXCEPTION_ACCESS_VIOLATION, turn this off
                                     # first and the rest of the motion still works.
ANIM_MIST_DRIFT_FRAC = 0.08          # the plume's billows drift at this fraction of
                                     # ANIM_FLOW_KM_S. Spray churns but does not race the
                                     # torrent that made it - and the mist box is only ~20 km
                                     # across, so a large fraction sweeps the whole plume
                                     # downwind within the clip. This holds the drift near
                                     # 0.8 km/s, i.e. about one box-width over 24 s.
ANIM_MIST_RISE_KM_S = 0.25           # spray rises as it billows; slow, or it reads as smoke
ANIM_CLOUD_KM_S     = 3.0            # cloud shadows drifting downwind over the terrain: the
                                     # cheapest motion cue in the whole clip, and the only one
                                     # that moves ground which is otherwise completely static.
                                     # Sized against CLOUD_SCALE_KM (38 km): at 3 km/s a shadow
                                     # crosses the ~100 km frame in about half a minute, which
                                     # is weather; much slower and the drift is imperceptible
                                     # over the clip's length.

ANIM_CAMERA       = "push"           # "static" | "push"
ANIM_PUSH_LENS_MM = 186.0            # "push": a slow zoom CAM_LENS_MM -> this. Zooming IN only:
                                     # the 160 mm framing was chosen so the DEM's cut-edge void
                                     # sits just outside the frame, so anything wider exposes it.
                                     # The camera's POSITION and DIRECTION never change, which is
                                     # what keeps the billboarded labels, the distance haze and
                                     # the CLOUD_CLEAR_KM zones exactly as tuned for the figure.
                                     # (An orbit cannot: see ANIM_TYPE. The north arrow is a HUD
                                     # sized from the lens, so it is built at the FINAL lens and
                                     # only ever sits further inside the corner before that.)

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
    # location/rotation passed EXPLICITLY: bpy operator properties are sticky, so a bare
    # transform_apply(scale=True) inherits whatever the previous call used - see make_water.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

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
    # flood_advance: the 3D sheet alone carries the flood, so the ground must show
    # the PRE-flood desiccated colour - otherwise the painted-in flood is visible
    # before the water sheet arrives and the "flood rolls in" illusion collapses
    if ANIMATE and ANIM_TYPE == "flood_advance":
        dry_png = meta.get("colormap_dry_png")
        if dry_png and os.path.isfile(os.path.join(DATA_DIR, dry_png)):
            cmap_png = dry_png
            log("terrain colour: using pre-flood colormap (%s) for flood_advance" % dry_png)
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
    tex.name = "terrain_albedo_tex"    # animation handle: _animate_terrain_wetting reroutes
                                       # everything downstream of this through a dry/wet mix
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
    # location=False is LOAD-BEARING, not tidiness. bpy operator properties are sticky: a bare
    # transform_apply(scale=True) reuses the previous call's flags, which baked this plane's
    # sea-level z into its MESH and left object.location.z at 0. Everything that animates the
    # sea (ANIM_TYPE "flood_rise", and the sea fill in "flood_advance") writes world heights
    # into location.z, so
    # with the height hidden in the mesh those writes were offsets on top of it and sank the
    # sea out of the frame instead of raising it.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

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
        # see SEA_PLUME_CORRIDOR_PNG: deliberately NOT the same mask the flood sheet now
        # uses, once MEGAFLOOD_SEA_GRACE_M has widened that one far into the basin.
        path = SEA_PLUME_CORRIDOR_PNG or os.path.join(DATA_DIR, corr)
        if os.path.exists(path):
            # check_existing: when this path is the SAME file the flood sheet already
            # loaded (see SEA_PLUME_CORRIDOR_PNG), reuse that ~90-megapixel image instead
            # of a second float-RGBA copy (~1.4 GB) - this machine is memory-starved enough
            # for that alone to be the difference between finishing and thrashing.
            cimg = bpy.data.images.load(path, check_existing=True)
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

            # gate by ALONG (G channel), not just coverage (R): R=255 the whole length of
            # the corridor, including the calm Sicily-hugging approach belt (measured: R=255,
            # B=0 foam at the coastal "controlling sill" bend, along~0.72) - keying the plume
            # on coverage alone painted a second turbid patch there, far from any real outflow.
            # Restrict it to near the actual plunge (along~0.906, see LIP_ALONG) and beyond,
            # same fix already used for the mist plume's MEGAFLOOD_MIST_LIP_WINDOW.
            pgate = nt.nodes.new("ShaderNodeMapRange")
            pgate.inputs["From Min"].default_value = float(SEA_PLUME_ALONG_ON)
            pgate.inputs["From Max"].default_value = float(SEA_PLUME_ALONG_FULL)
            pgate.inputs["To Min"].default_value = 0.0
            pgate.inputs["To Max"].default_value = 1.0
            try:
                pgate.clamp = True
            except AttributeError:
                pass
            nt.links.new(csep.outputs["Green"], pgate.inputs["Value"])

            pcov = nt.nodes.new("ShaderNodeMath")
            pcov.operation = "MULTIPLY"; pcov.use_clamp = True
            nt.links.new(csep.outputs["Red"], pcov.inputs[0])
            nt.links.new(pgate.outputs["Result"], pcov.inputs[1])

            pfac = nt.nodes.new("ShaderNodeMath")
            pfac.name = "sea_plume_strength"   # animation handle: the plume must not already
                                               # stain the Ionian before the flood reaches it
            pfac.operation = "MULTIPLY"; pfac.use_clamp = True
            pfac.inputs[1].default_value = float(SEA_PLUME_STRENGTH)
            nt.links.new(pcov.outputs["Value"], pfac.inputs[0])

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

    # crop to the wet corridor's own bbox (see FLOOD_SHEET_TARGET_QUAD_M above) if the
    # merge exported one; otherwise fall back to the old full-frame grid.
    crop = None
    bbox = meta.get("flood_sheet_bbox_uv")
    if bbox:
        u0, v0, u1, v1 = [float(t) for t in bbox]
        # raster (u, v) -> scene km, same convention as the mist box: row 0 (v=0) is the
        # NORTH edge (+y), so y needs the swap that x does not.
        cx0, cx1 = (u0 - 0.5) * width_km, (u1 - 0.5) * width_km
        cy0, cy1 = (0.5 - v1) * height_km, (0.5 - v0) * height_km
        crop_w, crop_h = cx1 - cx0, cy1 - cy0
        if crop_w > 0 and crop_h > 0:
            crop = (cx0, cx1, cy0, cy1, crop_w, crop_h)
    if crop is None:
        if bbox:
            log("flood sheet: degenerate flood_sheet_bbox_uv - using the full-frame grid")
        else:
            log("flood sheet: no flood_sheet_bbox_uv in meta - using the full-frame grid "
                "(re-run the merge for the cropped, higher-resolution sheet)")
        cuts_x = cuts_y = SUBDIVISIONS
        grid_size, grid_loc, y_scale = width_km, (0.0, 0.0, 0.0), height_km / width_km
    else:
        cx0, cx1, cy0, cy1, crop_w, crop_h = crop
        raw_x = crop_w * 1000.0 / FLOOD_SHEET_TARGET_QUAD_M
        raw_y = crop_h * 1000.0 / FLOOD_SHEET_TARGET_QUAD_M
        if raw_x * raw_y > FLOOD_SHEET_MAX_VERTS:
            # scale BOTH axes down together so quads stay square instead of clamping one
            # axis independently (which would happen for a bbox whose aspect ratio differs
            # a lot from 1:1, e.g. the video merge's much-wider grace-extended corridor)
            s = (FLOOD_SHEET_MAX_VERTS / (raw_x * raw_y)) ** 0.5
            raw_x *= s
            raw_y *= s
        cuts_x = max(round(raw_x), FLOOD_SHEET_MIN_SUBDIV)
        cuts_y = max(round(raw_y), FLOOD_SHEET_MIN_SUBDIV)
        grid_size = crop_w
        grid_loc = ((cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0, 0.0)
        y_scale = crop_h / crop_w

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=cuts_x, y_subdivisions=cuts_y,
        size=grid_size, location=grid_loc)
    sheet = bpy.context.active_object
    sheet.name = "flood_sheet"
    sheet.scale = (1.0, y_scale, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if crop is not None:
        # remap the default 0..1 UV (which now spans only the crop) onto the matching
        # sub-rectangle of the FULL-frame default UV space, so the displace modifiers
        # (texture_coords="UV") and the material's image lookups keep sampling the right
        # part of heightmap/corridor/relief textures that still cover the whole DEM.
        Uu0, Uu1 = 0.5 + cx0 / width_km, 0.5 + cx1 / width_km
        Vv0, Vv1 = 0.5 + cy0 / height_km, 0.5 + cy1 / height_km
        uv_layer = sheet.data.uv_layers.active
        if uv_layer is not None:
            for loop in uv_layer.data:
                lu, lv = loop.uv
                loop.uv = (Uu0 + lu * (Uu1 - Uu0), Vv0 + lv * (Vv1 - Vv0))
        log(f"flood sheet: cropped to {crop_w:.0f}x{crop_h:.0f} km "
            f"({cuts_x}x{cuts_y} cuts, quad ~{crop_w * 1000.0 / cuts_x:.0f} m)")

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
    that gates the reveal (source->lip) for the flood_advance animation.

    sea_z_km/merge_km fade the sheet out as its surface reaches the sea plane, so the
    torrent dissolves into the sea instead of ending at a material seam (FLOOD_SEA_MERGE_M).
    """
    img = bpy.data.images.load(corr_path, check_existing=True)
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

    # Gaussian feathers (EXTENT_SMOOTH_M etc.) never reach exactly zero - coverage(R) carries
    # a tiny, never-quite-zero tail across the WHOLE grid, decaying with distance from the
    # corridor. The mesh spans the entire regional frame regardless of coverage, so
    # thresholding it here is cheap insurance; cov_cut also gates Transmission Weight below,
    # which matters independently: alpha blending only mixes the surface result with what's
    # behind it, it does not disable the BSDF's own transmission lobe, so a coloured
    # Transmission keeps tinting the background through the "invisible" sheet everywhere the
    # geometry exists, even where alpha is already ~0.
    cov_cut = nt.nodes.new("ShaderNodeMapRange")
    cov_cut.inputs["From Min"].default_value = 0.02
    cov_cut.inputs["From Max"].default_value = 0.05
    cov_cut.inputs["To Min"].default_value = 0.0
    cov_cut.inputs["To Max"].default_value = 1.0
    try:
        cov_cut.clamp = True
    except AttributeError:
        pass
    nt.links.new(sep.outputs["Red"], cov_cut.inputs["Value"])

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
    trans_mul = nt.nodes.new("ShaderNodeMath"); trans_mul.operation = "MULTIPLY"
    trans_mul.inputs[1].default_value = float(FLOOD_SHEET_TRANSMISSION)
    nt.links.new(cov_cut.outputs["Result"], trans_mul.inputs[0])
    for _name in ("Transmission Weight", "Transmission"):
        try:
            nt.links.new(trans_mul.outputs["Value"], bsdf.inputs[_name])
            break
        except KeyError:
            continue
    else:
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
        hadd.inputs[1].default_value = 0.25      # was 0.6: foam-driven raised ridges
                                                 # caught the sky as hard white scratches
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

    # alpha = coverage(R, thresholded) * reveal * sea-dissolve * global alpha
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    nt.links.new(cov_cut.outputs["Result"], m1.inputs[0])
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
        # From Min is a NAMED Value node, not a bare float: ANIM_SEA_FILL keyframes the sea
        # plane's world Z (dry -> paleo_sea_level_m), and this dissolve has to track that same
        # curve. A fixed sea_z_km (the FINAL level) is fine for the still and for the old,
        # narrow ~19 km2 flood-sheet/sea overlap - but MEGAFLOOD_SEA_GRACE_M now lets the
        # sheet's coverage reach deep into the basin, where terrain sits well below both the
        # current AND final sea Z for most of the fill ramp. Dissolving against the FINAL level
        # while the sea object is still low left the sheet solid over water the sea hadn't
        # risen to yet - a pale, flatly-lit patch where the matte, non-reflective flood-sheet
        # material (FLOOD_SHEET_IOR=1.0) showed through instead of the glossy sea. See the
        # ANIM_SEA_FILL block, which keyframes this node's output with the identical curve.
        seaz = nt.nodes.new("ShaderNodeValue")
        seaz.name = "flood_sheet_sea_z"
        seaz.label = "flood sheet sea z"
        seaz.outputs[0].default_value = float(sea_z_km)
        nt.links.new(seaz.outputs[0], fade.inputs["From Min"])
        addm = nt.nodes.new("ShaderNodeMath"); addm.operation = "ADD"
        addm.inputs[1].default_value = float(merge_km)
        nt.links.new(seaz.outputs[0], addm.inputs[0])
        nt.links.new(addm.outputs["Value"], fade.inputs["From Max"])
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
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

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
    d3.name = "mist_density"           # animation handle: a spray plume standing over a DRY
                                       # canyon before the water arrives is the single most
                                       # obvious way to give the clip away as a fake
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
    noise.name = "cloud_noise"                  # animation handle: the drift is applied to the
                                                # NOISE branch only. Translating the plane itself
                                                # would drag the CLOUD_CLEAR_KM zones along with
                                                # it and un-protect the canyon it is shading.
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
    """Billboarded place-name labels - see the CONFIG block for the manuscript-checked
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


def _apply_anim_overrides():
    """Animation quality overrides, applied to the module globals BEFORE the scene is built.

    They have to be globals, not arguments: the build functions read the CONFIG block
    directly (that IS the configuration model here), and several of them run long before
    setup_animation() gets a say - make_north_arrow in particular sizes the HUD from
    RES_X/RES_Y and the camera lens, so it has to see the animation's values already.
    """
    global RES_X, RES_Y, SAMPLES, MIST_STEP_RATE
    global FLOOD_WAVE_STRENGTH, FLOOD_SWELL_STRENGTH, GEO_LABELS_LIST
    RES_X, RES_Y = ANIM_RES_X, ANIM_RES_Y
    SAMPLES = ANIM_SAMPLES
    MIST_STEP_RATE = ANIM_MIST_STEP_RATE
    # The sheet's bump amplitude is a look setting of the FIGURE, so it is raised here rather
    # than in the CONFIG block: the still keeps its published value, the clip gets a surface
    # that visibly moves. See ANIM_FLOOD_WAVE_BOOST for why amplitude and not speed.
    if ANIM_WATER_MOTION and ANIM_FLOOD_WAVE_BOOST != 1.0:
        FLOOD_WAVE_STRENGTH *= float(ANIM_FLOOD_WAVE_BOOST)
        FLOOD_SWELL_STRENGTH *= float(ANIM_FLOOD_WAVE_BOOST)
    if ANIM_GEO_LABEL_MOVES:
        GEO_LABELS_LIST = [(t, *ANIM_GEO_LABEL_MOVES.get(t, (x, y, e)))
                           for t, x, y, e in GEO_LABELS_LIST]
        for t in ANIM_GEO_LABEL_MOVES:
            log(f"anim: label '{t}' re-sited for the {ANIM_RES_X}x{ANIM_RES_Y} crop")


def _iter_fcurves(action):
    """Every f-curve of an action, on both the legacy and the layered (4.4+) data model.

    Blender 4.4 moved animation into layers/strips/channelbags and left Action.fcurves as a
    compatibility shim that can come back empty. Reading only .fcurves is how an eased curve
    silently stays on the default interpolation.
    """
    if action is None:
        return
    for fc in getattr(action, "fcurves", []) or []:
        yield fc
    for layer in getattr(action, "layers", []) or []:
        for strip in getattr(layer, "strips", []) or []:
            for bag in getattr(strip, "channelbags", []) or []:
                for fc in getattr(bag, "fcurves", []) or []:
                    yield fc


def _key(sock, keys, interp="LINEAR"):
    """Keyframe ONE socket: keys = [(frame, value), ...], then set the interpolation on that
    socket's own f-curve only. Per-socket rather than per-action, because the flood's eased
    reveal and the linear texture drift live in the same node tree and must not share a mode.
    """
    for fr, val in keys:
        sock.default_value = val
        sock.keyframe_insert("default_value", frame=int(round(fr)))
    try:
        path = sock.path_from_id("default_value")
    except Exception:
        return
    ad = getattr(sock.id_data, "animation_data", None)
    for fc in _iter_fcurves(ad.action if ad else None):
        if fc.data_path == path:
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def _key_loc_z(obj, keys, interp="BEZIER"):
    """Keyframe an object's world Z in km: keys = [(frame, z_km), ...].

    Separate from _key() because location is an object property with an ARRAY index rather
    than a node socket, so its f-curve is identified by (data_path 'location', array_index 2).
    Writing world heights here is only correct because make_water no longer bakes the plane's
    height into its mesh - see the transform_apply comment there.
    """
    for fr, val in keys:
        obj.location.z = val
        obj.keyframe_insert("location", index=2, frame=int(round(fr)))
    ad = getattr(obj, "animation_data", None)
    for fc in _iter_fcurves(ad.action if ad else None):
        if fc.data_path == "location" and fc.array_index == 2:
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def _insert_drift_mapping(nt, node):
    """Splice a Mapping node in front of `node`'s Vector input so the pattern can be
    translated per frame. Returns the Mapping, or None if the node has no Vector input.

    The coordinate SOURCE is preserved. A texture node with an unconnected Vector is on
    Generated coords (the implicit default: 0..1 across the object's bounding box), so a
    TexCoord/Generated is wired in explicitly - a no-op at zero offset. One that already
    reads something (the cloud noise reads Object coords) keeps that source. Either way
    every *_SCALE in the CONFIG block keeps exactly the meaning it has in the still.
    """
    vin = node.inputs.get("Vector")
    if vin is None:
        return None
    if vin.is_linked:
        src = vin.links[0].from_socket
    else:
        src = nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = (0.0, 0.0, 0.0)
    nt.links.new(src, mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], vin)     # replaces the old link, if any
    return mp


def _drift_material_noise(mat, delta, n_frames, skip_linked=True):
    """Advect every procedural noise layer of a material by `delta` over the clip.

    Only ShaderNodeTexNoise is touched, and by default only ones whose Vector is unconnected:
    the image textures in these materials are LOOKUPS (the corridor mask, the flood water
    surface, the mist mask), keyed to real positions on the DEM, and sliding one of those
    would move the water rather than its texture.

    Sign: a shader samples noise(P + L), so whatever sat at P+L now appears at P - the
    pattern travels along -L. Callers therefore pass the flow direction and this negates it.
    """
    if mat is None or not mat.use_nodes:
        return 0
    nt = mat.node_tree
    n = 0
    for node in [x for x in nt.nodes if x.type == "TEX_NOISE"]:
        if skip_linked and node.inputs["Vector"].is_linked:
            continue
        mp = _insert_drift_mapping(nt, node)
        if mp is None:
            continue
        _key(mp.inputs["Location"],
             [(1, (0.0, 0.0, 0.0)), (n_frames, tuple(-d for d in delta))], "LINEAR")
        n += 1
    return n


def _obj_span_km(obj, fallback=(1.0, 1.0, 1.0)):
    """The object's own bounding-box size in km - the denominator that turns a drift in km
    into the 0..1 units Generated coordinates actually use."""
    try:
        dim = [abs(v) for v in obj.dimensions]
    except Exception:
        return fallback
    return tuple(d if d > 1e-6 else f for d, f in zip(dim, fallback))


def animate_scene_motion(scene, meta, ve, width_km, height_km,
                         water=None, flood_sheet=None, mist=None, clouds=None):
    """Put the water, the spray and the weather in motion.

    Without this the clip is a wipe over a still photograph: the corridor's alpha rises but
    the ripples, the foam streaks and the plume never move. Everything here is an advected
    TEXTURE, never geometry - the cascade's standing-wave relief stays exactly where the
    merge put it, which is both correct (standing waves do not travel) and necessary (the
    relief is spatially gated to the chute; sliding it would smear rollers onto the flats).
    """
    if not ANIM_WATER_MOTION:
        return
    n_f = ANIM_FRAMES
    secs = ANIM_SECONDS
    br = math.radians(ANIM_FLOW_DIR_DEG)
    flow = (math.sin(br), math.cos(br))          # compass bearing -> scene XY unit vector

    moved = []
    if flood_sheet is not None:
        sx, sy, _ = _obj_span_km(flood_sheet, (width_km, height_km, 1.0))
        d = ANIM_FLOW_KM_S * secs
        n = _drift_material_noise(flood_sheet.active_material,
                                  (flow[0] * d / sx, flow[1] * d / sy, 0.0), n_f)
        if n:
            moved.append(f"flood sheet x{n} ({d:.0f} km)")

    if water is not None:
        sx, sy, _ = _obj_span_km(water, (width_km * WATER_MARGIN, height_km * WATER_MARGIN, 1.0))
        d = ANIM_SEA_KM_S * secs
        n = _drift_material_noise(water.active_material,
                                  (flow[0] * d / sx, flow[1] * d / sy, 0.0), n_f)
        if n:
            moved.append(f"sea x{n} ({d:.0f} km)")

    if mist is not None and ANIM_MIST_MOTION:
        sx, sy, sz = _obj_span_km(mist)
        d = ANIM_FLOW_KM_S * ANIM_MIST_DRIFT_FRAC * secs   # spray drifts, it does not race
        rise = ANIM_MIST_RISE_KM_S * secs
        n = _drift_material_noise(mist.active_material,
                                  (flow[0] * d / sx, flow[1] * d / sy, rise / sz), n_f)
        if n:
            moved.append(f"spray x{n} ({d:.0f} km + {rise * 1000:.0f} m rise)")

    if clouds is not None and ANIM_CLOUD_KM_S > 0:
        mat = clouds.active_material
        node = mat.node_tree.nodes.get("cloud_noise") if mat else None
        if node is not None:
            # Object coords on this plane ARE km, so no bounding-box division. And the drift
            # goes on the NOISE branch alone: the clear-zone ramps read the same TexCoord and
            # must stay nailed to the canyon they are keeping out of shadow.
            d = ANIM_CLOUD_KM_S * secs
            mp = _insert_drift_mapping(mat.node_tree, node)
            if mp is not None:
                _key(mp.inputs["Location"],
                     [(1, (0.0, 0.0, 0.0)), (n_f, (-flow[0] * d, -flow[1] * d, 0.0))], "LINEAR")
                moved.append(f"cloud shadows ({d:.0f} km)")

    log("anim motion: " + ("; ".join(moved) if moved else "nothing to move"))


def _animate_terrain_wetting(terrain, meta, keys, ease):
    """Paint the flooded colormap in behind the advancing front.

    flood_advance forces the DRY colormap onto the ground, so the merge's painted-in flood
    isn't already lying there before the water arrives. The price is that the ground then
    never wets at all: the sheet reads as cellophane over tan sediment, and the clip's final
    frame no longer matches the published figure, whose bed carries the scour and the
    braided channels the merge baked into colormap.png.

    So BOTH colormaps are loaded and cross-faded per pixel by exactly the reveal the sheet
    uses - corridor coverage (R) gated by the along-path coordinate (G) against the same
    advance value. Behind the front the bed is the figure's; ahead of it, untouched
    desiccated basin. Costs one extra full-size image in memory (see the mist's note on what
    Blender does to those), which is why it only ever runs for this one animation type.
    """
    if terrain is None:
        return
    mat = terrain.active_material
    if mat is None or not mat.use_nodes:
        return
    nt = mat.node_tree
    dry = nt.nodes.get("terrain_albedo_tex")
    wet_png = meta.get("colormap_png")
    corr_png = FLOOD_CORRIDOR_PNG or meta.get("flood_corridor_png")
    if dry is None or not wet_png or not corr_png:
        log("anim: no dry/wet ground reveal (needs the photoreal albedo material + both maps)")
        return
    wet_path = os.path.join(DATA_DIR, wet_png)
    corr_path = os.path.join(DATA_DIR, corr_png)
    if not (os.path.isfile(wet_path) and os.path.isfile(corr_path)):
        return

    # check_existing: the corridor mask is already loaded by the flood sheet and the sea, and
    # Blender expands every image to float RGBA - a third copy of a 6134x4533 mask is ~445 MB
    wet = nt.nodes.new("ShaderNodeTexImage")
    wet.image = bpy.data.images.load(wet_path, check_existing=True)
    wet.extension = "EXTEND"
    cimg = bpy.data.images.load(corr_path, check_existing=True)
    try:
        cimg.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    ctex = nt.nodes.new("ShaderNodeTexImage")
    ctex.image = cimg
    ctex.extension = "EXTEND"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(ctex.outputs["Color"], sep.inputs["Color"])

    adv = nt.nodes.new("ShaderNodeValue")
    adv.name = "terrain_advance_value"
    sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"
    nt.links.new(adv.outputs[0], sub.inputs[0])
    nt.links.new(sep.outputs["Green"], sub.inputs[1])
    rev = nt.nodes.new("ShaderNodeMath"); rev.operation = "MULTIPLY"; rev.use_clamp = True
    rev.inputs[1].default_value = 1.0 / max(FLOOD_FRONT_SOFTNESS, 1e-3)
    nt.links.new(sub.outputs["Value"], rev.inputs[0])
    fac = nt.nodes.new("ShaderNodeMath"); fac.operation = "MULTIPLY"; fac.use_clamp = True
    nt.links.new(sep.outputs["Red"], fac.inputs[0])
    nt.links.new(rev.outputs["Value"], fac.inputs[1])

    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    # Snapshot the dry albedo's consumers BEFORE rewiring: the link collection is live, so
    # relinking while iterating it would also catch the mix's own new input and feed the node
    # to itself.
    consumers = [l.to_socket for l in dry.outputs["Color"].links]
    for sock in consumers:
        nt.links.new(mix.outputs["Color"], sock)
    nt.links.new(dry.outputs["Color"], mix.inputs["Color1"])
    nt.links.new(wet.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(fac.outputs["Value"], mix.inputs["Fac"])

    _key(adv.outputs[0], keys, ease)
    log(f"anim: ground wets behind the front ({wet_png} over {len(consumers)} consumer(s))")


def _png_size(path):
    """(width, height) from a PNG's IHDR, without loading the pixels.

    Parsed by hand rather than via bpy.data.images.load: this runs before the render and
    must not add another multi-hundred-MB float image to a scene that already holds five.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return None
        return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
    except Exception:
        return None


def _check_resume_frames(anim_dir, lo, hi):
    """Warn if frames already on disk were rendered at a different resolution.

    ANIM_RESUME skips whatever is already there - which is exactly what makes a low-res test
    run into the default ANIM_DIR dangerous: those frames get kept, the rest come out at full
    size, and the mismatch only surfaces at encode time (video_gen skips odd-sized frames, so
    the clip silently loses them). Cheap to detect here, expensive to discover on Monday.
    """
    for n in range(lo, hi + 1):
        p = os.path.join(anim_dir, f"frame_{n:04d}.png")
        if not os.path.isfile(p):
            continue
        size = _png_size(p)
        if size and size != (RES_X, RES_Y):
            print(f"  WARNING: {os.path.basename(p)} is {size[0]}x{size[1]}, this run is "
                  f"{RES_X}x{RES_Y}.")
            print("           ANIM_RESUME will KEEP it. Empty the anim folder, or point "
                  "ANIM_DIR elsewhere.")
            return False
    return True


def _anim_phase_frames():
    """(hold_end, advance_end) in frames. The reveal must FINISH well before the clip does,
    or the cascade at full discharge - the actual subject - is never seen running."""
    hold = max(1.0, ANIM_HOLD_S * ANIM_FPS)
    adv = hold + max(1.0, ANIM_ADVANCE_S * ANIM_FPS)
    return min(hold, ANIM_FRAMES - 1), min(adv, ANIM_FRAMES)


def setup_animation(scene, meta, ve, water, cam, target, flood_sheet=None,
                    mist=None, clouds=None, terrain=None, width_km=1.0, height_km=1.0):
    """Keyframe the flood, the camera and the scene motion; configure the frame output."""
    scene.frame_start = max(1, int(ANIM_FRAME_START))
    scene.frame_end = ANIM_FRAMES
    scene.render.fps = ANIM_FPS
    hold_f, adv_f = _anim_phase_frames()
    ease = "BEZIER" if ANIM_ADVANCE_EASE else "LINEAR"

    # Denoising + a FIXED seed. With the seed held constant the residual noise pattern is
    # identical in every frame, so it reads as a static grain rather than boiling between
    # frames - which is exactly the artefact that makes a low-sample Cycles clip look cheap.
    try:
        scene.cycles.use_denoising = bool(ANIM_DENOISE)
        scene.cycles.use_animated_seed = False
        scene.render.use_motion_blur = False     # nothing here moves in geometry, and a
                                                 # science figure wants crisp frames anyway
    except AttributeError:
        pass
    # PERSISTENT DATA: keep the synced scene and its BVH resident between frames instead of
    # rebuilding them every time. This is exactly the case it exists for - NOTHING in this
    # animation moves in geometry: the terrain and the flood sheet are displaced once, the
    # camera never changes position (ANIM_CAMERA only touches the lens), and all the motion
    # lives in shader values. Rebuilding a BVH over ~12 M quads plus five large images for
    # every one of several hundred frames is pure repeated cost. It trades memory for it, so
    # if a run dies with an allocation failure this is the first switch to turn off.
    if ANIM_PERSISTENT_DATA:
        try:
            scene.render.use_persistent_data = True
        except AttributeError:
            pass

    if ANIM_TYPE in ("flood_rise", "flood_and_orbit") and water is not None:
        z0 = ve * FLOOD_RISE_FROM_M / 1000.0
        z1 = ve * meta["paleo_sea_level_m"] / 1000.0
        _key_loc_z(water, [(1, z0), (ANIM_FRAMES, z1)], "LINEAR")
        log(f"anim: flood rises {FLOOD_RISE_FROM_M:.0f} -> {meta['paleo_sea_level_m']:.0f} m")

    if ANIM_TYPE in ("orbit", "flood_and_orbit"):
        pivot = bpy.data.objects.new("orbit_pivot", None)
        bpy.context.collection.objects.link(pivot)
        pivot.location = target.location
        cam.parent = pivot
        cam.matrix_parent_inverse = pivot.matrix_world.inverted()
        pivot.rotation_euler = (0, 0, 0); pivot.keyframe_insert("rotation_euler", index=2, frame=1)
        pivot.rotation_euler = (0, 0, math.radians(360)); pivot.keyframe_insert("rotation_euler", index=2, frame=ANIM_FRAMES)
        # constant-speed orbit
        if pivot.animation_data:
            for fc in _iter_fcurves(pivot.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
        log("anim: camera orbit 360 deg (annotations/haze are NOT valid off-axis)")

    if ANIM_TYPE == "flood_advance" and flood_sheet is not None:
        mat = flood_sheet.active_material
        node = mat.node_tree.nodes.get("flood_advance_value") if mat else None
        if node is not None:
            a0 = float(ANIM_ADVANCE_FROM)
            adv_keys = [(1, a0), (hold_f, a0), (adv_f, 1.0), (ANIM_FRAMES, 1.0)]
            _key(node.outputs[0], adv_keys, ease)
            log(f"anim: flood advances {a0:.2f} -> 1 over frames {hold_f:.0f}-{adv_f:.0f}, "
                f"then {(ANIM_FRAMES - adv_f) / ANIM_FPS:.1f} s at full discharge")

            # the ground wets in step with the sheet - same curve, same reveal maths
            _animate_terrain_wetting(terrain, meta, adv_keys, ease)

            # The spray plume and the sea's turbidity plume are CONSEQUENCES of the water
            # arriving, so they cannot pre-date it. Both are held at zero and brought up as
            # the front reaches the lip - the sediment plume trailing, because a basin's
            # worth of suspended load takes time to spread into the Ionian.
            # lip_f is the frame at which the reveal value crosses LIP_ALONG (the corridor's
            # own along-path fraction at MEGAFLOOD_PATH's escarpment-lip waypoint, read off
            # flood_corridor.png's G channel there). Solved from reveal(t) = a0 + t*(1-a0):
            # the old hard-coded value is only ever valid for the along-parameterisation it
            # was measured against - it does not survive a0 changing OR the route's TOTAL
            # length changing, because along = cumulative-distance/total-length, so every
            # waypoint's along-value shifts when the route does. Re-measured after rerouting
            # the outflow tail (see MEGAFLOOD_PATH): 177/255 = 0.694 (was 231/255 = 0.906).
            LIP_ALONG = 0.694
            _a0 = float(ANIM_ADVANCE_FROM)
            _t_lip = min(max((LIP_ALONG - _a0) / max(1.0 - _a0, 1e-6), 0.0), 1.0)
            lip_f = hold_f + _t_lip * (adv_f - hold_f)
            if mist is not None:
                mnode = (mist.active_material.node_tree.nodes.get("mist_density")
                         if mist.active_material else None)
                if mnode is not None:
                    _key(mnode.inputs[1],
                         [(1, 0.0), (lip_f, 0.0),
                          (min(adv_f + 1.0 * ANIM_FPS, ANIM_FRAMES), float(MIST_DENSITY)),
                          (ANIM_FRAMES, float(MIST_DENSITY))], "BEZIER")
                    log("anim: spray plume builds with the cascade")
            if water is not None and water.active_material:
                pnode = water.active_material.node_tree.nodes.get("sea_plume_strength")
                if pnode is not None:
                    _key(pnode.inputs[1],
                         [(1, 0.0), (lip_f, 0.0),
                          (min(adv_f + 3.0 * ANIM_FPS, ANIM_FRAMES), float(SEA_PLUME_STRENGTH)),
                          (ANIM_FRAMES, float(SEA_PLUME_STRENGTH))], "BEZIER")
                    log("anim: sea turbidity plume builds after the plunge")

            # The Ionian is DRY until the torrent delivers it. Same rule as the two plumes -
            # a consequence must not pre-date its cause - but this one is the whole point of
            # the clip: without it the flood pours into a basin that is already full.
            if ANIM_SEA_FILL and water is not None:
                z_dry = ve * (meta["zmin_m"] if ANIM_SEA_FILL_FROM_M is None
                              else ANIM_SEA_FILL_FROM_M) / 1000.0
                z_full = ve * meta["paleo_sea_level_m"] / 1000.0
                fill0 = lip_f + ANIM_SEA_FILL_LAG_S * ANIM_FPS
                fill1 = min(fill0 + ANIM_SEA_FILL_S * ANIM_FPS, ANIM_FRAMES)
                _key_loc_z(water, [(1, z_dry), (fill0, z_dry),
                                   (fill1, z_full), (ANIM_FRAMES, z_full)], "BEZIER")
                log(f"anim: Ionian fills {z_dry / ve * 1000:.0f} -> "
                    f"{meta['paleo_sea_level_m']:.0f} m over frames "
                    f"{fill0:.0f}-{fill1:.0f}, then holds "
                    f"{(ANIM_FRAMES - fill1) / ANIM_FPS:.1f} s at the figure's level")

                # keep the flood sheet's sea-dissolve locked to the SAME curve as the sea
                # plane it is dissolving into - see the comment at flood_sheet_sea_z's
                # creation. Same keyframe values as the sea plane above, by construction.
                if flood_sheet is not None and flood_sheet.active_material:
                    seaz_node = flood_sheet.active_material.node_tree.nodes.get("flood_sheet_sea_z")
                    if seaz_node is not None:
                        _key(seaz_node.outputs[0], [(1, z_dry), (fill0, z_dry),
                                                    (fill1, z_full), (ANIM_FRAMES, z_full)], "BEZIER")
                        log("anim: flood-sheet sea-dissolve tracks the Ionian fill")
        else:
            log("anim: flood_advance requested but no flood sheet - falling back to a still hold")

    if ANIM_CAMERA == "push" and cam is not None and ANIM_TYPE not in ("orbit", "flood_and_orbit"):
        _key(_LensSock(cam.data), [(1, float(CAM_LENS_MM)), (ANIM_FRAMES, float(ANIM_PUSH_LENS_MM))],
             "BEZIER")
        log(f"anim: camera push {CAM_LENS_MM:.0f} -> {ANIM_PUSH_LENS_MM:.0f} mm")

    animate_scene_motion(scene, meta, ve, width_km, height_km,
                         water=water, flood_sheet=flood_sheet, mist=mist, clouds=clouds)

    anim_dir = ANIM_DIR or os.path.join(DATA_DIR, "anim")
    os.makedirs(anim_dir, exist_ok=True)
    if ANIM_FORMAT.upper() == "MP4":
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        try:
            scene.render.ffmpeg.constant_rate_factor = "HIGH"
        except (AttributeError, TypeError):
            pass
        scene.render.filepath = os.path.join(anim_dir, "flood.mp4")
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.join(anim_dir, "frame_")
        # Resume: skip frames already on disk. Blender only writes a frame's PNG once that
        # frame is complete, so an interrupted run leaves no half-written file to skip over -
        # which is also why use_placeholder stays OFF (placeholders are for splitting a job
        # across machines, and here they would let a crashed frame be mistaken for a done one).
        try:
            scene.render.use_overwrite = not ANIM_RESUME
            scene.render.use_placeholder = False
        except AttributeError:
            pass
    log(f"anim: {ANIM_FRAMES} frames @ {ANIM_FPS} fps -> {scene.render.filepath}")


class _LensSock:
    """Adapter that lets _key() drive camera.data.lens, which is a plain property rather than
    a node socket (same keyframe_insert/path_from_id contract, different attribute name)."""
    def __init__(self, cam_data):
        self._d = cam_data
        self.id_data = cam_data

    @property
    def default_value(self):
        return self._d.lens

    @default_value.setter
    def default_value(self, v):
        self._d.lens = float(v)

    def keyframe_insert(self, _prop, frame=1):
        self._d.keyframe_insert("lens", frame=frame)

    def path_from_id(self, _prop):
        return "lens"


def _render_anim_with_progress(n_frames, anim_dir):
    """Run bpy animation render with a tqdm-style progress bar.

    In QUIET mode the Blender per-sample console spam (Fra:N Mem:… Sample N/N …)
    is redirected to  <anim_dir>/blender_render.log  and our progress bar goes to
    the original terminal via the saved file descriptor.
    """
    import bpy.app.handlers as _hand

    state  = {"n": 0, "t0": time.time(), "term_fd": None}
    w_bar  = 28

    log_path = os.path.join(anim_dir, "blender_render.log")

    # Redirect Blender's C-level stdout/stderr to a log file (QUIET mode only)
    if QUIET:
        try:
            sys.stdout.flush(); sys.stderr.flush()
            term_fd = os.dup(1)            # keep a handle to the real terminal
            log_fd  = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(log_fd, 1)
            os.dup2(log_fd, 2)
            os.close(log_fd)
            state["term_fd"] = term_fd
        except Exception:
            state["term_fd"] = None        # fallback: let Blender spam the console

    def _post(scene, *args):
        state["n"] += 1
        n   = state["n"]
        t   = time.time() - state["t0"]
        rem = (t / n) * (n_frames - n) if n > 0 else 0
        f   = int(w_bar * n / n_frames)
        bar = "█" * f + "░" * (w_bar - f)
        msg = f"\r  [{bar}] {n}/{n_frames}  ~{int(rem//60)}:{int(rem%60):02d} left  "
        fd  = state["term_fd"]
        if fd is not None:
            os.write(fd, msg.encode("utf-8", errors="replace"))
        else:
            sys.stdout.write(msg); sys.stdout.flush()

    _hand.render_post.append(_post)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        if _post in _hand.render_post:
            _hand.render_post.remove(_post)
        fd = state["term_fd"]
        if fd is not None:
            try:
                sys.stdout.flush()
                os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
            except Exception:
                pass
            # print newline after the inline progress bar
            print()


def main():
    global DATA_DIR, OUTPUT_PNG
    # Blender resolves relative image paths against the .blend, not the working directory
    DATA_DIR, OUTPUT_PNG = os.path.abspath(DATA_DIR), os.path.abspath(OUTPUT_PNG)
    if QUIET:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR} (edit the CONFIG block)")
    if ANIMATE:
        _apply_anim_overrides()          # must precede the build - see its docstring
    meta = load_meta()

    bar = _Bar(10)
    clear_scene();                                  bar.step("scene")
    set_engine();                                   bar.step("engine")
    terrain, ve, wkm, hkm = make_terrain(meta);     bar.step("terrain")
    water = make_water(meta, ve, wkm, hkm);         bar.step("water")
    flood_sheet = make_flood_sheet(meta, ve, wkm, hkm); bar.step("flood")
    mist = make_plunge_mist(meta, ve, wkm, hkm);    bar.step("mist")
    make_sun();                                     bar.step("sun")
    clouds = make_cloud_shadows(ve, wkm, hkm)
    make_world();                                   bar.step("world")
    cam, target = make_camera(meta, ve, wkm, hkm);  bar.step("camera")
    # depsgraph must catch up on the camera's TRACK_TO constraint before matrix_world (and
    # therefore world_to_camera_view / the annotation billboarding below) is trustworthy.
    bpy.context.view_layer.update()
    if ANIMATE and ANIM_CAMERA == "push":
        # Build the HUD compass at the FINAL, tightest lens: it is placed at a fixed screen
        # fraction of the frustum, so sizing it for the wide start would walk it out of frame
        # as the push closes in. Sized for the end it only ever sits further into the corner.
        cam.data.lens = float(ANIM_PUSH_LENS_MM)
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

    if ANIMATE:
        anim_dir = ANIM_DIR or os.path.join(DATA_DIR, "anim")
        setup_animation(bpy.context.scene, meta, ve, water, cam, target, flood_sheet,
                        mist=mist, clouds=clouds, terrain=terrain,
                        width_km=wkm, height_km=hkm)
        lo, hi = int(ANIM_FRAME_START), int(ANIM_FRAMES)
        todo = hi - lo + 1
        # On a resume Blender skips frames already on disk WITHOUT firing render_post, so a
        # bar sized to the full range would sit at 3/600 and quote a nonsense ETA. Count what
        # is already there and size the bar to what will actually be rendered.
        done = 0
        if ANIM_RESUME and ANIM_FORMAT.upper() != "MP4":
            _check_resume_frames(anim_dir, lo, hi)
            done = sum(1 for n in range(lo, hi + 1)
                       if os.path.isfile(os.path.join(anim_dir, f"frame_{n:04d}.png")))
            todo = max(todo - done, 1)
        print(f"  rendering {todo} frames  [{ANIM_TYPE}  {ANIM_SECONDS:.0f} s @ {ANIM_FPS} fps  "
              f"{SAMPLES} spp"
              + (f"  resuming, {done} already done]" if done else
                 f"  {'resume on' if ANIM_RESUME else 'resume off'}]"))
        _render_anim_with_progress(todo, anim_dir)
        print(f"  animation done  →  {anim_dir}")
    else:
        print(f"  rendering still  [{SAMPLES} spp] …")
        bpy.ops.render.render(write_still=True)
        print(f"  done  →  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
