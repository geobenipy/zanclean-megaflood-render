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
from mathutils import Vector

# =====================================================================
# CONFIG
# =====================================================================
#DATA_DIR   = r"output_v11"
#OUTPUT_PNG = r"output_v11\render_waterfall.png"
DATA_DIR   = r"output_hero_v17"
OUTPUT_PNG = r"output_hero_v17\render_waterfall.png"

# --- console ---
QUIET = True                         # clean output: one progress bar + summary. False = verbose debug logs

# --- terrain ---
SUBDIVISIONS = 2400                  # grid resolution (cuts per side). 600-1000 for the
                                     # regional view; 1600 for the hero crop (70 km / 1600
                                     # = 44 m/quad, matches the 25 m heightmap better)
# VE         = 3                     # regional overview
VE           = None                  # None = ve_hint from metadata (hero merge writes 5)

# --- camera (oblique "fly-over" view) ---
#CAM_AZIMUTH_DEG  = 300.0          # regional overview: oblique from the SE, swung ~10 deg
                                  # more from the E (viewer's right) so the DEM cut-edge
                                  # void behind Etna leaves the frame
#CAM_ALTITUDE_DEG = 24.0
#CAM_DISTANCE_KM  = None           # None = auto (~1.6 x scene width)
CAM_AZIMUTH_DEG  = 250.0           # hero: from over the Ionian, looking WSW straight
                                    # at the escarpment wall + the falls in the notch
CAM_ALTITUDE_DEG = 24.0            # LOWER than the old 30: a waterfall is a VERTICAL event, and
                                    # at 30 deg the camera looked down ON the chute so the 1.8 km
                                    # drop foreshortened into a blue patch on a slope. But 18 deg
                                    # (tried) is too low for this crop - the camera sees PAST the
                                    # DEM's north-west edge and the frame fills with the cut-edge
                                    # void plus a 60 km flat lake blowing out white at grazing
                                    # incidence. 24 keeps the drop steep and the data in frame.
CAM_DISTANCE_KM  = 80.0            # hero: pulled back (was 70) so the WHOLE canyon fits
                                    # (hero_v6 was too tight)
#CAM_LENS_MM      = 160.0            # regional overview: zoomed into the escarpment/canyon
                                    # (was 118) so the waterfall reads and the far void goes

CAM_LENS_MM    = 88.0             # hero: TIGHT on the cascade (was 62). 62 fitted the whole
                                    # canyon but spent half the frame on the distant approach
                                    # lake - a flat plate that at a low camera angle reads as a
                                    # blown-out white slab with a straight DEM-edge cut. The
                                    # figure is about the FALL, so crop to it. (Zoom by lens, not
                                    # by CAM_DISTANCE_KM: it keeps the perspective compression.)

# --- sun ---
#SUN_AZIMUTH_DEG = 130.0           # regional overview: ESE (roughly BEHIND the camera) so
                                  # the sun's reflection glitters the foreground Ionian.
                                  # (was 315/NW relief light - swapped for the sea glitter)
SUN_AZIMUTH_DEG   = 85.0            # hero: ENE, ~behind the WSW-facing camera -> lights
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
TERRAIN_COLOR_VAR     = 0.22        # +-fraction albedo brightness variation (sediment patches)
TERRAIN_COLOR_SCALE   = 11.0        # patch scale (bigger number = smaller patches)
TERRAIN_BUMP_STRENGTH = 0.32        # micro rock bump (0 = smooth clay); keep subtle
TERRAIN_BUMP_FEATURE_M = 130.0      # GRAIN SIZE IN METRES. Was TERRAIN_BUMP_SCALE = 300, a
                                    # Generated-coordinate frequency - i.e. 1/300 of the object,
                                    # which is 930 m in the 280 km regional scene and 310 m in the
                                    # 92 km hero. Those are LANDFORMS, not grain: with the rock
                                    # bump multiplier on top they rendered the escarpment as
                                    # blisters / bubble-wrap. Object coordinates are in km, so this
                                    # is a real length and both panels get the same rock.
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
TERRAIN_ROCK_STRENGTH = 0.50        # max mix toward the rock colour on a vertical face
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
HAZE_STRENGTH    = 0.68             # max haze mix on the farthest terrain (0..1)
HAZE_EMIT_STRENGTH = 1.1            # haze brightness (match the horizon sky so terrain blends in)
# ABSOLUTE camera-distance thresholds (km), NOT relative to the shot: the same values keep
# the close hero (terrain ~35-105 km) fully crisp while the wide overview (terrain ~310-590
# km) keeps its NEAR ground crisp and only the FAR mountains recede into haze. START must
# sit near the overview's near edge (~310 km) or the whole overview washes out; the hero
# is entirely nearer than START so it stays haze-free. Lower START to push haze closer.
#HAZE_START_KM    = 340.0          # REGIONAL overview: no haze nearer than this (near-ground crisp)
#HAZE_END_KM      = 620.0          # full HAZE_STRENGTH at/beyond this distance (far mountains)
# HERO: the camera sits ~80 km out, so terrain runs from ~40 km (foreground escarpment) to
# ~140 km (far horizon). Haze from 85 km keeps the falls and the near canyon crisp while the
# far edge of the DEM - the straight cut border with its lit vertical face - dissolves into
# the sky instead of announcing itself as a cliff. Swap the comments for the overview.
HAZE_START_KM    = 85.0
HAZE_END_KM      = 150.0

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
SEA_SWELL_STRENGTH   = 0.30         # large slow swell bump added under the chop (0 = off)
SEA_SWELL_SCALE      = 4.0          # swell size (SMALL number = big rolling waves)
SEA_COLOR_LIGHT      = (0.10, 0.33, 0.60)   # lighter blue for the varying patches
SEA_COLOR_VARIATION  = 0.95         # 0 = uniform WATER_COLOR; ~0.8 = strong patchiness
SEA_COLOR_SCALE      = 1.1          # colour-patch size (small = broad ocean-scale patches)
# --- SUN GLITTER: a fine high-frequency normal layer breaks the mirror into countless
# wavelets so the sun reflection shatters into a glittering path (the single biggest
# "real ocean" cue). Needs a moderate-low WATER_ROUGHNESS so the sun stays a sharp
# highlight for the wavelets to chop up (too rough = the glitter smears into a dull sheen).
SEA_SPARKLE_STRENGTH = 0.18         # fine wavelet normal strength (0 = off). Too high = noisy
SEA_SPARKLE_SCALE    = 900.0        # wavelet frequency (BIG = fine sparkle; ~700-1200)
SEA_SPARKLE_DETAIL   = 3.0          # fractal octaves on the sparkle (crisper glints)
# --- REAL WAVE GEOMETRY: bump alone can't show waves at this scale (a flat plane reflects
# the uniform sky no matter the normal). So DISPLACE actual rolling swell into a subdivided
# sea plane -> the surface genuinely undulates, the horizon ripples, and the sun glitter
# breaks into a wavy field. Amplitude is VE-scaled and deliberately a bit exaggerated so it
# reads from altitude (real 1-3 m waves would be sub-pixel).
SEA_WAVE_GEO            = True       # False = flat plane (old behaviour)
SEA_WAVE_GEO_STRENGTH_M = 24.0      # swell amplitude in metres (VE-scaled). 0 = flat
SEA_WAVE_GEO_SCALE      = 3.5       # swell feature size in km (world/GLOBAL coords)
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
FLOOD_SHEET_ALPHA    = 0.95
FLOOD_SHEET_ROUGHNESS = 0.18         # 0.18, DOWN from 0.55. The high roughness existed to hide a
                                     # problem that no longer exists: with a geometrically FLAT
                                     # sheet, bump normals could not break the sky reflection, so
                                     # the fix was to blur the reflection by roughening the water.
                                     # But rough water under a bright sky scatters it diffusely,
                                     # and the ponded flood came out as a MILKY WHITE SLAB next to
                                     # a deep-blue Ionian - two water bodies, two materials.
                                     # MEGAFLOOD_LAKE_SWELL_M now puts REAL swell geometry on the
                                     # calm water (the same lesson SEA_WAVE_GEO learned), so the
                                     # geometry breaks the reflection and the roughness can go back
                                     # down to the sea's own value. Result: blue water with a sun
                                     # glitter path. Foam still drives roughness up to 0.9 at the
                                     # cascade via the mask's B channel, so whitewater stays matte.
FLOOD_SHEET_TRANSMISSION = 0.30      # light passes through to the painted flooded ground;
                                     # lower = the sheet's own dark blue asserts more (the
                                     # regional flood was washing out pale over light paint).
                                     # 0.30 (was 0.45): shallow bay water stays solid blue
                                     # instead of letting the tan bed mottle through
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
CASCADE_RELIEF_SCALE = 1.0           # multiplier on the baked amplitude - the whole point
                                     # of exporting the relief as its own map is that this
                                     # can be retuned WITHOUT re-running the ~10 min merge.
                                     # >1 = wilder whitewater; 0 = flat sheet (old look).

# --- SPRAY PLUME (3D volume where the cascade lands) ---
# The merge's PLUNGE_MIST_M apron is FLAT - it lies on the water. This is the real thing:
# a bounded Principled Volume box over the cascade whose density comes from flood_mist.png
# and fades out with height above the LOCAL water surface. Needs MEGAFLOOD_MIST_MASK=True.
# (A bounded OBJECT volume is safe; a world Volume Scatter is what turns this scene black.)
PLUNGE_MIST      = True
MIST_DENSITY     = 1.0               # volume density multiplier. The single knob for "how
                                     # much spray", and it is retunable WITHOUT re-running
                                     # the merge. 2.2 (first try) buried the whole canyon in
                                     # a fog bank; 0.25 was a thin haze. 1.0 = a spray veil
                                     # you can still see the cascade through. Scales with how
                                     # many km of volume a camera ray crosses, so a tighter
                                     # crop needs a bit more.
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
#CAM_TARGET_OFFSET_KM = (14.0, -9.0) # regional overview: recentred onto the escarpment
                                     # lip / canyon for the zoomed-in (lens 160) framing
CAM_TARGET_OFFSET_KM = (6.5, 7.0)   # hero crop: same look-at point as the old (0.5, 7.0), just
                                    # re-expressed for the WIDER crop box - the box centre moved
                                    # 6 km west, so the offset has to move 6 km east to keep the
                                    # falls framed where they were.
# to zoom: raise CAM_LENS_MM (telephoto, keeps perspective) or lower CAM_DISTANCE_KM (move in)

# --- animation ---
ANIMATE     = False                 # render an image sequence instead of a single still
ANIM_TYPE   = "flood_advance"       # "flood_rise" | "orbit" | "flood_and_orbit" | "flood_advance"
                                    # flood_advance = the corridor reveals source->lip: the
                                    # megaflood visibly rolls in and hits the falls (hero anim;
                                    # flood_rise only moves the distant sea plane a little)
ANIM_FRAMES = 30*25                 # 30 s @ 25 fps; Cycles needs minutes/frame - keep it short
ANIM_FPS    = 30
ANIM_FORMAT = "PNG"                 # "PNG" (frame sequence) | "MP4" (FFmpeg, built into Blender)
ANIM_DIR    = None                  # None = DATA_DIR/anim
FLOOD_RISE_FROM_M = -3500.0         # flood_rise: start level (deep) -> rises to paleo_sea_level_m

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
    that gates the reveal (source->lip) for the flood_advance animation.

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
    _set_input(bsdf, ["IOR"], WATER_IOR)

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


def setup_animation(scene, meta, ve, water, cam, target, flood_sheet=None):
    """Keyframe a rising flood and/or a camera orbit; configure sequence/video output."""
    scene.frame_start = 1
    scene.frame_end = ANIM_FRAMES
    scene.render.fps = ANIM_FPS

    if ANIM_TYPE in ("flood_rise", "flood_and_orbit") and water is not None:
        z0 = ve * FLOOD_RISE_FROM_M / 1000.0
        z1 = ve * meta["paleo_sea_level_m"] / 1000.0
        water.location.z = z0; water.keyframe_insert("location", index=2, frame=1)
        water.location.z = z1; water.keyframe_insert("location", index=2, frame=ANIM_FRAMES)
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
        if pivot.animation_data and pivot.animation_data.action:
            for fc in pivot.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
        log("anim: camera orbit 360 deg")
    if ANIM_TYPE == "flood_advance" and flood_sheet is not None:
        mat = flood_sheet.active_material
        node = mat.node_tree.nodes.get("flood_advance_value") if mat else None
        if node is not None:
            node.outputs[0].default_value = 0.0
            node.outputs[0].keyframe_insert("default_value", frame=1)
            node.outputs[0].default_value = 1.0
            node.outputs[0].keyframe_insert("default_value", frame=ANIM_FRAMES)
            ad = mat.node_tree.animation_data
            if ad and ad.action:
                for fc in ad.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"
            log("anim: flood advances (corridor reveal 0 -> 1)")
        else:
            log("anim: flood_advance requested but no flood sheet - falling back to a still hold")


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
    log(f"anim: {ANIM_FRAMES} frames @ {ANIM_FPS} fps -> {scene.render.filepath}")


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
    make_world();                                   bar.step("world")
    cam, target = make_camera(meta, ve, wkm, hkm);  bar.step("camera")
    setup_render();                                 bar.step("ready")

    eng = bpy.context.scene.render.engine.replace("BLENDER_", "")
    print(f"  scene {wkm:.0f}×{hkm:.0f} km  VE {ve}  |  {RES_X}×{RES_Y}  {eng}  {SAMPLES} spp  exp {EXPOSURE:+.0f}")

    if not RENDER_ON_RUN:
        print("  scene built (RENDER_ON_RUN=False). Press F12 to render.")
        return

    if ANIMATE:
        anim_dir = ANIM_DIR or os.path.join(DATA_DIR, "anim")
        setup_animation(bpy.context.scene, meta, ve, water, cam, target, flood_sheet)
        print(f"  rendering {ANIM_FRAMES} frames  [{ANIM_TYPE}  {SAMPLES} spp  {ANIM_FPS} fps]")
        _render_anim_with_progress(ANIM_FRAMES, anim_dir)
        print(f"  animation done  →  {anim_dir}")
    else:
        print(f"  rendering still  [{SAMPLES} spp] …")
        bpy.ops.render.render(write_still=True)
        print(f"  done  →  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
