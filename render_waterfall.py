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
DATA_DIR   = r"output"
OUTPUT_PNG = r"output\render_waterfall.png"

# --- console ---
QUIET = True                         # clean output: one progress bar + summary. False = verbose debug logs

# --- terrain ---
SUBDIVISIONS = 800                   # grid resolution (cuts per side). 600-1000 is a good range.
VE           = 3                     # vertical exaggeration; None = use ve_hint from metadata

# --- camera (oblique "fly-over" view) ---
CAM_AZIMUTH_DEG  = 310.0             # look direction, clockwise from north
CAM_ALTITUDE_DEG = 24.0             # camera height angle above the horizon
CAM_DISTANCE_KM  = None             # None = auto (~1.6 x scene width); or set in km
CAM_LENS_MM      = 150.0            # focal length (smaller = wider; larger = telephoto zoom)

# --- sun ---
SUN_AZIMUTH_DEG   = 315.0
SUN_ALTITUDE_DEG  = 32.0
SUN_STRENGTH      = 2.5

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
WATER_MARGIN = 1.01                  # plane size vs terrain (1.01 = just covers, no big skirt)

# --- render ---
RENDER_ON_RUN = True
RES_X, RES_Y  = 2400, 1600
SAMPLES       = 128
ENGINE_PREF   = "CYCLES"            # "EEVEE" (fast) or "CYCLES" (slower, photoreal)
BG_COLOR      = (0.66, 0.74, 0.84)  # flat sky colour (used when SKY_TYPE="flat")
BG_STRENGTH   = 0.35               # ambient fill for the flat/emission look

# --- photorealism ---
PHOTOREAL   = True                  # True: real sun + shadows + sky + reflective water (vs flat shadeless map)
SKY_TYPE    = "nishita"             # "nishita" (realistic sun-sky, also lights the scene) | "flat"
SKY_STRENGTH = 0.7                  # Nishita sky brightness: sky ambient + water reflection (low = sun lamp dominates)
EXPOSURE     = -2.0                 # global exposure in stops; lower if washed out, raise toward 0 if too dark
SUN_ANGLE_SOFT_DEG = 1.5            # sun angular size; >0 softens shadow edges (realism)
TERRAIN_ROUGHNESS  = 0.95           # matte ground

# --- realistic water (used when PHOTOREAL) ---
WATER_ROUGHNESS     = 0.15          # diffuse specular to avoid white glare on wide water surfaces
WATER_IOR           = 1.33
WATER_TRANSMISSION  = 0.65          # < 1.0 → base colour visible through water (less transparent)
WATER_WAVE_STRENGTH = 0.10          # subtle ripples — high values cause white specular blowout
WATER_WAVE_SCALE    = 40.0          # medium-scale waves

# --- zoom / framing ---
CAM_TARGET_OFFSET_KM = (0.0, 0.0)   # shift the look-at point (x,y km) to frame the canyon
# to zoom: raise CAM_LENS_MM (telephoto, keeps perspective) or lower CAM_DISTANCE_KM (move in)

# --- animation ---
ANIMATE     = False                 # render an image sequence instead of a single still
ANIM_TYPE   = "flood_rise"          # "flood_rise" | "orbit" | "flood_and_orbit"
ANIM_FRAMES = 120
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

    cmap_png = meta.get("colormap_png")
    has_cmap = cmap_png and os.path.isfile(os.path.join(DATA_DIR, cmap_png))
    if PHOTOREAL and has_cmap:
        _terrain_material_albedo(terrain, os.path.join(DATA_DIR, cmap_png))
    elif USE_COLORMAP and has_cmap:
        _terrain_material_colormap(terrain, os.path.join(DATA_DIR, cmap_png))
    else:
        _terrain_material(terrain, img, meta)
    return terrain, ve, width_km, height_km


def _terrain_material_albedo(obj, colormap_path):
    """Photoreal: baked colormap as diffuse albedo on a Principled BSDF, lit by the
    real sun so canyon walls cast shadows. (Bake the colormap flatter - lower the
    merge hillshade - if the combined shading looks too strong.)"""
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
        bsdf.inputs["Roughness"].default_value = TERRAIN_ROUGHNESS
        bsdf.inputs["Specular IOR Level"].default_value = 0.1
    except KeyError:
        pass
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
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
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, z))
    water = bpy.context.active_object
    water.name = "palaeo_water"
    water.scale = (width_km * WATER_MARGIN, height_km * WATER_MARGIN, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    if PHOTOREAL:
        _water_material_real(water)
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


def _water_material_real(water):
    """Reflective/refractive water with subtle ripples; reflects the sky (use a sky)."""
    mat = bpy.data.materials.new("water_real")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*WATER_COLOR, 1)
    bsdf.inputs["Roughness"].default_value = WATER_ROUGHNESS
    _set_input(bsdf, ["Transmission Weight", "Transmission"], WATER_TRANSMISSION)
    _set_input(bsdf, ["IOR"], WATER_IOR)
    # subtle ripples: noise -> bump -> normal
    if WATER_WAVE_STRENGTH > 0:
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = WATER_WAVE_SCALE
        try:
            noise.inputs["Detail"].default_value = 4.0
        except KeyError:
            pass
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = WATER_WAVE_STRENGTH
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # enable refraction in EEVEE (legacy + Next); Cycles needs nothing
    for attr, val in (("use_screen_refraction", True), ("use_raytrace_refraction", True)):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    water.data.materials.append(mat)


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
        try:
            sky.sky_type = "NISHITA"
            sky.sun_elevation = math.radians(SUN_ALTITUDE_DEG)
            sky.sun_rotation = math.radians(SUN_AZIMUTH_DEG)
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


def setup_animation(scene, meta, ve, water, cam, target):
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

    bar = _Bar(8)
    clear_scene();                                  bar.step("scene")
    set_engine();                                   bar.step("engine")
    terrain, ve, wkm, hkm = make_terrain(meta);     bar.step("terrain")
    water = make_water(meta, ve, wkm, hkm);         bar.step("water")
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
        setup_animation(bpy.context.scene, meta, ve, water, cam, target)
        print(f"  rendering {ANIM_FRAMES} frames  [{ANIM_TYPE}  {SAMPLES} spp  {ANIM_FPS} fps]")
        _render_anim_with_progress(ANIM_FRAMES, anim_dir)
        print(f"  animation done  →  {anim_dir}")
    else:
        print(f"  rendering still  [{SAMPLES} spp] …")
        bpy.ops.render.render(write_still=True)
        print(f"  done  →  {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
