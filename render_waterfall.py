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
import json
import math
from mathutils import Vector

# =====================================================================
# CONFIG
# =====================================================================
DATA_DIR   = r"output"          # folder with heightmap_16bit.png + heightmap_meta.json
OUTPUT_PNG = r"output\render_waterfall.png"

# --- terrain ---
SUBDIVISIONS = 800                   # grid resolution (cuts per side). 600-1000 is a good range.
VE           = 3                  # vertical exaggeration; None = use ve_hint from metadata
 
# --- camera (oblique "fly-over" view) ---
CAM_AZIMUTH_DEG  = 310              # look direction, clockwise from north (20 = looking NNE)
CAM_ALTITUDE_DEG = 24.0             # camera height angle above the horizon
CAM_DISTANCE_KM  = None             # None = auto (~1.6 x scene width); or set in km
CAM_LENS_MM      = 50.0             # focal length (smaller = wider)
 
# --- sun ---
SUN_AZIMUTH_DEG   = 315.0
SUN_ALTITUDE_DEG  = 32.0
SUN_STRENGTH      = 2.5
 
# --- colour rendering ---
VIEW_TRANSFORM = "AGX"          # "Standard" = literal/saturated; "AGX"/"Filmic" desaturate
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
SHOW_WATER  = True
WATER_COLOR = (0.05, 0.22, 0.42)
WATER_ALPHA = 0.75
WATER_MARGIN = 1.01                  # plane size vs terrain (1.01 = just covers, no big skirt)
 
# --- render ---
RENDER_ON_RUN = True
RES_X, RES_Y  = 2400, 1600
SAMPLES       = 64
ENGINE_PREF   = "EEVEE"             # "EEVEE" (fast) or "CYCLES" (slower, prettier)
BG_COLOR      = (0.66, 0.74, 0.84)  # sky colour
BG_STRENGTH   = 0.35               # ambient fill; low so the sun gives relief + colour
 
# =====================================================================
# implementation
# =====================================================================
 
def log(m):
    print(f"[blender] {m}")
 
 
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
    if USE_COLORMAP and cmap_png and os.path.isfile(os.path.join(DATA_DIR, cmap_png)):
        _terrain_material_colormap(terrain, os.path.join(DATA_DIR, cmap_png))
    else:
        _terrain_material(terrain, img, meta)
    return terrain, ve, width_km, height_km
 
 
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
 
    mat = bpy.data.materials.new("water_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*WATER_COLOR, 1)
        bsdf.inputs["Roughness"].default_value = 0.08
        try:
            bsdf.inputs["Alpha"].default_value = WATER_ALPHA
        except KeyError:
            pass
    # transparency (attribute name changed in 4.2 Eevee Next)
    for attr, val in (("blend_method", "BLEND"),
                      ("surface_render_method", "BLENDED")):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass
    water.data.materials.append(mat)
    return water
 
 
def make_sun():
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 50))
    sun = bpy.context.active_object
    sun.data.energy = SUN_STRENGTH
    az = math.radians(SUN_AZIMUTH_DEG)
    alt = math.radians(SUN_ALTITUDE_DEG)
    # point the sun: rotate so it shines from the given az/alt
    sun.rotation_euler = (math.radians(90) - alt, 0, -az)
    return sun
 
 
def make_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*BG_COLOR, 1)
        bg.inputs["Strength"].default_value = BG_STRENGTH
 
 
def make_camera(meta, ve, width_km, height_km):
    # target at scene centre, at mid terrain height
    tz = ve * ((meta["zmin_m"] + meta["zmax_m"]) / 2.0) / 1000.0
    target = bpy.data.objects.new("cam_target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0, tz)
 
    dist = CAM_DISTANCE_KM if CAM_DISTANCE_KM else 1.6 * max(width_km, height_km)
    az = math.radians(CAM_AZIMUTH_DEG)
    alt = math.radians(CAM_ALTITUDE_DEG)
    # look direction (toward az); place camera opposite, elevated
    look = Vector((math.sin(az), math.cos(az), 0))
    cam_xy = -dist * math.cos(alt) * look
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
    return cam
 
 
def setup_render():
    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = OUTPUT_PNG
 
 
def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR} (edit the CONFIG block)")
    log("clearing scene ...")
    clear_scene()
    set_engine()
    meta = load_meta()
    log(f"heightmap: {meta['ncols']}x{meta['nrows']}  z {meta['zmin_m']:.0f}..{meta['zmax_m']:.0f} m")
 
    terrain, ve, wkm, hkm = make_terrain(meta)
    log(f"terrain built: {wkm:.1f} x {hkm:.1f} km, VE={ve}")
    make_water(meta, ve, wkm, hkm)
    make_sun()
    make_world()
    make_camera(meta, ve, wkm, hkm)
    setup_render()
 
    if RENDER_ON_RUN:
        log(f"rendering -> {OUTPUT_PNG}")
        bpy.ops.render.render(write_still=True)
        log("render complete.")
    else:
        log("scene built (RENDER_ON_RUN=False). Press F12 to render.")
 
 
if __name__ == "__main__":
    main()
 