# Palaeo-DEM → 3D perspective render (Noto Canyon waterfall)

A three-step, install-free pipeline to build a Fig.-4-style oblique perspective
view of the fossil waterfall, by fusing your modern topo-bathymetry with the
interpreted MES surface and rendering it in Blender.

```
merge_paleo_dem.py   →  output_vN/   →  render_waterfall.py  →  PNG
   (plain Python)        (the only        (inside Blender)
                          contract)
                                       →  video_gen.py        →  MP4
```

The two heavy scripts never import each other. Their **only** contract is the
output directory, described by `heightmap_meta.json`. Everything else is a CONFIG
block edit at the top of each file.

This reproduces the core method of Madof et al. (Nat. Geosci. 2026): they
warped the modern topography-bathymetry where palaeo-isolines were sparse and
blended the restored surface in (their full pipeline used Global Mapper +
Houdini + World Machine + Substance Painter + Blender; this is the stripped-down,
free-software version — Blender only).

---

## 0. Blender without admin rights

You do **not** need an installer or admin:

1. Go to `blender.org/download`.
2. Instead of the `.msi` installer, take the **"Portable (.zip)"** Windows build.
3. Unzip it anywhere in your user space (Desktop / Documents).
4. Run `blender.exe` directly.

Add-ons (if you ever want BlenderGIS) install into your user config folder —
also no admin. If your IT blocks running `.exe` from user folders (AppLocker),
either ask IT for a one-time Blender install (free, trivial sell) or fall back
to the QGIS native 3D Map View, which you already have.

---

## 1. Merge — `merge_paleo_dem.py`

Needs Python with: `rasterio numpy scipy pillow matplotlib pandas imageio tqdm`
(`pip install --user ...`). An Anaconda `base` env usually has most of these.

**Geographic vs projected CRS (important):** if your modern DEM is geographic
(e.g. GEBCO in degrees) and your MES is projected (metres), the script detects
this and automatically reprojects the modern DEM into the MES's metric CRS, so
`FEATHER_M` and the Blender scale stay in metres. You can also force it with
`TARGET_CRS = "EPSG:32633"`.

Edit the **CONFIG block** at the top, then run:

```bash
python3 merge_paleo_dem.py
```

Key settings:

| Setting | What it does |
|---|---|
| `MODERN_DEM_PATH/FORMAT` | your regional topo-bathy (EMODnet + TINITALY/GEBCO). GeoTIFF or xyz. Defines output grid + CRS. |
| `MES_PATH/FORMAT` | the interpreted MES. GeoTIFF, or **xyz** for a Kingdom horizon export. |
| `MES_IS_DEPTH` | `True` if MES values are depth-below-sea (positive down). Converts to elevation internally. |
| `MES_CRS` | CRS of the MES (only needed for xyz; reprojected onto the modern grid). Set this carefully. |
| `CROP_BBOX` | limit to the study area (xmin,ymin,xmax,ymax in target CRS) so the heightmap stays manageable. |
| `FEATHER_M` | width of the blend ramp between modern and MES (default 3000 m). |
| `APPLY_SUBSIDENCE` | optional first-order loading correction: raises the deep landing zone back toward its 5.33 Ma elevation (`SUBSIDENCE_M`, default 500 m). Off by default. |
| `PALEO_SEA_LEVEL_M` | desiccated lowstand for the render's water plane (default −2000 m). |
| `VE_HINT` | suggested vertical exaggeration passed to Blender (default 3). |

**Important data point:** in the study area the script substitutes the **MES**,
*not* the modern seafloor — because the modern seafloor there is the
post-flood *filled* surface. The MES is the palaeo-surface the waterfall cut.

Outputs (in `OUT_DIR`, e.g. `output_v17/`):
- `merged_paleo_dem.tif` — georeferenced DEM (open in QGIS to sanity-check)
- `heightmap_16bit.png` — terrain, for Blender
- `heightmap_meta.json` — **the manifest**: names every other file and carries the
  physical scale (`zmin_m`/`zmax_m`, `width_m`, `pixel_m`, `paleo_sea_level_m`, `ve_hint`)
- `colormap.png` / `colormap_dry.png` — baked albedo, with and without the flood
- `flood_corridor.png` — RGB mask: **R** = coverage, **G** = along-path 0–1, **B** = foam
- `flood_surface_16bit.png` — the flood's water-surface elevation
- `flood_relief_16bit.png` — **signed** cascade relief in metres (see below)
- `flood_mist.png` — spray density for the Blender volume
- `hillshade_qc.png` — quick visual QC

Look at `hillshade_qc.png` first. The amphitheatre bowl should be obvious and
the seam between MES and modern should be smooth (no hard line). If the seam is
sharp, increase `FEATHER_M`.

---

## 2. Render — `render_waterfall.py`

1. In Blender → **Scripting** workspace → open this file.
2. Set `DATA_DIR` to your `out/` folder and `OUTPUT_PNG` to where you want the image.
3. Click **Run Script** (▶). The render appears and is saved.

Headless / cluster (no GUI):

```bash
blender -b -P render_waterfall.py
```

Knobs to turn:

| Setting | Effect |
|---|---|
| `VE` | vertical exaggeration (None = use the merge's `ve_hint`). Try 2–5. |
| `CAM_AZIMUTH_DEG` / `CAM_ALTITUDE_DEG` / `CAM_DISTANCE_KM` | frame the fly-over. 20° / 28° ≈ "view to the NNE". |
| `SUN_AZIMUTH_DEG` / `SUN_ALTITUDE_DEG` | relief shading; low sun = longer shadows. |
| `PALEO_SEA_LEVEL_M` (in the merge) + `SHOW_WATER` | the desiccated water level. |
| `SUBDIVISIONS` | terrain detail (600–1000). Higher = sharper + heavier. |
| `ENGINE_PREF` | `"EEVEE"` (fast, fine for drafts) or `"CYCLES"` (slower, nicer light). |

---

## The Zanclean megaflood corridor (the hero flood)

The automatic D8 rivers look like *rivers* because they are (flow accumulation).
The Zanclean flood was the single largest flood in Earth history — one colossal
torrent, not a dendritic network. So the route is **one hand-placed polyline**
(`MEGAFLOOD_PATH`, waypoints source→plunge) and the script rolls a wide flood
down it. Use `MEGAFLOOD_PATH_UNITS="crs"` and give target-CRS metres (read off
QGIS); the image-fraction mode exists but is crop-dependent.

`MEGAFLOOD_HALFWIDTH_KM` gives the half-width in km at each waypoint: wide where
it drains the deep trough, narrowing along the shelf, **pinched hard at the lip**
(a wide falling front drapes the cascade sideways along the escarpment instead of
down the canyon — more water in the canyon comes from `FILL_DEPTH`, i.e. a thicker
sheet, not a wider front).

You get: clean Atlantic blue at the inlet → turbid sediment toward the canyon →
white foam where it plunges over the escarpment, baked into `colormap.png` and
exported as `flood_corridor.png` (the mask).

### Measure the route, don't eyeball it

**Do not trace what looks right.** Twice now an eyeballed route has been overturned
by a **minimax / least-rise search** (Dijkstra where a path's cost is the *highest
cell it must cross*):

- The first route climbed a **+46 m** sill. The water had to pond ~100 m deep just
  to get over ground it never needed to cross — and *that*, not any shader setting,
  was why the flood rendered as an inland reservoir.
- Constraining the same search to the band along the Sicilian margin puts the river
  on the anastomosing channel the DEM actually shows. That costs head (the sill is
  **+6 m** there against **−48 m** out on the shelf), which is paid back by
  *narrowing* the corridor — the head goes into depth, not area.

**On a cliff, only the envelope can stop water.** Terrain confinement is useless on
an escarpment because the ground only falls. An earlier route ran along the Malta
escarpment crest with a 13 km half-width, so the corridor lapped over the rim and
the flood poured down the face as a *second cascade* 15 km south of the Noto Canyon.
The cure was structural (approach the canyon head from the north-west, west of the
crest), not cosmetic.

### The corridor is a FLOW boundary, not a water level

Say this in the caption. Measured on this DEM, the palaeo-topography here is a broad
**basin, not a valley**: at *any* water level the connected flooded area is an inland
sea (38 % of the frame even at −31 m). So the terrain **cannot** confine a river, and
a corridor with terrain-cut banks is not achievable at equilibrium — the honest
equilibrium picture is a lake.

The figure therefore shows the flood as a **snapshot of an advancing torrent** that
has crossed the dry basin and not yet filled it. On that reading the corridor edge is
the *flow front*, not a shoreline, and the envelope is the right tool.

### 3D water, cascade geometry, spray

With `MEGAFLOOD_WRITE_MASK=True`, the Blender script (`SHOW_FLOOD_SHEET=True`) builds a
real water sheet at the flood's water surface, clipped to the corridor.

A heightfield has **one z per cell**, so falling water can never be *painted* on it —
bump on a smooth sheet at km scale just returns the sky (the 1.8 km cataract rendered
as pale blobs). Instead the merge bakes the cascade's surface structure as a signed
relief in metres (`flood_relief_16bit.png`) and Blender displaces the sheet with it:
the wave crests become **real geometry** that catches the sun and shades its own
troughs. `CASCADE_RELIEF_SCALE` retunes the amplitude in the render, without a re-merge.

The spray plume at the foot of the fall is a bounded Principled Volume box whose
density comes from `flood_mist.png`. `MIST_DENSITY` is the one knob (≈1.0 is a veil you
can still see through; 2.2 buries the canyon in fog).

`ANIMATE=True, ANIM_TYPE="flood_advance"` sweeps the flood source→lip: the megaflood
advancing and going over.

Other knobs: `MEGAFLOOD_FEATHER_KM` / `MEGAFLOOD_FALLS_FEATHER_KM` (soft rim on the
flats vs. the tight rim on the falling cascade), `MEGAFLOOD_CONFINE_TO_TERRAIN`,
`MEGAFLOOD_POOLING`, `MEGAFLOOD_CONNECTED_FILL`, `MEGAFLOOD_FILL_DEPTH_M`.

**`MEGAFLOOD_SCOUR_M` — leave at 0.** It carves the corridor into the heightmap tapered
by the corridor weight, i.e. it *prints the corridor polygon into the DEM*: at 190 m,
78 % of the shoreline ended up sitting on the scour ramp rather than on real topography.
That is a data-falsifying edit of a terrain that is supposed to **be** the data.

---

## Traps that cost real time (read before debugging)

- **Never blur a one-sided field.** The shoreline is de-jagged by smoothing depth before
  thresholding it. Smoothing `clip(zwater − terrain, 0, None)` bleeds the water's depth
  *outward onto dry ground*, and the deeper the water at the shore, the further it bleeds.
  Against a steep bank this painted a translucent water film a **kilometre up the dry
  hillside**, which flashes white at a grazing sun and looks exactly like a water wall.
  Smooth the **signed** field: dry ground is negative and cancels the bleed, so the blur
  moves the zero *contour* instead of inflating the region. Same trap in `grey_closing`
  (it is *extensive*) — restrict it to cells genuinely enclosed by water.
- **Don't boost a bump on steep, VE-exaggerated geometry.** `ShaderNodeBump` differentiates
  in screen space, and on the escarpment the displaced quads are nearly vertical and hugely
  stretched — exactly where that derivative is least reliable. Multiplying the bump there
  rendered the cliff as blistered scales.
- **Procedural feature sizes must be physical.** Feed noise through *Object* coordinates
  (scene units are km) so a "grain size" is a real length. Generated coordinates normalise
  over the bounding box, so the same constant means 930 m in the regional shot and 310 m in
  the hero — and is anisotropic besides.
- **Cycles 5.1 hard-crashes** on an image texture inside a volume shader on a **shadow ray**
  (`EXCEPTION_ACCESS_VIOLATION`). It is size-dependent, so it looks like an OOM and is not.
  Fix: `box.visible_shadow = False`.
- **The Mist compositor pass does not apply headless**, and a world Volume Scatter turns a
  km-scale scene black. Atmospheric haze is done as a MixShader in the terrain material,
  keyed to camera distance in absolute km.

---

## Two kinds of shot (read this for real data)

A full GEBCO tile is ~500–600 km wide. Two consequences:

**1. Resolution.** GEBCO (~450 m) reprojected becomes the canvas resolution, so
your 20 m MES detail gets downsampled to ~450 m in the merged grid. Fine for a
regional overview; too coarse for the canyon itself.

**2. Scale.** A 2.2 km waterfall across a 500 km canvas is 0.4 % of the width —
even at VE 3 it is invisible. The waterfall **hero shot must be a crop.**

So:

| Shot | CROP_BBOX | TARGET_RES | VE | Why |
|---|---|---|---|---|
| Regional desiccated-basin overview (Euphrates-style) | None (full) | None (auto ~450 m) | 3 | context: exposed basin, Sicily Sill, Ionian |
| Noto waterfall close-up (hero) | a box around the canyon, in TARGET-CRS metres, e.g. `(490000, 4060000, 540000, 4100000)` | `50`–`100` | `5`–`8` | keeps the 20 m MES detail + makes the 2.2 km drop read |

`CROP_BBOX` is in the **target CRS (metres)** — read the canyon's extent off the
GeoTIFF in QGIS to set it.

**Two things to watch on your first run:**
- The line `MES coverage of grid: X %`. If it's ~0 %, the MES and the modern
  tile do **not** overlap (wrong tile, or the MES is from a different region) —
  the blend will just return GEBCO.
- `MES_IS_DEPTH`. Check `hillshade_qc.png`: the canyon must be a **bowl**. If it
  shows up as a **hill**, flip `MES_IS_DEPTH`.

---

- This is a **schematic reconstruction**, like the Euphrates Fig. 4 — label it
  as such; don't present the blended surface as a measured DEM.
- The "Nature look" (atmospheric haze, hand-placed labels, fine texturing) is
  designer craft on top of the geometry. This pipeline gets you the correct,
  well-composed geometry + colour + lighting — the last 10% of polish is
  Illustrator/Affinity work afterwards (labels, haze, annotations on the PNG).
- For a submission, correct + clear beats glossy. Polish can come later (and
  Nature/NGeo often provides art support for accepted papers).

---

## Files
- `merge_paleo_dem.py` — DEM fusion + megaflood + heightmap/mask export (plain Python)
- `render_waterfall.py` — Blender scene builder + render (**runs inside Blender**)
- `video_gen.py` — assembles a frame sequence into an MP4 (imageio + FFmpeg on PATH)
- `README.md` — this file

`output*/` is gitignored — the renders and DEMs are large and reproducible.

---

## 3. Video — `video_gen.py`

```bash
python video_gen.py
```

Needs `imageio tqdm` and FFmpeg on PATH. Point it at the frame sequence written by
`ANIMATE=True` in the render script.

---

## Verification

There are no tests, no linter and no build. **Verification is visual:**

1. After a merge, open `hillshade_qc.png`. The canyon must read as a **bowl** (if it is a
   hill, flip `MES_IS_DEPTH`), and the MES/modern seam must be smooth.
2. Then render. `DATA_DIR` in `render_waterfall.py` **must** point at the `OUT_DIR` the merge
   just wrote — pointing it at a stale directory is the single most common mistake here.

For a numeric question, write a throwaway `_probe_*.py` next to the scripts (it can
`import merge_paleo_dem` and call its functions on a decimated grid directly), then delete it.
That is far cheaper than a ~10 min merge plus a render.

Cost: a full regional merge is ~10 min, a hero merge ~5 min, and a Cycles still at the
script's default 4800×3200 / 256 spp is roughly an hour. Drop `RES_X/RES_Y` and `SAMPLES`
for drafts.