# Palaeo-DEM → 3D perspective render (Noto Canyon waterfall)

A two-step, install-free pipeline to build a Fig.-4-style oblique perspective
view of the fossil waterfall, by fusing your modern topo-bathymetry with the
interpreted MES surface and rendering it in Blender.

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

Outputs (in `out/`):
- `merged_paleo_dem.tif` — georeferenced DEM (open in QGIS to sanity-check)
- `heightmap_16bit.png` — for Blender
- `heightmap_meta.json` — real-world z/extent mapping (Blender reads this)
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
- `merge_paleo_dem.py` — DEM fusion + heightmap export
- `render_waterfall.py` — Blender scene builder + render
- `README.md` — this file