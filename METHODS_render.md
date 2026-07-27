# Methods — palaeotopographic reconstruction and 3D visualisation

Draft text for the manuscript. Two versions: a full subsection, and a condensed paragraph
for journals that cap the Methods length. Numbers are taken from the run that produced the
figure (`heightmap_meta.json`); update them if the merge is re-run with other settings.

---

## Full version

### Palaeotopographic reconstruction

The perspective view was built from a single palaeotopographic digital elevation model
(palaeo-DEM) that merges two independent surfaces. Present-day topography and bathymetry
outside the study area were taken from the GEBCO 2026 global grid; within the study area the
surface is the seismically interpreted Messinian Erosional Surface (MES), gridded at 20 m.
Both were resampled onto a common 25 m grid in UTM zone 33N (EPSG:32633) covering
280 × 200 km (11,200 × 8,000 cells) centred on the Hyblean margin and the adjacent Ionian
Basin. Depth-referenced MES values were converted to elevations, and residual gaps were
filled by distance-weighted interpolation from surrounding valid cells before merging.

The two surfaces were combined with a feathered blend, in which the weight assigned to the
MES rises smoothly from zero at the outer limit of the interpreted grid to unity in its
interior over a 100 km ramp. A wide ramp was used deliberately: a narrow transition leaves a
visible break in hillshade that a viewer reads as a scarp, and no such scarp exists in the
data. Post-Messinian differential subsidence was compensated by raising the deepest parts of
the basin by up to 200 m, scaled with depth. A single 50 m Gaussian filter was then applied
to the merged surface to suppress the contour-following terraces inherent to an interpolated
horizon, while leaving the metre-scale relief of the escarpment and the canyon intact. All
subsequent products — the render mesh, the hillshade quality-control image and the exported
GeoTIFF — derive from this same filtered surface, so the geometry shown is the geometry
measured.

Sea level was set to −2200 m relative to present, representing the desiccated Messinian
lowstand. At this level the Ionian Basin retains a residual water body while the Sicily Sill
and the adjacent shelf stand subaerially exposed, which is the configuration immediately
preceding the Zanclean reflooding.

### The megaflood corridor

The flood is shown as a single corridor rather than as a dendritic drainage network, because
the Zanclean event was one continuous torrent rather than a river system. The corridor
follows a polyline defined in projected coordinates; the route was not drawn by eye but
verified with a minimax ("least-rise") shortest-path search across the palaeo-DEM, in which
the cost of a path is the highest cell it must traverse. This identifies the lowest
continuous sill between the Sicily Sill and the canyon head, and was used to reject an
earlier route that would have required the water to surmount a +46 m barrier. The search was
additionally constrained to the northern part of the domain so that the route follows the
anastomosing channel system preserved along the Sicilian margin, which raises the controlling
sill to +6 m — a real hydraulic cost that is reflected in the reconstructed water-surface
elevations upstream.

Within this corridor the water surface was reconstructed topographically rather than imposed
as a flat plane. A surface sloping down-route was constructed from the minimum bed elevation
within a 2 km cross-route window, forced to be non-increasing downstream so that water ponds
behind sills and spills over them, and lowered beyond the corridor margin at 130 m km⁻¹. The
flooded area is then the set of cells lying below this surface and hydraulically connected to
the route, so that valleys fill, ridges remain as islands, and the flow plunges where the
ground falls away. Water depth, and hence the extent of the flooded area, was derived by
differencing this surface against the terrain; the shoreline was smoothed by filtering the
*signed* depth field before thresholding, which moves the zero contour without expanding the
water body onto dry ground.

It should be stated explicitly what the corridor margin represents. Measured on this
palaeo-DEM, the local palaeotopography is a broad basin rather than a confined valley: at any
water level the connected flooded area forms an inland sea. The terrain therefore cannot
laterally confine a river at equilibrium, and the corridor edge is consequently **not** a
shoreline. The figure depicts an instantaneous state of an advancing torrent that has crossed
the exposed basin floor and has not yet filled it, and the corridor margin is the flow front
of that advance.

The surface texture of the cascade was reconstructed as geometry rather than as a painted
pattern. A water sheet is a height field, so foam and streaks applied as colour or as a bump
map cannot produce the self-shadowing of a real cataract. Instead, the standing-wave train
and the flow-parallel structures of the chute were computed as a signed relief field in
metres (±150 m), gated by the large-scale slope of the water surface so that it develops only
where the water genuinely falls and the ponded reaches remain flat. This field was exported
as a separate layer and applied as true vertical displacement at render time.

### Three-dimensional rendering

Rendering was performed in Blender 5.1.2 using the Cycles path-tracing engine. The
palaeo-DEM was imported as a 16-bit height field and displaced onto a subdivided plane at
3200 subdivisions per side, with a vertical exaggeration of ×3; scene units correspond to
kilometres, so all distances, elevations and the camera position are physically scaled. The
reconstructed flood water surface and the cascade relief field were applied as a second
displaced surface confined to the corridor mask, and the spray plume at the foot of the fall
was rendered as a bounded participating-medium volume whose density is derived from the
computed plunge location and decays with height above the local water surface.

Surface colour is a hypsometric albedo map baked during the reconstruction step and applied
to a physically based shading model; slope-dependent shading derived from the surface normal
distinguishes bare rock on steep faces from sediment on the flats. The scene is lit by a
single directional source at 22° elevation, combined with a physical sky model, and
atmospheric attenuation is applied as a function of absolute camera distance. Scale bars,
place-names and the north arrow were placed as three-dimensional objects in world coordinates
and rendered with the scene, so that they follow the same perspective projection as the
terrain rather than being overlaid afterwards. The still frames were rendered at
4800 × 3200 px with 256 samples per pixel.

All processing was carried out with freely available software (GDAL/rasterio, SciPy, Blender).

### Supplementary animation

The supplementary video was produced from the same scene, camera and lighting as the still
figure, and differs from it only in that the flood is revealed progressively along the
corridor. The clip opens on the desiccated basin, the torrent then crosses the exposed
Sicily Sill and goes over the escarpment lip, and the remainder of the sequence shows the
cascade at full discharge. The flooded bed colouring, the spray plume and the sediment plume
delivered into the Ionian Basin are introduced in step with the arriving water, so that no
consequence of the flood is visible before the flood itself. Frames were rendered
individually at 1920 × 1280 px and assembled at 25 frames per second.

### A note on what the figure is

The reconstruction is a *visualisation* of measured palaeotopography together with a
geometrically and hydraulically constrained reconstruction of the flood surface. It is not
the output of a hydrodynamic simulation: no momentum equations are solved, and no discharge,
velocity or timescale should be inferred from the rendered water surface. The terrain, the
2.2 km rim-to-landing-zone relief and the >20 km width of the amphitheatre are measurements;
the water is a constrained reconstruction; the surface texture of the cascade is
illustrative.

---

## Condensed version (~180 words)

A palaeotopographic DEM was constructed by merging the GEBCO 2026 grid with the seismically
interpreted Messinian Erosional Surface on a common 25 m UTM 33N grid (280 × 200 km), using a
100 km feathered blend, a depth-scaled subsidence correction of up to 200 m, and a 50 m
Gaussian filter to suppress interpolation terraces. Sea level was set to the Messinian
lowstand of −2200 m. The megaflood corridor follows a route verified by a minimax
("least-rise") path search across the DEM, which identifies the lowest continuous sill
between the Sicily Sill and the canyon head. Within the corridor the water surface was
reconstructed topographically — sloping down-route, non-increasing downstream, and restricted
to cells below that surface and connected to it — so that valleys fill, ridges persist as
islands, and the flow plunges at the escarpment. The scene was rendered in Blender 5.1.2
(Cycles) at ×3 vertical exaggeration with physically scaled geometry and lighting. The
rendering is a constrained visualisation, not a hydrodynamic simulation; no discharge or
velocity should be inferred from it.
