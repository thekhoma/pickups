# Previz model provenance

Written 2026-10-21 for the Pickups public release. Covers every 3D model in `pickups/web/public/previz/`.

## Summary

Every model was built from scratch in code by an AI coding agent working for the project owner on Oct 19 and 20, 2026, as procedural three.js scripts, then exported with three.js GLTFExporter and compressed with gltfpack 1.3 (meshopt). No third-party meshes, textures, rigs, animations or HDRIs are used, and no AI 3D or image generator produced any geometry.

Byte check done today on the Mac Studio: all 105 `.glb` files under `pickups/web/public/previz/` are byte-identical to a file in the model-building project (kept outside this repo; the folders named below are there). 27 match `04_library_v1.4/exports/glb/` (the top-level GLTFExporter r186 exports) and 78 match `05_cursor_bundle/assets/` (the gltfpack 1.3 runtime copies of the same models). None came from anywhere else.

## How each part was made

| Part | How it was made | Source in the model-building project |
|---|---|---|
| Mannequin (person, person__lite, person__crowd) | Procedural three.js geometry (lathe, capsule, sphere and tube primitives, named joint pivots), written by hand in code. The head shape was matched by eye to a head reference image the owner supplied in chat; no geometry, texture or pixels from that image are in the model. I don't know where the image came from. | `03_mannequin_build/mannequin.js` (v1 and v2 archived in `_v1/`, `_v2/`), README there |
| Props, vehicles, animals, set pieces (57 models: car, SUV, truck, van, bus, dog, cat, horse, cow, bird, fountain, bench, tree, house and the rest) | One procedural three.js module per model, `create<Id>(options)`. car, suv, truck, van and bus share one brand-neutral vehicle profile in `car.js`. | `04_library_v1.4/assets/*.js` (registry `assets/index.js`), README there |
| Car interior | Procedural, from a script. | `04_library_v1.4/tools/make_car_interior.py` |
| GLB export and compression | three.js GLTFExporter (r186) export jobs, then gltfpack 1.3 for the runtime copies. | `04_library_v1.4/tools/jobs/*.mjs` (for example `final_phase1.mjs`, `export_v13.mjs`, `export_crowd.mjs`, `iv_export.mjs`), `04_library_v1.4/tools/render.mjs` |
| Scene kits (25) | Not baked models. Layout JSON written by scripts and assembled at runtime from the models above. | `04_library_v1.4/tools/make_layouts.py`, `make_layouts_v13.py`, `make_layouts_v14.py`; output in `05_cursor_bundle/data/kits/` |
| Poses (person and animal) | Joint angles in degrees, authored by hand and by script. No motion capture. | `04_library_v1.4/exports/person_poses.json`, `05_cursor_bundle/data/person_poses.json`, `animal_poses.json`, `04_library_v1.4/tools/jobs/dump_poses.mjs` |
| Walk, jog and run animation | Procedural gait code written by Cursor in the Pickups repo during Previz Fix B (Oct 21). No Mixamo or other clip files. | `pickups/web/src/previz/runtime/gait.js` (committed in 9c19e48) |
| Lighting environment | three.js `RoomEnvironment`, which is generated in code (MIT, part of three.js). No HDRI files. | `runtime/world.js` |
| Textures | None. All materials are flat colors (people #F4FF75 with black joints and white lines, props #ECEAE5, sets grey). | |
| Fonts | None are shipped with the models or the runtime. The QC contact sheet scripts use Inter and Instrument Serif (SIL OFL) and DejaVu Sans, which are installed on the box and are only used for review images that are not in the repo. | `04_library_v1.4/tools/fonts.py`, `03_mannequin_build/tools/compare.py` |

## Third-party code (not models)

three.js r170 (MIT) including GLTFLoader, BufferGeometryUtils, RoundedBoxGeometry and RoomEnvironment, and the meshoptimizer decoder (MIT). Build tools only, not shipped: gltfpack 1.3 (MIT), Playwright (Apache-2.0), Node.js, Python, TypeScript. Pickups uses its own three dependency (web/package.json); no vendored copy is shipped.

## Not in the repo

`01_concepts/`, `04_library_v1.4/style_frames/` and the render folders are concept and QC images for the owner's review. They are not shipped in Pickups, and this file doesn't vouch for their sources.

## License

Chosen by the owner on 2026-10-21 and committed in Pickups. Code is MIT. The Previz models, scene kits and poses are CC0 1.0 (public domain dedication, no credit required).
