# Start frame prompt template v2.2 (creative director, Oct 23 2026). Verbatim. Do not edit words.

v2.1 with a new paragraph 6 (light). Paragraph 7 is exactly the locked look text, with no exposure sentence spliced into it. Used word for word for method B3, the only method, for people and vehicles alike.

## POSITIVE
Eight plain-prose paragraphs, in this order, with the brackets filled per shot from the previz scene data:

1. Frame and counts: "A real photograph, a still frame from a live-action commercial shot on a large-format 65mm digital cinema camera. EXACTLY [N] [people / vehicles] in frame: [list each subject]."
2. Space and blocking: where each subject stands in frame (left, centre, right, foreground or background), with distances from camera in metres, and what each one is doing. Every grey figure in the layout is a real person and every grey shape is a real object: [name each object, e.g. weathered park bench, concrete parking structure].
3. Camera: "Camera at [height], [angle], [lens]mm lens" plus how the lens renders, e.g. "natural perspective, background softly out of focus". For moving vehicles: "filmed from a camera car driving alongside at about [X] km/h, road and background streaking with motion blur, wheels blurred".
4. Identity, one line per subject from its ref: hair, clothing, colours, materials.
5. World: real worn materials for the ground, walls and props, e.g. cracked asphalt, scuffed concrete, weathered timber, dry grass.
6. Light: "[Source and direction]. The light falls directly on the subjects and the ground around them. Exposed for the main subject; shadows stay open, with readable texture in the ground and foreground, never pure black."
7. Look (locked): "Tactile realism: readable colour, soft highlight roll-off, truthful skin colour, restrained fine grain over a sharp image. Texture comes from real surfaces and light, never a filter. Any signs or screens have no readable letters."
8. Test (locked): "If any surface looks clean, plastic, grey, untextured or rendered, the image is wrong."

BANNED in the positive: 8K, hyperreal, pore-level, previz, mannequin, blockout, proxy, cinematic. Strip them if a user's text adds them, or warn.

## SHARED NEGATIVE (needs CFG 3 or above)
3D render, CGI, clay render, grey mannequin, plastic skin, untextured surfaces, flat even lighting, video game, illustration, HDR, oversharpened, smeared face, text, letters, logos, watermark, rain, fog, haze, extra people, extra fingers, underexposed, crushed shadows, black foreground

## VEHICLE NEGATIVE APPEND (when any subject is a vehicle)
parked car, static background, frozen wheels, car on a white background

## SETTINGS (method B3, the only method)
Qwen-Image 2.1 fresh generation, 2048x1152, CFG 3, 30 steps, refs as image inputs. Sampler and scheduler from R&D's B3 job files.
Previz depth goes in over the whole frame. Soft canny goes in over the whole frame except inside the subject mask, so subjects get depth only.
Subject mask: every person and vehicle in the previz scene, dilated about 12 px, rendered from the previz object IDs (an ID or matte pass from the previz renderer), never from colour.
Joint circles on the mannequins stay masked.

## Run 2 addendum (creative director, Oct 24 2026, verbatim)
Cameras appeared as props (s01, s02, s04, s06). Never write the words camera, shot, filmed, lens flare or crew in a still prompt.
1. Frame and counts: "A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean. EXACTLY [N] [people / vehicles] in frame: [list each subject]."
3. Camera: "Viewpoint at [height], [angle], [lens]mm lens". For moving vehicles, no speed and no camera car: "seen side-on with the road and background streaking with motion blur, wheels blurred".
