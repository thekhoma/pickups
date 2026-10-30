# Start frame prompt template v2.1 (creative director, Oct 23 2026). Verbatim. Do not edit words.

v2 plus the exposure fix. Used word for word for both method B (people) and A2 (vehicles).

## POSITIVE
Eight plain-prose paragraphs, in this order, with the brackets filled per shot from the previz scene data:

1. Frame and counts: "A real photograph, a still frame from a live-action commercial shot on a large-format 65mm digital cinema camera. EXACTLY [N] [people / vehicles] in frame: [list each subject]."
2. Space and blocking: where each subject stands in frame (left, centre, right, foreground or background), with distances from camera in metres, and what each one is doing. Every grey figure in the layout is a real person and every grey shape is a real object: [name each object, e.g. weathered park bench, concrete parking structure].
3. Camera: "Camera at [height], [angle], [lens]mm lens" plus how the lens renders, e.g. "natural perspective, background softly out of focus". For moving vehicles: "filmed from a camera car driving alongside at about [X] km/h, road and background streaking with motion blur, wheels blurred".
4. Identity, one line per subject from its ref: hair, clothing, colours, materials.
5. World: real worn materials for the ground, walls and props, e.g. cracked asphalt, scuffed concrete, weathered timber, dry grass.
6. Light: one motivated source and its direction, e.g. "low late-afternoon sun from camera left", plus "Exposed for the subject's face; shadows stay open, with readable texture in the ground and foreground, never pure black."
7. Look (locked): "Tactile realism: readable colour, soft highlight roll-off, truthful skin colour, restrained fine grain over a sharp image. Texture comes from real surfaces and light, never a filter. Any signs or screens have no readable letters."
8. Test (locked): "If any surface looks clean, plastic, grey, untextured or rendered, the image is wrong."

BANNED in the positive: 8K, hyperreal, pore-level, previz, mannequin, blockout, proxy, cinematic. Strip them if a user's text adds them, or warn.

## SHARED NEGATIVE (needs CFG 3 or above)
3D render, CGI, clay render, grey mannequin, plastic skin, untextured surfaces, flat even lighting, video game, illustration, HDR, oversharpened, smeared face, text, letters, logos, watermark, rain, fog, haze, extra people, extra fingers, underexposed, crushed shadows, black foreground

## VEHICLE NEGATIVE APPEND (vehicle subject only)
parked car, static background, frozen wheels, car on a white background

## SETTINGS
Qwen-Image 2.1, 2048x1152, CFG 3, 30 steps for both arms. Sampler and scheduler from R&D's job files.
B (people): fresh generation with previz depth plus soft canny as image inputs, plus the refs (cap 2 refs).
A2 (vehicles): the app's edit graph with frame 0 as image 1, plus the refs.
Every arm masks the joint circles on the mannequins.
B3 (later) may only change the canny subject mask.
