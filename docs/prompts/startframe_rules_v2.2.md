# Start frame fill rules v2.2 (creative director, Oct 23 2026, verbatim)

1) PARAGRAPH 5, WORLD MATERIALS. One sentence made of a location base, then one phrase for each previz object present.
Location base, picked by a keyword in the user's prompt (default: outdoor street):
- street / city / sidewalk / bus: "cracked asphalt with faded lane paint, scuffed concrete kerbs and pavement, weathered building walls with stains and patched render"
- park / field / garden / forest: "real grass with worn patches and fallen leaves, packed gravel and dirt paths, trees with rough bark and uneven leaves"
- beach / sea / coast: "wet and dry sand with footprints, dark wet rocks, shallow water with real foam"
- rooftop: "tar and gravel roof surface with stains, weathered concrete parapet with rust streaks, galvanised metal vents"
- highway / road / freeway: "worn asphalt with tyre marks and cracks, concrete barriers with grime, dry roadside grass"
- home / room / interior / apartment: "oak floorboards with wear, painted plaster walls, fabric with real weave and creases"
- diner / bar / cafe / restaurant: "worn laminate and chrome counters, scuffed tile floor, vinyl seats with creases"
Per-object phrases, one for each object type present:
- bench: "a park bench of weathered timber slats on cast-iron legs"
- car / vehicle: "painted metal bodywork with reflections and road dust on the lower panels"
- table: "a scuffed metal table"
- chair: "a used chair with worn edges"
- tree: "real trees with rough bark"
- building / wall / box: "weathered concrete"
- planter / shrub: "a real shrub in a stone planter"
- anything else: "[object name] made of real, used materials"
2) PARAGRAPH 6, LIGHT. Direction comes from the previz sun or key light, relative to camera (camera left, camera right, behind camera, backlit); default camera left. Source, picked by a time cue in the user's prompt:
- dawn / sunrise: "low soft dawn sun from [dir], just above the horizon, cool shadows"
- morning: "clear morning sun from [dir], about 30 degrees high"
- noon / midday: "high midday sun from [dir], short hard shadows"
- afternoon: "late-afternoon sun from [dir], about 30 degrees high"
- golden hour / sunset: "low golden-hour sun from [dir], just above the horizon, long warm shadows"
- dusk / blue hour: "soft blue-hour skylight with warm practical lights coming on"
- night: "street lamps and lit windows as the light sources, from [dir]"
- overcast / cloudy: "soft overcast daylight from [dir], gentle shadows"
- interior with no cue: "soft daylight through a window at [dir]"
- exterior with no cue: the afternoon line.
Always follow it with: "The light falls directly on the subjects and the ground around them. Exposed for the main subject; shadows stay open, with readable texture in the ground and foreground, never pure black." Weather words (rain, fog, snow) are never written into the start frame.
3) PARAGRAPH 3, LENS RENDERING, picked by the previz focal length:
- 24mm or wider: "wide perspective with straight verticals, deep focus from foreground to background"
- 25 to 40mm: "natural perspective, background slightly soft"
- 41 to 65mm: "natural perspective with gentle compression, background softly out of focus"
- 66mm and longer: "compressed perspective, shallow depth of field, background melting into soft shapes"
If a vehicle subject is moving, add: "filmed from a camera car driving alongside at about [speed from the previz, else 80] km/h, road and background streaking with motion blur, wheels blurred".
4) GROUPS. Count people from the previz figures, never from the user's noun. Paragraph 1 reads "EXACTLY 2 people: person 1 on the left, person 2 on the right", numbered left to right by screen position. Each person gets their own line in paragraph 2 (position, distance, action) and paragraph 4 (identity). Use the ref name instead of 'person N' when there is one. With no ref, paragraph 4 says "Each person looks clearly different, with their own face, age, hair and everyday clothing in distinct colours". Vehicles work the same way: vehicle 1, vehicle 2.
5) COMMERCIAL-SAFE FALLBACK (Qwen-Image-Edit 2511). Same template v2.2 and negative, run the A2 way: previz frame 0 as image 1, refs as images 2+, CFG 3, 30 steps, joint circles masked. Add to the end of paragraph 2: "Replace every grey surface and figure with the real materials and people described; keep the exact layout, camera and framing."

## v2.3 addendum (creative director, Oct 23 2026, verbatim)

(a) Shared actions: if an action involves contact or is mutual (hug, embrace, handshake, kiss, dance together, hold hands, talk to each other), write it once as a group line naming everyone: "Person 1 and person 2 hug each other, [her face / his face] toward camera, every hand resting on the other's back and traceable to a shoulder." Leave out the bracket when the previz doesn't say who faces camera. A person with no action of their own gets "standing naturally, weight on one leg, hands relaxed." No person is ever left without an action line.
(b) Dedupe: keep the object phrase and drop the duplicate from the location base (e.g. trees in s04).
(c) A vehicle with no ref gets: "Vehicle 1 is a real, everyday [saloon / hatchback / SUV / pickup / van, picked from the previz vehicle type, default saloon] in a plain solid colour, with glass reflections, tyres with real tread and no readable badges." With two or more vehicles, add "each vehicle a different make and colour".
(d) Camera left stays the default for old shots.
(e) Positions: two people read left and right, three read left, centre and right; four or more read "far left, left of centre, right of centre, far right" or are numbered in screen order. A ref's type field always beats its name, so revert Claude's car-word guess. Commercial-safe size: generate at 1664x928, then Lanczos upscale to 2048x1152 (no VRAM test on the 24 GB test GPU, so use this path).
(f) Prop rule: "A Prop ref becomes a vehicle subject only when its name has a vehicle word AND it is placed on a vehicle-shaped previz object. Otherwise it stays a prop and is written only as an object phrase in paragraph 5."
(g) The paragraph 2 clause depends on what's in the shot:
- People and objects: "Every grey figure in the layout is a real person and every grey shape is a real object."
- Vehicles and objects only: "Every grey vehicle in the layout is a real car on a real road, and every other grey shape is a real object."
- Objects only: "Every grey shape in the layout is a real object."
- People and vehicles: "Every grey figure in the layout is a real person, every grey vehicle is a real car, and every other grey shape is a real object."
When the vehicle type is known (van, truck, bus), use it instead of "car".

## v2.4 addendum (creative director, Oct 23 2026, verbatim)

1. Ref captions: strip the ref-sheet framing before it goes into paragraph 4. Drop 'Car reference sheet on a plain light gray studio background:', 'the same', and any 'shown three/four views...' tail. Never cut mid-sentence; cut at the last full clause. c07 should read: 'Green coupe car is the one in image 3: a dark green 1970s fastback muscle coupe with chrome bumpers and a long sloping fastback roofline.'
3. Location bases. Add lake/pond/river/water = 'still lake water with real reflections, muddy shoreline with reeds, packed earth path'; forest = 'forest floor of leaf litter and roots, mossy trunks'; field = 'long uneven grass and wildflowers, dirt track'. When no keyword matches, infer from the previz objects before falling back to street: trees or grass with no road = park, road plus vehicles = street. s03 must come out lakeside.
4. Object phrases by location. On a rooftop, bench = 'a bench of weathered timber slats on metal legs' (never 'park bench' outside a park), and tree = 'a small tree in a large concrete planter'. A non-subject car object = 'a parked car with painted bodywork and road dust' (this replaces the bodywork phrase). The tree phrase everywhere = 'real trees with full, irregular canopies, rough bark and uneven branches'.
6. Light override: a user or showcase light line REPLACES the paragraph 6 source sentence. Paragraph 6 = the custom line, then the exposure sentence once. Remove the default 'Late-afternoon sun...' when an override exists.
7. Groups: every picked ref must go through (Jun was dropped, so check the ref list cap; if the 2-ref VRAM cap is what drops him, keep the cap but still give every picked person their described line and action, and report it). A group action in the user's words ('three dancers', 'they dance') applies to every person, so Kofi and Jun get 'dancing', not 'standing naturally'. For a mutual action, add to paragraph 4: '[A] and [B] are two different people.' Add 'duplicate person, twins, clone, third person' to the negative for any shot with 2+ people.
9. Final motion text: take the user's camera words first, then the previz path. If the previz path disagrees with the user's words (s06 cranes down vs tracks left), flag it in the UI as a previz mismatch instead of papering over it in text.
The two prop rules (v2.3 f) are approved as written.

## Run 2 addendum (creative director, Oct 24 2026, verbatim)
(a) World paragraph. The user's Look text wins about the ground and place: if it names ground or a place (track, path, gravel, dirt, grass, lawn, shore, sand, floor, roof, road, highway, sidewalk, street, counter, booth, wall), skip the location base sentence; keep only the object phrases for previz objects the Look text does not already name. Street is the default only when the previz has the road or cars; otherwise 'worn real ground, weathered paint and real used materials'. Rural row before the street row: track, farm, farmhouse, country, rural, outskirts, dirt, meadow = 'pale packed earth with sparse dry grass and faint wheel ruts, weathered painted clapboard and timber, flaking paint'. Lake row: 'still lake water with real reflections, wet pebbled granite shoreline, packed earth path'. Anything else: '[object name], weathered and real'. Buildings: 'a weathered building with real wall texture'; with a farmhouse or house in the Look text, 'a small weathered house with painted clapboard or plaster walls and a shingle roof'. A sentence naming the previz is dropped whole.
(b) Write what is there, never what is not: 'a smooth rounded tail with a plain flat trunk lid', 'clean unbadged bodywork', 'a smooth tail with a plain flat lid', 'a smooth bare roof', 'bare-headed', 'a bare face', 'blank unmarked signs'.
(c) The previz distance wins: when paragraph 2 gives a distance, the user's metres and frame-width sentences go. No km/h in a still. Paragraph 2 adds the previz size: 'his head is about 6 percent of the frame height', 'the car fills about 12 percent of the frame width'.
