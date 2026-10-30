"""Make start frame from previz (step 17b): the creative director's start-frame template v2.2
(docs/prompts/start_frame_template_v2_2.md), made with method zbuf (Run 1, below) or its fallback B3, for people and
vehicles alike.

B3 is a fresh Qwen-Image 2.1 generation (node for node R&D's run_b3/<shot>_v3_B3_api.json): image 1 is the playblast's
frame-0 depth (Depth Anything 3) over the whole frame, image 2 its soft canny over the whole frame except inside the
subject mask (so the subjects get depth only), then up to two identity references (three spill a 24 GB card: 905 s in
the test). qwen_image_2.1_int8_convrot + QwenImage21Cache, qwen3vl_8b_int8_convrot, qwen_image_2.1_vae_bf16,
TextEncodeQwenImage21 at 1536 (2048x1152), euler / simple, 30 steps, CFG 3, denoise 1. Never the turbo checkpoint or
the Lightning LoRA: at CFG 1 the negative does nothing.

The subject mask is every person and vehicle in the previz scene, from the previz renderer's subject matte (runtime.ts
subjectMatte: those meshes white, everything else black), grown 12 px as R&D grew theirs. Never from colour: a car is the
same grey as the road. The mannequins' joint circles are masked out of the canny too.

The prompt is the template's eight paragraphs with the brackets filled from the previz (frame 0's subjects, distances,
camera, lens, speed, key light) and the cast, by the creative director's fill rules (docs/prompts/startframe_rules_v2.2.md):
location bases, object phrases, light sources, lens bands and groups are tables in CONFIG, and the fill code only looks
them up. Precedence: the user's own words, then the previz scene data, then the rules' defaults. Locked lines are word
for word; nothing here writes prompt words of its own beyond the facts that fill a bracket.

With the commercial-safe setting on, the same words run the A2 way on Qwen-Image-Edit 2511 (fallback_graph).

Method zbuf (the default when the shot has its depth pass and ComfyUI has the 2602 ControlNet; R&D's l2r_round4 recipe):
the previz page's own z-buffer of frame 0 (runtime.ts depthPasses: view-space z, sky 0, no AA) is rank-mapped onto the
tones of Depth Anything 3's depth of the same frame (near bright, sky 0), blurred 4 px, and drives Z-Image Turbo through
Fun ControlNet Union 2.1 2602-8steps: 8 steps, one sampler, CFG 1, control on all 8 steps. Strength by the hero subject
(people 0.75, vehicles 0.65); one retry, at the other strength when the size is far off either way (size_state) or at the
next seed when the people count is wrong, with try 1 kept beside it, then a flag. Z-Image
has no reference-image input: identity is the cast's words only. B3 stays the fallback (method="B3", or a shot without the
depth pass or the model).

Every positive goes through prompt_words.clean_positive (no previz words, no negations); the negative keeps them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pickups import prompt_words

# ======================================================================================================================
# CONFIG: the one place for the start frame's words and method. Template v2.2, word for word.
# ======================================================================================================================
TEMPLATE_VERSION = "v2.2"
TEMPLATE_FILE = "docs/prompts/start_frame_template_v2_2.md"
METHOD = "B3"
ZBUF = "zbuf"
DEFAULT_METHOD = ZBUF

# Run 2 (d): no camera words in a still (they drew cameras into hands and corners: s01, s02, s04, s06).
FRAME_LINE = ("A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean. "
              "EXACTLY {count} in frame: {subjects}.")  # {count}: the template's "[N] [people / vehicles]"
# v2.3 (g): the paragraph 2 grey clause, by what's in the shot; the previz objects in frame follow it after a colon.
GREY_LINES = {
    "people": "Every grey figure in the layout is a real person and every grey shape is a real object.",
    "vehicles": "Every grey vehicle in the layout is a real {vehicle} on a real road, and every other grey shape is a real object.",
    "objects": "Every grey shape in the layout is a real object.",
    "people and vehicles": ("Every grey figure in the layout is a real person, every grey vehicle is a real {vehicle}, and "
                            "every other grey shape is a real object."),
}
GREY_VEHICLE = "car"  # unless every vehicle is one known type:
GREY_VEHICLE_TYPES = ("van", "truck", "bus")
CAMERA_LINE = "Viewpoint at {height}, {angle}, {lens}mm lens"
# Run 2 (d): never in a still's positive; the camera words a still's facts or the user's words use read as the viewer's.
NO_CAMERA_CLAUSE = r"(?i)\b(filmed|film crew|crew|lens flare|camera rig|tripod|shot on)\b"
VIEWER_WORDS: tuple[tuple[str, str], ...] = (
    (r"\bcamera (left|right)\b", r"frame \1"),
    (r"\bbehind (the )?camera\b", "behind the viewer"),
    (r"\b(toward|towards|to|at|from|facing|faces?|face) (the )?camera\b", r"\1 the viewer"),
    (r"\bto the lens\b", "toward the viewer"),
    (r"\bthe camera\b", "the viewer"),
    (r"\bcamera\b", "viewer"),
)
CAMERA_NEGATIVE = "camera, film camera, camera rig, tripod, film crew"
RULES_FILE = "docs/prompts/startframe_rules_v2.2.md"  # the fill rules below, word for word

# Rule 3, paragraph 3's lens rendering, by the previz focal length (mm): (up to and including, words); the last is open.
LENS_BANDS: tuple[tuple[float, str], ...] = (
    (24, "wide perspective with straight verticals, deep focus from foreground to background"),
    (40, "natural perspective, background slightly soft"),
    (65, "natural perspective with gentle compression, background softly out of focus"),
    (float("inf"), "compressed perspective, shallow depth of field, background melting into soft shapes"),
)
# Run 2 (c): no speed and no camera car in a still; side-on only when the previz has it side-on.
VEHICLE_LINE = "seen side-on with the road and background streaking with motion blur, wheels blurred"
VEHICLE_LINE_END_ON = "the road and background streaking with motion blur, wheels blurred"  # seen from behind or the front
SPEED_WORDS = r"(?i)\b\d+\s*(km/h|kph|mph)\b|\bkilometres an hour\b|\bmiles an hour\b"

# Rule 1, paragraph 5: a location base by a keyword in the user's words (else the previz place), then one phrase per
# previz object present. Keys are whole words; the first row that matches wins.
LOCATION_BASE: tuple[tuple[tuple[str, ...], str], ...] = (
    # Run 2 (a) 3: the rural row before the street row (c09's farm track came out with lane paint).
    (("track", "farm", "farmhouse", "country", "rural", "outskirts", "dirt", "meadow"), "pale packed earth with sparse dry grass and faint wheel ruts, weathered painted clapboard and timber, flaking paint"),
    (("street", "city", "sidewalk", "bus"), "cracked asphalt with faded lane paint, scuffed concrete kerbs and pavement, weathered building walls with stains and patched render"),
    (("park", "garden"), "real grass with worn patches and fallen leaves, packed gravel and dirt paths, trees with rough bark and uneven leaves"),
    # v2.4 3: lake, forest and field rows (forest and field leave the park row). Run 2 (a) 4: no reeds (drawn on granite).
    (("lake", "pond", "river", "water"), "still lake water with real reflections, wet pebbled granite shoreline, packed earth path"),
    (("forest",), "forest floor of leaf litter and roots, mossy trunks"),
    (("field",), "long uneven grass and wildflowers, dirt track"),
    (("beach", "sea", "coast"), "wet and dry sand with footprints, dark wet rocks, shallow water with real foam"),
    (("rooftop",), "tar and gravel roof surface with stains, weathered concrete parapet with rust streaks, galvanised metal vents"),
    (("highway", "road", "freeway"), "worn asphalt with tyre marks and cracks, concrete barriers with grime, dry roadside grass"),
    (("home", "room", "interior", "apartment"), "oak floorboards with wear, painted plaster walls, fabric with real weave and creases"),
    (("diner", "bar", "cafe", "restaurant"), "worn laminate and chrome counters, scuffed tile floor, vinyl seats with creases"),
)
LOCATION_DEFAULT = "street"  # "default: outdoor street", Run 2 (a) 2: only when the previz has the road or cars
NEUTRAL_KEY, NEUTRAL_BASE = "neutral", "worn real ground, weathered paint and real used materials"
# Run 2 (a) 1: the user's Look text naming the ground or a place wins: no location base at all.
LOOK_GROUND = ("track", "path", "gravel", "dirt", "grass", "lawn", "shore", "sand", "floor", "roof", "road", "highway",
               "sidewalk", "street", "counter", "booth", "wall")
# Run 11 (R&D, Run 10 check 1): the Look names where they stand ("stands on a pier"): that sentence goes beside the
# subject's line, and the location base keeps only its parts that aren't ground ("still lake water"), with the place's
# own surface first. "Wet pebbled granite shoreline, packed earth path" as the only ground put the man beside the pier.
STANDING_PLACES: dict[str, str] = {
    "pier": "the weathered timber boards of the pier underfoot", "jetty": "the weathered timber boards of the jetty underfoot",
    "dock": "the weathered timber boards of the dock underfoot", "boardwalk": "the weathered timber boards of the boardwalk underfoot",
    "pontoon": "the weathered boards of the pontoon underfoot", "deck": "weathered timber deck boards underfoot",
    "quay": "the worn stone of the quay underfoot", "wharf": "the worn timber and stone of the wharf underfoot",
    "bridge": "the bridge's worn deck and railings", "balcony": "the balcony's worn floor and railing",
    "terrace": "the terrace's worn stone paving", "porch": "the porch's painted timber boards",
    "veranda": "the veranda's painted timber boards", "platform": "the worn concrete edge of the platform",
    "steps": "worn stone steps", "stairs": "worn stone stairs", "ledge": "a rough weathered stone ledge",
    "cliff": "rough weathered rock at the cliff edge", "rooftop": "", "roof": "",  # the rooftop base row says the roof
}
# Someone standing, sitting or walking there: "the sun ... across the rooftop" is the light's, not a place to stand.
STAND_ON = (r"(?i)\b(?:stands?|standing|stood|sits?|sitting|sat|waits?|waiting|leans?|leaning|kneels?|kneeling|lies|lying|"
            r"walks?|walking|strolls?|strolling|poses?|posing|perched)\b[^.!?]*?\b(?:on|onto|upon|atop|at|across|along)\s+"
            r"(?:(?:the|a|an|this|that|its|his|her|their)\s+)?(?:[\w-]+\s+){{0,3}}?{word}s?\b")
GROUND_PART = (r"(?i)\b(shoreline|shore|path|paths|pavement|asphalt|kerbs?|grass|gravel|earth|sand|floor|floorboards|tiles?|"
               r"leaf litter|track|ruts|roof surface|ground)\b")
# v2.4 3: with no keyword, the previz objects decide before the street default: trees or grass with no road is a park;
# a road with vehicles is a street.
NATURE_OBJECTS = ("trees", "grass")
ROAD_OBJECTS = ("the road",)
VEHICLE_OBJECTS = ("cars",)
INTERIOR_KEYS = ("home", "room", "interior", "apartment", "diner", "bar", "cafe", "restaurant")
# Per-object phrases, keyed by the previz object's word (inView.ts wordFor). Ground words are the location base's job.
OBJECT_PHRASE: dict[str, str] = {
    "a bench": "a bench of weathered timber slats on metal legs",  # v2.4 4: never "park bench" outside a park
    "cars": "a parked car with painted bodywork and road dust",  # v2.4 4: a non-subject car (replaces the bodywork phrase)
    "a table": "a scuffed metal table",
    "chairs": "a used chair with worn edges",
    "trees": "real trees with full, irregular canopies, rough bark and uneven branches",  # v2.4 4: everywhere
    "buildings": "a weathered building with real wall texture",  # Run 2 (a) 6
    "walls": "weathered concrete",
    "boxes": "weathered concrete",
    "a planter": "a real shrub in a stone planter",
    "hedges": "a real shrub in a stone planter",
}
# Run 2 (a) 6: buildings when the Look text names a house or farmhouse.
HOUSE_WORDS, HOUSE_PHRASE = ("house", "farmhouse"), "a small weathered house with painted clapboard or plaster walls and a shingle roof"
OBJECT_OTHER = "{name}, weathered and real"  # "anything else" (Run 2 (a) 5)
LOOK_COVERS = ("buildings", "walls", "boxes")  # R2-7: the place's own fabric, which a Look that describes the place covers
NOT_OBJECTS = ("people", "the road", "a path", "the pavement", "grass", "water", "sand dunes", "snow", "mountains")
# v2.4 4: an object's phrase in one location (by the location row's first keyword), over OBJECT_PHRASE.
OBJECT_PHRASE_AT: dict[str, dict[str, str]] = {
    "park": {"a bench": "a park bench of weathered timber slats on cast-iron legs"},
    "rooftop": {"a bench": "a bench of weathered timber slats on metal legs", "trees": "a small tree in a large concrete planter"},
}

# Rule 2, paragraph 6: the source by a time cue in the user's words (else the previz time), its direction from the
# previz sun or key light relative to camera (default camera left), then the locked LIGHT_LINE on every shot.
LIGHT_SOURCE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dawn", "sunrise"), "low soft dawn sun from {dir}, just above the horizon, cool shadows"),
    (("morning",), "clear morning sun from {dir}, about 30 degrees high"),
    (("noon", "midday"), "high midday sun from {dir}, short hard shadows"),
    (("afternoon",), "late-afternoon sun from {dir}, about 30 degrees high"),
    (("golden hour", "golden", "sunset"), "low golden-hour sun from {dir}, just above the horizon, long warm shadows"),
    (("dusk", "blue hour"), "soft blue-hour skylight with warm practical lights coming on"),
    (("night",), "street lamps and lit windows as the light sources, from {dir}"),
    (("overcast", "cloudy"), "soft overcast daylight from {dir}, gentle shadows"),
)
LIGHT_INTERIOR = "soft daylight through a window at {dir}"  # interior, no cue
LIGHT_EXTERIOR = "afternoon"  # exterior, no cue: the afternoon line
LIGHT_DIRECTIONS = ("camera left", "camera right", "behind camera", "backlit")
# Step 20 / v2.4 6: a sentence of the user's Look (or a showcase's) about the light REPLACES the paragraph 6 source
# sentence: paragraph 6 is the custom line, then the exposure sentence once (LIGHT_LINE's last), no default sun line.
USER_LIGHT = r"\b(sun|sunlight|sunset|sunrise|light|lights|lit|lamps?|shadows?|golden[- ]hour|blue hour|dusk|dawn|backlit|overcast|moonlight)\b"
EXPOSURE = "Exposed for the main subject; shadows stay open, with readable texture in the ground and foreground, never pure black."
LIGHT_DIRECTION_DEFAULT = "camera left"
WEATHER = ("rain", "rainy", "raining", "fog", "foggy", "snow", "snowy", "snowing")  # never written into the start frame

# Rule 4, groups: people and vehicles numbered left to right by screen position.
GROUP_WORD = {"person": "person", "vehicle": "vehicle"}
GROUP_PLACE = ("on the left", "in the centre", "on the right")
# v2.3 (e): four read far left, left of centre, right of centre, far right; five or more are numbered in screen order.
GROUP_PLACE_4 = ("on the far left", "left of centre", "right of centre", "on the far right")
GROUP_NO_REF = "Each person looks clearly different, with their own face, age, hair and everyday clothing in distinct colours."

# v2.3 (a): a contact or mutual action is one group line naming everyone. Keys: the previz action or a word in the
# sentence; values: the verb phrase from the rule's list. The hug's own tail (and the face bracket, only when the previz
# says who faces camera) follow the rule's example; anyone with no action of their own gets DEFAULT_ACTION.
SHARED_ACTIONS: dict[str, str] = {
    "hug": "hug each other", "hugging": "hug each other", "embrace": "embrace", "embracing": "embrace",
    "handshake": "handshake", "kiss": "kiss", "kissing": "kiss", "dance together": "dance together",
    "hold hands": "hold hands", "holding hands": "hold hands", "talk to each other": "talk to each other",
}
SHARED_TOUCH = {"hug each other": "every hand resting on the other's back and traceable to a shoulder",
                "embrace": "every hand resting on the other's back and traceable to a shoulder"}
FACE_TOWARD = "{face} toward the viewer"  # "[her face / his face] toward camera", Run 2 (d): no camera words
DEFAULT_ACTION = "standing naturally, weight on one leg, hands relaxed"
# v2.4 7: a group action in the user's words ("three dancers", "they dance") applies to every person.
GROUP_ACTION_WORDS = r"(?i)\b(?:they|both|all|everyone|together)\s+(\w+)|\b(\w+)ers\b"
# v2.4 7: a mutual action adds this to paragraph 4, and any shot with 2+ people adds GROUP_NEGATIVE.
DIFFERENT_PEOPLE = "{names} are {count} different people."
COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
GROUP_NEGATIVE, GROUP_NEGATIVE_PAIR = "duplicate person, twins, clone", "third person"  # the pair's only
# Run 3 (s06, the director): with two or more people the count is said again after the blocking, in positive words ("no one
# else" and "nobody" are filtered as negations).
ONLY_PEOPLE = "These {n} are the only people in the frame; the rest of the frame is open ground."
ONLY_PEOPLE_AT = {"rooftop": "These {n} are the only people on the roof; the rest of the roof is open tar, gravel and sky."}
# Run 3 (s04, the director): a lone person under SMALL_HERO of the frame height is said small, with no number, no distance and
# no gesture: far across the place with the set around them, and "a small distant man in <one or two garments>".
SMALL_HERO = 0.15
SMALL_BLOCKING = "{label}: {where}, far across the {place}, a small figure with {around} filling the frame around {pronoun}."
SMALL_PERSON = "{label}: a small distant {noun}{worn}, {doing}."
# The place's noun and what fills the frame around a small figure there, by a word in the user's words, the sentence or
# the previz place (the most particular first: "a city park" is a park).
SMALL_PLACE: tuple[tuple[tuple[str, ...], tuple[str, str]], ...] = (
    (("pier", "jetty", "dock", "boardwalk"), ("pier", "pier and water")), (("bridge",), ("bridge", "bridge and view")),
    (("park", "lawn", "garden"), ("park", "lawn and trees")), (("rooftop", "roof"), ("roof", "roof and skyline")),
    (("lake", "pond", "shore"), ("lake", "shore and water")), (("beach", "sea", "coast"), ("beach", "sand and sea")),
    (("forest", "woods"), ("forest", "trees")), (("field", "meadow"), ("field", "long grass")),
    (("track", "farm", "farmhouse", "country"), ("fields", "track and fields")),
    (("highway", "freeway", "road"), ("road", "road and verges")), (("street", "sidewalk", "city"), ("street", "street and buildings")),
)
SMALL_PLACE_DEFAULT = ("open ground", "open ground")
MATERIALS = ("wool", "cotton", "linen", "leather", "suede", "denim", "corduroy", "tweed", "silk", "nylon", "fleece", "knit", "cashmere")
COLOUR_WORDS = ("black", "white", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "tan", "beige", "cream",
                "ivory", "navy", "olive", "khaki", "mustard", "teal", "burgundy", "maroon", "gold", "silver", "grey", "gray",
                "charcoal", "dark", "light", "pale", "bright")
HAIR_WORDS = ("hair", "beard", "braid", "bob", "curls", "moustache", "mustache", "stubble", "head", "ponytail", "bun", "afro", "puff")
# v2.3 (b): an object phrase keeps its words; the location base drops its part that names the same thing.
# Object word -> the word its base part names (only trees repeat today: the park base and the tree phrase).
BASE_DUPES: dict[str, str] = {"trees": "trees"}
# v2.3 (c): a vehicle with no ref. Its type from the previz vehicle's word, default saloon.
VEHICLE_NO_REF = ("{label} is a real, everyday {type} in a plain solid colour, with glass reflections, tyres with real tread "
                  "and no readable badges.")
VEHICLE_TYPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hatchback",), "hatchback"),
    (("suv", "jeep", "4x4"), "SUV"),
    (("pickup", "truck"), "pickup"),
    (("van", "minivan", "bus"), "van"),
    (("saloon", "sedan", "car", "coupe", "taxi"), "saloon"),
)
VEHICLE_TYPE_DEFAULT = "saloon"
MANY_VEHICLES = "each vehicle a different make and colour"  # two or more vehicles

# Rule 5, the commercial-safe fallback (Qwen-Image-Edit 2511, the A2 way): added to the end of paragraph 2.
FALLBACK_LINE = "Replace every grey surface and figure with the real materials and people described; keep the exact layout, camera and framing."
# What the positive says (prompt_words: no grey): the rule's line without the colour.
FALLBACK_LINE_CLEAN = "Turn every surface and figure into the real materials and people described; keep the exact layout, viewpoint and framing."
# v2.3 (e): the fallback generates at this size, then a Lanczos upscale to SIZE.
FALLBACK_SIZE = (1664, 928)

LIGHT_LINE = ("The light falls directly on the subjects and the ground around them. Exposed for the main subject; shadows "
              "stay open, with readable texture in the ground and foreground, never pure black.")
LOOK_LINE = ("Tactile realism: readable colour, soft highlight roll-off, truthful skin colour, restrained fine grain over a "
             "sharp image. Texture comes from real surfaces and light, never a filter. Any signs or screens have no readable "
             "letters.")
TEST_LINE = "If any surface looks clean, plastic, grey, untextured or rendered, the image is wrong."
BANNED = ("8K", "hyperreal", "pore-level", "previz", "mannequin", "blockout", "proxy", "cinematic")
NEGATIVE = ("3D render, CGI, clay render, grey mannequin, plastic skin, untextured surfaces, flat even lighting, video game, "
            "illustration, HDR, oversharpened, smeared face, text, letters, logos, watermark, rain, fog, haze, extra people, "
            "extra fingers, underexposed, crushed shadows, black foreground")
VEHICLE_NEGATIVE = "parked car, static background, frozen wheels, car on a white background"  # when any subject is a vehicle

# A cast reference is used for its look, not its studio: whenever references go in, so does this (step 18b: c08 came out
# on a white backdrop, the sedan's sheet background).
REF_NEGATIVE = "studio background, white background, seamless backdrop"
# A cast reference's saved type decides what it is (v2.3 (e): the type always beats the name). Cast kinds are person,
# place, prop and look. v2.3 (f): a prop is a vehicle subject only when its name has a vehicle word (the name's last word,
# so "car keys" is keys, and not after PROP_NOT_VEHICLE: "toy truck") AND it sits on a vehicle figure in frame 0;
# otherwise it stays a prop, written only as PROP_PHRASE in paragraph 5. Places and looks are not subjects.
PROP_NOT_VEHICLE = ("toy", "model", "miniature", "scale")
PROP_PHRASE = "{name} as in image {image}"
VEHICLE_REF_WORDS = ("car", "sedan", "saloon", "coupe", "suv", "van", "truck", "bus", "taxi", "jeep", "pickup", "vehicle",
                     "motorbike", "motorcycle", "bike", "boat", "convertible", "hatchback", "wagon", "limo", "lorry")
# Step 20 (R&D's c07, c08): a cast line is a description. A generator's caption ("Car reference sheet on a plain light
# gray studio background: the same ...") loses its framing words; a line cut at a saving limit with no sentence end keeps
# its whole clauses only, or goes. With no line left, a ref is just "X is the one in image N."
CAPTION_LEAD = r"(?i)^[^:]{0,90}\b(reference sheet|character sheet|turnaround|model sheet|studio background|backdrop)\b[^:]*:\s*(the same\s+)?"
# v2.4 1: and any "shown three/four views..." tail (or what's left of it at a cut: "show"); the description then reads
# "a dark green 1970s fastback muscle coupe ..." (the caption's "the same" goes, an article stands in).
CAPTION_TAIL = r"(?i)\s+sh(o(wn?)?)?(\s+(two|three|four|five|six|\d+)\b.*)?$"
CUT_AT = (140, 200)  # where older saves cut a line (the Images page's 140, a sheet's 200)
REF_ONLY = "{label} is the one in image {image}."
# No person or vehicle in frame 0 and no cast reference to stand in: refused, in the blank-frame guard's words.
NO_SUBJECTS = "Frame 1 shows no person or vehicle. Aim the camera at the subject, then Make shot again, or add a cast reference."
REMAKE = "Remake this shot"  # the one action for an old shot whose frame 0 can't be re-rendered here

# Facing from the previz figure's rotation at frame 0 (runtime.ts facingOf: camera, away, left or right on screen). The
# previz wins over the user's words (facing_conflicts flags it).
FACING_PERSON = {"camera": "facing the viewer, face toward the viewer", "away": "seen from behind, back to the viewer",
                 "left": "seen in profile, head and body turned to frame left",
                 "right": "seen in profile, head and body turned to frame right"}
FACING_VEHICLE = {"camera": "seen dead on from the front, squared to the viewer, headlights evenly either side of the centre line",
                  "away": "seen directly from behind, driving away from the viewer",
                  "left": "seen side-on in profile, its front end and headlights pointing to frame left",
                  "right": "seen side-on in profile, its front end and headlights pointing to frame right"}
# What the user's words say about facing, by the previz value they mean (first match wins).
FACING_WORDS: tuple[tuple[str, str], ...] = (
    ("away", r"(?i)\b(from behind|backs? to (the )?(camera|viewer)|facing away|walks? away|drives? away|seen from the back)\b"),
    ("camera", r"(?i)\b(toward(s)? (the )?(camera|viewer)|facing (the )?(camera|viewer)|faces? (the )?(camera|viewer)|to camera|head-on|dead on)\b"),
    ("left", r"(?i)\b(facing|toward(s)?|nose to) (frame )?left\b"),
    ("right", r"(?i)\b(facing|toward(s)?|nose to) (frame )?right\b"),
)

MAX_REFS = 2
STEPS, CFG, DENOISE, RESOLUTION, SAMPLER, SCHEDULER = 30, 3.0, 1.0, 1536, "euler", "simple"
SIZE = (2048, 1152)  # what RESOLUTION gives at 16:9
DA3_MODEL, DA3_RESOLUTION, CANNY_SOFT = "depth_anything_3_mono_large.safetensors", 1512, (0.08, 0.25)
SUBJECT_GROW = 12  # px the subject matte grows by (R&D's make_subjmask_b3.py: a 3x3 square, 12 times)

# Method zbuf (R&D's l2r_round4 RESULTS.md, "Setup notes" and "Still graph").
ZBUF_PATCH = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"  # models/model_patches
ZBUF_STEPS, ZBUF_SIZE, ZBUF_BLUR = 8, (1920, 1088), 4
DEPTH_FAR = 1000.0  # metres at the depth pass's full 24-bit code (runtime.ts DEPTH_FAR)
DEPTH_FILE = "previz_depth_{:04d}.png"  # the page's depth pass of one frame, in the shot folder (previz_maker)
STRENGTH = {"person": 0.75, "vehicle": 0.65}  # by the hero subject; the one retry uses the other
# Size bands, render over previz height of the hero subject (head top to feet): within 15% passes, 15 to 50% over warns,
# more than 50% over fails, more than 15% under warns.
SIZE_PASS, SIZE_FAIL = 0.15, 0.50
# Run 2: paragraph 2's size sentence from the previz figure's screen box (the director's "his head is about 6 percent of the
# frame height", "the car fills about 12 percent of the frame width"). A head is about 1/7.5 of a standing figure.
SIZE_PERSON = "{label}'s head is about {pct} percent of the frame height."
SIZE_VEHICLE = "{label} fills about {pct} percent of the frame width."
HEAD_STANDING, HEAD_SITTING = 0.13, 0.18
# Run 2 (c): with a previz distance in paragraph 2, the user's own distance or size sentence goes (the previz wins).
USER_DISTANCE = (r"(?i)\b(metres?|meters?)\b|\bfills?\s+(about|roughly|around|nearly|almost)\b.*\bof the frame\b"
                 r"|\bpercent of the frame\b|\bfills?\s+(?:(?:a|an|one|two)\s+)?(?:quarter|half|third)s?\s+(?:of\s+)?the frame\b")
TAKEN_OUT = 'Taken out of your words: "{clause}" ({why}).'
# ======================================================================================================================

DOING = {"walk": "walking", "run": "running", "jog": "jogging", "sit": "sitting", "stand": "standing", "idle": "standing",
         "wave": "waving", "dance": "dancing", "drive": "driving", "talk": "talking", "hug": "hugging", "jump": "jumping",
         "skate": "skateboarding", "ride": "riding", "read": "reading", "type": "typing", "kneel": "kneeling", "lie": "lying down",
         "sing": "singing", "sip": "drinking", "hold_phone": "holding a phone", "argue": "arguing", "point": "pointing"}


@dataclass
class Subject:
    kind: str
    label: str
    where: str
    place: str  # "on the left", "in the centre", "on the right"
    distance_m: float | None  # None: a reference standing in for a figure the scene didn't record
    x: float = 0.5
    doing: str | None = None
    word: str = ""  # the previz's noun for it ("sedan", "van")
    faces: str | None = None  # "her face" / "his face" when the previz says this one faces camera
    speed_ms: float | None = None
    line: str | None = None  # identity, from a cast entry
    image: int | None = None  # its reference image's number in the graph
    facing: str | None = None  # camera / away / left / right, from the previz figure's rotation
    h: float | None = None  # its height and width on screen at frame 0, as fractions of the frame (runtime.ts)
    w: float | None = None


@dataclass
class Plan:
    """What a start frame will be made of: the words and the references in order (shown to the user and an agent before
    anything runs)."""
    positive: str
    negative: str
    subjects: list[Subject] = field(default_factory=list)
    refs: list[Path] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)
    matte: Path | None = None  # the previz renderer's subject matte (white = a person or a vehicle)
    method: str = METHOD
    action: str | None = None  # REMAKE when the shot's frame 0 lacks data that only the previz page can render
    flags: list[str] = field(default_factory=list)  # where the user's words disagreed with the previz (the previz won)


class NoSubjects(ValueError):
    """Nothing to make a start frame of: NO_SUBJECTS, said to the user as is."""


def _strip(text: str, words: tuple[str, ...]) -> str:
    out = text
    for w in words:
        out = re.sub(rf"(?i)(?<![\w-]){re.escape(w)}(?![\w-])", "", out)
    return re.sub(r"\s+([,.;:])", r"\1", re.sub(r"[ \t]{2,}", " ", out)).strip(" ,;")


# Run 3 (R2-8): never a whole sentence. Only the "where the previz places it" clause goes, then the bare previz words
# ("A white car is parked where the previz places it, on the lane." -> "A white car is parked on the lane.").
PREVIZ_CLAUSE = r"(?i),?\s*where the (?:previz|pre-viz) [^,.]*,?"
PREVIZ_NAMES = r"(?i)(?<![\w-])(previz|pre-viz|mannequins?|blockout|block-out|proxy|proxies)(?![\w-])"


def strip_banned(text: str) -> str:
    """User or cast words without the previz clause and the previz words, then the template's other banned words and
    the weather words ("a cinematic rainy look" -> "a look")."""
    out = re.sub(PREVIZ_CLAUSE, " ", text or "")
    out = re.sub(PREVIZ_NAMES, "", out)
    return _strip(out, BANNED + WEATHER)


# Run 3 (R2-12): a sentence is split into clauses (commas, semicolons, " and ", " with ") and only a matching clause
# goes. When the sentence's own subject went with it, a following "with X" reads "The <subject> has X".
CLAUSE_SPLIT = r"(\s*[;,]\s*|\s+and\s+|\s+with\s+)"
SUBJECT_VERB = r"(?i)^(.+?)\s+(?:is|are|was|were|sits|stands|stays|waits|fills?)\b"
# Run 4: an " and " before the sentence's first verb joins a noun phrase ("A man in a navy coat and grey cap stands ..."),
# so it never splits; and a clause that keeps its subject and its own verb loses only what follows the verb ("A man ...
# stands about three metres from the viewer." -> "A man ... stands.").
CLAUSE_VERB = (r"(?i)\b(is|are|was|were|sits?|stands?|stays?|waits?|fills?|walks|runs|jogs|drives|rides|leans|looks|faces|"
               r"holds|has|have|moves|waves|dances|curves|lies|rests|parked|sat|stood)\b")
KEEPS_VERB = r"(?i)^(.+?\s+(?:sits|stands|stays|waits|waves|leans|rests|(?:is|are)\s+(?:parked|standing|sitting|waiting)))\s+(\S.*)$"


def _clauses(sent: str) -> list[str]:
    """CLAUSE_SPLIT's parts (clause, separator, clause, ...), with a noun phrase's " and " joined back."""
    parts = re.split(CLAUSE_SPLIT, sent)
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        if re.fullmatch(r"\s+and\s+", parts[i]) and not re.search(CLAUSE_VERB, "".join(out)):
            out[-1] += parts[i] + parts[i + 1]
        else:
            out += [parts[i], parts[i + 1]]
    return out


def strip_clauses(text: str, pattern: str) -> tuple[str, list[str]]:
    """The words without the clauses that match pattern, and the clauses taken out. A sentence with nothing left goes;
    anything left stays."""
    out, gone = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        if not sent or not re.search(pattern, sent):
            out.append(sent)
            continue
        end = sent[-1] if sent[-1:] in ".!?" else "."
        parts = _clauses(sent.rstrip(".!?"))
        subject = re.match(SUBJECT_VERB, parts[0])
        kept: list[str] = []
        lost_subject = False
        for i in range(0, len(parts), 2):
            clause = parts[i]
            held = re.match(KEEPS_VERB, clause)
            if held and re.search(pattern, held.group(2)) and not re.search(pattern, held.group(1)):
                gone.append(held.group(2).strip())
                clause = held.group(1)
            elif clause.strip() and re.search(pattern, clause):
                gone.append(clause.strip())
                lost_subject = lost_subject or i == 0
                continue
            if not clause.strip():
                continue
            if not kept and lost_subject and subject:
                lead = subject.group(1)
                clause = re.sub(r"(?i)^with\s+", f"{lead} has ", clause) if re.match(r"(?i)with\s", clause) else (
                    clause if re.match(r"(?i)(the|a|an|his|her|their|its|he|she|they|it)\b", clause) else f"{lead} {clause}")
            kept.append((parts[i - 1] if kept else "") + clause)
        left = "".join(kept).strip(" ,;")
        if left:
            out.append(_cap(left) + end)
    return " ".join(x for x in out if x), gone


def _where(x: float) -> str:
    return "left of frame" if x < 0.38 else "right of frame" if x > 0.62 else "centre of frame"


PLACE_WHERE = {GROUP_PLACE[0]: "left of frame", GROUP_PLACE[1]: "centre of frame", GROUP_PLACE[2]: "right of frame",
               GROUP_PLACE_4[0]: "far left of frame", GROUP_PLACE_4[1]: "left of centre", GROUP_PLACE_4[2]: "right of centre",
               GROUP_PLACE_4[3]: "far right of frame"}


def _place(x: float) -> str:
    return GROUP_PLACE[0] if x < 0.38 else GROUP_PLACE[2] if x > 0.62 else GROUP_PLACE[1]


def _metres(m: float) -> str:
    return f"about {round(m)} metres" if m >= 2 else f"about {round(m, 1):g} metres"


def _height(m: float) -> str:
    return f"about {round(m * 100)} cm above the ground" if m < 1 else f"about {round(m, 1):g} metres above the ground"


def _angle(pitch: float) -> str:
    if pitch <= -30:
        return "a high angle looking down"
    if pitch <= -8:
        return "looking slightly down"
    if pitch >= 8:
        return "a low angle looking up"
    return "level"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _has(text: str, key: str) -> bool:
    return re.search(rf"(?i)(?<![\w-]){re.escape(key)}(?![\w-])", text or "") is not None


def _first(words: list[str], table) -> tuple[str, str] | None:
    """The first table row whose key is a whole word in the first text that has one (the user's words before the
    previz data): (key, value)."""
    for text in words:
        for keys, value in table:
            for k in keys:
                if _has(text, k):
                    return k, value
    return None


def lens_words(focal_mm: float) -> str:
    return next(words for top, words in LENS_BANDS if focal_mm <= top)


def location(user: str, sentence: str, place: str, objects: list[str] | None = None) -> tuple[str, str]:
    """(key, base) for paragraph 5: the user's words, then the sentence, then the previz place, then what the previz
    objects say (v2.4 3), else a street."""
    hit = _first([user, sentence, place], LOCATION_BASE)
    if not hit and objects is not None:
        road = any(w in ROAD_OBJECTS for w in objects)
        if not road and any(w in NATURE_OBJECTS for w in objects):
            hit = next(("park", v) for keys, v in LOCATION_BASE if "park" in keys)
        elif not road and not any(w in VEHICLE_OBJECTS for w in objects):
            hit = (NEUTRAL_KEY, NEUTRAL_BASE)  # Run 2 (a) 2: no road, no cars: never the street
    return hit or (LOCATION_DEFAULT, next(v for keys, v in LOCATION_BASE if LOCATION_DEFAULT in keys))


def _noun(word: str) -> str:
    """A previz object's noun, singular: "a bench" -> bench, "trees" -> tree."""
    w = re.sub(r"(?i)^(a|an|the)\s+", "", word).strip().lower()
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def standing_place(user: str) -> tuple[str, str] | None:
    """(the place word, its sentence) when the Look says where they stand ("A man ... stands on a pier at dusk."), the
    first such sentence; None when it names none (STANDING_PLACES)."""
    for sent in re.split(r"(?<=[.!?])\s+", user or ""):
        for word in STANDING_PLACES:
            if re.search(STAND_ON.format(word=re.escape(word)), sent):
                return word, sent.strip()
    return None


def world_sentence(user: str, sentence: str, place: str, objects: list[str], standing_on: str | None = None) -> str:
    """Paragraph 5: the location base, then one phrase per previz object present (each phrase once), in the location's
    own wording where it has one (OBJECT_PHRASE_AT). Run 2 (a) 1: when the user's Look text names the ground or the
    place, it is the base: no base sentence, and only the objects it doesn't name ("" when nothing is left). Run 11:
    standing_on (standing_place's word): the base keeps only its parts that aren't ground, after the place's own surface."""
    key, base = location(user, sentence, place, objects)
    # A light sentence ("sun rakes across the rooftop") goes to paragraph 6 and describes no ground.
    said = " ".join(s for s in re.split(r"(?<=[.!?])\s+", user or "") if not re.search(USER_LIGHT, s, re.I))
    own_ground = any(_has(said, w) or _has(said, w + "s") for w in LOOK_GROUND)
    at = OBJECT_PHRASE_AT.get(key if key == NEUTRAL_KEY else next(keys[0] for keys, v in LOCATION_BASE if v == base), {})
    if standing_on and not _has(base, standing_on) and not (standing_on in ("roof", "rooftop") and key == "rooftop"):
        kept = [b for b in base.split(", ") if not re.search(GROUND_PART, b)]
        base = ", ".join([x for x in [STANDING_PLACES.get(standing_on, ""), *kept] if x]) or base
    house = any(_has(said, w) for w in HOUSE_WORDS)
    phrases: list[str] = []
    kept = [w for w in objects if w not in NOT_OBJECTS]
    for w in kept:
        if own_ground and (_has(said, _noun(w)) or _has(said, _noun(w) + "s")):
            continue  # the user's own words already describe it
        house_here = w == "buildings" and house
        if own_ground and not house_here and not at.get(w) and (w in LOOK_COVERS or w not in OBJECT_PHRASE):
            continue  # R2-7: the Look describes the place, so no generic "weathered and real" paragraph for it
        p = (HOUSE_PHRASE if house_here else None) or at.get(w) or OBJECT_PHRASE.get(w) or OBJECT_OTHER.format(name=w)
        if p not in phrases:
            phrases.append(p)
    # The object phrase wins: the base's part about the same thing goes (v2.3 (b), trees in s04).
    dupes = [BASE_DUPES[w] for w in kept if w in BASE_DUPES]
    parts = [] if own_ground else [b for b in base.split(", ") if not any(re.search(rf"(?i)\b{re.escape(d)}\b", b) for d in dupes)]
    return _cap(", ".join([*parts, *phrases])) + "." if parts or phrases else ""


def light_sentence(user: str, sentence: str, framing: dict, interior: bool) -> str:
    """Paragraph 6: the source by the time cue (the user's words, then the sentence, then the previz time), its direction
    from the previz key light (default camera left), then the locked LIGHT_LINE. Weather never enters it."""
    d = (framing.get("light") or {}).get("dir")
    direction = d if d in LIGHT_DIRECTIONS else LIGHT_DIRECTION_DEFAULT
    time = str((framing.get("world") or {}).get("time") or "")
    hit = _first([user, sentence, time], LIGHT_SOURCE)
    if hit:
        source = hit[1]
    elif interior:
        source = LIGHT_INTERIOR
    else:
        source = next(v for keys, v in LIGHT_SOURCE if LIGHT_EXTERIOR in keys)
    return f"{_cap(source.format(dir=direction))}. {LIGHT_LINE}"


def ref_kind(c: dict | None) -> str | None:
    """person, vehicle (saved as one), "prop vehicle" (a prop whose name is a vehicle: a vehicle only on a vehicle figure)
    or None, by the cast entry's saved type."""
    if not c:
        return None
    kind = c.get("kind") or "person"
    if kind in ("person", "vehicle"):
        return kind
    words = re.findall(r"[a-z0-9]+", c.get("name", "").lower())
    head = {words[-1], words[-1].removesuffix("s"), words[-1].removesuffix("es")} if words else set()
    if kind == "prop" and head & set(VEHICLE_REF_WORDS) and not set(words) & set(PROP_NOT_VEHICLE):
        return "prop vehicle"
    return None


def props_of(ref_cast: list[dict | None], subs: list["Subject"], first_ref: int) -> list[str]:
    """v2.3 (f): the prop references that are not subjects, as paragraph 5 object phrases."""
    used = {s.label for s in subs}
    return [PROP_PHRASE.format(name=c["name"].strip(), image=first_ref + i) for i, c in enumerate(ref_cast)
            if c and c.get("kind") == "prop" and c.get("name", "").strip() and c["name"].strip() not in used]


def subjects_of(framing: dict, cast: list[dict], sentence: str, ref_cast: list[dict | None] | None = None) -> list[Subject]:
    """Frame 0's people and vehicles in view, counted from the previz figures (never the sentence's noun) and numbered left
    to right per kind: "person 1", "vehicle 2", or the cast name the sentence gives that one (people take person entries,
    vehicles prop entries, in the sentence's order)."""
    named = sorted(((m.start(), c) for c in cast for m in [re.search(rf"(?i)(?<!\w)@?{re.escape(c.get('name', '').strip())}\b", sentence)]
                    if c.get("name", "").strip() and m), key=lambda t: t[0])
    names = {"person": [c for _, c in named if ref_kind(c) == "person"],
             "vehicle": [c for _, c in named if ref_kind(c) in ("vehicle", "prop vehicle")]}
    # The attached references name the figures too (after the ones the sentence names), each by its own kind.
    for c in ref_cast or []:
        k = {"prop vehicle": "vehicle"}.get(ref_kind(c) or "", ref_kind(c))
        if k and c not in names[k]:
            names[k].append(c)
    box = framing.get("subject")
    box_x = (float(box[0]) + float(box[2])) / 2 if isinstance(box, (list, tuple)) and len(box) == 4 else 0.5
    out: list[Subject] = []
    figures = [s for s in framing.get("subjects") or [] if isinstance(s, dict)]
    group = next((s.get("doing") for s in figures if s.get("doing") and re.search(r"(?i)(s|people|crowd|group|couple)$", str(s.get("word") or ""))), None)
    group = group or group_doing(sentence)
    for kind in ("person", "vehicle"):
        seen = sorted(({**s, "doing": s.get("doing") or (group if kind == "person" else None)}
                       for s in figures if s.get("kind") == kind and s.get("in_view")), key=lambda s: float(s.get("x", 0.5)))
        # A thin scene (an older shot: a box and a horizon, no figures) still has its references: each attached one of
        # this kind beyond the figures becomes a subject, the first where the box is. A prop only names a vehicle figure
        # that is there (v2.3 (f)), so it never adds one.
        attached = [c for c in ref_cast or [] if ref_kind(c) == kind]
        for i, c in enumerate(attached[len(seen):]):
            seen.append({"kind": kind, "x": box_x if not seen and i == 0 else 0.5, "in_view": True, "_ref": c,
                         "doing": "drive" if kind == "vehicle" and re.search(r"(?i)\b(driv|speed|rac|cruis|chase)\w*", sentence or "") else None})
        # A pair reads left and right, a trio left, centre and right (the rule's own "person 1 on the left, person 2 on
        # the right"), even when they stand close together; bigger groups and lone subjects go by where they are.
        relative = {1: None, 2: (GROUP_PLACE[0], GROUP_PLACE[2]), 3: GROUP_PLACE, 4: GROUP_PLACE_4}.get(len(seen), ())
        for n, s in enumerate(seen):
            c = s.get("_ref") or (names[kind][n] if n < len(names[kind]) else None)
            x = float(s.get("x", 0.5))
            place = relative[n] if relative else (_place(x) if relative is None else "")
            out.append(Subject(kind=kind, label=c["name"].strip() if c else f"{GROUP_WORD[kind]} {n + 1}",
                               where=PLACE_WHERE.get(place) or PLACE_WHERE[_place(x)], place=place,
                               word=str(s.get("word") or ""), faces=s.get("faces_camera"),
                               distance_m=float(s["distance_m"]) if s.get("distance_m") is not None else None, x=x,
                               doing=DOING.get(str(s.get("doing") or "")), speed_ms=s.get("speed_ms"),
                               line=cast_line(c.get("line", "")) if c else None,
                               facing=s.get("facing") if s.get("facing") in FACING_PERSON else None,
                               h=_frac(s.get("h")), w=_frac(s.get("w"))))
    # An older shot has no size per figure: the hero (the first in view) takes the framing box.
    same = [s for s in out if s.kind == hero_kind(framing) and s.distance_m is not None]
    hero = min(same, key=lambda s: abs(s.x - box_x)) if same else None
    if hero and hero.h is None and isinstance(box, (list, tuple)) and len(box) == 4:
        hero.w, hero.h = _frac(float(box[2]) - float(box[0])), _frac(float(box[3]) - float(box[1]))
    return out


def _frac(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0 < f <= 1.5 else None


def size_words(s: Subject) -> str | None:
    """Run 2: the previz size as a frame-relative number, so a far subject stays small (s04 came out 4x): a person's
    head against the frame height, a vehicle against the frame width. None when the previz has no size for it."""
    if s.kind == "person" and s.h:
        pct = round(s.h * (HEAD_SITTING if s.doing in ("sitting", "kneeling", "lying down") else HEAD_STANDING) * 100)
        return SIZE_PERSON.format(label=_cap(s.label), pct=max(1, pct))
    if s.kind == "vehicle" and s.w:
        return SIZE_VEHICLE.format(label=_cap(s.label), pct=max(1, round(min(s.w, 1.0) * 100)))
    return None


def _person_noun(s: Subject) -> str:
    """man, woman, boy, girl (from the cast line's lead, else the previz word), else person."""
    m = re.match(r"(?i)^(?:a|an)\s+(?:young\s+|old(?:er)?\s+)?(man|woman|boy|girl)\b", s.line or "")
    w = (m.group(1) if m else s.word or "").lower()
    return w if w in ("man", "woman", "boy", "girl") else "person"


def worn_words(line: str | None) -> str:
    """One or two garments of a cast line with their colour and no material, under one article: " in a navy peacoat and
    grey flat cap" ("" when the line names none)."""
    rest = re.sub(r"(?i)^(?:a|an)\s+(?:young\s+|old(?:er)?\s+)?(?:man|woman|boy|girl|person)(?:\s+in\s+(?:his|her|their)\s+\w+)?", "", line or "")
    segs = [s.strip() for s in re.split(r"(?i),\s*|\s+and\s+|\s+with\s+|\s+in\s+|\s+over\s+", rest) if s.strip()]
    worn = []
    for seg in segs:
        words = seg.split()
        if any(w.lower().strip(".") in HAIR_WORDS for w in words) or not any(w.lower() in COLOUR_WORDS for w in words):
            continue
        words = [("grey" if w.lower() == "gray" else w) for w in words if w.lower() not in MATERIALS + ("a", "an", "the")]
        worn.append(" ".join(words))
    worn = worn[:2]
    if not worn:
        return ""
    return " in " + ("an " if worn[0][0].lower() in "aeiou" else "a ") + " and ".join(worn)


def small_blocking(s: Subject, words: list[str], seat: bool) -> str:
    """Paragraph 2 for a small hero (SMALL_HERO): far across the place, the set filling the frame around them."""
    hit = _first(words, SMALL_PLACE)
    place, around = hit[1] if hit else SMALL_PLACE_DEFAULT
    items = (["bench"] if seat else []) + re.split(r"\s+and\s+|,\s*", around)
    things = "the " + (items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1])
    noun = _person_noun(s)
    pronoun = "him" if noun in ("man", "boy") else "her" if noun in ("woman", "girl") else "them"
    return SMALL_BLOCKING.format(label=_cap(s.label), where=s.where, place=place, around=things, pronoun=pronoun)


def small_person(s: Subject, seat: bool) -> str:
    """Paragraph 4 for a small hero: "Marcus: a small distant man in a navy peacoat and grey flat cap, sitting on the
    bench." No age, no face and no gesture: at that size they only blur."""
    doing = s.doing or "standing"
    if seat and doing == "sitting":
        doing = "sitting on the bench"
    return SMALL_PERSON.format(label=_cap(s.label), noun=_person_noun(s), worn=worn_words(s.line), doing=doing)


def group_doing(sentence: str) -> str | None:
    """v2.4 7: the action a group's words give everyone ("they dance", "three dancers"), as a previz action, else None."""
    for m in re.finditer(GROUP_ACTION_WORDS, sentence or ""):
        w = (m.group(1) or m.group(2) or "").lower()
        for k in (w, w.removesuffix("s"), w.removesuffix("ing"), w.removesuffix("ing") + "e", w + "e", w[:-1] if w[-2:-1] == w[-1:] else ""):
            if k and k in DOING:
                return k
    return None


def cast_line(text: str) -> str | None:
    """A cast entry's line for paragraph 4: its words without a generator caption's framing, and never cut mid-sentence
    (a line cut at a saving limit keeps its whole clauses, else goes)."""
    raw = " ".join(str(text or "").split())
    line = re.sub(CAPTION_LEAD, "", raw)
    tail = 0
    if line != raw:  # a sheet caption
        line, tail = re.subn(CAPTION_TAIL, "", line)
        if line and not re.match(r"(?i)(a|an|the)\b", line):
            line = ("an " if line[0].lower() in "aeiou" else "a ") + line
    if len(raw) in CUT_AT and not tail and not re.search(r"[.!?]$", line):
        cut = max(line.rfind(", "), line.rfind("; "), line.rfind(". "))
        line = line[:cut] if cut > 0 else ""
    line = strip_banned(line).strip(" ,;.")
    return line or None


def _names(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def shared_action(people: list["Subject"], sentence: str) -> str | None:
    """v2.3 (a): two or more people in a contact or mutual action, as one line naming everyone; None otherwise. The
    action comes from the previz (a figure's action) or a word in the sentence."""
    if len(people) < 2:
        return None
    raw = [str(s.doing or "") for s in people]
    verb = next((SHARED_ACTIONS[k] for k in SHARED_ACTIONS for d in raw if d and (d == k or DOING.get(k) == d)), None)
    verb = verb or next((v for k, v in SHARED_ACTIONS.items() if _has(sentence, k)), None)
    if not verb:
        return None
    line = f"{_cap(_names([s.label for s in people]))} {verb}"
    faces = [s.faces for s in people if s.faces]
    tail = [FACE_TOWARD.format(face=faces[0])] if len(faces) == 1 else []
    if verb in SHARED_TOUCH:
        tail.append(SHARED_TOUCH[verb])
    return ", ".join([line, *tail]) + "."


def vehicle_type(word: str) -> str:
    """v2.3 (c): the vehicle's type from the previz word, default saloon."""
    return next((t for keys, t in VEHICLE_TYPES if word and word.lower() in keys), VEHICLE_TYPE_DEFAULT)


def grey_line(subs: list["Subject"], objects: list[str]) -> str:
    """v2.3 (g): the grey clause for what's in the shot, the vehicle's type for "car" when every vehicle is one known
    type, then the objects in frame."""
    kinds = {s.kind for s in subs}
    key = {frozenset({"person"}): "people", frozenset({"vehicle"}): "vehicles",
           frozenset({"person", "vehicle"}): "people and vehicles"}.get(frozenset(kinds), "objects")
    types = {next((t for t in GREY_VEHICLE_TYPES if re.search(rf"(?i)\b{t}e?s?\b", f"{s.word} {s.label}")), "")
             for s in subs if s.kind == "vehicle"}
    line = GREY_LINES[key].format(vehicle=types.pop() if len(types) == 1 and "" not in types else GREY_VEHICLE)
    return line[:-1] + ": " + ", ".join(objects) + "." if objects else line


def _count(subs: list[Subject]) -> str:
    """[N] [people / vehicles]: "1 person", "2 people", "1 person and 1 vehicle"."""
    def one(kind: str, n: int) -> str:
        return f"{n} " + (("person" if n == 1 else "people") if kind == "person" else ("vehicle" if n == 1 else "vehicles"))
    people = sum(1 for s in subs if s.kind == "person")
    cars = sum(1 for s in subs if s.kind == "vehicle")
    if people and cars:
        return f"{one('person', people)} and {one('vehicle', cars)}"
    return one("vehicle", cars) if cars else one("person", people)


def facing_words(s: Subject) -> str | None:
    """The previz facing in words for this subject, or None when the previz didn't record one."""
    return (FACING_VEHICLE if s.kind == "vehicle" else FACING_PERSON).get(s.facing or "")


def said_facing(text: str) -> str | None:
    """camera / away / left / right when the words name a facing, else None."""
    return next((v for v, rx in FACING_WORDS if re.search(rx, text or "")), None)


# Run 2: who a facing claim can be about: a subject's name, a pronoun, or "the car" / "the woman" (a bench turned toward
# camera or a shadow toward frame left is not a facing claim).
SUBJECT_PRONOUNS = ("he", "she", "they", "both", "the car", "the vehicle", "the coupe", "the sedan", "the saloon", "the suv",
                    "the van", "the truck", "the woman", "the man", "the girl", "the boy", "the couple", "the dancers")
_CLAUSES = r"(\s*[,;]\s*|\s+(?:and|but|while|as)\s+)"


def _led_by(clause: str, names: list[str]) -> bool:
    """Does the clause start with one of the subjects (after a leading "and", "then" or "@")?"""
    c = re.sub(r"(?i)^\s*(and|but|then|while|as)\s+", "", clause).lstrip("@ ")
    return any(re.match(rf"(?i){re.escape(n)}\b", c) for n in names)


def facing_conflicts(text: str, subs: list[Subject]) -> tuple[str, list[str]]:
    """The previz wins over the user's words: (the words without the clauses naming a facing that no subject has in the
    previz, one flag per clause taken out). Only a clause about a subject counts (its name, he, she, they or the car, its
    own or the sentence's lead), and only that clause goes; a light or shadow sentence is never a facing claim. A pair
    facing two ways agrees with words for either."""
    faced = [s for s in subs if s.facing]
    if not faced or not text:
        return text, []
    names = sorted({*(s.label for s in subs), *SUBJECT_PRONOUNS}, key=len, reverse=True)
    out, flags = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not said_facing(sent) or re.search(USER_LIGHT, sent, re.I):
            out.append(sent)
            continue
        end = sent[-1] if sent[-1:] in ".!?" else ""
        parts = re.split(_CLAUSES, sent[:-1] if end else sent)
        lead = _led_by(parts[0], names)
        kept: list[str] = []
        for i in range(0, len(parts), 2):
            clause = parts[i]
            said = said_facing(clause)
            if said and (lead or _led_by(clause, names)) and all(s.facing != said for s in faced):
                one = faced[0]
                flags.append(f"Facing: the previz has {one.label} {facing_words(one)}; the words said otherwise, so the previz wins."
                             if len(faced) == 1 else "Facing: no one in the previz faces the way the words said, so the previz wins.")
                continue
            kept.append((parts[i - 1] if kept and i else "") + clause)
        left = "".join(kept).strip(" ,;")
        if left:
            out.append(_cap(left) + (end or "."))
    return " ".join(x for x in out if x), flags


def no_camera(text: str) -> str:
    """Run 2 (d): a still's positive never names a camera, a crew or a shot: clauses about filming go, and the facts'
    camera words read as the viewer's ("toward camera" -> "toward the viewer", "camera left" -> "frame left")."""
    paras = []
    for para in text.split("\n\n"):
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if re.search(NO_CAMERA_CLAUSE, sent):
                end = sent[-1] if sent[-1:] in ".!?" else "."
                parts = re.split(r"(\s*[,;]\s*)", sent.rstrip(".!?"))
                keep = [p for i, p in enumerate(parts) if not i % 2 and p.strip() and not re.search(NO_CAMERA_CLAUSE, p)]
                sent = (_cap(", ".join(keep)) + end) if keep else ""
            for rx, to in VIEWER_WORDS:
                sent = re.sub(rf"(?i){rx}", to, sent)
            if sent:
                out.append(_cap(sent))
        if out:
            paras.append(" ".join(out))
    return "\n\n".join(paras)


def _moving(s: Subject, sentence: str) -> bool:
    return bool(s.speed_ms) or s.doing == "driving" or re.search(r"(?i)\b(driv|speed|rac|cruis|zoom)\w*", sentence or "") is not None


def plan(spec: dict, framing: dict, cast: list[dict], refs: list[Path], ref_cast: list[dict | None], user_look: str = "",
         matte: Path | None = None, method: str = METHOD, at_key: bool = False) -> Plan:
    """The template filled for one shot by the fill rules. refs: the reference files in the order the user added them;
    ref_cast: the cast entry each one is (None for an uploaded picture); matte: the shot's subject matte, if the previz
    rendered one; method: B3, zbuf (no images: the depth pass only, so references are named in words), or "commercial-safe"
    (the A2 way on Qwen-Image-Edit 2511: frame 0 is image 1). at_key: a later keyframe's plan, so frame 0's size and
    facing sentences stay out (R2-1)."""
    # Only MAX_REFS go as images; every picked cast member still names a figure, with its line (R&D's s06: Jun).
    picked = list(ref_cast)
    refs = [] if method == ZBUF else list(refs)[:MAX_REFS]
    ref_cast = picked[: len(refs)]
    sentence = spec.get("notes", "")
    todo: list[str] = [TAKEN_OUT.format(clause=c.strip(" ,"), why="the previz places it")
                       for c in re.findall(PREVIZ_CLAUSE, user_look or "")]
    # The Look box as shown (auto or edited): its fixed look line goes (paragraph 1 says it) and its "grey figure"
    # sentence is said as what is there, so the time, the wardrobe and the place reach the still.
    user = strip_banned(prompt_words.look_words(user_look))
    subs = subjects_of(framing, cast, sentence, picked)
    if not subs:
        raise NoSubjects(NO_SUBJECTS)
    user, flags = facing_conflicts(user, subs)
    user, gone = strip_clauses(user, SPEED_WORDS)  # Run 2: no speed in a still
    todo += [TAKEN_OUT.format(clause=c, why="a still has no speed") for c in gone]
    if any(s.distance_m is not None for s in subs):  # Run 2 (c): the previz distance wins, said once
        user, gone = strip_clauses(user, USER_DISTANCE)
        todo += [TAKEN_OUT.format(clause=c, why="the previz sets the distance and size") for c in gone]
    # Run 11: where they stand, said beside them (paragraph 2), never left to the light paragraph for its "at dusk".
    whole_look = user
    stand_at = standing_place(user)
    if stand_at:
        user = " ".join(x for x in re.split(r"(?<=[.!?])\s+", user) if x and x.strip() != stand_at[1])
    first_ref = 2 if method == "commercial-safe" else 3  # B3: depth, canny, then the refs; A2 way: frame 0, then the refs
    for s in subs:
        for i, c in enumerate(ref_cast):
            if c and c.get("name", "").strip() == s.label:
                s.image = first_ref + i
    listed = ", ".join(f"{s.label} {s.place}".strip() for s in subs)
    p1 = FRAME_LINE.format(count=_count(subs), subjects=listed).replace(" in frame: .", " in frame.")

    view = framing.get("view") or {}
    shown = [w for w, v in sorted((view.get("share") or {}).items(), key=lambda kv: -kv[1]) if v >= 0.005 and w != "people"]
    people = [s for s in subs if s.kind == "person"]
    shared = shared_action(people, sentence)
    key, _ = location(whole_look, sentence, str(framing.get("place") or ""), shown)
    small = people[0] if len(people) == 1 and people[0].h and people[0].h < SMALL_HERO else None
    seat = "a bench" in shown or "bench" in ((spec.get("environment") or {}).get("props") or []) or _has(f"{sentence} {whole_look}", "bench")
    blocking = []
    for s in subs:
        if s is small:
            blocking.append(small_blocking(s, [whole_look, sentence, str(framing.get("place") or "")], seat))
            continue
        line = f"{_cap(s.label)}: {s.where}" + (f", {_metres(s.distance_m)} from the viewer" if s.distance_m is not None else "")
        if facing_words(s) and not at_key:
            line += f", {facing_words(s)}"
        if s.kind == "person" and shared:
            pass  # the group line below is their action
        elif s.doing:
            line += f", {s.doing}"
        elif s.kind == "person":
            line += f", {DEFAULT_ACTION}"
        blocking.append(line + ".")
        if size_words(s) and not at_key:
            blocking.append(size_words(s))
    if stand_at:
        blocking.insert(1, _cap(stand_at[1]) if re.search(r"[.!?]$", stand_at[1]) else _cap(stand_at[1]) + ".")
    if shared:
        blocking.append(shared)
    if len(people) > 1:  # Run 3 (s06): the count again, in positive words
        blocking.append(ONLY_PEOPLE_AT.get(key, ONLY_PEOPLE).format(n=COUNT_WORDS.get(len(people), str(len(people)))))
    # The layout line names the stand-ins without their colour (prompt_words: a positive never says grey). Method zbuf
    # has no layout image at all: the objects are paragraph 5's.
    layout = [] if method == ZBUF else [re.sub(r"(?i)\bgrey\s+", "", grey_line(subs, shown))]
    p2 = " ".join(blocking + layout + ([FALLBACK_LINE_CLEAN] if method == "commercial-safe" else []))

    cam = framing.get("camera")
    lens = (cam or {}).get("focal_mm") or (spec.get("lens") or {}).get("focal_mm")
    if cam:
        p3 = CAMERA_LINE.format(height=_height(float(cam["height_m"])), angle=_angle(float(cam["pitch_deg"])), lens=lens)
        p3 += f", {lens_words(float(lens))}." if lens else "."
    else:
        p3 = ""
        todo.append("camera height and angle: this shot was made before frame 0 recorded the camera (make the previz again)")
    movers = [s for s in subs if s.kind == "vehicle" and _moving(s, sentence)]
    if movers:
        line = VEHICLE_LINE_END_ON if movers[0].facing in ("away", "camera") else VEHICLE_LINE
        p3 = (p3[:-1] + ", " + line if p3 else _cap(line)) + "."

    lines = [small_person(s, seat) if s is small and not s.image else
             (f"{_cap(s.label)} is the one in image {s.image}: {s.line}." if s.line else REF_ONLY.format(label=_cap(s.label), image=s.image))
             if s.image else f"{_cap(s.label)}: {s.line}." for s in subs if s.line or s.image or s is small]
    if len(people) > 1 and any(not s.line and not s.image for s in people):
        lines.append(GROUP_NO_REF)
    if shared:  # v2.4 7: a mutual action names them as different people
        lines.append(DIFFERENT_PEOPLE.format(names=_cap(_names([s.label for s in people])), count=COUNT_WORDS.get(len(people), "")).replace("  ", " "))
    vehicles = [s for s in subs if s.kind == "vehicle"]
    lines += [VEHICLE_NO_REF.format(label=_cap(s.label), type=vehicle_type(s.word)) for s in vehicles if not s.line and not s.image]
    if len(vehicles) > 1:
        lines.append(_cap(MANY_VEHICLES) + ".")
    p4 = " ".join(lines)

    p5 = world_sentence(whole_look, sentence, str(framing.get("place") or ""), shown, standing_on=stand_at[0] if stand_at else None)
    props = props_of(ref_cast, subs, first_ref)
    if props:
        p5 = p5[:-1] + ", " + ", ".join(props) + "."
    p6 = light_sentence(whole_look, sentence, framing, interior=key in INTERIOR_KEYS)  # "at noon" beside them still lights it
    said = [x for x in re.split(r"(?<=[.!?])\s+", user) if len(x.split()) >= 3 and re.search(USER_LIGHT, x, re.I)]
    if said:  # the user's light replaces the default source sentence
        user = " ".join(x for x in re.split(r"(?<=[.!?])\s+", user) if x and x not in said)
        p6 = " ".join(_cap(x) if re.search(r"[.!?]$", x) else _cap(x) + "." for x in said) + " " + EXPOSURE
    if matte is None and method not in ("commercial-safe", ZBUF):
        todo.append("subject mask: this shot was made before the previz rendered its subject matte, so the canny keeps "
                    "the subjects (make the previz again)")
    paras = (p1, p2, p3, p4, user, p5, p6, LOOK_LINE, TEST_LINE)
    positive = prompt_words.clean_positive(no_camera("\n\n".join(x for x in paras if x)))
    # Run 2 (b): what the words said isn't there goes to the negative (the positive says what is there instead).
    negated = prompt_words.negated("\n".join([user, *(s.line or "" for s in subs)]))
    # Run 3: never the unwanted count's word for three or more (s06's "third person"); R2-4: a parked hero is no
    # "parked car".
    group_neg = GROUP_NEGATIVE + (", " + GROUP_NEGATIVE_PAIR if len(people) == 2 else "")
    vehicle_neg = VEHICLE_NEGATIVE if movers else VEHICLE_NEGATIVE.replace("parked car, ", "")
    negative = ", ".join([NEGATIVE, CAMERA_NEGATIVE] + ([group_neg] if len(people) > 1 else [])
                         + ([vehicle_neg] if any(s.kind == "vehicle" for s in subs) else [])
                         + ([REF_NEGATIVE] if refs else []) + negated)
    thin = (matte is None and method not in ("commercial-safe", ZBUF)) or not framing.get("camera") or not framing.get("subjects")
    return Plan(positive=positive, negative=negative, subjects=subs, refs=refs, todo=todo, matte=matte, method=method,
                action=REMAKE if thin else None, flags=flags)


# --- the canny mask: the subjects (from the matte) and the mannequins' joint circles -------------------------------------

def _dilate(m: np.ndarray, iterations: int, square: bool = False) -> np.ndarray:
    """Binary dilation, as scipy.ndimage.binary_dilation: the 4-neighbour cross by default, or the 3x3 square."""
    out = m.copy()
    for _ in range(iterations):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        if square:
            grown[1:, 1:] |= out[:-1, :-1]
            grown[1:, :-1] |= out[:-1, 1:]
            grown[:-1, 1:] |= out[1:, :-1]
            grown[:-1, :-1] |= out[1:, 1:]
        out = grown
    return out


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected labels (scipy.ndimage.label's default), by scanline runs and union-find."""
    h, w = mask.shape
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs: list[tuple[int, int, int, int]] = []  # row, start, end (exclusive), label
    prev: list[tuple[int, int, int]] = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        cur = []
        j = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            lab = 0
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                other = find(prev[k][2])
                if lab == 0:
                    lab = other
                elif other != lab:
                    parent[max(other, lab)] = min(other, lab)
                    lab = min(other, lab)
                k += 1
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            cur.append((s, e, lab))
            runs.append((y, s, e, lab))
        prev = cur
    labels = np.zeros((h, w), np.int32)
    remap: dict[int, int] = {}
    for y, s, e, lab in runs:
        root = find(lab)
        labels[y, s:e] = remap.setdefault(root, len(remap) + 1)
    return labels, len(remap)


def joint_mask(frame: Image.Image) -> np.ndarray:
    """The joint circles on the mannequins: dark blobs (15 to 40,000 px) whose 3 px ring is over a quarter mannequin
    yellow, grown by 7 px. R&D's thresholds (joint_mask_canny_box.py), so the canny is masked as in the test."""
    a = np.asarray(frame.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    dark = a.max(-1) < 60
    yellow = (r > 140) & (g > 140) & (b < 110) & (r - b > 80)
    labels, n = _components(dark)
    if not n:
        return np.zeros(dark.shape, bool)
    yd = _dilate(yellow, 4)
    flat = labels.ravel()
    area = np.bincount(flat, minlength=n + 1)
    # Each blob's pixels from one sort (a frame can hold thousands of dark blobs: shadows, windows).
    lit = np.flatnonzero(flat)
    lit = lit[np.argsort(flat[lit], kind="stable")]
    ends = np.cumsum(area[1:])
    keep = np.zeros(dark.shape, bool)
    w = dark.shape[1]
    for i in np.flatnonzero((area >= 15) & (area <= 40000)):
        if i == 0:
            continue
        pix = lit[ends[i - 1] - area[i]: ends[i - 1]]
        ys, xs = pix // w, pix % w
        y0, y1, x0, x1 = max(0, ys.min() - 4), ys.max() + 5, max(0, xs.min() - 4), xs.max() + 5
        c = labels[y0:y1, x0:x1] == i
        ring = _dilate(c, 3) & ~c
        if (ring & yd[y0:y1, x0:x1]).sum() / max(1, ring.sum()) > 0.25:
            keep[y0:y1, x0:x1] |= c
    return _dilate(keep, 7)


def subject_mask(matte: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Every person and vehicle in the previz scene, from the renderer's subject matte, grown SUBJECT_GROW px (R&D's
    3x3 square, 12 times). size: (width, height) of the canny it masks."""
    m = np.asarray(matte.convert("L").resize(size, Image.NEAREST)) > 127
    return _dilate(m, SUBJECT_GROW, square=True)


def canny_mask(frame: Image.Image, size: tuple[int, int], matte: Image.Image | None) -> np.ndarray:
    """What the canny loses: the subjects (when the shot has a matte) and the joint circles."""
    joints = joint_mask(frame.convert("RGB").resize(size))
    return joints | subject_mask(matte, size) if matte is not None else joints


def masked_canny(canny: Image.Image, frame: Image.Image, matte: Image.Image | None) -> Image.Image:
    """B3's image 2: the soft canny, black inside the subject mask and on the joint circles."""
    c = np.asarray(canny.convert("RGB")).copy()
    c[canny_mask(frame, (c.shape[1], c.shape[0]), matte)] = 0
    return Image.fromarray(c)


# --- running it -------------------------------------------------------------------------------------------------------

def graph_for(p: Plan, depth: object, canny: object, seed: int, refs: list | None = None, prefix: str = "pickups/start_frame") -> dict:
    """B3's generation graph (inputs are ComfyUI asset names or paths)."""
    from pickups import graphs
    return graphs.start_frame_fresh(depth, canny, p.positive, p.negative, seed=seed, refs=list(refs if refs is not None else p.refs),
                                    steps=STEPS, cfg=CFG, denoise=DENOISE, resolution=RESOLUTION, sampler=SAMPLER,
                                    scheduler=SCHEDULER, prefix=prefix)


def make(runner, p: Plan, frame0: Path, out: Path, seed: int = 42) -> Path:
    """Run the plan on ComfyUI (runner: .client and .run, a TaskContext or the quick runner): the depth and the soft canny
    of frame 0, the canny masked in Python, then the generation."""
    from pickups import graphs
    work = out.parent
    assets = runner.client.assets
    with Image.open(frame0) as f0:
        frame = f0.convert("RGB")
    job = runner.run(graphs.start_frame_maps(assets.from_file(frame0), da3_model=DA3_MODEL, da3_resolution=DA3_RESOLUTION, canny=CANNY_SOFT))
    depth, canny = work / "start_depth.png", work / "start_canny.png"
    job.get_outputs("PU_Depth")[0].to_file(depth)
    job.get_outputs("PU_Canny")[0].to_file(canny)
    matte = Image.open(p.matte) if p.matte and Path(p.matte).is_file() else None
    with Image.open(canny) as c:
        masked_canny(c, frame, matte).save(work / "start_canny_masked.png")
    g = graph_for(p, assets.from_file(depth), assets.from_file(work / "start_canny_masked.png"), seed,
                  refs=[assets.from_file(r) for r in p.refs])
    runner.run(g).get_outputs("PU_Plate")[0].to_file(out)
    return out


def painted_frame(frame: Image.Image) -> Image.Image:
    """The fallback's image 1: frame 0 with the joint circles painted over in the mannequins' own yellow (their median),
    so the edit isn't asked to keep black dots on a person."""
    a = np.asarray(frame.convert("RGB")).copy()
    m = joint_mask(frame)
    if m.any():
        i = a.astype(int)
        yellow = (i[..., 0] > 140) & (i[..., 1] > 140) & (i[..., 2] < 110) & (i[..., 0] - i[..., 2] > 80)
        if yellow.any():
            a[m] = np.median(a[yellow], axis=0).astype(np.uint8)
    return Image.fromarray(a)


def fallback_graph(p: Plan, frame0: object, seed: int, refs: list | None = None, prefix: str = "pickups/start_frame") -> dict:
    """The commercial-safe start frame (rule 5): Qwen-Image-Edit 2511 the A2 way, frame 0 as image 1 and the references
    as images 2 and 3, the template's positive and negative, CFG 3 and 30 steps, no Lightning LoRA."""
    from pickups import graphs
    refs = list(refs if refs is not None else p.refs)
    g = graphs.clean_plate(frame0, p.positive, seed=seed, fast=False, prefix=prefix, size=FALLBACK_SIZE,
                           image2=refs[0] if refs else None, image3=refs[1] if len(refs) > 1 else None,
                           steps=STEPS, cfg=CFG, negative=p.negative)
    # v2.3 (e): generated at FALLBACK_SIZE, then a Lanczos upscale to the start frame's size.
    g["upscale"] = {"class_type": "ImageScale", "inputs": {"image": g["PU_Plate"]["inputs"]["images"], "upscale_method": "lanczos",
                                                         "width": SIZE[0], "height": SIZE[1], "crop": "disabled"}}
    g["PU_Plate"]["inputs"]["images"] = ["upscale", 0]
    return g


def make_fallback(runner, p: Plan, frame0: Path, out: Path, seed: int = 42) -> Path:
    """Run the commercial-safe fallback: frame 0 with its joints painted over, then the 2511 edit."""
    work = out.parent
    assets = runner.client.assets
    with Image.open(frame0) as f0:
        painted_frame(f0.convert("RGB")).save(work / "start_frame0_painted.png")
    g = fallback_graph(p, assets.from_file(work / "start_frame0_painted.png"), seed, refs=[assets.from_file(r) for r in p.refs])
    runner.run(g).get_outputs("PU_Plate")[0].to_file(out)
    return out


# --- method zbuf: the previz's own depth -------------------------------------------------------------------------------

def decode_depth(png: Path | Image.Image) -> np.ndarray:
    """The page's depth pass (runtime.ts depthPasses: z / DEPTH_FAR in 24 bits, high byte in red) as metres; the sky (pure
    black) is inf."""
    im = png if isinstance(png, Image.Image) else Image.open(png)
    a = np.asarray(im.convert("RGB")).astype(np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    z = (r / 255.0 + g / 65025.0 + b / 16581375.0) * DEPTH_FAR
    return np.where((r == 0) & (g == 0) & (b == 0), np.inf, z).astype(np.float32)


def da3_tones(depth: Path | Image.Image) -> np.ndarray:
    """The tones of a Depth Anything 3 depth image (near bright), its sky (2 and under) left out, brightest first."""
    im = depth if isinstance(depth, Image.Image) else Image.open(depth)
    a = np.asarray(im.convert("L")).astype(np.float64)
    return np.sort(a[a > 2].ravel())[::-1]


def depth_control(z: np.ndarray, tones: np.ndarray | None = None, blur: float = ZBUF_BLUR) -> Image.Image:
    """The ControlNet's depth image: the z-buffer's exact order, rank-mapped onto the DA3 control's tones (nearest gets the
    brightest), sky 0, then a Gaussian blur of `blur` px (R&D's mkdepth.py). With no tones, an even ramp 255 to 32."""
    seen = np.isfinite(z)
    out = np.zeros(z.shape, np.float64)
    if seen.any():
        tones = tones if tones is not None and tones.size else np.linspace(255.0, 32.0, 256)
        zv = z[seen]
        q = np.empty(zv.size)
        q[np.argsort(zv, kind="stable")] = np.arange(zv.size) / max(1, zv.size - 1)
        out[seen] = tones[np.clip((q * (tones.size - 1)).round().astype(int), 0, tones.size - 1)]
    img = Image.fromarray(out.round().astype(np.uint8)).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


def hero_kind(framing: dict) -> str:
    """person or vehicle: the user-placed hero subject (the shot's first subject in view) decides the strength."""
    subs = [s for s in framing.get("subjects") or [] if isinstance(s, dict) and s.get("kind") in STRENGTH]
    hero = next((s for s in subs if s.get("in_view")), subs[0] if subs else None)
    return hero["kind"] if hero else "person"


def strengths(kind: str) -> tuple[float, float]:
    """(first try, the one retry) for the hero subject's kind."""
    first = STRENGTH.get(kind, STRENGTH["person"])
    return first, next(v for k, v in STRENGTH.items() if v != first)


def size_state(ratio: float | None) -> str:
    """pass, warn, fail (too big) or small (too small) for render / previz height; unmeasured when the match wasn't
    sure."""
    if ratio is None or not ratio > 0:
        return "unmeasured"
    if ratio > 1 + SIZE_FAIL:
        return "fail"
    if ratio < 1 / (1 + SIZE_FAIL):
        return "small"
    if ratio > 1 + SIZE_PASS or ratio < 1 - SIZE_PASS:
        return "warn"
    return "pass"


def needs_retry(state: str, ghost: bool = False) -> bool:
    """The one retry at the other strength: when the still is far too big or (R2-5) far too small, or a ghost was
    found."""
    return state in ("fail", "small") or bool(ghost)


def size_check_now(runner, shot_dir: Path, framing: dict, still: Path) -> dict:
    """R2-3: a still's size against the previz by the start-frame endpoint's own measure (size_from_depth, seeded from
    the subject matte), for a still the app didn't make: frame 0's DA3 (the one the start frame made, else made now) and
    the still's. Unmeasured when the shot has no box or ComfyUI can't run DA3."""
    from pickups import graphs
    box = framing.get("subject")
    frame0 = shot_dir / "previz_frame0.jpg"  # previz_maker.FRAME0
    if not box or not frame0.is_file() or not Path(still).is_file():
        return size_check_none()
    looks = shot_dir / "looks"
    looks.mkdir(exist_ok=True)

    def da3(src: Path, out: Path) -> Path:
        if not out.is_file():
            runner.run(graphs.start_frame_maps(runner.client.assets.from_file(src), da3_model=DA3_MODEL,
                                               da3_resolution=DA3_RESOLUTION, canny=CANNY_SOFT)).get_outputs("PU_Depth")[0].to_file(out)
        return out

    try:
        made = looks / "start_zbuf_control_da3.png"
        d0 = made if made.is_file() else da3(frame0, looks / "frame0_da3.png")
        d1 = da3(Path(still), looks / f"check_{Path(still).stem}_da3.png")
    except Exception:  # noqa: BLE001 - ComfyUI down or no DA3: the size is unmeasured, never an error
        return size_check_none()
    return size_from_depth(d0, d1, box, shot_dir / "previz_subjects0.png")


def size_from(got: dict) -> dict:
    """The size bands for an edge-match result (framing.compare: subject_h, subject_sure)."""
    h = got.get("subject_h")
    ratio = round(h[1] / h[0], 3) if got.get("subject_sure") and h and h[0] else None
    state = size_state(ratio)
    return {"ratio": ratio, "state": state, "warning": size_warning(state, ratio)}


# Run 4: a warning with no retry says the size itself ("about 1.5x"), not a percent off, which read as an error
SIZE_WORDS = {"warn": "The subject is about {x:.1f}x the previz size.", "fail": "The subject is {pct}% bigger than the previz.",
              "small": "The subject is {pct}% smaller than the previz."}


def size_warning(state: str, ratio: float | None) -> str | None:
    """The line the user sees for a size state, or None when it passed or wasn't measured."""
    words = SIZE_WORDS.get(state)
    return words.format(x=ratio or 1, pct=round(abs((ratio or 1) - 1) * 100)) if words else None
COUNT_WORDS_FLAG = "The still has {got} people; the previz has {want}."

# Run 2 (R&D #5): the edge match was never sure on a photo, so every still was "unmeasured" and the retry never fired.
# The hero is measured like for like instead, on Depth Anything 3's depth of frame 0 and of the still. Run 3 (R2-5): the
# seed is the median depth under the subject matte inside the hero's box (never the box's nearest ground), the region
# at that depth may grow only DEPTH_GROW box sizes past the box, and a region that runs the frame's width (the ground) is
# no measure. Checked on R&D's Run 2 and 3 stills against their rulers: s04 1.80 / 1.72 / 1.60 / 1.60 for 1.79 / 1.76 /
# 1.53 / 1.60, s06 1.03 for 1.01, s02 1.01 for about 1.06; c07, c08 and c09 come back unmeasured.
DEPTH_W = 384  # px the depth maps are measured at
DEPTH_TOL = 0.07  # the depth band, a fraction of the map's range
DEPTH_GROW = 1.0  # how far past the previz box the region may reach, in box sizes on each side
DEPTH_SEED_PCT = 75  # the seed depth: this percentile of the depth under the matte in the box
DEPTH_LEAK = 0.95  # a region taller or wider than this ran into the ground or the sky: not a measure


def _map(im: Path | Image.Image, size: tuple[int, int] | None = None, mode=Image.BILINEAR) -> np.ndarray:
    im = im if isinstance(im, Image.Image) else Image.open(im)
    w0, h0 = im.size
    size = size or (DEPTH_W, max(8, round(DEPTH_W * h0 / max(1, w0))))
    return np.asarray(im.convert("L").resize(size, mode)).astype(np.float32)


def depth_height(depth: Path | Image.Image, box: list | tuple, matte: Path | Image.Image | None = None) -> float | None:
    """The hero's height (a fraction of the frame) on a depth map (near bright): the connected region at the depth under
    the subject matte in its previz box (the box's middle third with no matte), kept within the grown box; None when that
    region isn't one thing (it spans the frame, or the box holds no depth)."""
    d = _map(depth)
    h, w = d.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    bx0, bx1 = int(np.clip(x0 * w, 0, w - 1)), int(np.clip(np.ceil(x1 * w), 1, w))
    by0, by1 = int(np.clip(y0 * h, 0, h - 1)), int(np.clip(np.ceil(y1 * h), 1, h))
    if (by1 - by0) * (bx1 - bx0) < 4 or float(d.max() - d.min()) < 8:
        return None
    inbox = np.zeros(d.shape, bool)
    inbox[by0:by1, bx0:bx1] = True
    seeds = (_map(matte, (w, h), Image.NEAREST) > 127) & inbox if matte is not None else np.zeros(d.shape, bool)
    if not seeds.any():
        tx = (bx1 - bx0) // 3
        ty = (by1 - by0) // 3
        seeds[by0 + ty:by1 - ty or by1, bx0 + tx:bx1 - tx or bx1] = True
    gx, gy = DEPTH_GROW * (x1 - x0), DEPTH_GROW * (y1 - y0)
    grown = np.zeros(d.shape, bool)
    grown[int(max(0, y0 - gy) * h):int(np.ceil(min(1, y1 + gy) * h)), int(max(0, x0 - gx) * w):int(np.ceil(min(1, x1 + gx) * w))] = True
    # The subject is the near part of what the matte covers: on a smaller still the matte also holds the far background.
    seed = float(np.percentile(d[seeds], DEPTH_SEED_PCT))
    band = (np.abs(d - seed) <= max(4.0, DEPTH_TOL * float(d.max() - d.min()))) & grown
    labels, _ = _components(band)
    hits = labels[seeds & band]
    if not hits.size:
        return None
    region = labels == np.bincount(hits).argmax()
    xs, ys = np.flatnonzero(region.any(0)), np.flatnonzero(region.any(1))
    wide = (xs[-1] - xs[0] + 1) / w > DEPTH_LEAK or (xs[0] == 0 and xs[-1] == w - 1)
    tall = (ys[-1] - ys[0] + 1) / h
    return None if wide or tall > DEPTH_LEAK else round(float(tall), 4)


def size_from_depth(depth0: Path | Image.Image, depth1: Path | Image.Image, box: list | tuple,
                    matte: Path | Image.Image | None = None) -> dict:
    """size_check from the two depth maps: still over frame 0, the same measure on both."""
    if matte is not None and not isinstance(matte, Image.Image):
        matte = Image.open(matte) if Path(matte).is_file() else None
    a, b = depth_height(depth0, box, matte), depth_height(depth1, box, matte)
    ratio = round(b / a, 3) if a and b else None
    state = size_state(ratio)
    return {"ratio": ratio, "state": state, "measure": "depth",
            "warning": size_warning(state, ratio)}


def zbuf_control(runner, depth_png: Path, frame: Path, out: Path) -> Path:
    """The control image for one frame: Depth Anything 3 of the previz frame for the tones (one ComfyUI job), the page's
    depth pass for the order (depth_control)."""
    from pickups import graphs
    job = runner.run(graphs.start_frame_maps(runner.client.assets.from_file(frame), da3_model=DA3_MODEL,
                                             da3_resolution=DA3_RESOLUTION, canny=CANNY_SOFT))
    da3 = out.with_name(out.stem + "_da3.png")
    job.get_outputs("PU_Depth")[0].to_file(da3)
    depth_control(decode_depth(depth_png), da3_tones(da3)).save(out)
    return out


def zbuf_patch(conn) -> str | None:
    """The 2602 ControlNet as this ComfyUI names it (top level or once in a subfolder of model_patches) when Z-Image Turbo
    is there too, else None (the caller falls back to B3)."""
    from pickups.app import manifest
    from pickups.images import _by_basename, _resolve
    try:
        have, _ = conn.model_scan()
    except Exception:  # noqa: BLE001 - offline: the fallback path reports it
        return None
    by_base = _by_basename(have)
    found = _resolve(ZBUF_PATCH, have, by_base)
    return found if found and all(_resolve(f, have, by_base) for f in manifest.START_DEPTH[1:]) else None


def zbuf_graph(p: Plan, control: object, seed: int, strength: float, patch: str = ZBUF_PATCH,
               prefix: str = "pickups/start_frame") -> dict:
    """Method zbuf's generation graph (control: a ComfyUI asset name or path)."""
    from pickups import graphs
    return graphs.start_frame_zbuf(control, p.positive, seed=seed, strength=strength, patch=patch, steps=ZBUF_STEPS,
                                   size=ZBUF_SIZE, prefix=prefix)


def make_zbuf(runner, p: Plan, depth_png: Path, frame0: Path, framing: dict, out: Path, seed: int = 42,
              patch: str = ZBUF_PATCH, ghost=None, matte: Path | None = None, count=None) -> dict:
    """Method zbuf with its one retry: DA3 of frame 0 for the tones, the control, the still at the hero's strength, the
    size check (seeded from the subject matte) and the people count, then one retry: the other strength when the size
    is far off (or `ghost(still)` says so), the next seed when `count(still)` differs from the previz's people. Try 1 is
    kept as <out>_try1.png before any retry (R2-10); the better of the two is the start still, and a second miss is
    flagged. Returns {"strength", "seed", "tries", "size": size_check, "people", "flag"}."""
    from pickups import graphs
    control = zbuf_control(runner, depth_png, frame0, out.parent / "start_zbuf_control.png")
    asset = runner.client.assets.from_file(control)
    da3_0 = control.with_name(control.stem + "_da3.png")  # frame 0's DA3, made for the control's tones
    box = framing.get("subject")
    want = sum(1 for s in framing.get("subjects") or [] if isinstance(s, dict) and s.get("kind") == "person" and s.get("in_view"))
    first, other = strengths(hero_kind(framing))
    tries: list[dict] = []
    strength, try_seed = first, seed
    for i in range(2):
        still = out if i == 0 else out.with_name(out.stem + "_retry.png")
        runner.run(zbuf_graph(p, asset, try_seed, strength, patch=patch)).get_outputs("PU_Plate")[0].to_file(still)
        check = size_check_none()
        if box and da3_0.is_file():  # Run 2: the still's DA3 against frame 0's, like for like
            da3 = still.with_name(still.stem + "_da3.png")
            runner.run(graphs.start_frame_maps(runner.client.assets.from_file(still), da3_model=DA3_MODEL,
                                               da3_resolution=DA3_RESOLUTION, canny=CANNY_SOFT)).get_outputs("PU_Depth")[0].to_file(da3)
            check = size_from_depth(da3_0, da3, box, matte)
        found = bool(ghost(still)) if ghost else False
        got = count(still) if count and want else None
        miscount = got is not None and got != want
        tries.append({"strength": strength, "seed": try_seed, "size": check, "ghost": found, "people": got,
                      "miscount": miscount, "file": still})
        resize = needs_retry(check["state"], found)
        if i or not (resize or miscount):
            break
        keep = out.with_name(out.stem + "_try1.png")  # R2-10: the retry never overwrites try 1
        keep.write_bytes(out.read_bytes())
        tries[0]["file"] = keep
        strength, try_seed = (other if resize else first), (seed + 1 if miscount else seed)

    def miss(t: dict) -> bool:
        return t["miscount"] or needs_retry(t["size"]["state"], t["ghost"])

    best = min(tries, key=lambda t: (t["miscount"], needs_retry(t["size"]["state"], t["ghost"]), abs(np.log(t["size"]["ratio"] or 1.0))))
    if best["file"] != out:
        out.write_bytes(best["file"].read_bytes())
    flag = None
    if all(miss(t) for t in tries):
        flag = "count" if best["miscount"] else "ghost" if best["ghost"] else "size"
    return {"strength": best["strength"], "seed": best["seed"], "tries": [{k: v for k, v in t.items() if k != "file"} for t in tries],
            "size": best["size"], "people": best["people"], "flag": flag,
            **({"warning": COUNT_WORDS_FLAG.format(got=best["people"], want=want)} if flag == "count" else {})}


def size_check_none() -> dict:
    return {"ratio": None, "state": "unmeasured", "warning": None}
