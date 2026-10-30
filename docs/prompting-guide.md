# Prompting guide

Each model in Pickups reads its prompt in its own way. LTX-2.5 wants a short paragraph that reads like a film
description. Qwen-Image wants an edit instruction. SCAIL-2 wants a description of the finished video. FLUX wants
a full photo caption. This page covers, for each model, what to write, where the prompt goes and what length works,
with a template to fill in and one good and one bad example.

Every rule here comes from the model makers' own guides, model cards and prompt-enhancer system prompts, or from
the official ComfyUI templates (see [Sources](#sources)). Rules marked *(our practice)* come from our own testing.

## Cheat sheet

| Task | Model | You write | Shape | Length |
|---|---|---|---|---|
| Generate or continue video | LTX-2.5 | What the viewer sees and hears, in order | One paragraph, present tense | 4–8 sentences |
| Render previz | LTX-2.5 Layout-To-Render | The finished shot | Plain sentences | 1–2 sentences |
| Look frame for previz | Qwen-Image 2.1 edit | An edit instruction | Starts with the verb | 2–4 sentences |
| Clean plate (Remove) | Qwen-Image 2.1 edit | What goes, and what fills the hole | Instruction | 2 sentences |
| Fill on moving shots (Remove) | LTX-2.5 In/Outpaint IC-LoRA | Only what fills the hole | A phrase | 5–20 words |
| Recast | SCAIL-2 | The finished video with the new person in it | Description, never an instruction | 90–140 words |
| New person for Recast | FLUX 3 klein | A full studio-photo caption | One prose block | 120–200 words |
| Weather, time of day, season | Qwen-Image 2.1, then LTX-2.5 Union Control | An instruction for frame 1, then a paragraph about the look | See [section 6](#6-change-weather-time-of-day-or-season) | |
| HDR, 4K upscale | LTX SDR-to-HDR, SeedVR2 | Nothing | | |

With **Setup → Licenses → Frame editor** switched to the alternate editor, the three Qwen-Image 2.1 rows above run on
it instead. Write the same instructions: Pickups renames `<image1>`, `<image2>` to the "Picture 1", "Picture 2" that
the alternate editor reads, and a previz look frame uses the previz frame plus the first two references (it reads
three images).

### Rules for every model

1. **Write what should be there, not what shouldn't.** Every Pickups job runs at CFG 1 (distilled or turbo models),
   so the negative prompt is never used. Putting "text, watermark" in the negative does nothing. Writing
   "plain grey backdrop" in the prompt itself works.
2. **Use plain, physical words.** "Low warm sun from the left, long shadows" works better than "epic, stunning,
   cinematic, 8k, masterpiece". One quality phrase is fine (LTX's own enhancer adds "film-grade colour, crisp
   detail"). A list of tags is not.
3. **Write prose.** Use no tag lists, bullet points, shot lists, headings or labels such as "Audio:" or "Style:".
4. **Name things by what they are and where they are in the frame**, for example "the man in the black jacket at
   the right end of the pier". Never refer to app state such as "the selected object" or "the reference image".
   The models can't see the app.
5. **For edits, keep the "stays the same" part short.** Use one closing clause such as "keep everything else
   exactly as it is". Qwen regenerates whatever you describe in detail, so a long description of the parts to keep
   makes them drift.

---

## 1. LTX-2.5 video prompt

**Where it goes:** `PU_Prompt` (`CLIPTextEncode`, Gemma 4 12B encoder) in every LTX graph: `drive`, `retake`,
`retake_two_stage`, `smoke_t2v`.

LTX-2.5's own prompt enhancer (ComfyUI node `TextGenerateLTX2Prompt`) rewrites prompts into the style of the
model's training captions. Its system prompt is the most precise spec available:

- **Start with the action or the picture.** Never open with "The scene opens…", "We see…" or "There is…".
- **Name exactly one shot type:** extreme wide, wide, medium, medium close-up, close-up or extreme close-up.
- **Always state the camera,** in prose: what it does, relative to whom, and where it ends up ("pushes in until
  she fills a medium close-up, then holds"). If it doesn't move, write "the camera remains static".
- **Give a viewpoint:** front-facing, side view, over-the-shoulder, low angle, high angle or top-down.
- **Describe only what can be seen.** Write "his brow furrows and his jaw tightens", not "he is angry". For people,
  give age range, hair, build and clothing.
- **Keep events in order,** joined with "then", "as" or "a moment later". Give every sentence an active verb.
- **Place each sound where it happens:** specific sounds ("tyres hiss on the wet road"), and dialogue in quotes
  with the voice ("she says quietly, 'We're late.'"). Only add sound in graphs that generate audio.
- **Use plain colour and light words,** with one consistent lighting setup per shot. Lightricks' guide asks for
  4–8 sentences for a single shot; the enhancer writes 150–220 words.

**Image-to-video.** Every Pickups LTX job starts from a real frame. Open with "Use the provided start image as the
first frame." (the official ComfyUI LTX-2.5 template's wording). Follow with one sentence that matches the frame
(shot, subject, place), then use the rest of the prompt for what happens next. Keep it to one continuous take: a
prompt that contradicts the frame makes LTX cut.

**Avoid** chaotic physics (pile-ups, splashing everywhere), long on-screen text, and cuts you didn't ask for. LTX-2.5
times the clip to the action you describe, so if you want a pause, write the pause.

**Negative prompt.** It is ignored at CFG 1. `DEFAULT_NEGATIVE` only applies when `dev_first_pass=True` (the dev
model at CFG 3). The official image-to-video template ends its main prompt with "No text, no black frames.", which
is the right place for that kind of guard.

```
Use the provided start image as the first frame. {shot_type} from a {viewpoint} of {subject_with_look} {where}.
{first_action}, then {second_action}. The camera {camera_motion_and_end_state}. {light_and_colour},
{texture_or_atmosphere}. {sound_with_quoted_dialogue} No text, no black frames.
```

**Good**
> Use the provided start image as the first frame. A wide shot from a low front three-quarter angle shows a silver
> sports sedan on a wet coastal road at dusk. The car accelerates toward the camera with its headlights on, then
> sweeps past frame-left as the camera pans to follow it into the bend and holds on its red taillights. Blue-hour
> sky over the sea, warm reflections on the wet asphalt, fine spray lifting from the tyres, crisp detail and
> film-grade colour. The motor's whine rises and fades, tyres hiss on the wet road, waves break below the cliffs.
> No text, no black frames.

**Bad**
> cinematic car commercial, epic, 8k, masterpiece, dramatic lighting, fast, cool camera move

It is a list of tags with no shot type, no order of events, no end point for the camera and no sound, so LTX
invents the audio.

**Prompt enhancer.** The Gemma 4 E2B enhancer (installed) does this rewrite for you. It adds 1–2 minutes and helps
most with short, rough prompts. Check any dialogue it writes.

## 2. Render previz: Layout-To-Render IC-LoRA

**Where it goes:** `PU_Prompt` in `graphs.layout_render`, which is the Previz task's `describe` field.

The previz sets the camera and blocking, and the look frame sets the palette, light and materials. The prompt
only names the finished shot. The LoRA was trained on short, simple prompts.

- Write **one or two sentences** describing the finished shot, matching the look frame.
- **Never write "clay", "3D" or "Unreal"** (model card). For the same reason, also leave out "previz",
  "grey", "Blender", "render", "CG" and "low-poly" *(our practice)*.
- A short camera phrase is fine. The model card's own example ends "...as the camera slowly pulls back from the
  basin."
- Don't contradict the look frame. If the frame is at dusk, don't write noon.

```
{subjects_with_key_look} {action_or_state} {place}, {light_or_time_of_day}{, as the camera {camera_move}}.
```

**Good:** "A red sports car speeds along a coastal road between sunlit cliffs and a deep blue sea as the camera car
leads it through the bend."
**Good** (Lightricks' example): "Two men in dark coats stand with their backs to the camera in a night apartment."
**Bad:** "Unreal Engine 3D render of a clay car, cinematic, photoreal, 8k, ultra detailed."

Settings: LoRA strength 1.0, CFG 1, 1920×1088, 8k+1 frames. If you feed the grey frame in place of a look frame,
the result comes out grey.

## 3. Qwen-Image 2.1 edit instructions

**Where it goes:** `PU_Prompt.prompt` (`TextEncodeQwenImage21`) in `graphs.qwen21_edit`. Pickups uses it for look
frames, clean plates and the first frame of environment changes.

Comfy's Qwen-Image 2.1 edit template includes the official prompt enhancer (Qwen 3.5 9B,
`qwen3.5_9b_qwen_image_2.1_pe_i2i`, installed) and its system prompt. Its rules, condensed:

- **Lead with the operation:** "Remove…", "Change…", "Convert…", "Replace…". Write an instruction, not a
  description of the finished picture.
- **Change exactly what you named, fully, and nothing else.** There are two ways to fail: *leakage* (changing
  something you didn't name) and *under-editing* (a result that could be mistaken for the input). Make the change
  unmistakable.
- **Say what stays, without describing it.** Use one closing clause that names the kept parts by type and position
  ("every other person and object, the framing, the light"). Don't describe how they look, because Qwen redraws
  anything you describe in detail.
- **Identity is the hardest thing to keep.** That covers faces, personal accessories, a product's exact design and
  markings, and the image's medium (a photo stays a photo). When identity comes from a reference image, point at
  that image; don't describe the face in words.
- **Ask only for what you want.** No "and tidy up the background".
- **When something is removed or revealed,** say what fills the space so the result still makes physical sense.
- **With two or more images, refer to them as `<image1>`, `<image2>`.** This is mandatory. Never write "image 1" or
  "the first image". State each image's role: which one is the canvas, and what to take from the other. With a
  single image, use no tags and just say "the image".
- **Write one paragraph,** with no line breaks, no resolution and no aspect ratio. Phrase requirements positively
  ("keep the background as it is", not "don't change the background"). Put any text that should appear in the
  image in double quotes, exactly as it should read.

Comfy's docs add a shortcut for local edits: paint over the area in a solid colour on image 1, then name it ("change
the jacket in the red area").

Settings: CFG 1 (the negative prompt is ignored) and 25 steps (the official pipeline uses 40–50, and whole-frame
edits need at least 25). `resolution` 0 keeps the frame's own size. If you run the enhancer, turn `thinking` on: it
was trained with it.

### 3a. Look frame for previz

```
Convert <image1>, a flat grey 3D viewport frame, into a photographic film still in the look of <image2>.
{describe_the_finished_frame}. Take the characters' identity and wardrobe, the palette, the light and the grade
from <image2>; take the composition from <image1>. Keep <image1>'s exact framing, camera angle and horizon, and the
position, size and pose of every person and object, with real material texture on every surface.
```

Without a look reference, remove `<image2>` and all the tags: "Convert the image, a flat grey 3D viewport frame,
into a photographic film still: {describe}. Keep the exact framing…". For the middle and last key frames, use the
first look frame as `<image2>`: "…in exactly the look of `<image2>`: the same place, the same subjects with the same
colours and details, the same light and grade."

"3D" is fine here, because the no-"3D" rule only applies to the Layout-To-Render prompt. Naming the grey look is
what tells Qwen to change it; otherwise it keeps the image's medium.

**Good**
> Convert \<image1\>, a flat grey 3D viewport frame, into a photographic film still in the look of \<image2\>. A red
> sports car drives along a coastal road in late-afternoon sun, sandstone cliffs on the right, a deep blue sea on the
> left, the asphalt dark and slightly worn. Take the palette, the light and the grade from \<image2\>; take the
> composition from \<image1\>. Keep \<image1\>'s exact framing, camera angle and horizon, and the position, size and
> pose of every object, with real material texture on every surface.

**Bad:** "Make image 1 look like image 2, photoreal, cinematic." Without tags Qwen can't tell the two images apart,
and the instruction describes nothing and keeps nothing.

### 3b. Clean plate (Remove)

```
Remove the {object} {where_in_frame}{ and its {shadow|reflection} on {surface}}, and fill the area it covered with
{what_continues_behind_it}, matching the surrounding perspective, texture and light. Keep everything else in the
image exactly as it is: the framing, lighting, colours, textures and every other person and object.
```

Painted-mask version (paint the mask in pure red on image 1): "Remove everything inside the red area and fill it
with {what_continues_behind_it}, matching the surrounding perspective, texture and light. Keep everything outside
the red area exactly as it is."

**Good**
> Remove the man in the black jacket standing at the right end of the pier and his shadow on the planks, and fill
> that area with the continuing weathered wooden planks, railing and calm grey water, matching the surrounding
> perspective, texture and light. Keep everything else in the image exactly as it is: the framing, lighting,
> colours, textures and every other person and object.

**Bad:** "Remove the selected object." Qwen never sees the app's selection, so it has to guess what to remove.

### 3c. Fill on moving shots (LTX In/Outpaint IC-LoRA)

This step uses LTX, not Qwen, but it serves the same task. According to Lightricks' model card, an empty or very
short prompt is enough. To steer the fill, **describe only the masked area**: describing the whole scene makes the
model duplicate people and objects. Grow the mask over shadows and reflections too, or the object comes back.

```
{what_the_revealed_area_shows}, {surface_and_texture}, {light}.
```

**Good:** "Weathered wooden pier planks and calm grey water in soft overcast light."
**Bad:** "A pier with people fishing and a man in a black jacket, the same place with nothing there." It names
people, so they get painted back in.

## 4. Recast: SCAIL-2

**Where it goes:** `PU_Prompt` in `graphs.scail2`, which is Recast's `describe` field. The SAM 3.1 labels (`person`
and `ref_person`, both "human" by default) are separate: they only choose who gets masked. In a crowd, make them
specific ("woman in red coat").

SCAIL-2's authors publish a replacement prompt enhancer (`prompt_enhancer.py`). Its rules are the spec:

- **Describe the finished video after the swap.** Don't write an instruction: no "replace", "swap", "edit" or
  "the person from the reference image".
- **Take the new person's look from the reference image,** and describe their visible clothing and appearance in
  detail.
- **Take everything else from the original person except identity and clothes:** their movement, pose, timing,
  position in the frame and interactions.
- **Name the objects the person touches or stays near** (steering wheel, door, cup, railing, bicycle). These help
  the model place the new person.
- **Keep the source shot's** environment, lighting, camera angle, shot scale and background.
- Use concrete verbs, and never mention masks, segmentation or prompts.
- Write **one English paragraph of about 90–140 words**. SCAIL-2 was trained on long, detailed prompts; short or
  empty prompts still run, but the results look worse.
- From Comfy's template: if swaps come out distorted, add proportion facts ("a normal-sized head, no hat").

```
{age_range} {gender} with {hair} and {build_or_features}, wearing {top}, {bottom}, {footwear}{, accessories}, is
{main_action} {where}. Positioned {frame_position} and {orientation_to_camera}, {pronoun} {action_beat_1}, then
{action_beat_2}, {interaction_with_objects}. {proportion_locks} {background_elements}, {lighting}, seen in a
{shot_scale} from {camera_angle}{, as the camera {camera_motion}}.
```

**Good**
> A woman in her early thirties with shoulder-length auburn hair and a slim build, wearing a mustard-yellow raincoat
> over a grey knit sweater, dark blue jeans and white canvas sneakers, is walking across a cobblestone square in an
> old European town. Positioned right of center and facing frame-left, she walks at an easy pace, glances up at the
> clock tower, then adjusts the strap of a brown leather shoulder bag with her right hand. She has a normal-sized
> head and no hat. Pastel stone facades, café tables with white umbrellas and a few distant pedestrians fill the
> background under soft overcast daylight, seen in a medium-wide shot from eye level as the camera tracks slowly
> alongside her.

**Bad:** "Replace the man with the woman in the reference image." It is an instruction and describes nothing.

Settings: CFG 1 with the Lightx2v LoRA (the negative prompt is unused), DPO LoRA at 1.0, and up to 81 frames per
segment. Width and height must be multiples of 32; replacement works best around 704p.

## 5. New person for Recast: FLUX 3 klein

**Where it goes:** `PU_Prompt` in `graphs.still`, which is the Recast "describe someone" path, at 832×1216.

Black Forest Labs says klein expects full scene captions, that short prompts give generic output, and to "not use
the model without a prompt upsampler". Pickups runs the distilled klein (15 steps as in Comfy's template, no guidance) without the
upsampler, so the prompt has to read like the upsampler's output already. Alternatively, run the upsampler first
(`qwen3.5_9b_flux3_prompt_upsampler`, installed). The upsampler follows these rules:

- **Open with the medium,** for example "Studio photograph, …". Use "photograph" for photos and keep
  "photorealistic" for renders.
- **Order the caption** as setting, then light and camera, then the subject with concrete colours, materials and
  condition.
- **For every person in focus,** give ethnicity, gender, age range, skin tone, hair and clothing.
- **Place things in the frame** (centered, frame-left, foreground), so the layout could be rebuilt from the words.
- **Be definite.** No hedging and no judgments of quality or mood. Describe what is present: "plain grey
  backdrop", not "no background".
- **Add a few ordinary, imperfect details** (a crease, scuffed toes) instead of a stock-photo ideal.
- **Use no real names, celebrities or brands.**

A good SCAIL-2 reference also needs a few things *(our practice)*:

- The whole body, from the top of the head to the shoes, is in frame, facing the camera.
- The arms are slightly away from the body, so SAM can separate them.
- The hands are empty, unless the person in the source holds the same thing.
- The backdrop is plain and contrasts with the clothes.
- The light comes from the same side as the source shot's light.

```
Studio photograph, full-length shot of a {age_range} {ethnicity} {gender} with {skin_tone} skin and {hair},
{build}, standing centered and facing the camera in a relaxed neutral stance, arms slightly away from the body,
feet shoulder-width apart, the whole figure from the top of the head to the shoes inside the frame. {Pronoun} wears
{top_with_colour_and_material}, {bottom}, {footwear}{, accessories}. {ordinary_details}. Soft, even light from the
front and slightly {light_side}, a gentle shadow under the chin. Plain light-grey seamless backdrop and floor, a
faint contact shadow under the feet.
```

**Good**
> Studio photograph, full-length shot of a white woman in her early thirties with fair skin and shoulder-length
> auburn hair, slim build, standing centered and facing the camera in a relaxed neutral stance, arms slightly away
> from the body, feet shoulder-width apart, the whole figure from the top of the head to the shoes inside the frame.
> She wears a mustard-yellow cotton raincoat, open over a grey knit sweater, dark blue straight-leg jeans and white
> canvas sneakers, with a brown leather shoulder bag on her left side. The raincoat has a soft crease across the
> belt line and the sneakers are slightly scuffed at the toes. Soft, even light from the front and slightly left, a
> gentle shadow under the chin. Plain light-grey seamless backdrop and floor, a faint contact shadow under the feet.

**Bad:** "Full-body photo of a woman in a yellow raincoat, standing, plain light background, natural light,
photoreal." It is too short, and the person has no profile, no position in the frame and no material details.

## 6. Change weather, time of day or season

The recipe has two steps. First, Qwen-Image 2.1 restyles frame 1. Then LTX-2.5 with the Union Control IC-LoRA
(`graphs.drive`) starts on the restyled frame and follows the structure of the real footage. Each step takes its
own prompt. **Use the same look words in both.**

**Frame 1 (Qwen):**
```
Change the {what_changes} in the image to {look_name}: {qwen_look_clause}. Keep the exact framing, camera angle
and geometry, every person's face, pose and clothing, and every vehicle's exact design, colour and markings; only
the {what_changes} change.
```

`{what_changes}` is "weather and light" for snow, rain and fog, "time of day and light" for night and golden hour,
and "season and light" for autumn.

**Video (LTX).** Lightricks' IC-LoRA guide says the motion comes from the reference video, so the prompt should
describe the look. Camera moves or choreography ("camera pans left", "character walks forward") conflict with the
control video and make the result worse. Weather that moves is the exception: the reference has none, so describe
it *(our practice)*.
```
Use the provided start image as the first frame. {ltx_look_clause} {subjects_as_they_appear}. {ltx_motion_clause}
```

| Look | Qwen look clause | LTX look clause + motion | Control *(our practice)* |
|---|---|---|---|
| **Snow** | fresh snow covers the ground, roofs, ledges, branches and the tops of vehicles, the sky is overcast white-grey, the light is soft and cool with a pale blue tint, and light snowflakes hang in the air | A snowy winter day under a flat white-grey sky, fresh snow on the ground, rooftops and car roofs, soft cool shadowless light. / Snowflakes fall steadily and drift sideways in a light wind. | depth |
| **Rain** | the ground and road are wet and dark with mirror-like reflections and shallow puddles, raindrops bead on car paint and glass, the sky is heavy grey, and the light is soft and cool | Steady rain under a heavy grey sky, wet reflective asphalt, soft cool diffused light. / Fine rain streaks fall across the frame, rings spread across the puddles, and droplets run down the car windows. | canny |
| **Fog** | thick white fog fills the scene so contrast fades with distance, the background dissolves into soft grey silhouettes, the foreground stays clear, and the light is diffuse and directionless | Dense fog, the background fading into soft grey silhouettes, muted low-contrast colour. / Slow wisps of fog drift through the frame, and lamps and headlights glow with soft halos. | depth |
| **Night** | the sky is dark navy to black, street lamps cast warm orange pools of light, shop windows are lit, the vehicles' headlights and taillights are on, and deep blue shadows fill the gaps between the lights | A realistic night scene lit by street lamps, lit windows and car lights, warm pools of light between deep blue shadows. / Headlight beams sweep across the road and highlights glint on paint and glass. | canny |
| **Golden hour** | a low sun from {side} casts warm orange-gold light and long soft shadows, people and vehicles get a warm rim light on the sun side, and the sky grades from pale gold at the horizon to soft blue above | Golden hour, low warm sunlight from {side} raking across the scene, long soft shadows, a gold rim light on people and bodywork, rich film-grade colour. / Dust glints in the backlit air. | canny |
| **Autumn** | the trees' leaves turn orange, red and yellow, fallen leaves scatter across the ground and verges, the grass is duller, and the light is soft, warm and low | Mid-autumn, orange, red and yellow foliage, fallen leaves on the ground, soft warm low sunlight. / Leaves drift down from the trees and skitter across the road in the breeze. | depth |

In the table, "/" separates the LTX look clause from the motion clause. *(Our practice)* Use depth when surfaces
change (snow, fog, autumn): canny edges would pull back the old road markings and leaves. Use canny when the
geometry stays the same.

**Good pair (snow, European square):**
> **Qwen:** Change the weather in the image to a snowy winter day: fresh snow covers the cobblestones, rooftops,
> window ledges, café tables and the tops of the parked cars, the sky is overcast white-grey, the light is soft and
> cool with a pale blue tint, and light snowflakes hang in the air. Keep the exact framing, camera angle and
> geometry, every person's face, pose and clothing, and every vehicle's exact design, colour and markings; only the
> weather and light change.
>
> **LTX:** Use the provided start image as the first frame. A snowy winter day in an old town square under a flat
> white-grey sky, fresh snow on the cobblestones, rooftops and café tables, soft cool shadowless light. A few
> pedestrians in dark coats and two parked hatchbacks dusted with snow. Snowflakes fall steadily and drift sideways
> in a light wind.

**Bad:** Qwen "make it winter", then LTX "the camera pans across a snowy square as people walk". The Qwen prompt
says nothing about where the snow is or what to keep, and the LTX prompt describes motion that conflicts with the
footage.

If you want people dressed for the new weather, ask for it in the Qwen instruction ("dress every person in a dark
winter coat"). Otherwise Qwen keeps their clothes, as designed.

## 7. No prompt needed

| Model | Prompt |
|---|---|
| **LTX SDR-to-HDR** | None. It was trained on a single fixed caption, and Pickups loads that caption's scene embedding (`ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors`). |
| **SeedVR2** | None. The ComfyUI nodes have no text input; it restores and upscales what is already in the footage. |
| **Refine Details IC-LoRA** | Rendering words only, for example "sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain". Never name subjects: every tile receives the whole prompt, so a named subject gets painted into tiles that don't contain it. |

---

## Prompt Director (future)

The Prompt Director would be a small local LLM that turns a plain request ("make it snow", "swap him for a woman in
a yellow raincoat") into the model-specific prompt described on this page, and shows the result for editing before
the job runs. Most of the pieces are already on the GPU machine:

- **LTX-2.5:** the official Gemma 4 E2B enhancer, run by `TextGenerateLTX2Prompt`, with separate text-to-video and
  image-to-video system prompts.
- **Qwen-Image 2.1:** the official Qwen 3.5 9B enhancer models for edits and generation (`…_pe_i2i`, `…_pe_t2i`).
  Their system prompts are in Comfy's templates.
- **FLUX 3 klein:** the official Qwen 3.5 9B upsampler, with its system prompt in Comfy's template.
- **SCAIL-2:** an official two-step recipe: caption 8 frames of the source shot, then write the replacement
  prompt from that caption plus the reference image. It uses Gemini; a local Qwen3-VL 8B (installed) could run the
  same prompts.
- **Layout-To-Render and the In/Outpaint fill:** no LLM needed, just a one-sentence rewrite rule.

A Prompt Director node would pick the system prompt for the task, attach the frame or frames, and return one
paragraph. The unfinished ComfyPromptNode prototype on the GPU PC is the natural place to build it. It is out of
scope for this build.

## Where the code doesn't match this guide yet

| Where | Now | Should be |
|---|---|---|
| `tasks/previz.py` `LOOK_INSTRUCTION`, `KEY_INSTRUCTION` | "image 1", "image 2" | `<image1>`, `<image2>`; also state each image's role ([3a](#3a-look-frame-for-previz)) |
| `tasks/remove.py` `plate_instruction` | "Remove the selected object…" | Name the object and where it is, or paint the mask red and say "the red area"; say what fills the hole ([3b](#3b-clean-plate-remove)) |
| `graphs.remove` default and `tasks/remove.py` prompt | "The same place with nothing there…" | Describe only what fills the hole ([3c](#3c-fill-on-moving-shots-ltx-inoutpaint-ic-lora)) |
| `tasks/recast.py` `graphs.still(...)` | "Full-body photo of …, photoreal." | The FLUX caption template, or run the FLUX upsampler first ([5](#5-new-person-for-recast-flux-3-klein)) |
| `tasks/recast.py` default SCAIL prompt | "The person from the reference image performs…" | A description of the finished video ([4](#4-recast-scail-2)) |
| `graphs.DEFAULT_NEGATIVE` | Passed to every LTX job | Ignored at CFG 1 (only `dev_first_pass` uses it). Put guards in the main prompt instead. |

## Sources

- LTX-2.5 prompting guide: <https://docs.ltx.io/open-source-model/usage-guides/prompting-guide>
- LTX-2.5 prompt-enhancer system prompts (`LTX24_T2V_SYSTEM_PROMPT`, `LTX24_I2V_SYSTEM_PROMPT`): ComfyUI
  `comfy_extras/nodes_textgen.py`
- ComfyUI LTX-2.5 templates `video_ltx2_5_i2v`, `video_ltx2_5_t2v`, and the docs at
  <https://docs.comfy.org/tutorials/video/ltx/ltx-2-5>
- IC-LoRA prompting (style, not motion): <https://ltx.io/model/model-blog/how-to-use-ic-lora-in-ltx-2>
- Layout-To-Render: <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render>
- In/Outpainting: <https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting>, Clean Plate:
  <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate>
- SDR-To-HDR:
  <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR>, Refine Details:
  <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details>
- Qwen-Image 2.1 edit enhancer system prompt: ComfyUI template `image_qwen_image_2_1_image_edit` (node 479);
  docs <https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1>; model card <https://huggingface.co/Qwen/Qwen-Image-2.1>
- SCAIL-2: <https://github.com/zai-org/SCAIL-2> (README "Prompt Semantics", `prompt_enhancer.py`,
  `examples/prompt_examples.txt`); ComfyUI template `video_wan21_scail2_character_replacement`;
  <https://docs.comfy.org/tutorials/video/zai/scail2>
- FLUX 3 klein: <https://huggingface.co/black-forest-labs/flux-3-image-klein>; upsampler system prompt in the ComfyUI
  template `image_flux3_klein_t2i`
- SeedVR2: ComfyUI template `utility_seedvr2_3b_int8_upscale_video` (no text-encoder node)
