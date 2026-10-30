# Pickups

**Power your creativity, locally.**

Pickups puts powerful open video models behind one simple app for creators and editors. Drop in a clip and remove what
you don't want, replace a person, change the weather or the time of day, turn a Blender previz into a finished shot, or
finish it to 4K HDR. It all runs on your own GPU through ComfyUI, and every result comes back as a new take you can
compare, approve and deliver as ProRes or MP4.

| | What it does | How |
|---|---|---|
| **Remove** | Erase a person, boom, sign or crew member (and their shadow, and the bag they carry) | Click it. SAM 3.1 follows it (each person of a group on their own) through the shot. On locked-off or slowly drifting shots (up to 10 s) the background is rebuilt from *real pixels in other frames*, aligned frame by frame; AI (Qwen-Image 2.1) only paints what was never seen. Moving shots use LTX-2.5 in/outpainting. |
| **Replace a person** | Same performance, new person | Pickups finds who's in the shot by itself (anywhere in the first 12 s; or click someone), then pick who's in: a cast member, a photo or character sheet, or a description (FLUX 3 klein makes a reference still). A person is painted onto the first frame of a crop that follows them, on a clean plate, guided by their pose. Only the subject is replaced. SCAIL-2 (More options) follows the exact performance instead. |
| **Change the weather** | Snow, rain or golden hour on a real shot | Qwen-Image 2.1 restyles the first frame; LTX-2.5 with the Union Control IC-LoRA re-renders the shot from it, steered by the footage's own depth (Depth Anything 3) or edges, so your camera move and every performance stay yours. |
| **Relight** | New light on the whole shot: sunset backlight, golden side light, cool low sun, midday or afternoon, or your own words | LTX-2.3 with Lightricks' Relight IC-LoRA re-renders the shot at 1280×704 (up to 5 s), lit to match a light-direction ball Pickups places in the corner and the model paints out. Needs a 24 GB card with nothing else loaded (it peaks around 25.5 GB). It changes light, not weather. |
| **Previz to final** | Turn a Blender/Unreal previz into a finished shot | LTX-2.5 Layout-To-Render follows your camera and blocking; the look comes from a look frame painted over the previz by Qwen-Image 2.1, with your character sheets and location plate as numbered references. 4 s per pass on a 24 GB card, 7 s in one pass on an 80 GB+ card (RTX PRO 6000). |
| **Finish** | 4K, HDR or both, for delivery | SeedVR2 3B restores and upscales to UHD; LTX-2.5's SDR-to-HDR reconstructs the highlights, merged so every pixel below the highlights stays your footage. BT.2020 HLG master + ACEScg EXR. |

Nothing outside your selection changes: every result reports the number of pixels changed outside the matte (it is 0),
and every take shows **what changed**: which pixels are your footage, which were borrowed from other frames, and which
were generated.

## Install

Pickups runs on any computer (Mac, Windows or Linux) and renders with a ComfyUI anywhere: on the same computer, on a
PC on your network or Tailscale, or on a rented GPU. Where Pickups runs you need [uv](https://docs.astral.sh/uv/) (it
brings Python 3.12+) and ffmpeg (`brew install ffmpeg` on a Mac, `sudo apt install ffmpeg` on Ubuntu,
`winget install Gyan.FFmpeg` on Windows). Where it renders you need ComfyUI 0.41+ with [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) on an NVIDIA GPU (tested: RTX 4090 24 GB, RTX PRO
6000 96 GB).

```bash
git clone <repo URL> pickups && cd pickups
```

```bash
uv run pickups
```

Pickups opens at http://127.0.0.1:7788. If that address is already taken, start with `uv run pickups --port 7790` and
open http://127.0.0.1:7790 instead. The first screen asks where ComfyUI runs:

- **This computer:** Pickups finds it on port 8188 (or ComfyUI Desktop's 8000).
- **Another computer:** type its address or Tailscale name, like `http://gpu-pc:8188`. Pickups tests it before switching
  and remembers your last five. ComfyUI there must listen on the network, so start it with `--listen`:
  - Windows portable, in its folder: `.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --listen 0.0.0.0 --port 8188`
  - Mac or Linux, in the ComfyUI folder: `python main.py --listen 0.0.0.0 --port 8188`
  - ComfyUI Desktop: Settings → Server-Config → Host `0.0.0.0`, then restart.
- **RunPod:** paste the pod's `https://<pod-id>-8188.proxy.runpod.net` address (see below).

**Examples.** Home's example gallery (before/after loops, and the clips "Try it" opens) comes in the examples pack,
`pickups_examples_v1.zip`. Put it next to the `pickups` folder (or start Pickups from the folder that has it) and it
installs on first run; or import it later in Setup → Examples (the zip, or the folder it unzips to). Without it, Home
shows the built-in character and image examples only.

ComfyUI needs no CORS settings: your browser only talks to Pickups, and Pickups talks to ComfyUI (ComfyUI's own page
opens through it at http://127.0.0.1:7788/comfy/). Then Pickups reads the GPU and shows what each feature can do on it,
with the clip length and render time we measured on that GPU, or "not tested on this GPU". Switch servers any time
from Setup → Server.

**Models.** Every feature is optional. Setup → Models shows what each one needs. When ComfyUI is on the same computer,
Get downloads them into its models folder with progress. When it's on another computer, run the command Setup shows
there, from a Pickups folder (it fills in that ComfyUI's models folder), for example:

```bash
uv run pickups models --task recast --dir "C:\ComfyUI_windows_portable\ComfyUI\models"
```

Some LTX and FLUX files are gated on Hugging Face: open the model page Setup links to and accept the license, then
download the file in your browser into the folder Setup shows (or run `hf auth login` and the `hf download` lines Setup
gives you, on the GPU computer). Pickups never asks for, stores or sends a Hugging Face token.

**RunPod (no GPU of your own).** Deploy a pod with a 24 GB+ GPU and any ComfyUI template that has: ComfyUI on port 8188
exposed as HTTP, ComfyUI-LTXVideo installed, and a network volume mounted for the models (so they survive a stop). Copy the HTTP address for port
8188 from the pod's Connect menu and paste it into Pickups. RunPod bills by the minute while the pod runs, idle or not:
stop it in RunPod when you're done.

To skip the first screen (scripts, a fixed setup), `uv run pickups --comfy http://gpu-pc:8188` still works.

Image models: `uv run pickups models --dir /path/to/ComfyUI/models --task image` adds FLUX 3 klein (accept Black Forest
Labs' license on Hugging Face first); `--task image-<model>` adds the others (`image-qwen-image-2.1-turbo`,
`image-krea2-turbo`, `image-z-image-turbo`, `image-ideogram4`, `image-hidream-o1`).

## Try it without a GPU

Start Pickups (`uv run pickups`) and press **Try the demo** on the first screen: the whole app on example shots, no
ComfyUI needed. **Exit demo** in the top bar takes you back. You can also open http://127.0.0.1:7788/?mock=1 directly,
or http://127.0.0.1:7788/?tour=1 to watch it drive itself.

## Images

Pick a model, an aspect ratio and a resolution, write a prompt, generate. Add reference images (or @-mention your cast)
and the same models edit instead. Generated images live in the project: one click makes one a cast entry, a previz
look frame, a Change weather first frame or a Recast reference.

Open-weights image models, run locally: **FLUX 3 klein** (Black Forest Labs), **Qwen-Image 2.1** and **2.1 Turbo**,
**Krea 2 Turbo**, **Z-Image Turbo**, **Ideogram v4** and **HiDream O1**, each wired exactly as in Comfy's official
templates (`src/pickups/images.py`), all tested on a 24 GB card.

## Projects, cast and delivery

- **Projects** hold shots, cast, images and deliveries. Every shot keeps a stack of takes (`TAKE 01` is always the
  original) with its lineage (`TAKE 03 · FINISH · FROM TAKE 02`).
- **Cast** are the people, places, props and looks of a project: an image, a name and one describing line. **Build
  character sheet** turns one photo, or a few words, into a front | back | close-up sheet, so Recast and Render previz
  see the character from every side. Write
  `@Maya` in any prompt and Pickups expands it for each model, because each model reads prompts differently: Qwen-Image
  gets the cast image as `<image2>`, LTX-2.5 gets "Maya, a woman in her thirties with short black hair and a red wool coat…", and
  SCAIL-2 gets the image as its reference. The **prompt receipt** under every prompt field shows exactly what each model
  will receive. The same characters stay consistent across Recast, Render previz, Change weather and Images.
- **Approve** takes (`⇧A`), then **Deliveries** lists every approved take with its files. **Finish all approved**
  queues 4K + HDR masters for all of them.
- **Notes:** press `N` to leave a note at the playhead on any take. Notes show on the scrubber and in the take panel,
  and Export timeline carries them into the edit at the same frame: timeline markers in the XML (tested in DaVinci
  Resolve Studio) plus clip markers on each take for Premiere, and Avid locators in the EDL.
- Exports are NLE-ready: self-describing file names (`harbour_pier_T02_remove_ProRes422HQ_UHD.mov`), the source's
  frame rate and **start timecode** (plus any trim offset), so a take drops straight over the original in Resolve or
  Premiere.

## Into your edit, out to the world

- **Export timeline:** a project's approved takes become an FCP7 XML (DaVinci Resolve, Premiere Pro) or a CMX3600 EDL,
  each take placed at its source timecode with a self-describing clip name, so the edit lands on top of the originals.
  Tested by importing into DaVinci Resolve Studio.
- **Open in ComfyUI:** every take keeps the exact ComfyUI workflows that made it (API format, real inputs), so you can
  open any result in ComfyUI and keep tweaking. Nothing is a black box.
- **Share:** one click makes a social-ready before/after: 9:16 stacked or a 1:1 wipe, with the slate burned in (what was
  done, how much of the picture is still your footage, which models) and "Made locally with Pickups".
- **Receipts:** every take has a printable receipt page to hand a client (before and after, how much is still your
  footage, the models and their licenses, the prompts each model received, source timecode, files and workflows), and
  Deliveries exports one for every approved take as a zip, with the ComfyUI workflows alongside.

## Prompt Director

Write a plain idea ("old fisherman on the pier, camera pushes in, he sighs") and press **Enhance**: the Prompt Director
rewrites it into the prompt that model reads best, and shows it to you as a suggestion (use it, keep yours, or try
again). It uses each model maker's own prompt enhancer and system prompt, read live from Comfy's official templates:
Qwen-Image 2.1's enhancers for image prompts and edits, the FLUX 3 prompt upsampler, and LTX-2.5's own director for
video. @-mentions of your cast are expanded first, and for edits and first frames the director sees the image.

It runs where you choose (Setup → Prompt director):
- **A local LLM server** such as [LM Studio](https://lmstudio.ai) or Ollama: press *Find my LM Studio*, pick a model,
  *Test & save*. Fast (a few seconds) and keeps the GPU free for video.
- **The GPU machine**: the official enhancer models in ComfyUI (slower while video jobs are running).
- **Off**: the Enhance buttons disappear; turn it back on any time.

## Prompting

Every task writes its prompts the way its model was trained to read them (LTX-2.5, Layout-To-Render, Qwen-Image 2.1,
SCAIL-2, FLUX 3 klein). [docs/prompting-guide.md](docs/prompting-guide.md) is the cheat sheet, with templates and good
and bad examples for each.

## Delivery

Every result comes in the formats people actually hand over:

| Format | Sizes | For |
|---|---|---|
| **ProRes 422 HQ** `.mov` (10-bit 4:2:2) | UHD 3840×2160 (4K-class sources), HD 1920×1080, and the native-resolution master | Editing and grading |
| **MP4 · H.264** (High profile) | UHD and HD | Plays everywhere, uploads anywhere |
| **MP4 · H.265 / HEVC** (10-bit, `hvc1`) | Top size | About half the size of H.264 at the same quality; modern Macs, PCs and phones |
| **HDR master** (HLG, BT.2020, HEVC 10-bit) + **EXR** (ACEScg) | Finish only | HDR delivery and grading |

Other aspect ratios are delivered in their own shape. Previz renders 16:9. Originals are never
modified; every result is a new take. On a 4-second 4K shot: ProRes UHD ≈ 370 MB, H.264 UHD ≈ 11 MB, H.265 UHD ≈ 6 MB
(both MP4s ≈ 48–49 dB PSNR against the ProRes, i.e. visually transparent).

## System requirements and benchmarks

### The ComfyUI computer (renders)

| | Minimum | Recommended | Best |
|---|---|---|---|
| GPU | NVIDIA, 24 GB | NVIDIA RTX 4090, 24 GB | NVIDIA, 80 GB or more (RTX PRO 6000, 96 GB) |
| Features | Any one | All 7 | All 7, longer and sharper |
| Longest take | 4 to 6 s | 4 to 6 s | Up to 9 s |
| System RAM | 64 GB | 64 GB | 64 GB |
| Disk for models | 20 GB (Image) or 60 GB (one video feature) | 135 GB | 135 GB |
| ComfyUI | 0.41 or newer, with ComfyUI-LTXVideo | 0.41 or newer, with ComfyUI-LTXVideo | 0.41 or newer, with ComfyUI-LTXVideo |
| OS | Windows 11 or Linux | Windows 11 | Linux (RunPod) or Windows 11 |

Mac, AMD or under 24 GB: run Pickups on that computer and render on another one, over your network, Tailscale or RunPod.
Pickups checks this per feature, in decimal GB (an RTX 4090 reads 25.8): 23.5 GB of VRAM for every feature and 25.0 GB
for Relight. An NVIDIA card below that says "Not tested on this GPU"; another GPU says "Not supported on this computer".

| Feature | Models |
|---|---|
| Remove a person | 59.4 GB |
| Replace a person | 62.7 GB |
| Previz to final | 60.3 GB |
| Change the weather | 60.7 GB |
| Relight | 42.9 GB |
| Finish | 44.0 GB (4K alone: SeedVR2, 4.0 GB) |
| New image (FLUX 3 klein) | 20.3 GB |
| All 7 | 134 GB |
| Remove, Replace, Previz, Weather and Finish together | 70.9 GB |

Files shared by several features are counted once. Keep 10 GB free beyond the models for a video feature (20 GB for
Finish, 5 GB for Image): Fix in Setup checks the models drive before downloading.

### The computer running Pickups

| | |
|---|---|
| OS | Mac, Windows or Linux |
| Software | Python 3.12 or newer, uv, ffmpeg (ffmpeg 8 for HDR and wide-gamut sources) |
| GPU | Not needed |
| Browser | Current Chrome, Edge, Safari or Firefox |
| Disk | About 1.1 GB per shot |

Tested: Pickups on a Mac Studio (M1 Ultra); ComfyUI on an RTX 4090 PC (64 GB RAM, Windows 11), or on a rented RTX PRO
6000 Blackwell.

<!-- bench:start -->
Measured end to end through the app: the median of 99 finished runs on 2 GPUs, updated 2026-10-17 by `pickups bench --md README.md`. GPU time is ComfyUI's share; the rest is compositing and encoding on the client.

| Task | GPU | Shot | Runs | GPU time | Total | Per frame | Fastest – slowest |
|---|---|---|--:|--:|--:|--:|---|
| Remove, locked-off | RTX 4090 | 97 frames, 3840×2748 | 1 | 13 s | 8.1 min | 5.0 s | – |
| Remove, locked-off | RTX 4090 | 202 frames, 3840×2748 | 8 | 30 s | 8.4 min | 2.5 s | 7.7 min – 12.7 min |
| Remove, locked-off | RTX PRO 6000 Blackwell Server Edition | 202 frames, 3840×2748 | 1 | 1.7 min | 7.8 min | 2.3 s | – |
| Remove, locked-off | RTX 4090 | 80 frames, 1920×1080 | 2 | 1.8 min | 2.6 min | 2.0 s | 1.8 min – 3.5 min |
| Remove, moving camera | RTX 4090 | 97 frames, 3840×2160 | 5 | 5.4 min | 8.3 min | 5.1 s | 7.8 min – 46.8 min |
| Remove, moving camera | RTX 4090 | 74 frames, 1920×1080 | 2 | 3.3 min | 4.2 min | 3.0 s | 4.0 min – 4.5 min |
| Recast (LTX-2.5) | RTX 4090 | 97 frames, 3840×2748 | 8 | 3.1 min | 5.8 min | 3.6 s | 4.2 min – 9.0 min |
| Recast (LTX-2.5) | RTX PRO 6000 Blackwell Server Edition | 81 frames, 3840×2160 | 9 | 3.4 min | 5.9 min | 4.3 s | 5.8 min – 6.6 min |
| Recast (LTX-2.5) | RTX 4090 | 57 frames, 1920×1080 | 2 | 4.1 min | 4.9 min | 4.3 s | 4.2 min – 5.7 min |
| Recast (SCAIL-2) | RTX 4090 | 81 frames, 3840×2160 | 7 | 3.4 min | 5.1 min | 3.8 s | 4.1 min – 11.1 min |
| Recast (SCAIL-2) | RTX 4090 | 77 frames, 1920×1080 | 2 | 2.6 min | 3.3 min | 2.5 s | 3.2 min – 3.4 min |
| Change weather | RTX 4090 | 97 frames, 1920×1024 | 15 | 2.4 min | 3.2 min | 2.0 s | 2.3 min – 5.6 min |
| Render previz | RTX 4090 | 97 frames, 1920×1088 | 12 | 3.6 min | 3.9 min | 2.4 s | 3.5 min – 45.4 min |
| Render previz | RTX PRO 6000 Blackwell Server Edition | 97 frames, 1920×1088 | 3 | 4.6 min | 5.5 min | 3.4 s | 2.8 min – 5.5 min |
| Render previz | RTX PRO 6000 Blackwell Server Edition | 113 frames, 1920×1088 | 10 | 5.2 min | 6.5 min | 3.2 s | 5.5 min – 9.3 min |
| Finish, best quality | RTX 4090 | 97 frames, 3840×2160 | 3 | 21.0 min | 25.2 min | 17 s | 24.7 min – 41.5 min |
| Character sheet | RTX 4090 | 3 views, 832×1280 | 9 | 20 s | 21 s | 7.0 s | 21 s – 26 s |
<!-- bench:end -->

Image (FLUX 3 klein) on the RTX 4090: 4.7 s for a 1344×768 image, 12.8 s at 2048×1152.

Every job records its speed in `benchmarks.jsonl` in your Pickups folder: the task, the GPU, the size, the time in each
stage, the GPU time and the total. `pickups bench` sums it up per task and GPU, and Setup → Benchmarks shows the same
table. `pickups bench --md README.md` rewrites the table above. Each run also says where the GPU was: `local` (ComfyUI
on this computer), `remote` (a PC on your network), or `runpod` (a RunPod address). `PICKUPS_GPU_LABEL` names a ComfyUI on a local address, such as a tunnel.

## ComfyUI on another computer, and Pickups on your phone

**ComfyUI on another computer.** Start ComfyUI there with `--listen` (for example `python main.py --listen 0.0.0.0
--port 8188`), then in Pickups' setup type its address on your network or its Tailscale name (`http://gpu-pc:8188`).
`--listen` opens ComfyUI to that network, so only do it on a network you trust, or reach it over Tailscale. **Never
forward ComfyUI's port to the internet**: it runs any workflow it's sent and has no login.

**Pickups on your phone.** With Tailscale on this computer and your phone, Setup → Use on your phone → Allow phone
access serves Pickups on this computer's Tailscale address as well (never on every network interface), and shows the
address with a QR code. Only devices in your tailnet can open it. It's off until you turn it on;
`PICKUPS_TAILSCALE=0` hides it.

## Your library and the ComfyUI machine

Everything Pickups keeps lives in one folder: `library/` inside your Pickups folder (`~/.pickups` by default, or
`PICKUPS_HOME`). Setup → Storage shows where it is, how big it is, and opens it.

- **Trash.** Move to Trash on a take or a shot moves its folder into `library/trash/`, with Undo. Assets → Trash shows
  everything there with Restore; Empty Trash removes it from this computer for good.
- **Failed and canceled jobs** leave nothing behind: their unfinished take folder is removed, and on start Pickups clears
  any left from a crash (older than an hour). Set `PICKUPS_KEEP_WORK=1` to keep them for debugging.

**On the ComfyUI machine.** ComfyUI keeps a copy of every file a job sends it, and of every frame and video it saves:

- `ComfyUI/input/`: the clips, masks and pictures Pickups uploaded.
- `ComfyUI/output/pickups/`: every frame and video a graph saved (final and intermediate).
- `ComfyUI/output/hdr_exr/`: Finish's HDR EXR frames.

Pickups copies each result back to your library when a take finishes and never reads these again, so they're safe to
clear once no job is running. Pickups doesn't remove them itself; Empty Trash says how many files stay there.

## Development

`uv sync --group dev` and `cd web && npm install` (Node.js 22.6+), then `uv run pytest` (unit tests, no GPU; the few
that run the web code skip until `npm install` has run) and `cd web && npm run build && npm run test:previz` for the web
app. `uv run python scripts/smoke_test.py` checks a running app. Before publishing anything, `uv run python scripts/audit_repo.py` checks the repo for keys, local paths,
private IPs, emails and any names you list in `PICKUPS_AUDIT_TERMS`.

**Make a reel.** `uv run python scripts/make_reel.py --format both` cuts a silent, looping 15-second before/after reel
from your example gallery (9:16 and 16:9, with a contact sheet of each) for social posts; `--cta "<repo or handle>"` adds
a last line to its end card.

## How it works

Every task is a set of ComfyUI workflows (API format) submitted through the **Comfy SDK** to Comfy's API proxy in front
of your ComfyUI. Compositing, the receipt and the ProRes encodes run on the machine running Pickups.

Agents and scripts use the same `POST /api/jobs` calls as the page (`docs/ui-spec.md`). Previz to final renders from the
start frame alone: keyframes are off and not supported in this release, so leave `params.keyframes` unset.

## Licenses

The default build is open weights only and runs fully local: every model runs on your own ComfyUI, and Pickups uses no
API or partner nodes. NVIDIA RTX Video (a faster 4K preview) is optional and off by default, because NVIDIA's library is
proprietary (Setup → Licenses).

Open weights don't always mean free for paid work. Licenses as published on Hugging Face, checked Oct 26, 2026:

| Model | Used for | License | Paid work |
|---|---|---|---|
| LTX-2.5 22B, its text encoder and IC-LoRAs (Layout-To-Render, Union Control, Relight, SDR-to-HDR, In/Outpainting, Refine-Details) | Video: previz to final, weather, Recast, HDR, Text and Image to video | LTX-2.x Community License (LTX-2.3 LoRAs: LTX-2 Community License) | Free under US$10M yearly revenue (with affiliates); a paid license above |
| Qwen-Image 2.1 (+ Turbo, prompt enhancers) | The default frame editor (clean plates, look and first frames); Images | Qwen Research License | No: research and evaluation only |
| Alternate frame editor (optional, off by default) | Frame editor when Setup → Licenses → Frame editor is switched to it | See its model card in Setup → Licenses | Check its own license |
| FLUX 3 klein | Images (default), Recast reference stills | FLUX Kommunity License | Non-commercial; commercial only under US$5M revenue (per the model card; the license text itself is gated and unverified) |
| Krea 2 Turbo (+ style-reference LoRA) | Images, style reference | Krea 2 Community License | Only under US$1M yearly revenue (with affiliates), with content filtering |
| Ideogram 4 | Images (lettering); plain words are written up as its JSON caption by the local Qwen 3.5 prompt enhancer first | Ideogram 4 Non-Commercial (gated: accept the terms on Hugging Face to download) | No: non-commercial only |
| Z-Image Turbo | Images, the start frame | Apache-2.0 | Yes |
| Z-Image Turbo Fun ControlNet Union 2.1 | The start frame (follows the previz depth) | Apache-2.0 | Yes |
| Depth Anything 3 Mono Large | Depth for the start frame, weather and Recast | Apache-2.0 | Yes |
| HiDream-O1 | Images, edits | MIT | Yes |
| SAM 3.1 | Click to select, tracking | SAM License | Within Meta's trade-control and end-use limits |
| SCAIL-2 (Wan 2.1 14B) | Recast (SCAIL-2 engine) | MIT | Yes |
| SeedVR2 3B | Finish to 4K | Apache-2.0 | Yes |
| Text encoders and VAEs (Qwen3-VL, Qwen3-4B, Gemma 4 E2B, FLUX 2 VAE, Qwen-Image VAE, ae) | The models above | Apache-2.0 (the FLUX 3 prompt upsampler: Apache-2.0 per the FLUX card, unverified) | Yes |

For paid work, switch **Setup → Licenses → Frame editor** to the alternate editor (check its own license first) and pick
an image model marked Yes.
Each take's Details list the models it ran with their licences. Not used by the app: a Qwen-Image 2.1 Fun ControlNet
file (license unverified, likely non-commercial). Pickups itself is MIT licensed.

## Credits

- **Pickups code** (app, Previz runtime, tools and tests): MIT, see `LICENSE`.
- **Previz models, scene kits and poses** (`web/public/previz/` and the built copy in `src/pickups/web/dist/previz/`): original work made for Pickups, CC0 1.0. See `web/public/previz/LICENSE.md`.
- **Third-party code in the Previz runtime:**
  - [three.js](https://github.com/mrdoob/three.js) 0.170.0, MIT.
  - [meshoptimizer](https://github.com/zeux/meshoptimizer) mesh decoder, MIT.
