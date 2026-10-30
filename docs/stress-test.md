# Stress test

`scripts/stress_test.py` runs a matrix of shots × characters × tasks × seeds through Pickups, one take at a time. It grades
every take and writes a report that opens straight from disk.

## Run it

```sh
uv run python scripts/stress_test.py stress/matrix_stage1.json --dry-run       # the plan; changes nothing
uv run python scripts/stress_test.py stress/matrix_stage1.json                 # render every cell not finished yet
uv run python scripts/stress_test.py stress/matrix_stage1.json --judge         # the same, plus a verdict from the local vision model
uv run python scripts/stress_test.py stress/matrix_stage1.json --retry-failed  # run failed, timed-out and unstartable cells again
uv run python scripts/stress_test.py stress/matrix_stage1.json --rescore --judge  # new pictures, scores and verdicts; no GPU
uv run python scripts/stress_test.py stress/matrix_stage1.json --report-only   # rebuild report.html and report.csv
```

Other flags: `--only 'skye-walker__alien*,*__t-rex__*'` (cell id patterns), `--limit N` (render at most N cells),
`--force` (render the chosen cells again, as new takes), `--out DIR`, `--server URL`, `--no-wait`, `--idle-timeout S`,
`--judge-url`, `--judge-model`, `--judge-timeout S`.

Running the same command again carries on where it stopped. Finished cells are skipped. A finished take that is missing
its pictures (or its verdict, with `--judge`) gets them without rendering again. If that take was deleted in the app, the
cell becomes an error ("Lost its take") and the run carries on; `--retry-failed` or `--force` renders it again. Ctrl-C
cancels the cell in flight.
Exit codes: 0 when done, 2 when the run had to stop (ComfyUI stuck, or busy for too long), 130 after Ctrl-C.

## How a cell runs

Cells go through the app's own job queue inside the harness process: `JobManager` with the same `TASKS` that
`POST /api/jobs` uses, as in `scripts/run_task.py`. Pictures are uploaded with `Store.add_upload`, which is what
`POST /api/uploads` does. Takes land in the normal library, so each one can be opened, compared and approved in the app.

**Why in-process rather than the server API:** restarting the server (to ship a fix) kills its in-flight jobs. A matrix
that runs for hours has to survive that, and every run should use the code on disk. The cost is a second GPU queue, so
the harness keeps out of the app's way:

- Before each cell it waits until the app's `/api/jobs` has nothing queued or running and ComfyUI's `/queue` is empty.
  If anything else used the GPU since the last cell, it frees ComfyUI's memory first, as the app does between tasks.
- If the app starts rendering during a cell, the cell is marked *contended* and its time isn't counted.
- One cell at a time. A cell gets `max(cell_timeout_s, 3 × the task's estimate)` and is then cancelled. If the cancel
  doesn't land within 2 minutes, ComfyUI counts as stuck and the run stops.

**Shots:** a matrix shot given by `shot_id` is copied once into the matrix's project ("Stress test" by default), so the
user's own shot gets no stress takes. Set `"copy": false` to use the shot as it is. A shot given by `path` imports that
file. Copies and uploads are remembered in `results/_library.json`, so later runs and stages reuse them.

**Characters** go in as an uploaded picture (`ref_upload_id`), as when someone drops a photo or sheet into Recast. A
character with only a `describe` uses Recast's FLUX 3 klein still instead.

## The matrix file

```json
{
 "stage": "stage1",
 "title": "Cast smoke: one easy shot, every character",
 "about": "One line for the report.",
 "project": "Stress test",
 "seeds": [42],
 "tasks": ["recast", {"task": "remove", "params": {"text": "car"}}],
 "cell_timeout_s": 2400,
 "shots": [
  {"id": "skye-walker", "name": "Skye ridge walker", "shot_id": "shot_…", "copy": true,
   "params": {"frame": 40, "points": [], "text": "man", "scene": "On a high grassy ridge…"}},
  {"id": "empty-beach", "path": "clips/empty_beach.mov", "params": {"frame": 0, "text": "person"},
   "expect": {"status": "failed", "within_s": 60, "message": "nobody|no one|anyone"}}
 ],
 "characters": [
  {"from_cast": "cast.json", "variant": "chosen"},
  {"name": "Ghost lady", "describe": "a Victorian ghost lady in a grey silk dress"},
  {"name": "Alien sheet", "image": "sheets/alien_turnaround.png", "describe": "a slim green alien in a silver flight suit"}
 ],
 "thresholds": {"halo_max": 2.0}
}
```

| Field | Meaning |
| --- | --- |
| `stage` | Folder name under `results/`, and the report's heading. |
| `tasks` | Task names, or `{task, params, characters, seeds, expect}`. Recast runs once per character; other tasks run once per shot unless `"characters": true`. |
| `shots[].params` | Task params for that shot: the pick (`frame`, `points`, `text`) and Recast's `scene`. `task_params: {task: {...}}` overrides per task. |
| `shots[].expect` | An expected outcome: `status` `"failed"`, `within_s`, `message` (a regular expression). |
| `characters[]` | `from_cast`: the fun-cast file (`variant`: `"chosen"`, a picture number from 0, or a list). Otherwise `image` (a file), `ref_upload_id` or `cast_image: {project, id}`, plus `describe` and extra `params`. |
| `"@upload:<path>"` | Anywhere in params: uploaded at run time and replaced by its id. |
| `thresholds` | Overrides for the limits below. |

A cell's id is `<shot>__<character or none>__<task>__s<seed>`. A cast file named in `from_cast` is read when the run starts,
so it can be written after the matrix.

## What each cell writes

In `results/<stage>/<cell>/`:

- `result.json`: shot and take ids, params, timings (queue wait, run time, estimate, GPU), QA, verdict, error, grade
  and reasons, and the time the code last changed.
- `contact.jpg`: 8 frames across the take with Before and After rows. When the change is local, two close-up rows
  follow it frame by frame, and a red box on the After row shows where they look.
- `sbs.mp4`: 2 s from the middle, before on the left and after on the right, 540 px high. `sbs.gif` is the same as a
  small GIF.
- `judge_1-3.jpg` (close-ups at 20, 50 and 80 % from the full-resolution master) and `reference.jpg`.

In `results/<stage>/`: `report.html` and `report.csv`. The report has a grade count, a shot × character grid, filters
and one card per cell with its contact sheet, numbers, reasons and links to the cell's files. Thumbnails live inside
the page and the links are relative, so the folder can be moved or zipped as a whole.

## Checks and grades

Automatic checks on a finished Recast take:

| Check | Passes when | From |
| --- | --- | --- |
| Coverage | 0.7–1.6 | `details._qa.coverage` (the new character's area against the old one's) |
| Halo | under 2.0 | `details._qa.halo` |
| Flicker | under 1.5 | `details._qa.flicker` |
| Empty frames | 0 | `details._qa.empty_new_frames` (frames where the new character wasn't found) |
| Outside the selection | 0 px | `details.pixels_changed_outside_selection` |
| Time | at most 1.5 × the estimate | submit-to-finish time against the task's starting estimate |

Other tasks are checked on Outside the selection (when the take reports it) and Time.

The halo, flicker and coverage limits are first guesses, not yet measured on real takes: after the first stage, compare
the numbers with what the contact sheets show and set `thresholds` in the matrix before trusting the grades.

Each Recast card also says, in one line, how the take was made (from `details._qa`, not graded): the reference path
("front view cropped from a 3-view sheet"), whether ComfyUI refused our reference mask and SAM's was used instead, how
many people shared the crop and which one was re-rendered, and SCAIL-2's working size. The same fields are columns in
`report.csv`.

| Grade | When |
| --- | --- |
| **A** | Every automatic check was measured and passed, and the judge gave identity and artefacts 4 or more. Cells without a character need no verdict. |
| **B** | Nothing failed, but something stops it being an A: a judge score of 3, no verdict, QA scores missing, or a time that couldn't be counted. |
| **F** | The cell failed, timed out or couldn't start; a measured check failed; or the judge gave 2 or less. |
| **–** | Not run yet, or interrupted (it runs again next time). |

Grades are worked out again whenever the report is written, so after a threshold change `--report-only` is enough.
Each grade comes with its reasons in words. A failure whose message reads like code (an error class, a traceback,
ffmpeg) gets an extra reason, because a person can't act on it.

**Expected failures** (`expect.status: "failed"`) grade A when the take stops within `within_s` (default 120 s) with a
plain message that matches `message`. They grade F if a take is made anyway, if it stops too slowly, or if the message
is technical or different.

## The judge

`--judge` asks the vision model set in Setup → Director (`director.url` and `director.vision_model`, else `model`) for
a verdict. That can be LM Studio or any OpenAI-compatible server. It sends the reference picture and the three
close-ups, and asks for `{"identity_match": 1-5, "artefacts": 1-5, "notes": "…"}`:

- It asks for structured output (a JSON schema) and asks again without it if the server answers HTTP 400.
- Only a JSON object with whole-number scores from 1 to 5 counts. Prose, other scores, or notes in another script
  (CJK, as in `director.py`) earn one plain retry, then the cell is left unjudged with the reason.
- Each call times out after `--judge-timeout` (180 s by default, because the first call may load the model).
- Before the run, the harness checks the server lists the model (5 s). If it doesn't, the run carries on unjudged, and
  a later `--judge` run judges those takes without rendering again. A broken judge never fails a cell.

## Stage 1, and what comes next

`matrix_stage1.json` (kept with the stress material, outside the repo) is the cast smoke test. The Skye ridge walker is
a single person on a still camera who turns round and walks toward the camera (picked on frame 40 as "man"). They are
recast as each of the nine fun-cast characters, with each character's chosen picture and seed 42: about 45 minutes of GPU.

Later stages add shots for the risks the design review ranked:

1. **Non-human characters:** Stage 1 already covers them (astronaut, knight, T-rex costume, android, alien, skeleton,
   lizard explorer). Watch Coverage and Empty frames, which show SAM's "person" missing the new character, and identity.
2. **Contact sheets as the reference:** characters whose `image` is a multi-view sheet, next to the same character's
   front still.
3. **A second person in the crop:** the couple shots (touching and walking together).
4. **Close-ups:** the face close-up and the man at the table. Watch artefacts for softness.
5. **Small figures:** the castle path walker.
6. **Inputs people drop:** a vertical phone clip, 60 fps, a 12 s clip and a clip with nobody in it. Each is a shot with
   an `expect`, so each must end fast with a worded outcome.
