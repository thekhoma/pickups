// The shot-spec builder. The sentence is already a spec; this only places it.
import type { CameraPick } from "./compile.ts";
import { ensurePoses, shotLibrary } from "./library";
import { createShotBuilder } from "./runtime/builder.js";
import { frameShot } from "./runtime/camera.js";
import { aimGaze } from "./runtime/motion.js";
import { guardShot } from "./runtime/shotGuard.js";
import type { BuiltShot, CameraSpec, KitLayout, ShotSpec } from "./runtime/types";
import { pickFor, readShot, shotSpecOn } from "./shotLex.ts";
import type { PrevizSpec } from "./spec.ts";

const layouts = new Map<string, KitLayout>();

async function getLayout(id: string): Promise<KitLayout> {
  const cached = layouts.get(id);
  if (cached) return cached;
  const res = await fetch(`/previz/data/kits/${id}.json`);
  if (!res.ok) throw new Error("kit " + id + " " + res.status);
  const layout = (await res.json()) as KitLayout;
  layouts.set(id, layout);
  return layout;
}

let builder: ReturnType<typeof createShotBuilder> | null = null;
let seq = 0;

export function shotSeq(): number {
  return seq;
}

function shots(): ReturnType<typeof createShotBuilder> {
  if (!builder) builder = createShotBuilder({ lib: shotLibrary(), getLayout });
  return builder;
}

/** One builder per runtime, so a gate bake cannot tear down the editor's shot. */
export function createRuntimeBuilder(): ReturnType<typeof createShotBuilder> {
  return createShotBuilder({ lib: shotLibrary(), getLayout });
}

/** Focal length that renders `fov` degrees tall. Wide frames use a film height of 36mm / aspect, the same lens the preview shows. */
function focalOf(fov: number, aspect: number): number {
  const filmH = 36 / Math.max(aspect, 1);
  return Math.round(((filmH / 2) / Math.tan((fov * Math.PI) / 360)) * 10) / 10;
}

/** A camera button replaces the sentence's shot, angle or move. A vehicle with no angle stays through the windshield. */
export function shotFraming(shot: ShotSpec, pick: CameraPick | null): ShotSpec {
  if (!pick) return shot;
  const camera: CameraSpec = { ...shot.camera };
  if (pick === "close") camera.shot = "close";
  else if (pick === "wide") camera.shot = "wide";
  else if (pick === "close-wide") {
    camera.shot = "close";
    camera.move = "pull_out";
  } else if (pick === "dolly-in") camera.move = "push_in";
  else if (pick === "dolly-out") camera.move = "pull_out";
  else if (pick === "orbit") camera.move = "orbit";
  else if (pick === "drone") camera.angle = "overhead";
  else if (pick === "fpv") camera.angle = "pov";
  else if (pick === "track") camera.move = "track";
  else if (pick === "parallel") camera.move = "dolly_along";
  else if (pick === "locked") camera.move = "static";
  return { ...shot, camera };
}

/**
 * A walker coming toward the lens grows in frame: the camera body holds while its aim follows them. It only gives ground
 * (camShare of their travel) when they would otherwise come closer than 3 m.
 */
function holdForApproach(built: BuiltShot, shot: ShotSpec, states: { pos: number[] }[], duration: number): void {
  const move = shot.camera.move;
  if (shot.camera.facing !== "toward" || !(move == null || move === "push_in")) return;
  const subj = built.subject?.object;
  const gait = subj?.userData.gait as { follow?: boolean; start: [number, number]; yaw: number; speed: number; maxDist: number; camShare?: number } | undefined;
  if (!gait || !gait.follow || !states.length) return;
  const pace = shot.camera.subjectSpeed ?? 1;
  const travel = Math.min(gait.maxDist ?? Infinity, Math.abs(gait.speed) * pace * duration);
  const cam = states[states.length - 1].pos;
  const dir = [Math.sin(gait.yaw), Math.cos(gait.yaw)];
  const end = [gait.start[0] + dir[0] * travel, gait.start[1] + dir[1] * travel];
  // how far the lens is still ahead of them along their walk at the end (negative: they would have passed it)
  const gap = (cam[0] - end[0]) * dir[0] + (cam[2] - end[1]) * dir[1];
  gait.camShare = travel > 1e-3 ? Math.max(0, Math.min(1, (3 - gap) / travel)) : 1;
}

/**
 * A couple in a dance hold turns on the spot with the camera's circle, about 30 degrees less than the camera travels, so
 * the camera still visibly circles them but stays on their side or three-quarter and never ends behind one of them.
 */
function turnWithCamera(built: BuiltShot, keys: { pos: number[] }[]): void {
  const dancers = built.actors.map((a) => a.object).filter((o) => (o.userData.motion as { kind?: string } | undefined)?.kind === "sway");
  const m0 = dancers[0]?.userData.motion as { center?: number[]; spin?: number } | undefined;
  if (dancers.length !== 2 || !m0?.center || keys.length < 2) return;
  const c = m0.center;
  const azOf = (p: number[]) => Math.atan2(p[0] - c[0], p[2] - c[2]);
  let arc = 0;
  for (let i = 1; i < keys.length; i++) {
    let d = azOf(keys[i].pos) - azOf(keys[i - 1].pos);
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    arc += d;
  }
  const lag = Math.sign(arc) * Math.min(Math.abs(arc), (30 * Math.PI) / 180);
  // start them turned so the view swings through a true side-on at mid-shot: lag/2 one way, lag/2 the other
  const pa = dancers[0].position;
  const pb = dancers[1].position;
  const axis = Math.atan2(pb.x - pa.x, pb.z - pa.z);
  const wrap = (v: number) => Math.atan2(Math.sin(v), Math.cos(v));
  const base = azOf(keys[0].pos) + lag / 2 - axis;
  const turn0 = [wrap(base - Math.PI / 2), wrap(base + Math.PI / 2)].sort((u, v) => Math.abs(u) - Math.abs(v))[0];
  for (const o of dancers) Object.assign(o.userData.motion as object, { spin: arc - lag, turn0 });
}

export function currentShot(): BuiltShot | null {
  return shots().current;
}

export interface AssembledShot {
  built: BuiltShot;
  spec: PrevizSpec;
  label: string;
}

/** Which walkers the camera rides with, from the move (a crane or a truck keeps up with a few people; a top-down
 * crowd holds its frame while the crowd crosses it). */
function followFlags(built: BuiltShot, shot: ShotSpec, framing: ReturnType<typeof shotFraming>): void {
  const move = framing.camera.move;
  const segs = shot.camera.segments || [];
  const people = (shot.actors || []).filter((a) => a.kind === "person").reduce((n, a) => n + (a.count || 1), 0);
  const stays = (move === "boom_up" || move === "boom_down" || move === "truck_left" || move === "truck_right") && people < 6 && shot.camera.facing !== "toward";
  if (stays) for (const seg of segs) if ((seg.tracking ?? 0) < 0.5) seg.tracking = 1;
  const follow = segs.length
    ? segs.some((s) => (s.tracking ?? 0) >= 0.5) || stays
    : move == null || move === "push_in" || move === "pull_out" || move === "follow" || move === "track" || move === "fpv" || move === "aerial" || move === "body_mount" || stays;
  const overheadCrowd = framing.camera.angle === "overhead" && people >= 6;
  for (const actor of built.actors) if (actor.object.userData.gait && !actor.object.userData.extra) actor.object.userData.gait.follow = follow && !overheadCrowd;
}

/** Place the whole spec. The camera subject is lite, or the hero mesh on a close-up. Every other person is the crowd mesh. */
export async function assembleShot(spec: PrevizSpec, builder: ReturnType<typeof createShotBuilder> | null = null): Promise<AssembledShot | null> {
  if (!shotSpecOn()) return null;
  const shot = readShot(spec.notes);
  if (!shot) return null;
  await ensurePoses();
  const pick = pickFor(spec);
  const close = shot.camera.shot === "close" || shot.camera.shot === "extreme_close" || pick === "close" || pick === "close-wide";
  const built = await (builder || shots()).build(shot, { subjectTier: close ? "hero" : "lite", liteCount: 1, crowdCount: 12, instancing: true });
  const aspect = spec.size[0] / spec.size[1];
  if (spec.camera.free && spec.camera.keys.length >= 2) {
    // Run 7: a path bent in Top view that still rides with the subject sets the people up as the framed shot did
    // (who the camera follows, how much of the walk it shares, a couple turning with it). Old free cameras never set
    // follow and build as before.
    if (spec.camera.follow) {
      const framing = shotFraming(shot, pick);
      followFlags(built, shot, framing);
      holdForApproach(built, framing, spec.camera.keys, spec.duration_s || 4);
      turnWithCamera(built, spec.camera.keys);
    }
    const guarded = guardShot({
      built,
      keys: spec.camera.keys.map((k) => ({ t: k.t, pos: k.pos, look: k.look, focal_mm: k.focal_mm ?? spec.lens.focal_mm })),
      aspect,
      sentence: spec.notes || "",
      shot,
      locked: true,
    });
    aimGaze(built, spec.camera.keys[spec.camera.keys.length - 1].pos);
    seq += 1;
    const timing = {
      sentence: spec.notes,
      label: "locked",
      start: spec.camera.keys[0].pos,
      end: spec.camera.keys[spec.camera.keys.length - 1].pos,
      ...built.timing,
      place: shot.place.id,
      standin: !!shot.place.standin,
      triangles: built.stats.triangles,
      drawCalls: built.stats.drawCalls,
      seq,
      guard: guarded.notes,
    };
    const win = window as Window & { __previzShot?: typeof timing; __previzBuilt?: BuiltShot };
    win.__previzShot = timing;
    win.__previzBuilt = built;
    return { built, spec, label: "locked" };
  }
  const framing = shotFraming(shot, pick);
  followFlags(built, shot, framing);
  const framed = frameShot(built, framing, { aspect });
  const states = framed.keys && framed.keys.length >= 2 ? framed.keys : framed.mid ? [framed.start, framed.mid, framed.end] : [framed.start, framed.end];
  holdForApproach(built, framing, states, spec.duration_s || 4);
  const roll = shot.camera.dutch || 0;
  let keys = states.map((state, i) => ({
    t: typeof state.t === "number" ? state.t : states.length === 1 ? 0 : i / (states.length - 1),
    pos: state.pos as [number, number, number],
    look: state.target as [number, number, number],
    focal_mm: state.focal_mm ?? focalOf(state.fov, aspect),
    ...(roll ? { roll } : {}),
  }));
  // a lens the sentence names ("35mm") is the lens, unless the move is a lens ramp (dolly zoom)
  const lensMm = (shot.camera as { lensMm?: number }).lensMm;
  const ramped = keys.some((k) => Math.abs(k.focal_mm - keys[0].focal_mm) > 0.5);
  if (lensMm && !ramped) keys = keys.map((k) => ({ ...k, focal_mm: lensMm }));
  const guarded = guardShot({ built, keys, aspect, sentence: spec.notes || "", shot: framing });
  keys = guarded.keys;
  turnWithCamera(built, keys);
  // a head turn "to the camera" ends on the lens where the camera really finishes
  aimGaze(built, keys[keys.length - 1].pos);
  const chained = (shot.camera.segments || []).length > 1;
  const subjectSpeed = shot.camera.subjectSpeed;
  const next: PrevizSpec = {
    ...spec,
    subjects: subjectSpeed ? spec.subjects.map((s) => (s.kind === "humanoid" ? { ...s, speed: subjectSpeed } : s)) : spec.subjects,
    lens: { ...spec.lens, focal_mm: keys[0].focal_mm },
    camera: {
      ...spec.camera,
      ease: framed.label === "fpv follow" || chained ? "linear" : "in_out",
      taper: chained ? 0 : spec.camera.taper,
      ...(shot.camera.segments?.length ? { segments: shot.camera.segments } : {}),
      ...(shot.camera.handheld ? { handheld: shot.camera.handheld } : {}),
      keys,
    },
  };
  seq += 1;
  const timing = {
    sentence: spec.notes,
    label: framed.label,
    start: framed.start.pos,
    mid: framed.mid?.pos ?? (framed.keys && framed.keys.length > 2 ? framed.keys[Math.floor(framed.keys.length / 2)].pos : null),
    end: framed.end.pos,
    keyCount: keys.length,
    ...built.timing,
    place: shot.place.id,
    standin: !!shot.place.standin,
    triangles: built.stats.triangles,
    drawCalls: built.stats.drawCalls,
    seq,
    guard: guarded.notes,
  };
  const win = window as Window & { __previzShot?: typeof timing; __previzBuilt?: BuiltShot };
  win.__previzShot = timing;
  win.__previzBuilt = built;
  return { built, spec: next, label: framed.label };
}
