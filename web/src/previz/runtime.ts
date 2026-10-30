// The renderer. renderFrame(i) never reads the clock: frame i is always the same picture.
import * as THREE from "three";
import { BokehPass } from "three/examples/jsm/postprocessing/BokehPass.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import type { CameraPose } from "./cameraMath";
import { assetsFor, ensureLibrary, libraryOn, preloadCore, previzTimes } from "./library";
import { frameShot } from "./runtime/camera.js";
import { createWorld, disposeEnvironment } from "./runtime/world.js";
import { facingOf } from "./facing";
import { frameView, wordFor, type FrameView } from "./inView";
import { buildScene, type Built, type TopFrame } from "./scene";
import { fitBounds } from "./pathEdit";
import { TOP_PX } from "./topView";

/** Frame 0's framing, sent with Make shot: the main subject's box, the horizon, what the frame shows, and the place. */
export interface Frame0 {
  subject?: [number, number, number, number];
  horizon?: number;
  view?: FrameView;
  place?: string;
  /** Things out of frame 0 that come into view later, when and on which side (for the motion prompt). */
  reveals?: Reveal[];
  /** The camera's real move: where it is and what it looks at (the main subject) at the first and the last frame. The
   * spec's keys are the template's until the page frames the shot, so the motion words read this instead. */
  move?: CameraMove;
  /** The camera at frame 0: height above the ground, pitch (negative looks down) and lens. */
  camera?: { height_m: number; pitch_deg: number; focal_mm: number };
  /** Each subject at frame 0, in the shot's order: what it is, where across the frame (0 left, 1 right), how far. */
  subjects?: Frame0Subject[];
  /** The sentence's time of day and weather words, if it has them. */
  world?: { time?: string | null; weather?: string | null };
  /** Where the key light (the sun) comes from, relative to the camera at frame 0. */
  light?: { dir: "camera left" | "camera right" | "behind camera" | "backlit" };
}

export interface Frame0Subject {
  kind: "person" | "vehicle" | "animal";
  x: number;
  distance_m: number;
  in_view: boolean;
  /** The sentence's verb for it ("walk", "sit", "wave", "drive"). */
  doing?: string;
  /** The sentence's noun for it ("woman", "sedan"). */
  word?: string;
  /** How fast it travels, metres a second (a car's drive, a walker's gait, a board's cruise). */
  speed_ms?: number;
  /** Which way it faces at frame 0, from its rotation (facingOf). */
  facing?: "camera" | "away" | "left" | "right";
  /** Its width and height on screen at frame 0, as fractions of the frame. */
  w?: number;
  h?: number;
}

/** The depth pass's far plane: z / DEPTH_FAR is written in 24 bits (start_frame.py DEPTH_FAR reads it back). */
export const DEPTH_FAR = 1000;
/** The frames the depth pass is made for: the start still, and the rescue keyframes' 48 and 96 (previz_maker.DEPTH_FRAMES). */
export const DEPTH_FRAMES = [0, 48, 96];


export interface CameraMove {
  start: { pos: [number, number, number]; look: [number, number, number] };
  end: { pos: [number, number, number]; look: [number, number, number] };
}

/** The frame-0 camera in the template's terms: height in metres, pitch in degrees, focal length on a 36 mm gauge. */
function cameraAt0(cam: THREE.PerspectiveCamera): { height_m: number; pitch_deg: number; focal_mm: number } {
  const dir = new THREE.Vector3();
  cam.getWorldDirection(dir);
  const r = (n: number, k = 10) => Math.round(n * k) / k;
  return { height_m: r(cam.position.y, 100), pitch_deg: r((Math.asin(Math.max(-1, Math.min(1, dir.y))) * 180) / Math.PI), focal_mm: Math.round(cam.getFocalLength()) };
}

/** The scene's key light (its sun: the shadow-casting directional light) relative to the camera: in front of the lens
 * is backlit, behind it is behind camera, otherwise the side it comes from. Null when the scene has none. */
function keyLightDir(scene: THREE.Scene, cam: THREE.PerspectiveCamera): "camera left" | "camera right" | "behind camera" | "backlit" | null {
  let key: THREE.DirectionalLight | null = null;
  scene.traverse((o) => {
    const l = o as THREE.DirectionalLight;
    if (l.isDirectionalLight && l.visible && (!key || (l.castShadow && !key.castShadow) || l.intensity > key.intensity)) key = l;
  });
  if (!key) return null;
  const light = key as THREE.DirectionalLight;
  const from = light.getWorldPosition(new THREE.Vector3()).sub(light.target.getWorldPosition(new THREE.Vector3())).normalize();
  const fwd = cam.getWorldDirection(new THREE.Vector3());
  const right = new THREE.Vector3().crossVectors(fwd, cam.up).normalize();
  const ahead = from.dot(fwd);
  if (ahead > 0.5) return "backlit";
  if (ahead < -0.5) return "behind camera";
  return from.dot(right) >= 0 ? "camera right" : "camera left";
}

const VEHICLE_ASSET = /car|suv|van|truck|bus|sedan|coupe|taxi|jeep|pickup|vehicle|boat|motorbike|motorcycle/i;

/** The scene objects that are the shot's people and vehicles (the actor list also holds props: a bench, a phone). */
function subjectsAt0Roots(objs: THREE.Object3D[]): THREE.Object3D[] {
  return objs.filter((o) => {
    const ud = o.userData || {};
    const asset = String(ud.asset || ud.libraryId || "");
    if (ud.drive || VEHICLE_ASSET.test(asset)) return true;
    return !/^(dog|cat|horse|bird)/.test(asset) && (ud.asset === "person" || /^person|crowd/.test(asset) || !!ud.gait);
  });
}

/** The shot's subjects at frame 0: its people, animals and vehicles (the runtime's actor list also holds the props,
 * a bench or a phone: those are left out). */
type Parsed = { actors?: { kind?: string; pose?: string | null; asset?: string; word?: string }[]; props?: { kind?: string; word?: string }[] } | null;

function subjectsAt0(objs: THREE.Object3D[], cam: THREE.PerspectiveCamera, seen: number[], shot?: Parsed): Frame0Subject[] {
  const out: Frame0Subject[] = [];
  const parsed = shot?.actors;
  // The sentence's nouns, in order: people for the people, vehicle props for the vehicles.
  const people = (shot?.actors ?? []).filter((a) => a.kind !== "animal").map((a) => a.word);
  const vehicles = (shot?.props ?? []).filter((p) => p.kind === "vehicle").map((p) => p.word);
  let nPeople = 0;
  let nVehicles = 0;
  objs.forEach((o, i) => {
    const box = new THREE.Box3().setFromObject(o);
    const c = box.isEmpty() ? o.getWorldPosition(new THREE.Vector3()) : box.getCenter(new THREE.Vector3());
    const p = c.clone().project(cam);
    const ud = o.userData || {};
    const ride = o.parent?.userData?.cruise as { speed?: number } | undefined;
    const asset = String(ud.asset || ud.libraryId || parsed?.[i]?.asset || "");
    const drive = ud.drive as { speed?: number } | undefined;
    const animal = /^(dog|cat|horse|bird)/.test(asset);
    const person = !animal && (ud.asset === "person" || /^person|crowd/.test(asset) || !!ud.gait);
    const kind: Frame0Subject["kind"] | null = drive || VEHICLE_ASSET.test(asset) ? "vehicle" : animal ? "animal" : person ? "person" : null;
    if (!kind) return;
    const speed = drive?.speed ?? ride?.speed ?? (ud.gait as { speed?: number } | undefined)?.speed;
    const doing = (kind === "person" ? parsed?.[i]?.pose : null) || (drive ? "drive" : (ud.gait as { pose?: string } | undefined)?.pose) || undefined;
    const word = kind === "person" ? people[nPeople++] : kind === "vehicle" ? vehicles[nVehicles++] : undefined;
    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(o.getWorldQuaternion(new THREE.Quaternion()));
    const toCam = cam.position.clone().sub(c);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion);
    const facing = facingOf({ x: fwd.x, z: fwd.z }, { x: toCam.x, z: toCam.z }, { x: right.x, z: right.z });
    // Run 2: its size on screen (the box's corners projected), for the still's "head about 6 percent of the frame height".
    let size: { w: number; h: number } | null = null;
    if (!box.isEmpty()) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const cx of [box.min.x, box.max.x]) for (const cy of [box.min.y, box.max.y]) for (const cz of [box.min.z, box.max.z]) {
        const q = new THREE.Vector3(cx, cy, cz).project(cam);
        xs.push(Math.max(-1, Math.min(1, q.x)));
        ys.push(Math.max(-1, Math.min(1, q.y)));
      }
      const r3 = (n: number) => Math.round(n * 1000) / 1000;
      size = { w: r3((Math.max(...xs) - Math.min(...xs)) / 2), h: r3((Math.max(...ys) - Math.min(...ys)) / 2) };
    }
    out.push({
      ...(size && size.w > 0 && size.h > 0 ? size : {}),
      kind,
      ...(word ? { word } : {}),
      x: Math.round(((p.x + 1) / 2) * 100) / 100,
      distance_m: Math.round(cam.position.distanceTo(c) * 10) / 10,
      in_view: seen.includes(i),
      ...(doing ? { doing } : {}),
      ...(speed != null && speed > 0 ? { speed_ms: Math.round(speed * 10) / 10 } : {}),
      facing,
    });
  });
  return out;
}

export interface Reveal {
  word: string;
  at_s: number;
  side: "left" | "right" | "middle";
}
import { assembleShot, createRuntimeBuilder, shotFraming } from "./shotScene";
import { LIBRARY_FAILED, plainError } from "./errors";
import { ensureShotLexicon, pickFor, readShot, shotLibraryFailed, shotSpecOn } from "./shotLex.ts";
import { shownPlace } from "./compile.ts";
import { shotFrames, type PrevizSpec } from "./spec";

type ShotBuild = {
  root: THREE.Object3D;
  actors: THREE.Object3D[];
  kitWorld?: object;
  /** The place kit's name ("Lake", "Park"), for the start frame's words. */
  place?: string;
  look?: { time?: "day" | "golden" | "dusk" | "night" | null; weather?: "clear" | "overcast" | "rain" | "fog" | "snow" | null };
} | null;

export type Axis = "x" | "y" | "z";

/** What the pointer is on in Top view. */
export type TopHit =
  | { kind: "handle" }
  | { kind: "camera" }
  | { kind: "point"; index: number }
  | { kind: "subject"; index: number }
  | { kind: "path"; u: number }
  | { kind: "none" };

/** The parsed shot, so "pu" and "pus" that mean the same thing do not rebuild. */
function sentenceKey(notes: string): unknown {
  const shot = readShot(notes || "");
  if (!shot) return notes || "";
  return {
    place: shot.place?.id ?? null,
    world: shot.world ?? null,
    camera: shot.camera ?? null,
    actors: (shot.actors || []).map((a) => ({ id: a.id, asset: a.asset, kind: a.kind, pose: a.pose, count: a.count })),
    props: (shot.props || []).map((p) => ({ id: p.id, asset: p.asset, count: p.count })),
    beats: shot.beats || [],
    facing: shot.camera?.facing ?? null,
    segments: (shot.camera?.segments || []).map((s) => ({ move: s.move, from_t: s.from_t, to_t: s.to_t, tracking: s.tracking })),
    speed: shot.camera?.subjectSpeed ?? null,
    lens: shot.camera?.lensMm ?? null,
  };
}

/** What forces a new scene. A free camera's knots, and the lens in shot mode, ride on the live spec. */
function layoutKey(spec: PrevizSpec, shotMode = false): string {
  const free = !!spec.camera.free;
  const pick = spec.camera.pick ?? null;
  return JSON.stringify({
    template: spec.template,
    size: spec.size,
    props: spec.environment.props,
    subjects: spec.subjects.map((s) => ({ id: s.id, kind: s.kind, height_m: s.height_m, anim: s.anim })),
    sentence: sentenceKey(spec.notes),
    camera: {
      ease: spec.camera.ease,
      pick,
      move: pick,
      free,
      ...(shotMode ? {} : { focal: spec.lens.focal_mm }),
      ...(free
        ? {}
        : {
            keys: spec.camera.keys.map((k) => ({
              t: k.t,
              look: k.look,
              roll: k.roll ?? 0,
              ...(shotMode ? {} : { focal_mm: k.focal_mm }),
            })),
          }),
    },
  });
}

export class PrevizRuntime {
  readonly renderer: THREE.WebGLRenderer;
  private built: Built | null = null;
  private place: string | undefined;
  /** The sentence as the page sent it (the assembled spec doesn't keep it): frame 0's nouns and words come from it. */
  private sentence = "";
  private spec: PrevizSpec | null = null;
  private layout = "";
  private raycaster = new THREE.Raycaster();
  private drag: {
    index: number;
    axis: Axis;
    origin: [number, number, number];
    grab: number;
    anchor: THREE.Vector3;
    camPos: THREE.Vector3;
    camQuat: THREE.Quaternion;
  } | null = null;
  private shown = 0;
  private view: "eye" | "split" | "top" = "eye";
  /** Top view's map. Kept across rebuilds of the same scene; null fits it on the next draw. */
  private topFrame: TopFrame | null = null;
  private topSelected: number | null = null;
  private clean = false;
  /** "rack focus" in the sentence: a depth-of-field pass whose focus pulls from the background to the subject. */
  private rack: { composer: EffectComposer; bokeh: BokehPass; scene: THREE.Scene; far: number } | null = null;
  private generation = 0;
  private closed = false;
  private shotMode = false;
  private timer = 0;
  private resolvers: Array<() => void> = [];
  private queued: { source: PrevizSpec; frame: number; pixels?: [number, number] } | null = null;
  private builder = createRuntimeBuilder();
  /** One line when a shot falls back to the grey scene. */
  fallback: ((reason: string) => void) | null = null;
  /** Quiet lines after a shot is corrected. Never a popup. */
  onGuard: ((notes: string[]) => void) | null = null;
  onLabel: ((label: string) => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, alpha: false });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.VSMShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    this.renderer.shadowMap.needsUpdate = true;
    this.renderer.setPixelRatio(1);
    void preloadCore();
  }

  setSpec(source: PrevizSpec, frame = 0, pixels?: [number, number]): Promise<void> {
    const spec = structuredClone(source);
    const nextKey = layoutKey(spec, this.shotMode);
    if (this.built && this.spec && nextKey === this.layout) {
      const focal = spec.lens.focal_mm;
      const ramp = spec.camera.keys.some((k, i) => i > 0 && k.focal_mm && k.focal_mm !== spec.camera.keys[0].focal_mm);
      const before = this.spec.subjects.map((s) => s.pos.join(","));
      const live: PrevizSpec = this.shotMode && !spec.camera.free
        ? {
            ...spec,
            lens: { ...spec.lens, focal_mm: focal },
            camera: {
              ...spec.camera,
              keys: this.spec.camera.keys.map((k, i) => {
                const seat = (window as unknown as { __previzBuilt?: { subject?: { object?: { userData?: { seatFrame?: { wideLook: number; headTop: number; tableY: number } } } } } }).__previzBuilt?.subject?.object?.userData?.seatFrame;
                const look = seat
                  ? [k.look[0], focal >= 70 ? (seat.headTop + seat.tableY) / 2 : seat.wideLook, k.look[2]] as typeof k.look
                  : k.look;
                const kept = ramp ? (spec.camera.keys[i]?.focal_mm ?? k.focal_mm ?? focal) : focal;
                return { ...k, focal_mm: kept, look };
              }),
            },
          }
        : spec;
      this.spec = live;
      this.built.live.spec = live;
      const moved = live.subjects.some((s, i) => before[i] !== s.pos.join(","));
      if (!this.shotMode || moved) live.subjects.forEach((s, i) => this.built?.place(i, s.pos));
      if (this.shotMode && moved && !spec.camera.free) this.reframe();
      this.renderFrame(frame);
      return Promise.resolve();
    }
    this.queued = { source, frame, pixels };
    return new Promise((resolve) => {
      this.resolvers.push(resolve);
      window.clearTimeout(this.timer);
      this.timer = window.setTimeout(() => this.flush(), 120);
    });
  }

  private flush(): void {
    const job = this.queued;
    const waiters = this.resolvers.splice(0);
    this.queued = null;
    if (!job || this.closed) {
      waiters.forEach((resolve) => resolve());
      return;
    }
    const done = this.runSpec(job.source, job.frame, job.pixels);
    void done.finally(() => waiters.forEach((resolve) => resolve()));
  }

  private runSpec(source: PrevizSpec, frame = 0, pixels?: [number, number]): Promise<void> {
    const spec = structuredClone(source);
    this.sentence = source.notes || "";
    const generation = ++this.generation;
    if (!previzTimes.boot) previzTimes.boot = performance.now();
    const finish = (live: PrevizSpec, shot: ShotBuild) => {
      if (this.closed || generation !== this.generation) return;
      this.spec = live;
      this.shotMode = !!shot;
      this.layout = layoutKey(spec, this.shotMode);
      this.drag = null;
      this.built?.dispose();
      const started = performance.now();
      this.built = buildScene(live, shot);
      // a new scene: the map fits it on the next Top draw
      this.topFrame = null;
      this.place = shot?.place;
      if (shot) {
        createWorld(this.renderer, this.built.scene, { quality: "fast" }).apply(shot.kitWorld || {}, shot.look || {});
        this.built.update(0);
        this.renderer.compile(this.built.scene, this.built.camera);
        this.renderer.shadowMap.needsUpdate = true;
        this.renderer.render(this.built.scene, this.built.camera);
        this.renderer.shadowMap.needsUpdate = true;
        this.renderer.render(this.built.scene, this.built.camera);
        const shotWin = (window as unknown as { __previzShot?: { guard?: string[]; label?: string } }).__previzShot;
        this.onGuard?.(shotWin?.guard || []);
        if (shotWin?.label) this.onLabel?.(shotWin.label);
        this.fallback?.("");
      }
      previzTimes.lastBuildMs = performance.now() - started;
      if (!previzTimes.firstMs) previzTimes.firstMs = performance.now() - previzTimes.boot;
      else if (!previzTimes.lastLoadMs) previzTimes.rebuildMs = previzTimes.lastBuildMs;
      const times = {
        firstMs: previzTimes.firstMs,
        coldLoadMs: previzTimes.coldLoadMs,
        lastLoadMs: previzTimes.lastLoadMs,
        lastBuildMs: previzTimes.lastBuildMs,
        rebuildMs: previzTimes.rebuildMs,
      };
      (window as Window & { __previzTimes?: typeof times }).__previzTimes = times;
      const [w, h] = pixels ?? [live.size[0], live.size[1]];
      this.renderer.setSize(w, h, false);
      this.renderFrame(frame);
    };
    const legacy = () => {
      this.shotMode = false;
      if (!libraryOn()) {
        finish(spec, null);
        return Promise.resolve();
      }
      return ensureLibrary(assetsFor(spec))
        .catch((err: unknown) => {
          this.fallback?.(plainError(err, "build"));
        })
        .then(() => finish(spec, null));
    };
    if (!shotSpecOn() || !libraryOn()) return legacy();
    const ready = readShot(spec.notes) ? Promise.resolve() : ensureShotLexicon();
    return ready
      .then(() => assembleShot(spec, this.builder))
      .then((assembled) => {
        if (this.closed || generation !== this.generation) return;
        if (!assembled) {
          this.fallback?.("That sentence didn't build a shot, so this is the plain grey scene.");
          return legacy();
        }
        const actors = [...assembled.built.actors, ...assembled.built.props].map((item) => item.object);
        const parsed = readShot(spec.notes);
        const look = parsed?.world;
        // Run 3: a place the parser only defaulted to is no place (the start frame then reads the previz objects).
        const place = shownPlace(spec.notes, assembled.built.layout.label, parsed?.place?.source);
        finish(assembled.spec, { root: assembled.built.root, actors, kitWorld: assembled.built.layout.world, place, look });
      })
      .catch((err: unknown) => {
        if (!this.closed && generation === this.generation) {
          this.fallback?.(shotLibraryFailed() ? LIBRARY_FAILED : plainError(err, "build"));
          return legacy();
        }
        console.error(err);
      });
  }

  /** Move a shot actor, then frame again unless the camera is free. */
  private reframe(): void {
    const spec = this.spec;
    const built = (window as Window & { __previzBuilt?: Parameters<typeof frameShot>[0] }).__previzBuilt;
    if (!spec || !built || spec.camera.free || !this.built) return;
    const shot = readShot(spec.notes);
    if (!shot) return;
    const aspect = spec.size[0] / spec.size[1];
    const framing = shotFraming(shot, pickFor(spec));
    const framed = frameShot(built, framing, { aspect });
    const states = framed.keys && framed.keys.length >= 2 ? framed.keys : framed.mid ? [framed.start, framed.mid, framed.end] : [framed.start, framed.end];
    const filmH = 36 / Math.max(aspect, 1);
    const focal = (fov: number) => Math.round(((filmH / 2) / Math.tan((fov * Math.PI) / 360)) * 10) / 10;
    const ramp = spec.camera.keys.some((k, i) => i > 0 && k.focal_mm && k.focal_mm !== spec.camera.keys[0].focal_mm);
    const keys = states.map((state, i) => ({
      t: states.length === 1 ? 0 : i / (states.length - 1),
      pos: state.pos as [number, number, number],
      look: state.target as [number, number, number],
      focal_mm: ramp ? (spec.camera.keys[Math.min(i, spec.camera.keys.length - 1)].focal_mm ?? focal(state.fov)) : (spec.lens.focal_mm || focal(state.fov)),
    }));
    const next = { ...spec, camera: { ...spec.camera, keys } };
    this.spec = next;
    this.built.live.spec = next;
  }

  getSpec(): PrevizSpec | null {
    return this.spec;
  }

  /** How many frames this spec is, capped by the GPU. */
  count(cap: number): number {
    if (!this.spec) return 1;
    return shotFrames(this.spec.duration_s, this.spec.fps, cap);
  }

  /** Hold one camera pose for every frame. Used while setting a start or an end. */
  setPose(pose: CameraPose | null): void {
    if (!this.built) return;
    this.built.live.pose = pose;
    this.renderFrame(this.shown);
  }

  /** The pose held for every frame (while setting a start or an end, or flying), or null. */
  heldPose(): CameraPose | null {
    return this.built?.live.pose ?? null;
  }

  currentView(): "eye" | "split" | "top" {
    return this.view;
  }

  /** The camera's real move, first frame to last, each end looking at the main subject (or 10 m ahead with none). */
  private moveOf(built: NonNullable<typeof this.built>, cam: THREE.PerspectiveCamera): CameraMove {
    const r = (v: THREE.Vector3): [number, number, number] => [v.x, v.y, v.z].map((n) => Math.round(n * 1000) / 1000) as [number, number, number];
    const at = (frame: number) => {
      this.renderFrame(frame);
      cam.updateMatrixWorld(true);
      const main = built.actors[0]?.object;
      const box = main ? new THREE.Box3().setFromObject(main) : null;
      const dir = new THREE.Vector3();
      cam.getWorldDirection(dir);
      const look = box && !box.isEmpty() ? box.getCenter(new THREE.Vector3()) : cam.position.clone().addScaledVector(dir, 10);
      return { pos: r(cam.position), look: r(look) };
    };
    const end = at(Math.max(0, this.count(100000) - 1));
    const start = at(0);
    return { start, end };
  }

  /** Frame 0 through the shot camera: the main subject's box and the horizon height, as fractions of the frame (0 is the
   * top). Sent with Make shot so a start frame pinned later can be checked against it. Call with the eye view on. */
  framing0(): Frame0 | null {
    const built = this.built;
    if (!built || !this.spec) return null;
    this.renderFrame(0);
    const cam = built.camera;
    cam.updateMatrixWorld(true);
    const out: Frame0 = {};
    const main = built.actors[0]?.object;
    if (main) {
      const box = new THREE.Box3().setFromObject(main);
      if (!box.isEmpty()) {
        let x0 = 1, y0 = 1, x1 = 0, y1 = 0, seen = false;
        for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
          const p = new THREE.Vector3(x, y, z).project(cam);
          if (p.z < -1 || p.z > 1) continue;
          seen = true;
          const u = (p.x + 1) / 2;
          const v = (1 - p.y) / 2;
          x0 = Math.min(x0, u); x1 = Math.max(x1, u); y0 = Math.min(y0, v); y1 = Math.max(y1, v);
        }
        const clip = (n: number) => Math.round(Math.min(1, Math.max(0, n)) * 10000) / 10000;
        if (seen && clip(x1) > clip(x0) && clip(y1) > clip(y0)) out.subject = [clip(x0), clip(y0), clip(x1), clip(y1)];
      }
    }
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    const flat = new THREE.Vector3(dir.x, 0, dir.z);
    if (flat.lengthSq() > 1e-6) {
      // Any point level with the lens lies on the horizon line; 10 m ahead stays inside the far plane.
      const far = cam.position.clone().add(flat.normalize().multiplyScalar(10)).project(cam);
      if (far.z >= -1 && far.z <= 1) out.horizon = Math.round(Math.min(2, Math.max(-1, (1 - far.y) / 2)) * 10000) / 10000;
    }
    // What frame 0 shows and what it doesn't: the start frame is made from these words, so nothing out of frame is added.
    out.view = frameView(built.scene, cam, built.actors.map((a) => a.object));
    out.camera = cameraAt0(cam);
    out.move = this.moveOf(built, cam);
    const key = keyLightDir(built.scene, cam);
    if (key) out.light = { dir: key };
    const parsed = readShot(this.sentence || this.spec.notes || "");    out.subjects = subjectsAt0(built.actors.map((a) => a.object), cam, out.view.actors, parsed);
    if (parsed?.world && (parsed.world.time || parsed.world.weather)) out.world = { time: parsed.world.time ?? null, weather: parsed.world.weather ?? null };
    if (this.place) out.place = this.place;
    if (out.view.outside.length) out.reveals = this.revealsOf(out.view.outside);
    this.renderFrame(0);
    return out.subject || out.horizon != null || out.view ? out : null;
  }

  /**
   * Frame 0's subject matte for the start frame's canny mask (start_frame.py): every person and vehicle in the scene
   * (the sentence's and the place's own: extras, parked cars, a crowd) white, everything else black, at the export size.
   * An ID pass from the scene itself, never from colour: a grey car on a grey road can't be told apart by its pixels.
   * A PNG data URL, or null before a scene is built.
   */
  subjectMatte(width = 1920, height = 1088): string | null {
    const built = this.built;
    if (!built || !this.spec || typeof document === "undefined") return null;
    this.renderFrame(0);
    const { scene, camera } = built;
    const renderer = this.renderer;
    const roots = new Set(subjectsAt0Roots(built.actors.map((a) => a.object)));
    const MOVERS = new Set(["people", "cars", "a boat", "a bike"]);
    const isSubject = (o: THREE.Object3D): boolean => {
      for (let p: THREE.Object3D | null = o; p; p = p.parent) {
        if (roots.has(p)) return true;
        if (/^(kit_.*|set|items)$/.test(p.name)) return false;
        const w = wordFor(p.name);
        if (w && MOVERS.has(w)) return true;
      }
      return false;
    };
    const white = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const black = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const swapped: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
    const hidden: THREE.Object3D[] = [];
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        const mat = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.Material | undefined;
        // A see-through stand-in (dust, a light cone) stays out rather than turning into a black wall.
        if (mat && (mat.visible === false || (mat.transparent && mat.opacity < 0.05))) {
          if (mesh.visible) hidden.push(mesh);
          mesh.visible = false;
          return;
        }
        swapped.push([mesh, mesh.material]);
        mesh.material = isSubject(mesh) ? white : black;
      } else if (((o as THREE.Line).isLine || (o as THREE.Points).isPoints || (o as THREE.Sprite).isSprite) && o.visible) {
        hidden.push(o);
        o.visible = false;
      }
    });
    const background = scene.background;
    const fog = scene.fog;
    const toneMapping = renderer.toneMapping;
    const aspect = camera.aspect;
    scene.background = new THREE.Color(0x000000);
    scene.fog = null;
    renderer.toneMapping = THREE.NoToneMapping;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    const target = new THREE.WebGLRenderTarget(width, height);
    const px = new Uint8Array(width * height * 4);
    try {
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(target, 0, 0, width, height, px);
    } finally {
      renderer.setRenderTarget(null);
      for (const [mesh, mat] of swapped) mesh.material = mat;
      for (const o of hidden) o.visible = true;
      scene.background = background;
      scene.fog = fog;
      renderer.toneMapping = toneMapping;
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      target.dispose();
      white.dispose();
      black.dispose();
      this.renderFrame(0);
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const img = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      const from = (height - 1 - y) * width * 4; // WebGL rows run bottom to top
      img.data.set(px.subarray(from, from + width * 4), y * width * 4);
    }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL("image/png");
  }

  /**
   * The start still's depth (start_frame.py, method zbuf): a z-buffer of frames 0, 48 and 96 through the shot camera, from
   * the scene itself. View-space z over DEPTH_FAR in 24 bits (red the high byte), no anti-aliasing (a render target), the
   * sky black (0): the sky dome, the path line, gizmos and see-through stand-ins are hidden. Frame number to a PNG data URL;
   * null before a scene is built.
   */
  depthPasses(width = 1920, height = 1088): Record<string, string> | null {
    const built = this.built;
    if (!built || !this.spec || typeof document === "undefined") return null;
    const { scene, camera } = built;
    const renderer = this.renderer;
    const last = Math.max(0, this.count(100000) - 1);
    const frames = [...new Set(DEPTH_FRAMES.map((f) => Math.min(f, last)))];
    const mat = new THREE.MeshBasicMaterial();
    mat.side = THREE.DoubleSide;
    mat.fog = false;
    mat.toneMapped = false;
    mat.customProgramCacheKey = () => "pickups_depth";
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = "varying float vDepthZ;\n" + sh.vertexShader.replace("#include <project_vertex>", "#include <project_vertex>\n  vDepthZ = -mvPosition.z;");
      sh.fragmentShader = `varying float vDepthZ;
void main() {
  float d = clamp(vDepthZ / ${DEPTH_FAR.toFixed(1)}, 0.0, 0.999999);
  float hi = floor(d * 255.0) / 255.0;
  float r1 = (d - hi) * 255.0;
  float mid = floor(r1 * 255.0) / 255.0;
  float lo = (r1 - mid) * 255.0;
  gl_FragColor = vec4(hi + 0.25 / 255.0, mid + 0.25 / 255.0, lo, 1.0);
}`;
    };
    const hidden: THREE.Object3D[] = [];
    scene.traverse((o) => {
      if (!o.visible) return;
      const mesh = o as THREE.Mesh;
      const m = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.Material | undefined;
      const sky = mesh.isMesh && (o.name === "world_sky" || (m as THREE.ShaderMaterial | undefined)?.type === "ShaderMaterial");
      const seeThrough = mesh.isMesh && !!m && (m.visible === false || (m.transparent && m.opacity < 0.05));
      if (sky || seeThrough || (o as THREE.Line).isLine || (o as THREE.Points).isPoints || (o as THREE.Sprite).isSprite) {
        hidden.push(o);
        o.visible = false;
      }
    });
    const background = scene.background;
    const fog = scene.fog;
    const toneMapping = renderer.toneMapping;
    const aspect = camera.aspect;
    const clear = renderer.getClearColor(new THREE.Color());
    const clearAlpha = renderer.getClearAlpha();
    const target = new THREE.WebGLRenderTarget(width, height);
    const px = new Uint8Array(width * height * 4);
    const out: Record<string, string> = {};
    try {
      for (const f of frames) {
        this.renderFrame(f);
        scene.background = null;
        scene.fog = null;
        scene.overrideMaterial = mat;
        renderer.toneMapping = THREE.NoToneMapping;
        renderer.setClearColor(0x000000, 1);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setRenderTarget(target);
        renderer.clear();
        renderer.render(scene, camera);
        renderer.readRenderTargetPixels(target, 0, 0, width, height, px);
        renderer.setRenderTarget(null);
        scene.overrideMaterial = null;
        const url = this.pixelsToPng(px, width, height);
        if (url) out[String(f)] = url;
      }
    } finally {
      renderer.setRenderTarget(null);
      scene.overrideMaterial = null;
      for (const o of hidden) o.visible = true;
      scene.background = background;
      scene.fog = fog;
      renderer.toneMapping = toneMapping;
      renderer.setClearColor(clear, clearAlpha);
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
      target.dispose();
      mat.dispose();
      this.renderFrame(0);
    }
    return Object.keys(out).length ? out : null;
  }

  /** Render-target pixels (bottom row first) as a top-row-first PNG data URL, alpha forced opaque. */
  private pixelsToPng(px: Uint8Array, width: number, height: number): string | null {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const img = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      const from = (height - 1 - y) * width * 4; // WebGL rows run bottom to top
      img.data.set(px.subarray(from, from + width * 4), y * width * 4);
    }
    for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL("image/png");
  }

  /** When each word missing from frame 0 first covers 2% of the frame, every half second through the shot. */
  private revealsOf(words: string[]): Reveal[] {
    const built = this.built;
    const spec = this.spec;
    if (!built || !spec) return [];
    const fps = spec.fps || 24;
    const last = Math.max(0, Math.round(spec.duration_s * fps) - 1);
    const out: Reveal[] = [];
    const left = new Set(words);
    for (let f = Math.round(fps / 2); f <= last && left.size; f += Math.round(fps / 2)) {
      this.renderFrame(f);
      const v = frameView(built.scene, built.camera, built.actors.map((a) => a.object), 24, 14);
      for (const w of [...left]) {
        if ((v.share[w] ?? 0) < 0.02) continue;
        const x = v.x[w] ?? 0.5;
        out.push({ word: w, at_s: Math.round((f / fps) * 2) / 2, side: x < 0.4 ? "left" : x > 0.6 ? "right" : "middle" });
        left.delete(w);
      }
    }
    return out;
  }

  /** Why this frame might read as blank: the camera inside an object, or nothing but bare ground (see inView.ts). */
  viewAt(frame: number): FrameView | null {
    const built = this.built;
    if (!built || !this.spec) return null;
    this.renderFrame(frame);
    return frameView(built.scene, built.camera, built.actors.map((a) => a.object), 24, 14);
  }

  /** Where the picture is aimed right now, including a recompose. */
  viewPose(): CameraPose | null {
    const built = this.built;
    const aim = built?.live.aim;
    if (!built || !aim) return null;
    const dist = built.camera.position.distanceTo(new THREE.Vector3(aim.look[0], aim.look[1], aim.look[2]));
    const dir = new THREE.Vector3();
    built.camera.getWorldDirection(dir);
    const reach = dist > 0.05 ? dist : 1;
    return {
      pos: [built.camera.position.x, built.camera.position.y, built.camera.position.z],
      look: [
        built.camera.position.x + dir.x * reach,
        built.camera.position.y + dir.y * reach,
        built.camera.position.z + dir.z * reach,
      ],
    };
  }

  showHandles(index: number | null): void {
    this.built?.actors.forEach((actor, i) => {
      if (actor.handles) actor.handles.visible = i === index;
    });
    this.renderFrame(this.shown);
  }

  /** What the pointer is on: an arrow, or a subject. */
  pick(ndcX: number, ndcY: number): { index: number; axis: Axis | null } | null {
    if (!this.built) return null;
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.built.camera);
    const handles: THREE.Object3D[] = [];
    this.built.actors.forEach((actor) => {
      if (actor.handles?.visible) handles.push(actor.handles);
    });
    const arrow = this.raycaster.intersectObjects(handles, true)[0];
    if (arrow) {
      const axis = arrow.object.userData.axis as Axis | undefined;
      const index = this.owner(arrow.object);
      if (axis && index >= 0) return { index, axis };
    }
    const body = this.raycaster.intersectObjects(this.built.actors.map((actor) => actor.object), true)[0];
    if (!body) return null;
    const index = this.owner(body.object);
    return index >= 0 ? { index, axis: null } : null;
  }

  /** The middle of a subject, in the world. Orbit swings around this. */
  pivotOf(index: number): [number, number, number] | null {
    const actor = this.built?.actors[index];
    if (!actor) return null;
    actor.object.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(actor.object);
    if (box.isEmpty()) return null;
    const c = box.getCenter(new THREE.Vector3());
    return [c.x, c.y, c.z];
  }

  beginDrag(index: number, axis: Axis, ndcX: number, ndcY: number): void {
    const built = this.built;
    const actor = built?.actors[index];
    const origin = actor?.origin ?? actor?.home;
    const hit = actor ? this.worldOnAxis(axis, actor.object.position, ndcX, ndcY) : null;
    if (!built || !actor || !origin || hit == null) return;
    this.drag = {
      index,
      axis,
      origin: [...origin],
      grab: hit,
      anchor: actor.object.position.clone(),
      camPos: built.camera.position.clone(),
      camQuat: built.camera.quaternion.clone(),
    };
  }

  /** New blocking position while an arrow is dragged. */
  moveDrag(ndcX: number, ndcY: number): [number, number, number] | null {
    if (!this.drag || !this.built) return null;
    const cam = this.built.camera;
    const posed = cam.position.clone();
    const posedQuat = cam.quaternion.clone();
    cam.position.copy(this.drag.camPos);
    cam.quaternion.copy(this.drag.camQuat);
    cam.updateMatrixWorld(true);
    const hit = this.worldOnAxis(this.drag.axis, this.drag.anchor, ndcX, ndcY);
    cam.position.copy(posed);
    cam.quaternion.copy(posedQuat);
    if (hit == null) return null;
    const delta = hit - this.drag.grab;
    const next: [number, number, number] = [...this.drag.origin];
    const axis = this.drag.axis === "x" ? 0 : this.drag.axis === "y" ? 1 : 2;
    next[axis] = Math.min(40, Math.max(axis === 1 ? 0 : -40, this.drag.origin[axis] + delta));
    this.built.place(this.drag.index, next);
    this.renderFrame(this.shown);
    return next;
  }

  endDrag(): void {
    this.drag = null;
  }

  private debugLine(): void {
    const el = document.getElementById("previz-debug");
    if (!el) return;
    const shot = (window as Window & { __previzShot?: { place?: string; loadMs?: number; buildMs?: number; triangles?: number; drawCalls?: number; label?: string } }).__previzShot;
    if (!shot) return;
    const tris = shot.triangles ? `${(shot.triangles / 1000).toFixed(0)}k tris` : "";
    el.textContent = [shot.place, `load ${shot.loadMs ?? 0} ms`, `build ${shot.buildMs ?? 0} ms`, tris, shot.drawCalls != null ? `${shot.drawCalls} draws` : "", shot.label].filter(Boolean).join("  |  ");
  }

  private owner(obj: THREE.Object3D): number {
    return this.built?.actors.findIndex((actor) => {
      let node: THREE.Object3D | null = obj;
      while (node) {
        if (node === actor.object || node === actor.handles) return true;
        node = node.parent;
      }
      return false;
    }) ?? -1;
  }

  /** World position along an axis, through a fixed point, with the camera as it is right now. */
  private worldOnAxis(axis: Axis, anchor: THREE.Vector3, ndcX: number, ndcY: number): number | null {
    const built = this.built;
    if (!built) return null;
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), built.camera);
    const dir = new THREE.Vector3(axis === "x" ? 1 : 0, axis === "y" ? 1 : 0, axis === "z" ? 1 : 0);
    const eye = new THREE.Vector3();
    built.camera.getWorldDirection(eye);
    const normal = eye.clone().addScaledVector(dir, -eye.dot(dir));
    if (normal.lengthSq() < 1e-8) normal.crossVectors(dir, new THREE.Vector3(0, 1, 0));
    if (normal.lengthSq() < 1e-8) normal.crossVectors(dir, new THREE.Vector3(1, 0, 0));
    normal.normalize();
    const hit = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(normal, anchor), hit)) return null;
    return hit.dot(dir);
  }

  setView(view: "eye" | "split" | "top", fit = true): void {
    const from = this.view;
    this.view = view;
    // the map fits when it opens (Top and Split differ in width), then holds still; Make shot's own switch keeps it
    if (fit && view !== "eye" && from !== view) this.topFrame = null;
    this.renderFrame(this.shown);
    if (view !== "eye") this.onTopFrame?.(this.topScale());
  }

  // ---- Top view: a still map. It fits on a new scene, a view switch or Fit; otherwise only zoom and pan move it. ----

  /** The Top pane in CSS pixels (the canvas is drawn at the shot's size and scaled by the page). */
  private topPane(): { w: number; h: number } {
    const size = new THREE.Vector2();
    this.renderer.getSize(size);
    const el = this.renderer.domElement as HTMLCanvasElement | undefined;
    const cw = el?.clientWidth || size.x;
    const ch = el?.clientHeight || size.y;
    return { w: this.view === "split" ? cw - Math.floor(cw / 2) : cw, h: Math.max(1, ch) };
  }

  private topAspect(): number {
    const p = this.topPane();
    return p.w / p.h;
  }

  /** The map's frame, fitted first if there is none yet. */
  topFrameNow(): TopFrame {
    if (!this.topFrame) this.fitTop();
    return this.topFrame ?? { cx: 0, cz: 0, half: 8 };
  }

  /** Frame the map on the camera's path, its points and every subject, first frame to last. */
  fitTop(): void {
    const built = this.built;
    if (!built) return;
    this.topFrame = fitBounds(built.mapPoints(), { aspect: this.topAspect(), minHalf: 4, margin: 1.3 });
    this.topChanged();
  }

  /** Zoom the map about a point of the pane (NDC); factor over 1 zooms out. */
  zoomTop(factor: number, ndcX = 0, ndcY = 0): void {
    const before = this.groundAt(ndcX, ndcY);
    const f = this.topFrameNow();
    this.topFrame = { ...f, half: Math.min(250, Math.max(2, f.half * factor)) };
    this.applyTop();
    const after = this.groundAt(ndcX, ndcY);
    if (before && after && this.topFrame) this.topFrame = { ...this.topFrame, cx: this.topFrame.cx + before.x - after.x, cz: this.topFrame.cz + before.z - after.z };
    this.topChanged();
  }

  /** Slide the map so the ground under `from` (NDC) ends up under `to`. */
  panTop(from: { x: number; y: number }, to: { x: number; y: number }): void {
    const a = this.groundAt(from.x, from.y);
    const b = this.groundAt(to.x, to.y);
    const f = this.topFrameNow();
    if (!a || !b) return;
    this.topFrame = { ...f, cx: f.cx + a.x - b.x, cz: f.cz + a.z - b.z };
    this.topChanged();
  }

  /** Metres per CSS pixel on the map now (the scale bar). */
  topScale(): number {
    return (2 * this.topFrameNow().half) / this.topPane().h;
  }

  /** Called when the map moves or zooms, for the scale bar. */
  onTopFrame: ((metresPerPixel: number) => void) | null = null;

  private topChanged(): void {
    this.renderFrame(this.shown);
    this.onTopFrame?.(this.topScale());
  }

  /** Point the ortho camera at the map frame (picking needs it before the next draw). */
  private applyTop(): void {
    const built = this.built;
    if (!built || this.view === "eye") return;
    built.showPath(true, this.topAspect(), { frame: this.topFrameNow(), mpp: this.topScale(), selected: this.topSelected ?? null });
  }

  /** The highlighted path point in Top view (the rail's Remove point acts on it). */
  setTopSelection(index: number | null): void {
    this.topSelected = index;
    this.renderFrame(this.shown);
  }

  /**
   * What is under the pointer in Top view, nearest first in this order: the camera's turn handle, the camera, a path
   * point, a person or vehicle, the path line. Distances are in screen pixels, so it works the same at any zoom.
   */
  pickTop(ndcX: number, ndcY: number): TopHit {
    const built = this.built;
    if (!built || this.view === "eye") return { kind: "none" };
    this.applyTop();
    const pane = this.topPane();
    const v = new THREE.Vector3();
    const px = (x: number, z: number) => {
      v.set(x, 0, z).project(built.top);
      return { dx: ((v.x - ndcX) / 2) * pane.w, dy: ((v.y - ndcY) / 2) * pane.h };
    };
    const dist = (x: number, z: number) => {
      const d = px(x, z);
      return Math.hypot(d.dx, d.dy);
    };
    const mpp = this.topScale();
    const aim = built.live.aim;
    const yaw = Math.atan2(aim.look[0] - aim.pos[0], aim.look[2] - aim.pos[2]);
    const hx = aim.pos[0] + Math.sin(yaw) * TOP_PX.handleAt * mpp;
    const hz = aim.pos[2] + Math.cos(yaw) * TOP_PX.handleAt * mpp;
    if (dist(hx, hz) <= TOP_PX.handle + 6) return { kind: "handle" };
    // Run 10: a point under the pointer wins over the camera drawn on it (after a click adds a point the camera sits
    // there, and a drag made a second point instead of moving it), unless every point is in one place (a still
    // camera moves as one). The path and the people win over the camera's view wedge, which covers a push-in's path.
    const keys = built.drawnKeys();
    const still = keys.every((k) => Math.hypot(k.x - keys[0].x, k.z - keys[0].z) < 0.01);
    let best = -1;
    let bestD = TOP_PX.hit;
    keys.forEach((k, i) => {
      const d = dist(k.x, k.z);
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    });
    if (best >= 0 && !still) return { kind: "point", index: best };
    if (dist(aim.pos[0], aim.pos[2]) <= TOP_PX.hit + 2) return { kind: "camera" };
    if (best >= 0) return { kind: "point", index: best };
    for (const s of built.footprints()) {
      if (s.index < (this.spec?.subjects.length ?? 0) && dist(s.x, s.z) <= TOP_PX.subject + 4) return { kind: "subject", index: s.index };
    }
    const path = built.trajectory(96);
    let near = Infinity;
    let at = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const a = px(path[i][0], path[i][1]);
      const b = px(path[i + 1][0], path[i + 1][1]);
      const ex = b.dx - a.dx;
      const ey = b.dy - a.dy;
      const len2 = ex * ex + ey * ey || 1;
      const s = Math.max(0, Math.min(1, -(a.dx * ex + a.dy * ey) / len2));
      const d = Math.hypot(a.dx + ex * s, a.dy + ey * s);
      if (d < near) {
        near = d;
        at = (i + s) / (path.length - 1);
      }
    }
    if (near <= TOP_PX.hit - 2 && !still) return { kind: "path", u: at };
    // the wedge: anywhere inside the lens's triangle
    const along = TOP_PX.wedge * mpp;
    const g = this.groundAt(ndcX, ndcY);
    if (g) {
      const rx = g.x - aim.pos[0];
      const rz = g.z - aim.pos[2];
      const fwd = rx * Math.sin(yaw) + rz * Math.cos(yaw);
      const side = Math.abs(rx * Math.cos(yaw) - rz * Math.sin(yaw));
      const focal = built.camera.getFocalLength();
      const spread = Math.tan(Math.min(1.35, Math.atan(18 / Math.max(4, focal))));
      if (fwd > 0 && fwd <= along && side <= fwd * spread + 3 * mpp) return { kind: "camera" };
    }
    if (near <= TOP_PX.hit - 2) return { kind: "path", u: at };
    return { kind: "none" };
  }

  /**
   * Make the camera the user's: the keys on screen now (the framed ones, in shot mode) become the spec's, free, and
   * still riding with the subject if they did. No rebuild: the scene is the one on screen. Returns the camera.
   */
  adoptPath(): PrevizSpec["camera"] | null {
    const built = this.built;
    const spec = this.spec;
    if (!built || !spec) return null;
    if (spec.camera.free) return spec.camera;
    const follow = built.follows();
    const camera: PrevizSpec["camera"] = {
      ...spec.camera,
      keys: spec.camera.keys.map((k) => ({ ...k, pos: [...k.pos] as [number, number, number], look: [...k.look] as [number, number, number] })),
      free: true,
      ...(follow ? { follow: true } : {}),
    };
    const next = { ...spec, camera };
    this.spec = next;
    built.live.spec = next;
    this.layout = layoutKey(next, this.shotMode);
    return camera;
  }

  /** Key `index`'s ground position that draws it at (x, z) in Top view (the subject's travel and distance taken out). */
  keyGround(index: number, x: number, z: number): [number, number] | null {
    return this.built ? this.built.keyGround(index, x, z) : null;
  }

  /** For tests and agents: where Top view drew things, in CSS pixels from the canvas's top left (window.__previzTop). */
  private publishTop(): void {
    const built = this.built;
    if (!built || typeof window === "undefined" || this.view === "eye") return;
    const pane = this.topPane();
    const el = this.renderer.domElement as HTMLCanvasElement | undefined;
    const offset = this.view === "split" ? (el?.clientWidth || 0) - pane.w : 0;
    const v = new THREE.Vector3();
    const px = (x: number, z: number) => {
      v.set(x, 0, z).project(built.top);
      return { x: Math.round(offset + ((v.x + 1) / 2) * pane.w), y: Math.round(((1 - v.y) / 2) * pane.h) };
    };
    const mpp = this.topScale();
    const aim = built.live.aim;
    const yaw = Math.atan2(aim.look[0] - aim.pos[0], aim.look[2] - aim.pos[2]);
    (window as Window & { __previzTop?: unknown }).__previzTop = {
      frame: this.topFrame,
      mpp,
      camera: px(aim.pos[0], aim.pos[2]),
      handle: px(aim.pos[0] + Math.sin(yaw) * TOP_PX.handleAt * mpp, aim.pos[2] + Math.cos(yaw) * TOP_PX.handleAt * mpp),
      points: built.drawnKeys().map((k) => px(k.x, k.z)),
      path: built.trajectory(16).map(([x, z]) => px(x, z)),
      subjects: built.footprints().map((s) => ({ index: s.index, kind: s.kind, ...px(s.x, s.z) })),
      selected: this.topSelected ?? null,
    };
  }

  /** Key `index` as Top view draws it. */
  drawnKey(index: number): { x: number; z: number; u: number } | null {
    return this.built?.drawnKeys()[index] ?? null;
  }

  /** Shot time 0 to 1 of a frame. */
  timeOf(frame: number): number {
    if (!this.spec) return 0;
    const n = Math.max(1, Math.round(this.spec.duration_s * this.spec.fps + 1e-6));
    return n <= 1 ? 0 : Math.max(0, Math.min(n - 1, frame)) / (n - 1);
  }

  /** Ground point under the picture, for dragging a person in shot mode. */
  dropAt(ndcX: number, ndcY: number): { x: number; z: number } | null {
    const built = this.built;
    if (!built) return null;
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), built.camera);
    const hit = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) return null;
    return { x: Math.min(40, Math.max(-40, hit.x)), z: Math.min(40, Math.max(-40, hit.z)) };
  }

  /** Where a Top view point (NDC of the Top pane) meets the ground. */
  groundAt(ndcX: number, ndcY: number): { x: number; z: number } | null {
    const built = this.built;
    if (!built) return null;
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), built.top);
    const hit = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), hit)) return null;
    return { x: hit.x, z: hit.z };
  }

  /**
   * Export mode: every frame is the camera's eye with no editor graphics (the path, its knots, the move arrows or
   * anything else marked userData.overlay), whatever view or selection the editor has. exportPreviz turns it on for
   * its loop and off after, so a caller can't leak an overlay into the shot by forgetting to reset the view.
   */
  setClean(on: boolean): void {
    this.clean = on;
  }

  /** Editor-only objects in the scene (userData.overlay) that are showing now. */
  overlaysShowing(): THREE.Object3D[] {
    const out: THREE.Object3D[] = [];
    this.built?.scene.traverseVisible((n) => { if (n.userData?.overlay) out.push(n); });
    return out;
  }

  /** Run 2 (R&D #4): the tree proxies showing now (a pine template's clones and the kit's instanced trees). The layout
   * video leaves them out: past the start still, LTX drew them as ball topiary. The start still's depth keeps them. */
  treesShowing(): THREE.Object3D[] {
    const out: THREE.Object3D[] = [];
    this.built?.scene.traverseVisible((n) => {
      if (n.userData?.proxy === "pine" || /^inst_tree(__|$)/.test(n.name) || /^tree(__|$)/.test(String(n.userData?.asset ?? ""))) out.push(n);
    });
    return out;
  }

  renderFrame(i: number): void {
    if (!this.spec || !this.built) return;
    if (this.clean) {
      const hidden = [...this.overlaysShowing(), ...this.treesShowing()];
      hidden.forEach((n) => { n.visible = false; });
      const view = this.view;
      this.view = "eye";
      try {
        this.renderFrameAs(i);
      } finally {
        this.view = view;
        hidden.forEach((n) => { n.visible = true; });
      }
      return;
    }
    this.renderFrameAs(i);
  }

  private renderFrameAs(i: number): void {
    if (!this.spec || !this.built) return;
    this.shown = i;
    const n = Math.max(1, Math.round(this.spec.duration_s * this.spec.fps + 1e-6));
    const t = n <= 1 ? 0 : Math.max(0, Math.min(n - 1, i)) / (n - 1);
    this.built.update(t);
    if (this.built.scene.userData.moves) this.renderer.shadowMap.needsUpdate = true;
    const size = new THREE.Vector2();
    this.renderer.getSize(size);
    const w = size.x;
    const h = Math.max(1, size.y);
    const specAspect = this.spec.size[0] / this.spec.size[1];
    const fitEye = (aspect: number) => {
      if (this.built && this.built.camera.aspect !== aspect) {
        this.built.camera.aspect = aspect;
        this.built.camera.updateProjectionMatrix();
      }
    };
    if (this.view === "eye") {
      fitEye(specAspect);
      this.built.showPath(false);
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, w, h);
      if (this.shotMode && /\brack focus\b/i.test(this.spec.notes || "")) this.renderRack(t, w, h);
      else this.renderer.render(this.built.scene, this.built.camera);
      this.debugLine();
      return;
    }
    const view = { frame: this.topFrameNow(), mpp: this.topScale(), selected: this.topSelected ?? null };
    if (this.view === "top") {
      this.built.showPath(true, this.topAspect(), view);
      const fog = this.built.scene.fog;
      this.built.scene.fog = null;
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.render(this.built.scene, this.built.top);
      this.built.scene.fog = fog;
      this.publishTop();
      return;
    }
    // Split: the real shot on the left, letterboxed at the shot's own aspect (B8: never a narrower frame), the map right
    const half = Math.max(1, Math.floor(w / 2));
    this.renderer.setScissorTest(true);
    fitEye(specAspect);
    this.built.showPath(false);
    const eyeH = Math.min(h, Math.round(half / specAspect));
    const eyeY = Math.floor((h - eyeH) / 2);
    this.renderer.setScissor(0, 0, half, h);
    this.renderer.setViewport(0, 0, half, h);
    const clearColor = this.renderer.getClearColor?.(new THREE.Color());
    this.renderer.setClearColor?.(0x000000, 1);
    this.renderer.clear?.();
    if (clearColor) this.renderer.setClearColor?.(clearColor, 1);
    this.renderer.setViewport(0, eyeY, half, eyeH);
    this.renderer.setScissor(0, eyeY, half, eyeH);
    this.renderer.render(this.built.scene, this.built.camera);
    this.built.showPath(true, this.topAspect(), view);
    const fog = this.built.scene.fog;
    this.built.scene.fog = null;
    this.renderer.setViewport(half, 0, w - half, h);
    this.renderer.setScissor(half, 0, w - half, h);
    this.renderer.render(this.built.scene, this.built.top);
    this.built.scene.fog = fog;
    this.renderer.setScissorTest(false);
    this.publishTop();
  }

  /** Rack focus: sharp on the far background at the start, pulled onto the subject's face over the middle of the shot. */
  private renderRack(t: number, w: number, h: number): void {
    const built = this.built;
    if (!built) return;
    if (!this.rack || this.rack.scene !== built.scene) {
      this.rack?.composer.dispose();
      const composer = new EffectComposer(this.renderer);
      composer.addPass(new RenderPass(built.scene, built.camera));
      const bokeh = new BokehPass(built.scene, built.camera, { focus: 20, aperture: 0.0022, maxblur: 0.02 });
      composer.addPass(bokeh);
      composer.addPass(new OutputPass());
      // the far focus: the median distance to whatever the frame sees past the subject (from the first frame)
      const sub0 = (window as Window & { __previzBuilt?: { subject?: { object?: THREE.Object3D } } }).__previzBuilt?.subject?.object;
      const ray = new THREE.Raycaster();
      const hits: number[] = [];
      for (let gx = -2; gx <= 2; gx++) {
        for (let gy = -2; gy <= 2; gy++) {
          ray.setFromCamera(new THREE.Vector2(gx * 0.42, gy * 0.42), built.camera);
          const hit = ray.intersectObjects(built.scene.children, true).find((h) => {
            if (!(h.object as THREE.Mesh).isMesh || !h.object.visible || h.object.name === "world_sky") return false;
            let n: THREE.Object3D | null = h.object;
            while (n) { if (n === sub0) return false; n = n.parent; }
            return true;
          });
          if (hit) hits.push(hit.distance);
        }
      }
      hits.sort((a, b) => a - b);
      const far = hits.length ? Math.max(4, Math.min(40, hits[Math.floor(hits.length / 2)])) : 15;
      this.rack = { composer, bokeh, scene: built.scene, far };
    }
    const { composer, bokeh, far } = this.rack;
    composer.setSize(w, h);
    const sub = (window as Window & { __previzBuilt?: { subject?: { object?: THREE.Object3D } } }).__previzBuilt?.subject?.object;
    const head = sub?.getObjectByName("head");
    const near = head ? built.camera.position.distanceTo(head.getWorldPosition(new THREE.Vector3())) : 1.2;
    const k = Math.max(0, Math.min(1, (t - 0.25) / 0.5));
    const e = k * k * (3 - 2 * k);
    const u = bokeh.uniforms as unknown as Record<string, { value: unknown }>;
    u.focus.value = far + (near - far) * e;
    composer.render();
  }

  dispose(): void {
    if (this.closed) return;
    this.rack?.composer.dispose();
    this.rack = null;
    this.closed = true;
    window.clearTimeout(this.timer);
    this.resolvers.splice(0).forEach((resolve) => resolve());
    this.generation += 1;
    this.built?.dispose();
    this.built = null;
    disposeEnvironment(this.renderer);
    this.renderer.dispose();
    try { this.renderer.forceContextLoss(); } catch { /* the context is already gone */ }
  }
}

/** A blank frame is a failed WebGL draw. A gray layout always has some contrast. */
export function frameIsBlank(renderer: THREE.WebGLRenderer): boolean {
  const gl = renderer.getContext();
  const w = gl.drawingBufferWidth;
  const h = gl.drawingBufferHeight;
  if (w < 2 || h < 2) return true;
  const pix = new Uint8Array(4);
  const pts: Array<[number, number]> = [
    [w * 0.5, h * 0.5],
    [w * 0.25, h * 0.3],
    [w * 0.75, h * 0.72],
    [w * 0.5, h * 0.15],
    [w * 0.5, h * 0.88],
  ];
  let min = 255;
  let max = 0;
  for (const [x, y] of pts) {
    gl.readPixels(x | 0, y | 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
    const l = (pix[0] + pix[1] + pix[2]) / 3;
    if (l < min) min = l;
    if (l > max) max = l;
  }
  return max - min < 6;
}
