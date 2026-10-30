// Builds one gray scene from a spec. `update(t)` is the only way it moves; t is 0–1.
import * as THREE from "three";
import { applyCamera, sampleCamera, steadyUp } from "./camera";
import type { CameraPose } from "./cameraMath";
import { shotTimeAt } from "./pathEdit";
import { createTopOverlay, type TopSubject } from "./topView";
import { beginShot } from "./library";
import { makeMats } from "./primitives";
import { dressDescribed } from "./places";
import { dress, type Dress } from "./templates";
import type { PrevizSpec } from "./spec";
import type { Actor } from "./primitives";
import { updateLeash } from "./runtime/attach.js";
import { applyGait } from "./runtime/gait.js";
import { applyDrive, applyMotion, driveSide } from "./runtime/motion.js";
import { groundObject } from "./runtime/poser.js";
import { texturedGround } from "./runtime/groundtex.js";

/** The Top view's map: its middle on the ground and its half-height, in metres. */
export interface TopFrame {
  cx: number;
  cz: number;
  half: number;
}

export interface Built {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  top: THREE.OrthographicCamera;
  actors: Actor[];
  /** The spec the frame is sampled from. A position nudge updates this without a rebuild. */
  live: { spec: PrevizSpec; pose: CameraPose | null; aim: CameraPose; t: number };
  update(t: number): void;
  /** Top view on or off. On, it frames the map (it never fits itself) and draws the path, points, camera and people. */
  showPath(on: boolean, aspect?: number, view?: { frame: TopFrame; mpp: number; selected: number | null }): void;
  /** Where the camera really goes, seen from above (the subject's travel included), shot time 0 to 1. */
  trajectory(steps?: number): Array<[number, number]>;
  /** Each key as drawn in Top view, and the shot time the camera reaches it. */
  drawnKeys(): Array<{ x: number; z: number; u: number }>;
  /** The key position (on the ground) that puts key `index` under (x, z) in Top view. */
  keyGround(index: number, x: number, z: number): [number, number];
  /** People and vehicles as Top view rings: where they are on this frame. */
  footprints(): Array<TopSubject & { index: number }>;
  /** Everything Top view should fit: the camera's path, its keys, and every subject from the first frame to the last. */
  mapPoints(): Array<[number, number]>;
  /** Does the camera ride with a subject that travels? */
  follows(): boolean;
  place(index: number, pos: [number, number, number]): void;
  dispose(): void;
}

function skyTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 64;
  const g = c.getContext("2d");
  if (g) {
    const grd = g.createLinearGradient(0, 0, 0, 64);
    grd.addColorStop(0, "#c2c2c2");
    grd.addColorStop(0.55, "#8e8e8e");
    grd.addColorStop(1, "#5c5c5c");
    g.fillStyle = grd;
    g.fillRect(0, 0, 4, 64);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** QC renders only. User camera presets are left alone. */
function qcSide(): boolean {
  return typeof window !== "undefined" && new URLSearchParams(window.location.search).get("qc") === "side";
}

/** Locked side-on camera, so a walk can be judged. Nothing here is drawn for a normal shot. */
function applyQcSide(camera: THREE.PerspectiveCamera, actors: Actor[], spec: PrevizSpec): void {
  const walker = actors.find((a) => {
    const gait = a.object.userData.gait as { animal?: boolean; crowd?: boolean; yaw?: number; start?: [number, number]; speed?: number; maxDist?: number; baseY?: number; rigY?: number; hipAboveRig?: number } | undefined;
    return gait && !gait.animal && !gait.crowd;
  });
  const gait = walker?.object.userData.gait as { yaw: number; start: [number, number]; speed: number; maxDist: number; baseY: number; rigY: number; hipAboveRig: number } | undefined;
  if (!walker || !gait) return;
  const yaw = gait.yaw;
  const fx = Math.sin(yaw);
  const fz = Math.cos(yaw);
  const sx = Math.cos(yaw);
  const sz = -Math.sin(yaw);
  const travel = Math.min(gait.maxDist, gait.speed * (spec.duration_s || 4));
  const midX = gait.start[0] + fx * travel * 0.5;
  const midZ = gait.start[1] + fz * travel * 0.5;
  const hipY = gait.baseY + gait.rigY + (gait.hipAboveRig || 0.9);
  const aspect = spec.size[0] / Math.max(1, spec.size[1]);
  const filmH = 36 / Math.max(aspect, 0.1);
  const halfV = Math.atan(filmH / 2 / 35);
  const body = 1.72;
  const dist = (body / 0.6) / (2 * Math.tan(halfV));
  camera.filmGauge = 36;
  camera.setFocalLength(35);
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
  camera.up.set(0, 1, 0);
  camera.position.set(midX + sx * dist, hipY, midZ + sz * dist);
  camera.lookAt(midX, hipY, midZ);
  const deg = 180 / Math.PI;
  const joint = (name: string) => walker.object.getObjectByName(name);
  const ang = (name: string) => {
    const j = joint(name);
    return j ? [j.rotation.x * deg, j.rotation.y * deg, j.rotation.z * deg] : [0, 0, 0];
  };
  const rig = joint("person_rig");
  const world = (name: string) => {
    const j = joint(name);
    if (!j) return [0, 0, 0];
    const p = new THREE.Vector3();
    j.getWorldPosition(p);
    return [p.x, p.y, p.z];
  };
  (window as Window & { __qc?: Record<string, unknown> }).__qc = {
    yaw,
    dist,
    hipY,
    cam: [camera.position.x, camera.position.y, camera.position.z],
    look: [midX, hipY, midZ],
    focal: camera.getFocalLength(),
    fov: camera.fov,
    rigY: rig?.position.y ?? 0,
    kneeL: ang("knee_L"),
    hipL: ang("hip_L"),
    shL: ang("shoulder_L"),
    elL: ang("elbow_L"),
    wristL: ang("wrist_L"),
    wristR: ang("wrist_R"),
    hips: ang("hips"),
    chest: ang("chest"),
    neck: ang("neck"),
    speed: gait.speed,
    stride: (walker.object.userData.gait as { stride?: number }).stride ?? 0,
    root: [walker.object.position.x, walker.object.position.y, walker.object.position.z],
    ankleL: world("ankle_L"),
    ankleR: world("ankle_R"),
  };
}

export function buildScene(spec: PrevizSpec, shot?: { root: THREE.Object3D; actors: THREE.Object3D[] } | null): Built {
  beginShot(spec);
  const mats = makeMats();
  const scene = new THREE.Scene();
  // A built shot gets its sky, fog and lights from createWorld. This is the flag-off look.
  let sky: THREE.Texture | null = null;
  let sun: THREE.DirectionalLight | null = null;
  let fogFar = 400;
  if (!shot) {
    sky = skyTexture();
    scene.background = sky;
    const [fogNear, far] = spec.environment.fog;
    fogFar = far;
    scene.fog = new THREE.Fog(0x8a8a8a, fogNear, fogFar);
    scene.add(new THREE.HemisphereLight(0xc8c8c8, 0x3a3a3a, 0.65));
    sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(40, 80, 30);
    sun.castShadow = true;
    const span = Math.min(180, Math.max(24, fogFar * 0.2));
    sun.shadow.camera.left = -span;
    sun.shadow.camera.right = span;
    sun.shadow.camera.top = span;
    sun.shadow.camera.bottom = -span;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = span * 4;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.bias = -0.0003;
    scene.add(sun);
  }

  if (!shot && spec.environment.ground === "plane") {
    const ground = new THREE.Mesh(new THREE.CircleGeometry(Math.max(80, fogFar), 48), texturedGround(mats.dark.clone()));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
  }

  const actors: Actor[] = [];
  const drives: THREE.Object3D[] = [];
  if (shot) shot.root.traverse((n) => { if (n.userData?.drive) drives.push(n); });
  if (drives.length) scene.userData.moves = true;
  const api: Dress = { add: (obj) => scene.add(obj), actors };
  if (shot) {
    scene.add(shot.root);
    for (const obj of shot.actors) {
      const gait = obj.userData.gait as { follow?: boolean; camShare?: number } | undefined;
      const board = obj.parent && (obj.parent as THREE.Object3D).userData?.cruise ? (obj.parent as THREE.Object3D) : null;
      const ride = board?.userData?.cruise as { speed: number; yaw: number; start: [number, number] } | undefined;
      const start = board ? board.position.clone() : obj.position.clone();
      if (gait || ride || obj.userData.motion) scene.userData.moves = true;
      actors.push({
        object: obj as THREE.Group,
        travel: 0,
        home: [start.x, start.y, start.z],
        origin: [start.x, start.y, start.z],
        tracks: ride ? true : gait ? gait.follow !== false : undefined,
        camShare: gait?.camShare,
        update(t, speed) {
          void speed;
          if (ride && board) {
            const dur = live.spec.duration_s || 4;
            const dist = ride.speed * Math.max(0, Math.min(1, t)) * dur;
            board.position.x = ride.start[0] + Math.sin(ride.yaw) * dist;
            board.position.z = ride.start[1] + Math.cos(ride.yaw) * dist;
            const slope = (ride as { slope?: { y0: number; tan: number } }).slope;
            if (slope) board.position.y = slope.y0 - dist * slope.tan;
            board.updateMatrixWorld(true);
            const wp = new THREE.Vector3();
            board.getWorldPosition(wp);
            this.origin = [wp.x, wp.y, wp.z];
            return;
          }
          if (obj.userData.motion) applyMotion(obj, t, live.spec.duration_s || 4);
          if (!obj.userData.gait) return;
          const pace = live.spec.subjects.find((s) => s.kind === "humanoid" && s.speed)?.speed ?? 1;
          applyGait(obj, t, pace, live.spec.duration_s || 4);
          const wade = obj.userData.wade as { shoreZ: number } | undefined;
          if (wade && typeof wade.shoreZ === "number") {
            // into the sea: the water rises up the legs
            const into = wade.shoreZ - obj.position.z;
            if (into > 0) obj.position.y -= Math.min(0.5, into * 0.16);
          }
          const p = obj.position;
          this.origin = [p.x, p.y, p.z];
        },
      });
    }
  } else if (spec.template === "described") dressDescribed(spec, mats, api);
  else dress(spec, mats, api);

  const camera = new THREE.PerspectiveCamera(35, spec.size[0] / spec.size[1], 0.05, 4000);
  const top = new THREE.OrthographicCamera(-20, 20, 20, -20, 0.1, 500);
  top.up.set(0, 0, -1);
  // editor-only graphics: never in an exported frame (runtime.setClean, tests/previz/step25_units.mjs)
  const overlay = createTopOverlay();
  for (const name of ["camera_marker", "camera_path", "footprints"]) {
    const o = overlay.group.getObjectByName(name);
    if (o) o.userData.overlay = true;
  }
  scene.add(overlay.group);
  const height = spec.subjects[0]?.height_m ?? 2;
  const near = Math.max(0.02, Math.min(0.35, height * 0.0012));
  const far = Math.max(fogFar * 3, 100);

  type Motion = { shift: [number, number, number]; along: number; posShare: number; followed: boolean };
  let lastMotion: Motion = { shift: [0, 0, 0], along: 0, posShare: 1, followed: false };
  /** How far the followed subject has travelled (the actors are already at this frame). */
  const motionOf = (current: PrevizSpec, driveShift: [number, number, number] | null): Motion => {
    let along = 0;
    let shift: [number, number, number] = [0, 0, 0];
    let followed = false;
    let posShare = 1;
    const deltaOf = (index: number): [number, number, number] | null => {
      const actor = actors[index];
      const home = actor?.home;
      const origin = actor?.origin;
      if (!actor || actor.tracks === false || !home || !origin) return null;
      return [origin[0] - home[0], origin[1] - home[1], origin[2] - home[2]];
    };
    current.subjects.forEach((s, i) => {
      if (s.anim !== "track" && !s.anim.endsWith("_follow")) return;
      const delta = deltaOf(i);
      const actor = actors[i];
      if (!delta || !actor) return;
      shift = delta;
      along = actor.travel;
      posShare = actor.camShare ?? 1;
      followed = true;
    });
    if (!followed && driveShift) {
      shift = driveShift;
      followed = true;
    }
    if (!followed) {
      current.subjects.forEach((_, i) => {
        if (followed) return;
        const delta = deltaOf(i);
        if (!delta || (delta[0] === 0 && delta[1] === 0 && delta[2] === 0)) return;
        shift = delta;
        posShare = actors[i]?.camShare ?? 1;
        followed = true;
      });
    }
    return { shift, along, posShare, followed: followed && (shift[0] !== 0 || shift[1] !== 0 || shift[2] !== 0 || along !== 0) };
  };
  /** The camera at shot time u: the keys, plus the subject's travel unless the user's keys stand still (free, no follow). */
  const poseAt = (current: PrevizSpec, u: number, m: Motion) => {
    const sample = sampleCamera(current, u);
    if (!current.camera.free || current.camera.follow) {
      let { shift, along } = m;
      const posShare = m.posShare;
      const segs = current.camera.segments;
      if (segs && segs.length) {
        let tracking = 1;
        let from = 0;
        for (const seg of segs) {
          tracking = seg.tracking ?? 1;
          from = seg.from_t;
          if (u <= seg.to_t + 1e-4) break;
        }
        if (tracking < 0.5) {
          const scale = u > 1e-4 ? from / u : 0;
          shift = [shift[0] * scale, shift[1] * scale, shift[2] * scale];
          along *= scale;
        }
      }
      sample.pos = [sample.pos[0] + shift[0] * posShare, sample.pos[1] + shift[1] * posShare, sample.pos[2] + (shift[2] + along) * posShare];
      sample.look = [sample.look[0] + shift[0], sample.look[1] + shift[1], sample.look[2] + shift[2] + along];
    }
    if (current.camera.frame) sample.frame = current.camera.frame;
    return sample;
  };

  const live: Built["live"] = { spec, pose: null, aim: { pos: [0, 1, 0], look: [0, 1, 1] }, t: 0 };
  const update = (t: number) => {
    const u = t < 0 ? 0 : t > 1 ? 1 : t;
    live.t = u;
    const current = live.spec;
    for (const actor of actors) actor.update(u, current.speed ?? 1);
    if (shot) {
      const w = u * Math.PI * 2;
      shot.root.traverse((n) => {
        const home = n.userData?.kiteHome as THREE.Vector3 | undefined;
        if (!n.userData?.kiteFloat || !home) return;
        n.position.set(home.x + Math.sin(w) * 0.16, home.y + Math.sin(w * 2) * 0.1, home.z + Math.cos(w) * 0.12);
      });
      shot.root.updateMatrixWorld(true);
      shot.root.traverse((n) => {
        if (n.userData?.leash) updateLeash(n);
      });
    }
    // timed drives (a chase through traffic); the camera rides with the one marked follow
    let driveShift: [number, number, number] | null = null;
    for (const n of drives) {
      const r = applyDrive(n, u, live.spec.duration_s || 4);
      const d = n.userData.drive as { follow?: boolean; delay?: number } | undefined;
      if (r && d?.follow) driveShift = [r.dx * 1, 0, r.dz * 1];
      if (r && d?.follow && d.delay) {
        const a = n.userData.drive as { yaw: number; lanes?: [number, number, number][] };
        // the camera takes the whole lane change a beat after the car, so the weave reads in frame and the camera
        // never sits in the lane the car has just left (where the car it is passing still is)
        const late = driveSide(a, u - d.delay / (live.spec.duration_s || 4));
        const rx = Math.cos(a.yaw), rz = -Math.sin(a.yaw);
        driveShift = [r.dx + rx * (r.side - late), 0, r.dz + rz * (r.side - late)];
      }
    }
    const m = motionOf(current, driveShift);
    lastMotion = m;
    const sample = poseAt(current, u, m);
    const held = live.pose;
    if (held) {
      sample.pos = [held.pos[0], held.pos[1], held.pos[2]];
      sample.look = [held.look[0], held.look[1], held.look[2]];
      delete sample.frame;
    }
    live.aim = { pos: [sample.pos[0], sample.pos[1], sample.pos[2]], look: [sample.look[0], sample.look[1], sample.look[2]] };
    if (typeof window !== "undefined") (window as Window & { __previzAim?: unknown }).__previzAim = { ...live.aim, shift: m.shift, keys: current.camera.keys, distance: current.camera.distance, held: !!live.pose };
    applyCamera(camera, sample, current.size[0] / current.size[1], near, far, held ? undefined : steadyUp(current, u));
    const skyMesh = scene.getObjectByName("world_sky");
    if (skyMesh) skyMesh.position.copy(camera.position);
    if (qcSide()) applyQcSide(camera, actors, current);
    for (const actor of actors) {
      const handles = actor.handles;
      const base = handles?.userData.length as number | undefined;
      if (!handles || !base) continue;
      const dist = camera.position.distanceTo(actor.object.position);
      const target = Math.min(0.9, Math.max(0.16, dist * 0.045));
      handles.scale.setScalar(target / base);
    }
  };
  update(0);

  // The subject's travel at 49 times through the shot, so Top view can draw where the camera really goes without
  // moving the people for every point it draws. Kept until a subject, the length or the pace changes.
  const STEPS = 48;
  let motionCache: { key: string; m: Motion[]; seen: Array<[number, number]> } | null = null;
  const motions = (): Motion[] => {
    const key = JSON.stringify([live.spec.duration_s, live.spec.speed ?? 1, live.spec.subjects.map((s) => [s.pos, s.speed ?? 1, s.anim])]);
    if (motionCache && motionCache.key === key) return motionCache.m;
    const keep = live.t;
    const held = live.pose;
    live.pose = null;
    const out: Motion[] = [];
    const seen: Array<[number, number]> = [];
    const v = new THREE.Vector3();
    for (let i = 0; i <= STEPS; i++) {
      update(i / STEPS);
      out.push(lastMotion);
      // where every person and vehicle is, in the world, every quarter of the shot (the map fits them all)
      if (i % 12 === 0) for (const a of actors.slice(0, Math.max(1, live.spec.subjects.length))) {
        a.object.getWorldPosition(v);
        seen.push([v.x, v.z]);
      }
    }
    live.pose = held;
    update(keep);
    motionCache = { key, m: out, seen };
    return out;
  };
  const motionAt = (u: number): Motion => {
    const ms = motions();
    const f = Math.min(STEPS, Math.max(0, u * STEPS));
    const i = Math.min(STEPS - 1, Math.floor(f));
    const w = f - i;
    const a = ms[i];
    const b = ms[i + 1];
    const mix = (x: number, y: number) => x + (y - x) * w;
    return {
      shift: [mix(a.shift[0], b.shift[0]), mix(a.shift[1], b.shift[1]), mix(a.shift[2], b.shift[2])],
      along: mix(a.along, b.along),
      posShare: mix(a.posShare, b.posShare),
      followed: a.followed || b.followed,
    };
  };
  const trajectory = (steps = 96): Array<[number, number]> => {
    const out: Array<[number, number]> = [];
    for (let i = 0; i <= steps; i++) {
      const u = i / steps;
      const p = poseAt(live.spec, u, motionAt(u)).pos;
      out.push([p[0], p[2]]);
    }
    return out;
  };
  const drawnKeys = () => {
    const cam = live.spec.camera;
    return cam.keys.map((k) => {
      const u = shotTimeAt(cam, k.t);
      const p = poseAt(live.spec, u, motionAt(u)).pos;
      return { x: p[0], z: p[2], u };
    });
  };
  const keyGround = (index: number, x: number, z: number): [number, number] => {
    const cam = live.spec.camera;
    const k = cam.keys[index];
    if (!k) return [x, z];
    const u = shotTimeAt(cam, k.t);
    const shown = poseAt(live.spec, u, motionAt(u)).pos;
    const bare = sampleCamera(live.spec, u).pos;
    const d = cam.distance ?? 1;
    const tx = x - (shown[0] - bare[0]);
    const tz = z - (shown[2] - bare[2]);
    return [k.look[0] + (tx - k.look[0]) / d, k.look[2] + (tz - k.look[2]) / d];
  };
  const kindOf = (o: THREE.Object3D, i: number): TopSubject["kind"] => {
    const ud = o.userData || {};
    const asset = String(ud.asset || ud.libraryId || "");
    const subj = live.spec.subjects[i]?.kind;
    if (ud.drive || subj === "car" || subj === "boat" || /car|suv|van|truck|bus|sedan|taxi|jeep|pickup|vehicle|boat|motor/i.test(asset)) return "vehicle";
    if (ud.gait || ud.asset === "person" || /^person|crowd/.test(asset) || subj === "humanoid") return "person";
    return "other";
  };
  const footprints = () => {
    const out: Array<TopSubject & { index: number }> = [];
    const v = new THREE.Vector3();
    const q = new THREE.Quaternion();
    actors.forEach((a, index) => {
      const kind = kindOf(a.object, index);
      if (kind === "other") return;
      a.object.getWorldPosition(v);
      a.object.getWorldQuaternion(q);
      const f = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
      out.push({ index, x: v.x, z: v.z, kind, yaw: Math.hypot(f.x, f.z) > 0.2 ? Math.atan2(f.x, f.z) : null });
    });
    return out;
  };
  const mapPoints = (): Array<[number, number]> => {
    const pts = trajectory(24);
    for (const k of drawnKeys()) pts.push([k.x, k.z]);
    motions();
    pts.push(...(motionCache?.seen ?? []));
    for (const f of footprints()) if (f.index < Math.max(1, live.spec.subjects.length)) pts.push([f.x, f.z]);
    return pts;
  };
  const showPath = (on: boolean, aspect = 1, view?: { frame: TopFrame; mpp: number; selected: number | null }) => {
    overlay.group.visible = on;
    if (!on || !view) return;
    const { frame, mpp, selected } = view;
    const wide = Math.max(0.2, aspect);
    top.left = -frame.half * wide;
    top.right = frame.half * wide;
    top.top = frame.half;
    top.bottom = -frame.half;
    top.position.set(frame.cx, 120, frame.cz);
    top.lookAt(frame.cx, 0, frame.cz);
    top.updateProjectionMatrix();
    const keys = drawnKeys();
    const aimYaw = Math.atan2(live.aim.look[0] - live.aim.pos[0], live.aim.look[2] - live.aim.pos[2]);
    const focal = camera.getFocalLength();
    overlay.draw({
      path: trajectory(96),
      points: keys.map((k, i) => ({ x: k.x, z: k.z, role: i === 0 ? "start" : i === keys.length - 1 ? "end" : "mid", selected: i === selected })),
      camera: { x: live.aim.pos[0], z: live.aim.pos[2], yaw: aimYaw, hfov: 2 * Math.atan(18 / Math.max(4, focal)) },
      subjects: footprints(),
      mpp,
      frame: { ...frame, aspect: wide },
    });
  };

  return {
    scene,
    camera,
    top,
    actors,
    live,
    update,
    showPath,
    trajectory,
    drawnKeys,
    keyGround,
    footprints,
    mapPoints,
    follows: () => motions().some((m) => m.followed) && (!live.spec.camera.free || !!live.spec.camera.follow),
    place(index, pos) {
      const actor = actors[index];
      if (!actor) return;
      motionCache = null;
      actor.origin = [pos[0], pos[1], pos[2]];
      if (live.spec.subjects[index]) live.spec.subjects[index].pos = [pos[0], pos[1], pos[2]];
      if (shot) {
        actor.object.position.set(pos[0], pos[1], pos[2]);
        actor.object.updateMatrixWorld(true);
        try { groundObject(actor.object, 0); } catch { /* a prop has no body to ground */ }
        const g = actor.object.userData.gait as { start?: [number, number]; baseY?: number } | undefined;
        if (g) {
          g.start = [actor.object.position.x, actor.object.position.z];
          g.baseY = actor.object.position.y;
        }
        actor.home = [actor.object.position.x, actor.object.position.y, actor.object.position.z];
        actor.origin = [...actor.home];
      }
    },
    dispose() {
      if (shot) {
        shot.root.removeFromParent();
        const wsky = scene.getObjectByName("world_sky") as THREE.Mesh | undefined;
        if (wsky) {
          wsky.geometry.dispose();
          const mat = wsky.material;
          if (!Array.isArray(mat)) mat.dispose();
        }
        if (scene.environment && !scene.environment.userData.shared) scene.environment.dispose();
      } else {
        scene.traverse((obj) => {
          const m = obj as THREE.Mesh;
          if (m.geometry) m.geometry.dispose();
        });
      }
      sky?.dispose();
      overlay.dispose();
      mats.dispose();
      sun?.shadow.map?.dispose();
    },
  };
}
