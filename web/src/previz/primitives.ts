// Gray stand-ins. No textures: a video model copies whatever pattern it sees.
import * as THREE from "three";
import { spinWheels, spinY, walk, type Limbs } from "./animate";
import { claimPerson, libraryOn, modelSize, poseNameFor, takeModel } from "./library";
import { attachProp } from "./runtime/attach.js";
import { setAnimalPose, setPersonScale, setPose } from "./runtime/poser.js";

export interface Mats {
  gray: THREE.MeshStandardMaterial;
  dark: THREE.MeshStandardMaterial;
  light: THREE.MeshStandardMaterial;
  emit: THREE.MeshStandardMaterial;
  /** Light stand-in body, black joints, and the blue / orange side marks from the design. */
  skin: THREE.MeshStandardMaterial;
  socket: THREE.MeshStandardMaterial;
  seam: THREE.MeshStandardMaterial;
  markBlue: THREE.MeshStandardMaterial;
  markOrange: THREE.MeshStandardMaterial;
}

export function makeMats(): Mats & { dispose: () => void } {
  const gray = new THREE.MeshStandardMaterial({ color: 0x8a8a8a, roughness: 0.88, metalness: 0 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.92, metalness: 0 });
  const light = new THREE.MeshStandardMaterial({ color: 0xc8c8c8, roughness: 0.8, metalness: 0 });
  const emit = new THREE.MeshStandardMaterial({ color: 0xe4e4e4, roughness: 0.5, metalness: 0, emissive: 0xe4e4e4, emissiveIntensity: 0.7 });
  const skin = new THREE.MeshStandardMaterial({ color: 0xd4d0c8, roughness: 0.7, metalness: 0 });
  const socket = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.42, metalness: 0.04 });
  const seam = new THREE.MeshStandardMaterial({ color: 0x5e5a54, roughness: 0.75, metalness: 0 });
  const markBlue = new THREE.MeshStandardMaterial({ color: 0x3c92d8, roughness: 0.5, metalness: 0 });
  const markOrange = new THREE.MeshStandardMaterial({ color: 0xe08c38, roughness: 0.5, metalness: 0 });
  return {
    gray, dark, light, emit, skin, socket, seam, markBlue, markOrange,
    dispose() {
      gray.dispose();
      dark.dispose();
      light.dispose();
      emit.dispose();
      skin.dispose();
      socket.dispose();
      seam.dispose();
      markBlue.dispose();
      markOrange.dispose();
    },
  };
}

export interface Actor {
  object: THREE.Group;
  /** Extra metres added to the camera when this subject is a tracking shot. */
  travel: number;
  /** Where the sentence put them. Dragging changes `origin`, not this. */
  home?: [number, number, number];
  /** Where the user has placed them. Travel is added on top. */
  origin?: [number, number, number];
  /** False keeps a walking actor in a locked-off frame. Unset keeps the old follow rule. */
  tracks?: boolean;
  /** Share of this actor's travel the camera body follows (the aim always follows). Undefined = all of it. */
  camShare?: number;
  handles?: THREE.Group;
  update(t: number, speed: number): void;
}

function mesh(geo: THREE.BufferGeometry, material: THREE.Material, shadow = true): THREE.Mesh {
  const m = new THREE.Mesh(geo, material);
  m.castShadow = shadow;
  m.receiveShadow = shadow;
  return m;
}

function segment(length: number, radius: number, material: THREE.Material): THREE.Mesh {
  const cap = mesh(new THREE.CapsuleGeometry(radius, Math.max(0.02, length - radius * 2), 6, 12), material);
  cap.position.y = -length / 2;
  return cap;
}

function joint(radius: number, material: THREE.Material): THREE.Mesh {
  return mesh(new THREE.SphereGeometry(radius, 22, 16), material);
}

interface Figure {
  bob: THREE.Group;
  leftLeg: THREE.Group;
  rightLeg: THREE.Group;
  leftArm: THREE.Group;
  rightArm: THREE.Group;
  forearm: THREE.Group;
  foreLen: number;
  bendKnees(): void;
  /** Elbows bent, arms off the ribs. A straight hang reads as a gingerbread man. */
  relax(): void;
}

/** Slightly tapered limb. A ball joint at each end hides the flat cap. */
function bone(length: number, radiusTop: number, radiusBottom: number, material: THREE.Material): THREE.Mesh {
  const m = mesh(new THREE.CylinderGeometry(radiusTop, radiusBottom, length, 24), material);
  m.position.y = -length / 2;
  return m;
}

/** Blue on the figure's right, orange on the left, front and back, as in the stand-in design. */
function sideMarks(length: number, radius: number, side: number, mats: Mats): THREE.Group {
  const g = new THREE.Group();
  const mark = side < 0 ? mats.markBlue : mats.markOrange;
  const w = Math.min(length * 0.028, radius * 0.28);
  for (const z of [1, -1]) {
    const line = mesh(new THREE.BoxGeometry(w, length * 0.7, w * 0.4), mark, false);
    line.position.set(0, -length / 2, z * (radius + w * 0.2));
    g.add(line);
  }
  return g;
}

/** A line that lies on a unit sphere, so it reads as drawn on the head rather than stuck in front of it. */
function sphereLine(points: THREE.Vector3[], radius: number, material: THREE.Material): THREE.Mesh {
  const curve = new THREE.CatmullRomCurve3(points);
  return mesh(new THREE.TubeGeometry(curve, 28, radius, 6, false), material, false);
}

function faceMarks(material: THREE.Material): THREE.Group {
  const g = new THREE.Group();
  const tube = 0.014;
  const down: THREE.Vector3[] = [];
  for (let i = 0; i <= 22; i++) {
    const phi = 0.08 + (1.28 - 0.08) * (i / 22);
    down.push(new THREE.Vector3(0, Math.cos(phi), Math.sin(phi) * 0.992));
  }
  const across: THREE.Vector3[] = [];
  const y = 0.18;
  for (let i = 0; i <= 18; i++) {
    const x = -0.46 + 0.92 * (i / 18);
    const z = Math.sqrt(Math.max(0.02, 1 - y * y - x * x));
    across.push(new THREE.Vector3(x, y, z * 0.992));
  }
  g.add(sphereLine(down, tube, material), sphereLine(across, tube, material));
  return g;
}

/** Push the chest forward and the seat back so the torso is not a spun vase. */
function sculptTorso(geo: THREE.BufferGeometry, H: number): void {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const fade = (t: number) => {
    const u = Math.min(1, Math.max(0, t));
    return u * u * (3 - 2 * u);
  };
  const band = (y: number, a: number, b: number, c: number, d: number) => {
    if (y <= a || y >= d) return 0;
    if (y < b) return fade((y - a) / (b - a));
    if (y > c) return fade((d - y) / (d - c));
    return 1;
  };
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const chest = band(y, H * 0.17, H * 0.22, H * 0.29, H * 0.34);
    const seat = band(y, H * 0.015, H * 0.045, H * 0.09, H * 0.14);
    let nz = z;
    if (z > 0) nz = z * (1 + chest * 0.1) + chest * H * 0.008;
    if (z < 0) nz = z * (1 + seat * 0.12) - seat * H * 0.006;
    pos.setZ(i, nz);
  }
  geo.computeVertexNormals();
}

/**
 * The stand-in from the design sheet: light body, black ball joints, a smooth
 * head with a cross, mitten hands, and a blue / orange side mark.
 */
function buildFigure(height: number, mats: Mats): Figure {
  const H = height;
  const hipY = H * 0.5;
  const shoulderY = H * 0.755;
  const bob = new THREE.Group();

  const legOf = (side: number) => {
    const hip = new THREE.Group();
    hip.position.set(side * H * 0.078, hipY, 0);
    hip.add(joint(H * 0.036, mats.socket));
    const thighLen = H * 0.23;
    hip.add(bone(thighLen, H * 0.032, H * 0.026, mats.skin));
    hip.add(sideMarks(thighLen, H * 0.028, side, mats));
    const shin = new THREE.Group();
    shin.position.y = -thighLen;
    shin.add(joint(H * 0.03, mats.socket));
    const shinLen = H * 0.22;
    shin.add(bone(shinLen, H * 0.024, H * 0.016, mats.skin));
    shin.add(sideMarks(shinLen, H * 0.02, side, mats));
    const ankle = joint(H * 0.02, mats.socket);
    ankle.position.y = -shinLen;
    const foot = mesh(new THREE.SphereGeometry(1, 20, 14), mats.skin);
    const fp = foot.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < fp.count; i++) {
      const x = fp.getX(i) * H * 0.026;
      const y = fp.getY(i);
      let z = fp.getZ(i) * H * 0.055;
      if (z > 0) z *= 1.12;
      fp.setXYZ(i, x, y < 0 ? y * H * 0.008 : y * H * 0.022, z);
    }
    foot.geometry.computeVertexNormals();
    foot.position.set(0, -shinLen - H * 0.018, H * 0.012);
    shin.add(ankle, foot);
    hip.add(shin);
    hip.userData.shin = shin;
    return hip;
  };
  const leftLeg = legOf(-1);
  const rightLeg = legOf(1);

  const torsoGeo = new THREE.LatheGeometry(
    [
      new THREE.Vector2(H * 0.018, 0),
      new THREE.Vector2(H * 0.055, H * 0.02),
      new THREE.Vector2(H * 0.072, H * 0.055),
      new THREE.Vector2(H * 0.066, H * 0.095),
      new THREE.Vector2(H * 0.052, H * 0.14),
      new THREE.Vector2(H * 0.048, H * 0.175),
      new THREE.Vector2(H * 0.078, H * 0.22),
      new THREE.Vector2(H * 0.098, H * 0.27),
      new THREE.Vector2(H * 0.112, H * 0.31),
      new THREE.Vector2(H * 0.064, H * 0.355),
      new THREE.Vector2(H * 0.03, H * 0.39),
    ],
    48,
  );
  sculptTorso(torsoGeo, H);
  const torso = mesh(torsoGeo, mats.skin);
  torso.position.y = hipY - H * 0.03;

  const neck = mesh(new THREE.CylinderGeometry(H * 0.022, H * 0.03, H * 0.045, 20), mats.skin);
  neck.position.y = hipY + H * 0.34;

  const head = new THREE.Group();
  head.position.y = hipY + H * 0.445;
  head.scale.set(H * 0.066, H * 0.078, H * 0.068);
  head.add(mesh(new THREE.SphereGeometry(1, 40, 30), mats.skin));
  head.add(faceMarks(mats.seam));

  const armOf = (side: number) => {
    const upper = new THREE.Group();
    upper.position.set(side * H * 0.132, shoulderY, 0);
    upper.add(joint(H * 0.032, mats.socket));
    const upperLen = H * 0.16;
    upper.add(bone(upperLen, H * 0.02, H * 0.017, mats.skin));
    upper.add(sideMarks(upperLen, H * 0.018, side, mats));
    const fore = new THREE.Group();
    fore.position.y = -upperLen;
    fore.add(joint(H * 0.022, mats.socket));
    const foreLen = H * 0.145;
    fore.add(bone(foreLen, H * 0.016, H * 0.013, mats.skin));
    fore.add(sideMarks(foreLen, H * 0.014, side, mats));
    const wrist = joint(H * 0.016, mats.socket);
    wrist.position.y = -foreLen;
    const palm = mesh(new THREE.SphereGeometry(1, 16, 12), mats.skin);
    const pp = palm.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pp.count; i++) {
      pp.setXYZ(i, pp.getX(i) * H * 0.014, pp.getY(i) * H * 0.034, pp.getZ(i) * H * 0.01);
    }
    palm.geometry.computeVertexNormals();
    palm.position.y = -foreLen - H * 0.03;
    const thumb = mesh(new THREE.SphereGeometry(H * 0.009, 10, 8), mats.skin);
    thumb.scale.set(0.7, 1.25, 0.7);
    thumb.position.set(-side * H * 0.012, -foreLen - H * 0.01, H * 0.004);
    fore.add(wrist, palm, thumb);
    upper.add(fore);
    upper.userData.fore = fore;
    upper.userData.foreLen = foreLen;
    return upper;
  };
  const leftArm = armOf(-1);
  const rightArm = armOf(1);
  const rightFore = rightArm.userData.fore as THREE.Group;
  const foreLen = rightArm.userData.foreLen as number;

  bob.add(leftLeg, rightLeg, torso, neck, head, leftArm, rightArm);
  const bendKnees = () => {
    for (const leg of [leftLeg, rightLeg]) {
      const shin = leg.userData.shin as THREE.Object3D | undefined;
      if (shin) shin.rotation.x = Math.max(0, -leg.rotation.x) * 0.7;
    }
  };
  const relax = () => {
    leftArm.rotation.z = -0.16;
    rightArm.rotation.z = 0.16;
    (leftArm.userData.fore as THREE.Group).rotation.x = -0.1;
    rightFore.rotation.x = -0.1;
  };
  return { bob, leftLeg, rightLeg, leftArm, rightArm, forearm: rightFore, foreLen, bendKnees, relax };
}

export function humanoid(height: number, mats: Mats, moving: boolean): Actor {
  const g = new THREE.Group();
  const figure = buildFigure(height, mats);
  g.add(figure.bob);
  const limbs: Limbs = figure;
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      if (!moving) return;
      walk(limbs, t, speed, height);
      figure.relax();
      figure.bendKnees();
    },
  };
}

export function car(height: number, mats: Mats, metres: number): Actor {
  const lib = libraryFigure("car", height, ["wheel_FL", "wheel_FR", "wheel_BL", "wheel_BR"]);
  if (lib) {
    lib.travel = metres;
    return lib;
  }
  const g = new THREE.Group();
  const w = height * 1.12;
  const len = height * 2.65;
  const body = mesh(new THREE.BoxGeometry(w, height * 0.28, len), mats.gray);
  body.position.set(0, height * 0.36, 0);
  const skirt = mesh(new THREE.BoxGeometry(w * 1.02, height * 0.08, len * 0.78), mats.dark);
  skirt.position.set(0, height * 0.22, 0);
  const hood = mesh(new THREE.BoxGeometry(w * 0.92, height * 0.08, len * 0.32), mats.light);
  hood.position.set(0, height * 0.52, len * 0.26);
  const cabin = mesh(new THREE.BoxGeometry(w * 0.84, height * 0.28, len * 0.38), mats.light);
  cabin.position.set(0, height * 0.6, -len * 0.04);
  const glass = mesh(new THREE.BoxGeometry(w * 0.72, height * 0.16, len * 0.32), mats.dark);
  glass.position.set(0, height * 0.64, -len * 0.04);
  const bumper = mesh(new THREE.BoxGeometry(w * 0.9, height * 0.06, height * 0.08), mats.dark);
  bumper.position.set(0, height * 0.28, len * 0.5);
  g.add(body, skirt, hood, cabin, glass, bumper);
  for (const x of [-1, 1]) {
    const lamp = mesh(new THREE.SphereGeometry(height * 0.06, 8, 6), mats.light);
    lamp.position.set(x * w * 0.32, height * 0.4, len * 0.5);
    const mirror = mesh(new THREE.BoxGeometry(height * 0.04, height * 0.08, height * 0.14), mats.dark);
    mirror.position.set(x * w * 0.5, height * 0.62, len * 0.08);
    g.add(lamp, mirror);
  }
  const pivots: THREE.Object3D[] = [];
  const wr = height * 0.18;
  for (const [x, z] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    const pivot = new THREE.Group();
    pivot.position.set(x * w * 0.46, wr, z * len * 0.32);
    const wheel = mesh(new THREE.CylinderGeometry(wr, wr, wr * 0.4, 16), mats.dark);
    wheel.rotation.z = Math.PI / 2;
    const hub = mesh(new THREE.CylinderGeometry(wr * 0.38, wr * 0.38, wr * 0.44, 10), mats.light);
    hub.rotation.z = Math.PI / 2;
    pivot.add(wheel, hub);
    pivots.push(pivot);
    g.add(pivot);
  }
  return {
    object: g,
    travel: metres,
    update(t, speed) {
      spinWheels(pivots, t, speed);
    },
  };
}

export function product(height: number, mats: Mats): Actor {
  const g = new THREE.Group();
  const body = mesh(new THREE.BoxGeometry(height, height * 0.72, height), mats.light);
  body.position.y = height * 0.36;
  const cap = mesh(new THREE.SphereGeometry(height * 0.28, 16, 12), mats.gray);
  cap.position.y = height * 0.86;
  g.add(body, cap);
  return { object: g, travel: 0, update: (t, speed) => spinY(g, t, speed * 0.5) };
}

export function house(mats: Mats, w = 8, h = 6, d = 7): THREE.Group {
  const lib = fit("house", h, w, d);
  if (lib) return lib;
  const g = new THREE.Group();
  const walls = mesh(new THREE.BoxGeometry(w, h, d), mats.gray);
  walls.position.y = h / 2;
  const roof = mesh(new THREE.ConeGeometry(w * 0.72, h * 0.45, 4), mats.dark);
  roof.position.y = h + h * 0.18;
  roof.rotation.y = Math.PI / 4;
  g.add(walls, roof);
  return g;
}

export function tree(mats: Mats, h = 7): THREE.Group {
  const lib = fit("tree", h);
  if (lib) return lib;
  const g = new THREE.Group();
  const trunk = mesh(new THREE.CylinderGeometry(h * 0.028, h * 0.075, h * 0.42, 8), mats.dark);
  trunk.position.y = h * 0.21;
  const limb = (x: number, y: number, z: number, len: number, lean: number, yaw: number) => {
    const branch = mesh(new THREE.CylinderGeometry(h * 0.012, h * 0.028, len, 6), mats.dark);
    branch.position.set(x, y, z);
    branch.rotation.z = lean;
    branch.rotation.y = yaw;
    return branch;
  };
  const puff = (x: number, y: number, z: number, r: number, mat = mats.gray) => {
    const crown = mesh(new THREE.SphereGeometry(r, 12, 9), mat);
    crown.position.set(x, y, z);
    crown.scale.set(1.15, 0.72, 1);
    g.add(crown);
  };
  g.add(
    trunk,
    limb(h * 0.06, h * 0.4, 0, h * 0.2, -0.7, 0.2),
    limb(-h * 0.05, h * 0.36, h * 0.02, h * 0.16, 0.65, -0.4),
  );
  puff(0, h * 0.58, 0, h * 0.22);
  puff(-h * 0.16, h * 0.5, h * 0.06, h * 0.16, mats.dark);
  puff(h * 0.17, h * 0.52, -h * 0.08, h * 0.17);
  puff(h * 0.04, h * 0.68, h * 0.05, h * 0.14, mats.light);
  puff(-h * 0.08, h * 0.62, -h * 0.1, h * 0.13, mats.dark);
  puff(h * 0.1, h * 0.46, h * 0.1, h * 0.12);
  return g;
}

export function building(mats: Mats, w: number, h: number, d: number): THREE.Group {
  const lib = fit("building", h, w, d);
  if (lib) return lib;
  const g = new THREE.Group();
  const block = mesh(new THREE.BoxGeometry(w, h, d), mats.gray);
  block.position.y = h / 2;
  g.add(block);
  const floors = Math.max(2, Math.floor(h / 4));
  for (let i = 1; i < floors; i++) {
    const band = mesh(new THREE.BoxGeometry(w + 0.08, 0.18, d + 0.08), mats.dark, false);
    band.position.y = (i / floors) * h;
    g.add(band);
  }
  return g;
}

export function cloud(mats: Mats, scale: number): THREE.Group {
  const g = new THREE.Group();
  const puff = (x: number, y: number, z: number, r: number) => {
    const s = mesh(new THREE.SphereGeometry(r, 8, 6), mats.light, false);
    s.position.set(x, y, z);
    g.add(s);
  };
  puff(0, 0, 0, scale);
  puff(scale * 0.8, scale * 0.15, 0, scale * 0.7);
  puff(-scale * 0.7, scale * 0.05, scale * 0.2, scale * 0.6);
  return g;
}

export function road(mats: Mats, length: number, width = 12): THREE.Mesh {
  const m = mesh(new THREE.BoxGeometry(width, 0.08, length), mats.dark);
  m.position.y = 0.04;
  return m;
}

export function laneLine(mats: Mats): THREE.Mesh {
  return mesh(new THREE.BoxGeometry(0.18, 0.03, 2.6), mats.light, false);
}

export function ring(mats: Mats, radius: number): THREE.Mesh {
  const m = mesh(new THREE.TorusGeometry(radius, 0.07, 8, 28), mats.emit, false);
  return m;
}

function sipLift(t: number): number {
  if (t < 0.4 || t > 0.92) return 0;
  const u = (t - 0.4) / 0.52;
  return Math.sin(Math.min(1, u) * Math.PI);
}

function fit(id: string, height?: number, w?: number, d?: number): THREE.Group | null {
  const model = takeModel(id);
  if (!model) return null;
  const size = modelSize(id);
  if (size && size.y > 0 && (height || w || d)) {
    const uniform = height && !w && !d ? height / size.y : 0;
    const sx = w && size.x > 0 ? w / size.x : uniform || 1;
    const sy = height && size.y > 0 ? height / size.y : 1;
    const sz = d && size.z > 0 ? d / size.z : uniform || 1;
    model.scale.set(sx, sy, sz);
  }
  return model;
}

function libraryFigure(id: string, height: number, wheelNames: string[]): Actor | null {
  const model = fit(id, height);
  if (!model) return null;
  const g = new THREE.Group();
  g.add(model);
  const wheels = wheelNames.map((name) => model.getObjectByName(name)).filter((node): node is THREE.Object3D => !!node);
  const rest = wheels.map((wheel) => wheel.rotation.x);
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      const spin = t * Math.max(0.05, speed) * Math.PI * 28;
      wheels.forEach((wheel, i) => {
        wheel.rotation.x = rest[i] + spin;
      });
    },
  };
}

/** Gray body. `anim` is stand, walk, sit, sit_sip, or stand_sip. A sip parents a cup to the right hand. */
export function person(height: number, mats: Mats, anim: string): Actor {
  const posed = libraryOn() ? takeModel(claimPerson()) : null;
  if (posed) {
    const g = new THREE.Group();
    g.add(posed);
    setPersonScale(posed, height >= 1.5 ? 1 : 0.7);
    const pose = poseNameFor(anim);
    setPose(posed, pose);
    const propId = pose === "sip" ? "cup" : pose === "hold_phone" ? "phone" : "";
    const prop = propId ? takeModel(propId) : null;
    if (prop) attachProp(posed, prop, "attach_hand_R", { upright: pose === "sip" });
    return { object: g, travel: 0, update() {} };
  }
  const g = new THREE.Group();
  const figure = buildFigure(height, mats);
  g.add(figure.bob);
  const { bob, leftLeg, rightLeg, leftArm, rightArm, forearm, foreLen } = figure;
  const r = height * 0.046;
  const sipping = anim === "sit_sip" || anim === "stand_sip";
  let cup: THREE.Object3D | null = null;
  if (anim === "phone") {
    const handset = mesh(new THREE.BoxGeometry(r * 0.55, r * 1.15, r * 0.12), mats.dark);
    handset.position.set(0, -foreLen, r * 0.35);
    forearm.add(handset);
  }
  if (sipping) {
    cup = mesh(new THREE.CylinderGeometry(r * 0.48, r * 0.38, r * 1.2, 12), mats.light);
    cup.position.set(0, -foreLen * 0.92, 0);
    forearm.add(cup);
  }
  const limbs: Limbs = { leftLeg, rightLeg, leftArm, rightArm, bob };
  const bend = () => figure.bendKnees();
  const sitting = anim === "sit" || anim === "sit_sip";
  const gait = anim.startsWith("run") ? "run" : anim.startsWith("jog") ? "jog" : anim.startsWith("walk") ? "walk" : null;
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      limbs.bob.rotation.set(0, 0, 0);
      limbs.bob.position.set(0, 0, 0);
      limbs.leftLeg.rotation.set(0, 0, 0);
      limbs.rightLeg.rotation.set(0, 0, 0);
      limbs.leftArm.rotation.set(0, 0, 0);
      limbs.rightArm.rotation.set(0, 0, 0);
      forearm.rotation.set(0, 0, 0);
      if (cup) cup.rotation.set(0, 0, 0);
      if (gait) {
        walk(limbs, t, speed, height, gait);
        figure.relax();
        bend();
        return;
      }
      if (anim === "jump") {
        const crouch = t < 0.18 ? t / 0.18 : t > 0.78 ? (t - 0.78) / 0.22 : 0;
        const air = t >= 0.18 && t <= 0.78 ? Math.sin(((t - 0.18) / 0.6) * Math.PI) : 0;
        const tuck = (1 - air) * (0.35 + crouch);
        limbs.leftLeg.rotation.x = -0.9 * tuck;
        limbs.rightLeg.rotation.x = limbs.leftLeg.rotation.x;
        limbs.leftArm.rotation.x = -0.4 - air * 1.5;
        limbs.rightArm.rotation.x = -0.4 - air * 1.5;
        limbs.bob.position.y = air * height * 0.32 - crouch * height * 0.1;
        bend();
        return;
      }
      if (anim === "lay") {
        limbs.bob.rotation.x = Math.PI / 2;
        limbs.bob.position.set(0, height * 0.16, height * 0.42);
        bend();
        return;
      }
      if (anim === "kneel") {
        limbs.leftLeg.rotation.x = -1.55;
        limbs.rightLeg.rotation.x = -0.35;
        limbs.bob.position.y = -height * 0.32;
        bend();
        return;
      }
      if (anim === "crouch") {
        limbs.leftLeg.rotation.x = -1.15;
        limbs.rightLeg.rotation.x = -1.15;
        limbs.bob.position.y = -height * 0.28;
        bend();
        return;
      }
      if (anim === "wave") {
        limbs.rightArm.rotation.x = -2.4;
        limbs.rightArm.rotation.z = Math.sin(t * Math.PI * 8) * 0.45;
        bend();
        return;
      }
      if (sitting) {
        limbs.leftLeg.rotation.x = -1.45;
        limbs.rightLeg.rotation.x = -1.45;
        limbs.bob.position.y = -height * 0.46;
        limbs.leftArm.rotation.x = -0.35;
        if (sipping) holdCup(true, sipLift(t));
        else limbs.rightArm.rotation.x = -0.3;
        bend();
        return;
      }
      if (anim === "phone") {
        limbs.rightArm.rotation.x = -0.7;
        forearm.rotation.x = -1.1;
        bend();
        return;
      }
      if (sipping) holdCup(false, sipLift(t));
      else figure.relax();
      bend();

      function holdCup(seated: boolean, lift: number) {
        limbs.rightArm.rotation.x = seated ? -0.55 : -0.35;
        limbs.rightArm.rotation.z = seated ? -0.55 : -0.35;
        forearm.rotation.x = (seated ? -1.25 : -1.15) - lift * 0.7;
        if (cup) cup.rotation.x = -(limbs.rightArm.rotation.x + forearm.rotation.x);
      }
    },
  };
}

function animalLeg(len: number, radius: number, mats: Mats): THREE.Group {
  const hip = new THREE.Group();
  hip.add(segment(len, radius, mats.gray));
  const paw = mesh(new THREE.SphereGeometry(radius * 1.15, 10, 8), mats.dark);
  paw.scale.set(1, 0.55, 1.3);
  paw.position.y = -len;
  hip.add(paw);
  return hip;
}

function animalPoseName(pose: boolean | string | undefined, fallback: string): string {
  if (pose === true || pose === "sit" || pose === "sitting") return "sit";
  if (pose === false || pose == null || pose === "") return fallback;
  const name = pose.replace(/_follow$/, "");
  if (name === "jog" || name === "run") return "run";
  if (name === "lie" || name === "lay" || name === "lie_down") return "lie";
  if (name === "walk" || name === "stand" || name === "sit") return name;
  return "stand";
}

export function dog(height: number, mats: Mats, sitting: boolean | string = false): Actor {
  const posed = animalPoseName(sitting, "walk");
  const model = fit("dog", height);
  if (model) {
    setAnimalPose(model, posed);
    const g = new THREE.Group();
    g.add(model);
    return { object: g, travel: 0, update() {} };
  }
  const g = new THREE.Group();
  const bodyLen = height * 1.05;
  const bodyR = height * 0.28;
  const body = mesh(new THREE.CapsuleGeometry(bodyR, Math.max(0.05, bodyLen - bodyR * 2), 8, 14), mats.gray);
  body.rotation.x = Math.PI / 2;
  body.position.set(0, posed === "sit" ? height * 0.42 : height * 0.58, 0);
  const chest = mesh(new THREE.SphereGeometry(bodyR * 1.05, 14, 12), mats.gray);
  chest.position.set(0, body.position.y + height * 0.04, bodyLen * 0.28);
  const head = mesh(new THREE.SphereGeometry(height * 0.26, 16, 12), mats.light);
  head.position.set(0, body.position.y + height * 0.22, bodyLen * 0.52);
  const snout = mesh(new THREE.CapsuleGeometry(height * 0.09, height * 0.16, 6, 8), mats.gray);
  snout.rotation.x = Math.PI / 2;
  snout.position.set(0, head.position.y - height * 0.08, head.position.z + height * 0.22);
  const nose = mesh(new THREE.SphereGeometry(height * 0.045, 8, 6), mats.dark);
  nose.position.set(0, snout.position.y, snout.position.z + height * 0.12);
  g.add(body, chest, head, snout, nose);
  for (const x of [-1, 1]) {
    const ear = mesh(new THREE.SphereGeometry(height * 0.1, 10, 8), mats.dark);
    ear.scale.set(0.55, 1.3, 0.7);
    ear.position.set(x * height * 0.14, head.position.y + height * 0.08, head.position.z - height * 0.02);
    ear.rotation.z = x * 0.5;
    g.add(ear);
  }
  const legs: THREE.Object3D[] = [];
  const legLen = posed === "sit" ? height * 0.22 : height * 0.42;
  for (const [x, z] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    const leg = animalLeg(legLen, height * 0.07, mats);
    leg.position.set(x * height * 0.16, body.position.y - height * 0.02, z * bodyLen * 0.32);
    legs.push(leg);
    g.add(leg);
  }
  const tail = new THREE.Group();
  tail.position.set(0, body.position.y + height * 0.08, -bodyLen * 0.48);
  const tailMesh = mesh(new THREE.CapsuleGeometry(height * 0.045, height * 0.28, 5, 8), mats.dark);
  tailMesh.rotation.x = 1.1;
  tailMesh.position.set(0, height * 0.12, -height * 0.08);
  tail.add(tailMesh);
  g.add(tail);
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      if (posed === "sit") return;
      const swing = Math.sin(t * Math.max(0.05, speed) * Math.PI * 8);
      legs.forEach((leg, i) => {
        const pair = i === 0 || i === 3;
        leg.rotation.x = (pair ? swing : -swing) * 0.55;
      });
      tail.rotation.y = swing * 0.35;
      tail.rotation.x = Math.abs(swing) * 0.15;
    },
  };
}

export function cat(height: number, mats: Mats, pose?: string): Actor {
  const model = fit("cat", height);
  if (model) {
    setAnimalPose(model, animalPoseName(pose, "stand"));
    const g = new THREE.Group();
    g.add(model);
    const tail = model.getObjectByName("tail");
    const rest = tail?.rotation.z ?? 0;
    return {
      object: g,
      travel: 0,
      update(t, speed) {
        if (!tail) return;
        tail.rotation.z = rest + Math.sin(t * Math.max(0.05, speed) * Math.PI * 6) * 0.2;
      },
    };
  }
  const g = new THREE.Group();
  const body = mesh(new THREE.CapsuleGeometry(height * 0.22, height * 0.7, 8, 12), mats.gray);
  body.rotation.x = Math.PI / 2;
  body.position.set(0, height * 0.48, 0);
  const head = mesh(new THREE.SphereGeometry(height * 0.26, 16, 12), mats.light);
  head.scale.set(1, 0.92, 0.9);
  head.position.set(0, height * 0.7, height * 0.46);
  const cheek = mesh(new THREE.SphereGeometry(height * 0.1, 10, 8), mats.gray);
  cheek.scale.set(0.8, 0.6, 1.2);
  cheek.position.set(0, height * 0.62, height * 0.64);
  g.add(body, head, cheek);
  for (const x of [-1, 1]) {
    const ear = mesh(new THREE.ConeGeometry(height * 0.08, height * 0.2, 5), mats.dark);
    ear.position.set(x * height * 0.12, height * 1.02, height * 0.46);
    g.add(ear);
  }
  const tail = new THREE.Group();
  tail.position.set(0, height * 0.55, -height * 0.48);
  const tailA = mesh(new THREE.CapsuleGeometry(height * 0.04, height * 0.28, 5, 8), mats.dark);
  tailA.position.y = height * 0.16;
  tailA.rotation.x = -0.4;
  const tailB = mesh(new THREE.CapsuleGeometry(height * 0.03, height * 0.22, 5, 8), mats.gray);
  tailB.position.set(0, height * 0.38, height * 0.06);
  tailB.rotation.x = -0.8;
  tail.add(tailA, tailB);
  g.add(tail);
  const legs: THREE.Object3D[] = [];
  for (const [x, z] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
    const leg = animalLeg(height * 0.36, height * 0.045, mats);
    leg.position.set(x * height * 0.14, height * 0.42, z * height * 0.32);
    legs.push(leg);
    g.add(leg);
  }
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      const swing = Math.sin(t * Math.max(0.05, speed) * Math.PI * 6);
      tail.rotation.z = swing * 0.2;
      legs.forEach((leg, i) => {
        leg.rotation.x = (i < 2 ? swing : -swing) * 0.25;
      });
    },
  };
}

export function bicycle(height: number, mats: Mats, rider: boolean, moto = false): Actor {
  const lib = libraryFigure("bicycle", height, ["wheel_F", "wheel_B"]);
  if (lib) {
    if (rider) {
      const riderActor = person(height, mats, "sit");
      const seat = lib.object.getObjectByName("attach_seat");
      (seat ?? lib.object).add(riderActor.object);
      const spin = lib.update.bind(lib);
      lib.update = (t, speed) => {
        spin(t, speed);
        riderActor.update(t, speed);
      };
    }
    return lib;
  }
  const g = new THREE.Group();
  const wheelR = height * (moto ? 0.22 : 0.28);
  const pivots: THREE.Object3D[] = [];
  for (const z of [-0.55, 0.6]) {
    const pivot = new THREE.Group();
    pivot.position.set(0, wheelR, z * height);
    const wheel = mesh(new THREE.TorusGeometry(wheelR, wheelR * 0.16, 8, 16), mats.dark);
    wheel.rotation.y = Math.PI / 2;
    pivot.add(wheel);
    pivots.push(pivot);
    g.add(pivot);
  }
  const frame = mesh(new THREE.BoxGeometry(height * 0.08, height * 0.08, height * 1.15), mats.gray);
  frame.position.set(0, wheelR + height * 0.22, 0);
  frame.rotation.x = -0.35;
  const bar = mesh(new THREE.BoxGeometry(height * 0.46, height * 0.05, height * 0.05), mats.dark);
  bar.position.set(0, wheelR + height * 0.55, height * 0.42);
  const seat = mesh(new THREE.BoxGeometry(height * 0.12, height * 0.05, height * 0.28), mats.light);
  seat.position.set(0, wheelR + height * 0.48, -height * 0.15);
  g.add(frame, bar, seat);
  let riderActor: Actor | null = null;
  if (rider) {
    riderActor = person(height, mats, "sit");
    riderActor.object.position.set(0, wheelR + height * 0.5, -height * 0.02);
    g.add(riderActor.object);
  }
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      spinWheels(pivots, t, speed);
      riderActor?.update(t, speed);
    },
  };
}

export function skateboard(height: number, mats: Mats, rider: boolean): Actor {
  const lib = libraryFigure("skateboard", height, ["truck_F", "truck_B"]);
  if (lib) {
    if (rider) {
      const riderActor = person(height, mats, "stand");
      riderActor.object.position.y = modelSize("skateboard")?.y ?? 0.1;
      lib.object.add(riderActor.object);
      const spin = lib.update.bind(lib);
      lib.update = (t, speed) => {
        spin(t, speed);
        riderActor.update(t, speed);
        lib.object.rotation.z = Math.sin(t * Math.max(0.2, speed) * Math.PI * 2) * 0.04;
      };
    }
    return lib;
  }
  const g = new THREE.Group();
  const deck = mesh(new THREE.BoxGeometry(height * 0.22, height * 0.04, height * 0.7), mats.dark);
  deck.position.y = 0.08;
  g.add(deck);
  for (const z of [-0.22, 0.22]) {
    const truck = mesh(new THREE.BoxGeometry(height * 0.2, 0.04, 0.05), mats.gray);
    truck.position.set(0, 0.04, z * height);
    g.add(truck);
  }
  let riderActor: Actor | null = null;
  if (rider) {
    riderActor = person(height, mats, "stand");
    riderActor.object.position.y = 0.1;
    g.add(riderActor.object);
  }
  return {
    object: g,
    travel: 0,
    update(t, speed) {
      riderActor?.update(t, speed);
      g.rotation.z = Math.sin(t * Math.max(0.2, speed) * Math.PI * 2) * 0.04;
    },
  };
}

export function boat(mats: Mats, rider: boolean): Actor {
  const model = takeModel("boat");
  if (model) {
    const g = new THREE.Group();
    g.add(model);
    let riderActor: Actor | null = null;
    if (rider) {
      riderActor = person(1.75, mats, "stand");
      const seat = model.getObjectByName("attach_seat");
      (seat ?? g).add(riderActor.object);
    }
    return { object: g, travel: 0, update(t, speed) { riderActor?.update(t, speed); } };
  }
  const g = new THREE.Group();
  const hull = mesh(new THREE.BoxGeometry(1.5, 0.45, 3.2), mats.gray);
  hull.position.y = 0.28;
  const bow = mesh(new THREE.BoxGeometry(1.1, 0.28, 0.8), mats.light);
  bow.position.set(0, 0.42, 1.7);
  g.add(hull, bow);
  let riderActor: Actor | null = null;
  if (rider) {
    riderActor = person(1.75, mats, "stand");
    riderActor.object.position.set(0, 0.45, -0.2);
    g.add(riderActor.object);
  }
  return { object: g, travel: 0, update(t, speed) { riderActor?.update(t, speed); } };
}

export function fountain(mats: Mats): THREE.Group {
  const lib = takeModel("fountain");
  if (lib) return lib;
  const g = new THREE.Group();
  const basin = mesh(new THREE.CylinderGeometry(1.3, 1.45, 0.45, 18), mats.gray);
  basin.position.y = 0.28;
  const column = mesh(new THREE.CylinderGeometry(0.18, 0.22, 1.3, 10), mats.light);
  column.position.y = 1.05;
  const top = mesh(new THREE.SphereGeometry(0.28, 12, 10), mats.light);
  top.position.y = 1.75;
  g.add(basin, column, top);
  return g;
}

export function table(mats: Mats): THREE.Group {
  const lib = takeModel("table");
  if (lib) return lib;
  const g = new THREE.Group();
  const top = mesh(new THREE.BoxGeometry(1.4, 0.08, 0.8), mats.gray);
  top.position.y = 0.74;
  g.add(top);
  for (const [x, z] of [[-0.55, -0.28], [0.55, -0.28], [-0.55, 0.28], [0.55, 0.28]] as const) {
    const leg = mesh(new THREE.BoxGeometry(0.07, 0.74, 0.07), mats.dark);
    leg.position.set(x, 0.37, z);
    g.add(leg);
  }
  return g;
}

export function bench(mats: Mats): THREE.Group {
  const lib = takeModel("bench");
  if (lib) return lib;
  const g = new THREE.Group();
  const seat = mesh(new THREE.BoxGeometry(1.5, 0.08, 0.48), mats.gray);
  seat.position.set(0, 0.46, 0.02);
  const back = mesh(new THREE.BoxGeometry(1.5, 0.55, 0.08), mats.light);
  back.position.set(0, 0.82, -0.2);
  g.add(seat, back);
  for (const x of [-0.62, 0.62]) {
    const leg = mesh(new THREE.BoxGeometry(0.08, 0.46, 0.42), mats.dark);
    leg.position.set(x, 0.23, 0.02);
    g.add(leg);
  }
  return g;
}

export function pedestal(mats: Mats, topY: number): THREE.Group {
  const g = new THREE.Group();
  const col = mesh(new THREE.CylinderGeometry(0.35, 0.42, topY, 16), mats.dark);
  col.position.y = topY / 2;
  const top = mesh(new THREE.CylinderGeometry(0.7, 0.7, 0.08, 20), mats.gray);
  top.position.y = topY;
  g.add(col, top);
  return g;
}
