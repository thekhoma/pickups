// What Top view draws over the scene: the camera path, its points, the camera (a wedge that faces where the lens looks,
// with a turn handle), a ring on each person or vehicle, and a faint ground grid. All of it is editor-only
// (userData.overlay), so no exported frame ever shows it. Sizes are in screen pixels: mpp is metres per pixel.
import * as THREE from "three";

export const IVORY = 0xeee9e0;
const DARK = 0x0d0c0b;
const PERSON = 0xe8c547;
const VEHICLE = 0xb8b4ac;

/** Screen sizes, in CSS pixels. Hit testing in runtime.pickTop uses the same numbers. */
export const TOP_PX = {
  point: 6,
  pointSelected: 8,
  ring: 2,
  path: 2.5,
  wedge: 64,
  handle: 7,
  handleAt: 92,
  subject: 13,
  hit: 11,
};

export interface TopPoint {
  x: number;
  z: number;
  role: "start" | "mid" | "end";
  selected: boolean;
}

export interface TopSubject {
  x: number;
  z: number;
  kind: "person" | "vehicle" | "other";
  yaw: number | null;
}

export interface TopDraw {
  path: Array<[number, number]>;
  points: TopPoint[];
  camera: { x: number; z: number; yaw: number; hfov: number } | null;
  subjects: TopSubject[];
  mpp: number;
  frame: { cx: number; cz: number; half: number; aspect: number };
}

function mat(color: number, opacity = 1): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false, depthWrite: false, side: THREE.DoubleSide, fog: false, toneMapped: false });
}

/** Flat on the ground plane: geometry made in XY is laid on XZ. */
function flat(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  geo.rotateX(-Math.PI / 2);
  return geo;
}

export function createTopOverlay(): { group: THREE.Group; draw(d: TopDraw): void; dispose(): void } {
  const group = new THREE.Group();
  group.name = "top_overlay";
  group.userData.overlay = true;
  group.visible = false;
  const Y = 40;
  let order = 1000;
  const add = <T extends THREE.Object3D>(o: T, name: string): T => {
    o.name = name;
    o.renderOrder = order++;
    o.frustumCulled = false;
    group.add(o);
    return o;
  };

  // the grid sits on the ground under everything (depth-tested), only in Top view
  const gridMat = new THREE.LineBasicMaterial({ color: IVORY, transparent: true, opacity: 0.09, depthWrite: false, fog: false, toneMapped: false });
  let grid: THREE.LineSegments | null = null;
  let gridKey = "";
  const drawGrid = (f: TopDraw["frame"]) => {
    const step = f.half <= 22 ? 1 : f.half <= 70 ? 5 : 10;
    const reachX = f.half * f.aspect * 1.3;
    const reachZ = f.half * 1.3;
    const x0 = Math.floor((f.cx - reachX) / step) * step;
    const x1 = Math.ceil((f.cx + reachX) / step) * step;
    const z0 = Math.floor((f.cz - reachZ) / step) * step;
    const z1 = Math.ceil((f.cz + reachZ) / step) * step;
    const key = [step, x0, x1, z0, z1].join(",");
    if (key === gridKey && grid) return;
    gridKey = key;
    const pts: number[] = [];
    for (let x = x0; x <= x1; x += step) pts.push(x, 0.06, z0, x, 0.06, z1);
    for (let z = z0; z <= z1; z += step) pts.push(x0, 0.06, z, x1, 0.06, z);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    if (grid) {
      grid.geometry.dispose();
      grid.geometry = geo;
    } else {
      grid = new THREE.LineSegments(geo, gridMat);
      grid.name = "top_grid";
      grid.frustumCulled = false;
      group.add(grid);
    }
  };

  // the path: a ribbon a few pixels wide
  const MAXP = 256;
  const ribbonGeo = new THREE.BufferGeometry();
  const ribbonPos = new Float32Array(MAXP * 2 * 3);
  ribbonGeo.setAttribute("position", new THREE.BufferAttribute(ribbonPos, 3));
  const idx: number[] = [];
  for (let i = 0; i < MAXP - 1; i++) {
    const a = i * 2;
    idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  ribbonGeo.setIndex(idx);
  const ribbon = add(new THREE.Mesh(ribbonGeo, mat(IVORY, 0.7)), "camera_path");

  // points: a dark ring under an ivory mark; the start is a square, the end a triangle
  const disc = flat(new THREE.CircleGeometry(1, 28));
  const square = flat(new THREE.PlaneGeometry(1.7, 1.7));
  const tri = flat(new THREE.CircleGeometry(1.25, 3).rotateZ(Math.PI / 2));
  const ringMat = mat(DARK, 0.85);
  const pointMat = mat(IVORY, 0.92);
  const selectedMat = mat(IVORY, 1);
  const points = new THREE.Group();
  add(points, "path_points");

  // the camera: a filled wedge (the lens's field of view), a dark outline, an aim line and the turn handle
  const wedgeGeo = new THREE.BufferGeometry();
  wedgeGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(9), 3));
  const wedgeLineGeo = new THREE.BufferGeometry();
  wedgeLineGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
  const aimGeo = new THREE.BufferGeometry();
  aimGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
  aimGeo.setIndex([0, 1, 2, 1, 3, 2]);
  const marker = new THREE.Group();
  add(marker, "camera_marker");
  // a soft cone of what the lens sees, an ivory outline, and a small camera body at the lens, its front toward the cone
  const wedge = new THREE.Mesh(wedgeGeo, mat(IVORY, 0.22));
  const wedgeLine = new THREE.Line(wedgeLineGeo, new THREE.LineBasicMaterial({ color: IVORY, transparent: true, opacity: 0.95, depthTest: false, fog: false, toneMapped: false }));
  const aim = new THREE.Mesh(aimGeo, mat(IVORY, 0.75));
  const handleRing = new THREE.Mesh(disc, ringMat);
  const handle = new THREE.Mesh(disc, mat(IVORY, 1));
  const bodyGeo = flat(new THREE.PlaneGeometry(1, 1.35));
  const lensGeo = flat(new THREE.PlaneGeometry(0.6, 0.45).translate(0, 0.85, 0));
  const bodyRing = new THREE.Mesh(flat(new THREE.PlaneGeometry(1.32, 1.67)), ringMat);
  const body = new THREE.Mesh(bodyGeo, mat(IVORY, 1));
  const lens = new THREE.Mesh(lensGeo, mat(IVORY, 1));
  for (const [o, n] of [[aim, "camera_aim"], [wedge, "camera_wedge"], [wedgeLine, "camera_outline"], [bodyRing, "camera_body_ring"], [body, "camera_body"], [lens, "camera_lens"], [handleRing, "turn_ring"], [handle, "turn_handle"]] as const) {
    o.name = n;
    o.renderOrder = order++;
    o.frustumCulled = false;
    marker.add(o);
  }

  const subjects = new THREE.Group();
  add(subjects, "footprints");
  const ringGeo = flat(new THREE.RingGeometry(0.82, 1, 32));
  const tickGeo = flat(new THREE.PlaneGeometry(0.16, 0.7).translate(0, 1.2, 0));
  const personMat = mat(PERSON, 0.9);
  const vehicleMat = mat(VEHICLE, 0.9);
  // the points and the camera draw over the people
  order += 0;
  points.renderOrder = 1100;
  marker.renderOrder = 1200;
  subjects.renderOrder = 1050;

  const meshAt = (parent: THREE.Group, i: number, geo: THREE.BufferGeometry, m: THREE.Material, ro: number): THREE.Mesh => {
    let mesh = parent.children[i] as THREE.Mesh | undefined;
    if (!mesh) {
      mesh = new THREE.Mesh(geo, m);
      mesh.frustumCulled = false;
      parent.add(mesh);
    }
    mesh.geometry = geo;
    mesh.material = m;
    mesh.renderOrder = ro;
    mesh.visible = true;
    return mesh;
  };

  return {
    group,
    draw(d) {
      group.visible = true;
      drawGrid(d.frame);
      const mpp = d.mpp;
      // ribbon
      const n = Math.min(MAXP, d.path.length);
      const w = (TOP_PX.path * mpp) / 2;
      for (let i = 0; i < n; i++) {
        const [x, z] = d.path[i];
        const [ax, az] = d.path[Math.max(0, i - 1)];
        const [bx, bz] = d.path[Math.min(n - 1, i + 1)];
        let tx = bx - ax;
        let tz = bz - az;
        const len = Math.hypot(tx, tz) || 1;
        tx /= len;
        tz /= len;
        const nx = -tz * w;
        const nz = tx * w;
        ribbonPos.set([x + nx, Y, z + nz, x - nx, Y, z - nz], i * 6);
      }
      (ribbonGeo.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      ribbonGeo.setDrawRange(0, Math.max(0, n - 1) * 6);
      ribbon.visible = n > 1;

      // points: two meshes each (ring, mark)
      points.children.forEach((c) => { c.visible = false; });
      d.points.forEach((p, i) => {
        const r = (p.selected ? TOP_PX.pointSelected : TOP_PX.point) * mpp;
        const geo = p.role === "start" ? square : p.role === "end" ? tri : disc;
        const ring = meshAt(points, i * 2, geo, ringMat, 1100 + i * 2);
        ring.position.set(p.x, Y + 1, p.z);
        ring.scale.setScalar(r + TOP_PX.ring * mpp);
        const mark = meshAt(points, i * 2 + 1, geo, p.selected ? selectedMat : pointMat, 1101 + i * 2);
        mark.position.set(p.x, Y + 1.5, p.z);
        mark.scale.setScalar(r);
      });

      // camera marker
      marker.visible = !!d.camera;
      if (d.camera) {
        const { x, z, yaw, hfov } = d.camera;
        const L = TOP_PX.wedge * mpp;
        const fx = Math.sin(yaw);
        const fz = Math.cos(yaw);
        const half = Math.min(1.35, Math.max(0.12, hfov / 2));
        const side = Math.tan(half) * L;
        const rx = Math.cos(yaw);
        const rz = -Math.sin(yaw);
        const y = Y + 3;
        const ax = x + fx * L + rx * side;
        const az = z + fz * L + rz * side;
        const bx = x + fx * L - rx * side;
        const bz = z + fz * L - rz * side;
        (wedgeGeo.getAttribute("position") as THREE.BufferAttribute).set([x, y, z, ax, y, az, bx, y, bz]);
        wedgeGeo.getAttribute("position").needsUpdate = true;
        wedgeGeo.computeBoundingSphere();
        (wedgeLineGeo.getAttribute("position") as THREE.BufferAttribute).set([x, y + 0.1, z, ax, y + 0.1, az, bx, y + 0.1, bz, x, y + 0.1, z]);
        wedgeLineGeo.getAttribute("position").needsUpdate = true;
        const H = TOP_PX.handleAt * mpp;
        const hx = x + fx * H;
        const hz = z + fz * H;
        const aw = 0.75 * mpp;
        (aimGeo.getAttribute("position") as THREE.BufferAttribute).set([
          x + rx * aw, y - 0.5, z + rz * aw, x - rx * aw, y - 0.5, z - rz * aw,
          hx + rx * aw, y - 0.5, hz + rz * aw, hx - rx * aw, y - 0.5, hz - rz * aw,
        ]);
        aimGeo.getAttribute("position").needsUpdate = true;
        // the body is drawn in XY then laid flat (its +y is -z), so turn it by yaw + 180 degrees to face the cone
        for (const [o, dy] of [[bodyRing, 0.2], [body, 0.3], [lens, 0.3]] as const) {
          o.position.set(x, y + dy, z);
          o.rotation.set(0, yaw + Math.PI, 0);
          o.scale.setScalar(12 * mpp);
        }
        handleRing.position.set(hx, y + 0.4, hz);
        handleRing.scale.setScalar((TOP_PX.handle + TOP_PX.ring) * mpp);
        handle.position.set(hx, y + 0.5, hz);
        handle.scale.setScalar(TOP_PX.handle * mpp);
      }

      // a ring on each person or vehicle, with a tick for the way it faces
      subjects.children.forEach((c) => { c.visible = false; });
      d.subjects.forEach((s, i) => {
        const m = s.kind === "vehicle" ? vehicleMat : personMat;
        const r = Math.max(TOP_PX.subject * mpp, s.kind === "vehicle" ? 2.6 : 0.5);
        const ring = meshAt(subjects, i * 2, ringGeo, m, 1050 + i * 2);
        ring.position.set(s.x, Y - 2, s.z);
        ring.scale.setScalar(r);
        const tick = meshAt(subjects, i * 2 + 1, tickGeo, m, 1051 + i * 2);
        tick.visible = s.yaw != null;
        tick.position.set(s.x, Y - 2, s.z);
        tick.scale.setScalar(r);
        tick.rotation.set(0, (s.yaw ?? 0) + Math.PI, 0);
      });
    },
    dispose() {
      grid?.geometry.dispose();
      for (const g of [ribbonGeo, disc, square, tri, wedgeGeo, wedgeLineGeo, aimGeo, ringGeo, tickGeo, bodyGeo, lensGeo, bodyRing.geometry]) g.dispose();
      for (const m of [gridMat, ringMat, pointMat, selectedMat, personMat, vehicleMat]) m.dispose();
    },
  };
}
