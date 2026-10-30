// Director 30: parse, real builder, frameShot. Headless. No WebGL.
import './node_dom.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, parse, readJSON } from './common.mjs';
import { createLibrary } from '../../src/previz/runtime/loader.js';
import { createShotBuilder } from '../../src/previz/runtime/builder.js';
import { frameShot } from '../../src/previz/runtime/camera.js';
import { registerAnimalPoses, registerPoses } from '../../src/previz/runtime/poser.js';

const PUBLIC = path.resolve(ROOT, '../../public/previz');
const LOCAL = 'http://previz.local';
const OrigRequest = globalThis.Request;
globalThis.Request = class extends OrigRequest {
  constructor(input, init) {
    if (typeof input === 'string' && input.startsWith('/')) input = LOCAL + input;
    super(input, init);
  }
};
const origFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  const mark = LOCAL + '/previz/';
  if (url.startsWith(mark)) {
    const file = path.join(PUBLIC, url.slice(mark.length));
    if (!fs.existsSync(file)) return new Response('missing ' + url, { status: 404 });
    return new Response(fs.readFileSync(file));
  }
  return origFetch(input, init);
};

const ROWS = readJSON('tests/director30.json');
let builder = null;
function shots() {
  if (builder) return builder;
  registerPoses(readJSON('data/person_poses.json'));
  registerAnimalPoses(readJSON('data/animal_poses.json'));
  const layouts = new Map();
  builder = createShotBuilder({
    lib: createLibrary({ baseUrl: '/previz/assets/' }),
    getLayout: async (id) => {
      if (layouts.has(id)) return layouts.get(id);
      const layout = JSON.parse(fs.readFileSync(path.join(PUBLIC, 'data/kits', id + '.json'), 'utf8'));
      layouts.set(id, layout);
      return layout;
    },
  });
  return builder;
}

function has(list, asset) {
  return (list || []).some((p) => p.asset === asset || String(p.asset || '').startsWith(asset));
}

export async function runDirector() {
  let fail = 0;
  let pass = 0;
  for (const row of ROWS) {
    const problems = [];
    const shot = parse(row.text);
    const cam = shot.camera || {};
    const people = shot.actors.filter((a) => a.kind === 'person' || a.asset === 'person');
    if (row.place && shot.place.id !== row.place) problems.push('place ' + shot.place.id);
    if (row.move && cam.move !== row.move) problems.push('move ' + cam.move);
    if (row.angle && cam.angle !== row.angle) problems.push('angle ' + cam.angle);
    if (row.facing && cam.facing !== row.facing) problems.push('facing ' + cam.facing);
    if (row.lens && cam.lensMm !== row.lens) problems.push('lens ' + cam.lensMm);
    if (row.speed && cam.subjectSpeed !== row.speed) problems.push('speed ' + cam.subjectSpeed);
    if (row.shot && cam.shot !== row.shot) problems.push('shot ' + cam.shot);
    if (row.dutch && cam.dutch !== row.dutch) problems.push('dutch ' + cam.dutch);
    if (row.handheld && !cam.handheld) problems.push('handheld');
    if (row.world && shot.world && shot.world.time !== row.world && shot.world.weather !== row.world) problems.push('world');
    if (row.weather && (!shot.world || shot.world.weather !== row.weather)) problems.push('weather');
    if (row.pose && !people.some((a) => a.pose === row.pose)) problems.push('pose');
    if (row.count && !people.some((a) => a.count === row.count)) problems.push('count');
    if (row.prop && !has(shot.props, row.prop) && !has(shot.actors, row.prop)) problems.push('prop ' + row.prop);
    if (row.beat && !(shot.beats || []).some((b) => b.type === row.beat)) problems.push('beat');
    if (row.chip && !(shot.notes || []).includes(row.chip)) problems.push('chip');
    if (row.pair && !people.some((a) => a.pair && a.pair.id === row.pair)) problems.push('pair');
    if (row.noAirplane && (has(shot.props, 'airplane') || has(shot.actors, 'airplane'))) problems.push('airplane');
    if (row.noTruck && has(shot.props, 'truck')) problems.push('truck');
    if (row.angleDeg) {
      const seg = (cam.segments || []).find((s) => s.move === 'orbit');
      const angle = seg && seg.params && seg.params.angle;
      if (angle !== row.angleDeg) problems.push('angleDeg ' + angle);
    }
    try {
      const built = await shots().build(shot, { subjectTier: shot.camera.shot === 'close' || shot.camera.shot === 'extreme_close' ? 'hero' : 'lite', liteCount: 1, crowdCount: 12, instancing: true });
      const framed = frameShot(built, shot, { aspect: 1920 / 1088 });
      const keys = framed.keys && framed.keys.length >= 2 ? framed.keys : [framed.start, framed.end];
      if (!framed.start || !framed.end || keys.length < 2) problems.push('camera');
      if (row.move === 'static') {
        const a = keys[0].pos || framed.start.pos;
        const b = keys[keys.length - 1].pos || framed.end.pos;
        const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        if (d > 1e-4) problems.push('locked ' + d.toFixed(3));
      }
      if (row.angleDeg === 360) {
        let sweep = 0;
        const look = (k) => k.look || k.target;
        const pos = (k) => k.pos;
        for (let i = 1; i < keys.length; i++) {
          const a0 = Math.atan2(pos(keys[i - 1])[0] - look(keys[i - 1])[0], pos(keys[i - 1])[2] - look(keys[i - 1])[2]);
          const a1 = Math.atan2(pos(keys[i])[0] - look(keys[i])[0], pos(keys[i])[2] - look(keys[i])[2]);
          let d = a1 - a0;
          if (d > Math.PI) d -= Math.PI * 2;
          if (d < -Math.PI) d += Math.PI * 2;
          sweep += d;
        }
        if (Math.abs(Math.abs(sweep) * 180 / Math.PI - 360) > 5) problems.push('sweep ' + (sweep * 180 / Math.PI).toFixed(1));
      }
      if (row.move === 'dolly_zoom') {
        const f0 = (keys[0].focal_mm) || 0;
        const f1 = (keys[keys.length - 1].focal_mm) || 0;
        if (!(f0 > 50 && f1 > 0 && f0 / f1 >= 1.8)) problems.push('focal ' + f0 + '→' + f1);
      }
    } catch (err) {
      problems.push('build ' + (err && err.message ? err.message.split('\n')[0] : err));
    }
    if (problems.length) {
      fail += 1;
      console.log('FAIL  director ' + row.id + ' ' + problems.join('; '));
    } else pass += 1;
  }
  return { fail, line: `director30: ${pass}/${ROWS.length}` };
}
