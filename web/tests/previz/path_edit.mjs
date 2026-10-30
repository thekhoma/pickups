// Run 7: editing the camera path in Top view (src/previz/pathEdit.ts). Pure functions, no browser.
import { keyTime, sampleCamera } from '../../src/previz/cameraMath.ts';
import { aimPoint, canRemove, curveAt, fitBounds, insertPoint, keyAtTime, keyTimeAt, MAX_KEYS, movePath, movePoint, removePoint, setViewAt, shotTimeAt, turnPoint } from '../../src/previz/pathEdit.ts';

export function runPathEdit() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };
  const near = (a, b, tol) => Math.abs(a - b) <= tol;
  const keys = [
    { t: 0, pos: [-6, 1.6, 8], look: [0, 1.2, 0], focal_mm: 35 },
    { t: 0.5, pos: [0, 1.8, 10], look: [0, 1.2, 0], focal_mm: 35 },
    { t: 1, pos: [6, 1.6, 8], look: [0, 1.2, 0], focal_mm: 35 },
  ];
  const json = (k) => JSON.stringify(k);

  // B7: a click on the curve gets the curve's own key time, not the eased shot time
  const at = curveAt(keys, 0.1, 35).pos;
  const found = keyTimeAt(keys, at[0], at[2], 35);
  ok(near(found.t, 0.1, 0.02) && found.dist < 0.05, `keyTimeAt: ${JSON.stringify(found)}`);
  const eased = { ease: 'in_out', taper: 1, speed: 1 };
  ok(near(keyTime(eased, 0.25), 0.1035, 0.01), `the in_out ease maps a quarter of the shot to key time ${keyTime(eased, 0.25)}`);
  for (const t of [0, 0.1, 0.37, 0.5, 0.9, 1]) ok(near(keyTime(eased, shotTimeAt(eased, t)), t, 1e-4), `shotTimeAt is keyTime's inverse at ${t}`);
  ok(near(shotTimeAt({ ease: 'linear', speed: 2 }, 1), 0.5, 1e-4), 'a 2x move reaches its end half way through the shot');

  // insert: on the curve, in time order, the curve keeps its shape (B18)
  const ins = insertPoint(keys, 0.25, 35);
  ok(ins && ins.keys.length === 4 && ins.index === 1 && ins.keys[1].t === 0.25, `insertPoint: ${json(ins)}`);
  if (ins) {
    const on = curveAt(keys, 0.25, 35);
    ok(near(ins.keys[1].pos[0], on.pos[0], 0.002) && near(ins.keys[1].pos[1], on.pos[1], 0.002) && near(ins.keys[1].look[2], on.look[2], 0.002), 'the new point sits on the curve, height and aim included');
    let worst = 0;
    for (let i = 0; i <= 40; i++) {
      const a = curveAt(keys, i / 40, 35).pos;
      const b = curveAt(ins.keys, i / 40, 35).pos;
      worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    }
    ok(worst < 0.35, `inserting a point bends the path by ${worst.toFixed(3)} m`);
    ok(ins.keys.every((k, i, all) => i === 0 || k.t > all[i - 1].t) && ins.keys[0].t === 0 && ins.keys[ins.keys.length - 1].t === 1, 'times stay in order, 0 to 1');
    ok(json(ins.keys[0]) === json(keys[0]) && json(ins.keys[2]) === json(keys[1]), 'inserting leaves the other points as they were');
  }
  ok(insertPoint(keys, 0.5, 35) === null, 'no second point at the same time');
  const full = Array.from({ length: MAX_KEYS }, (_, i) => ({ t: i / (MAX_KEYS - 1), pos: [i, 1, 0], look: [i, 1, 5] }));
  ok(insertPoint(full, 0.51, 35) === null, `a path stops at ${MAX_KEYS} points`);

  // move: only that point, only on the ground (B1 regression)
  const moved = movePoint(keys, 1, 2, 12);
  ok(json(moved[0]) === json(keys[0]) && json(moved[2]) === json(keys[2]), 'moving a point leaves every other point byte for byte');
  ok(json(moved[1].pos) === json([2, 1.8, 12]) && json(moved[1].look) === json(keys[1].look) && moved[1].t === 0.5, `the moved point keeps its height, aim and time: ${json(moved[1])}`);
  ok(json(keys[1].pos) === json([0, 1.8, 10]), 'movePoint does not change its input');
  const slid = movePath(keys, 1, -2);
  ok(slid.every((k, i) => k.pos[0] === keys[i].pos[0] + 1 && k.look[2] === keys[i].look[2] - 2 && k.pos[1] === keys[i].pos[1]), 'movePath slides points and aims together');

  // aim and turn
  const aimed = aimPoint(keys, 0, -6, 0);
  const reach0 = Math.hypot(keys[0].look[0] - keys[0].pos[0], keys[0].look[2] - keys[0].pos[2]);
  ok(near(aimed[0].look[0], -6, 0.002) && near(Math.hypot(aimed[0].look[0] - -6, aimed[0].look[2] - 8), reach0, 0.01) && aimed[0].look[1] === 1.2, `aimPoint: ${json(aimed[0])}`);
  const turned = turnPoint(turnPoint(keys, 2, 0.4), 2, -0.4);
  ok(near(turned[2].look[0], keys[2].look[0], 0.01) && near(turned[2].look[2], keys[2].look[2], 0.01), 'turning there and back is no turn');

  // remove: never the ends, never below two
  ok(!canRemove(keys, 0) && !canRemove(keys, 2) && canRemove(keys, 1), 'only middle points can go');
  ok(removePoint(keys, 0) === keys && removePoint(keys, 2) === keys, 'removing an end does nothing');
  ok(removePoint(keys, 1).length === 2 && removePoint(removePoint(keys, 1), 1).length === 2, 'a path keeps two points');

  ok(keyAtTime(keys, 0.505, 0.01) === 1 && keyAtTime(keys, 0.3, 0.01) === -1, 'keyAtTime finds the point under the current frame');

  // B6: the map fits the subject, not just the path (a still camera 12 m from the subject)
  const box = fitBounds([[0, 12], [0, 12], [0, 0]], { aspect: 1920 / 1088 });
  ok(box.cz > 4 && box.cz < 8 && box.half >= 6 * 1.2 && box.half - Math.abs(box.cz) >= 0, `fitBounds: ${json(box)}`);
  ok(fitBounds([]).half === 4 && fitBounds([[3, 3]]).half === 4, 'an empty or one-point map is 8 m tall');

  // an old two-key spec samples exactly as before (keyTime is the same maths sampleCamera always had)
  const spec = { lens: { focal_mm: 35 }, camera: { interp: 'catmull_rom', ease: 'in_out', keys: [keys[0], keys[2]] } };
  const mid = sampleCamera(spec, 0.5).pos;
  ok(near(mid[0], 0, 1e-9) && near(mid[2], 8, 1e-9), `two keys, half way: ${json(mid)}`);

  // Run 14 (walkthrough 8): the camera moved by hand keys its frame; an existing key there takes the view, no other key moves
  const view = { pos: [1, 2.5, 9], look: [0, 1, 0] };
  const onKey = setViewAt(keys, 0.5, view, 0.01, 35);
  ok(onKey && onKey.keys.length === 3 && onKey.index === 1 && json(onKey.keys[1].pos) === json(view.pos) && json(onKey.keys[0]) === json(keys[0]) && json(onKey.keys[2]) === json(keys[2]), `setViewAt on a key: ${json(onKey)}`);
  const fresh = setViewAt(keys, 0.75, view, 0.01, 35);
  ok(fresh && fresh.keys.length === 4 && fresh.index === 2 && fresh.keys[2].t === 0.75 && json(fresh.keys[2].look) === json(view.look), `setViewAt between keys: ${json(fresh)}`);
  ok(json(keys[1].pos) === json([0, 1.8, 10]), 'setViewAt leaves the keys it was given alone');
  const packed = Array.from({ length: MAX_KEYS }, (_, i) => ({ t: i / (MAX_KEYS - 1), pos: [i, 1.6, 0], look: [0, 1, 0] }));
  ok(setViewAt(packed, 0.51, view, 0.001, 35) === null, 'setViewAt: no 25th key');
  // Run 14 (walkthrough 22): Curve shapes the path between keys; every key stays on it, 1 is the old smooth path
  const curved = (curve) => ({ lens: { focal_mm: 35 }, camera: { ease: 'linear', taper: 0, speed: 1, distance: 1, keys, ...(curve == null ? {} : { curve }) } });
  for (const c of [0, 0.5, 1, 1.5]) {
    for (const k of keys) {
      const at = sampleCamera(curved(c), k.t).pos;
      ok(near(at[0], k.pos[0], 1e-6) && near(at[2], k.pos[2], 1e-6), `curve ${c} passes through the key at t=${k.t}`);
    }
  }
  const old = sampleCamera(curved(), 0.25).pos;
  const one = sampleCamera(curved(1), 0.25).pos;
  ok(json(old) === json(one), 'curve 1 is the path as it was');
  const straight = sampleCamera(curved(0), 0.25).pos;
  ok(near(straight[0], -3, 0.02) && near(straight[2], 9, 0.02), `curve 0 is a straight line between keys: ${json(straight)}`);
  const round = sampleCamera(curved(1.5), 0.25).pos;
  ok(round[2] > one[2], `curve 1.5 bows out further than 1: ${json(round)} vs ${json(one)}`);
  return { fail: problems.length, problems, line: `path edit (key time, insert, move, aim, remove, 24 points, map fit): ${problems.length ? problems.length + ' failing' : 'all pass'}` };
}
