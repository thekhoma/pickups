// Run 3 (QA's list, the director's RUN3_FIXES and addendum, R&D's BUGS_RUN2): the video line names only the shot's own cast
// (R2-9) and never says "in in"; no keyframe control in the app (R2-1, R2-6); the place chip is empty until the words
// name or imply one, Park only on park, lawn, bench or path; traffic words only when the user typed them (c08); a car the
// sentence drives, filmed tracking, rides with a camera car so the road and the far background stream the same way (c07);
// an existing layout's kit trees stay out unless the sentence places them (R2-2).
import './node_dom.mjs';
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { parse } from './common.mjs';
import { shots } from './shot_guard.mjs';
import { inferPlace, shownPlace, compile } from '../../src/previz/compile.ts';
import { carWordsIn, loneCar, motionPromptFor, personSubject, vehicleSubject } from '../../src/previz/motionPrompt.ts';
import { frameShot } from '../../src/previz/runtime/camera.js';
import { CAMERA_CAR_SPEED } from '../../src/previz/runtime/builder.js';

const key = (pos, look = [0, 1, 0]) => ({ pos, look });
const push = { keys: [key([0, 1.5, 6]), key([0, 1.5, 3])], handheld: null };

export async function runRun3Units() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };

  // --- R2-9: the shot's own cast only, and never "in in" ---
  const maya = { id: 'm', name: 'Maya', kind: 'person', line: 'a woman in her 40s with a straight shoulder-length black bob, in a knee-length buttoned mustard yellow wool coat', url: '' };
  const marcus = { id: 'k', name: 'Marcus', kind: 'person', line: 'a man in his 60s in a navy wool peacoat and grey flat cap', url: '' };
  const s04 = { notes: 'A man sitting on a bench in the park. The camera pushes in.', camera: push, subjects: [] };
  const lone = { place: 'Park', subjects: [{ kind: 'person', word: 'man', x: 0.5, distance_m: 15, in_view: true }] };
  const none = motionPromptFor(s04, lone, { cast: [maya, marcus] });
  ok(/^A man sitting on a bench/.test(none) && !/woman|Maya|mustard|peacoat/.test(none), `R2-9: no pick keeps "A man", never the project cast: ${none}`);
  const picked = motionPromptFor(s04, lone, { cast: [maya, marcus], picked: [marcus] });
  ok(picked.startsWith('A man in his 60s in a navy wool peacoat and grey flat cap sitting on a bench'), `R2-9: the picked cast: ${picked}`);
  const named = motionPromptFor({ ...s04, notes: 'Marcus sits on a bench in the park.' }, lone, { cast: [maya, marcus] });
  ok(named.startsWith('A man in his 60s in a navy wool peacoat and grey flat cap sits'), `R2-9: a name in the words counts: ${named}`);
  for (const line of [maya.line, marcus.line, 'a young man in an orange hoodie, in grey joggers']) ok(!/\bin in\b/.test(personSubject(line)), `no "in in": ${personSubject(line)}`);

  // --- R2-1, R2-6: no keyframe control in the app ---
  const form = readFileSync(new URL('../../src/components/FormsCreative.tsx', import.meta.url), 'utf8');
  ok(!/keyframes/i.test(form) && !/Mid and last keyframes/.test(form), 'no keyframe switch and no keyframes param in the Previz form');

  // --- the place chip: empty by default, inferred from the user's words ---
  const want = [['A man waves.', null], ['blorp zingle', null], ['A car on the farm track.', 'Country'], ['An old farmhouse.', 'Country'],
    ['She walks down the sidewalk.', 'Street'], ['A man on the street.', 'Street'], ['Three friends on a roof.', 'Rooftop'],
    ['A woman on the shore.', 'Lake'], ['A man by the lake.', 'Lake'], ['A man in a booth.', 'Diner'], ['A diner at night.', 'Diner'],
    ['A dog on the lawn.', 'Park'], ['A man on a bench.', 'Park'], ['A jogger on the path.', 'Park'], ['A kid in the park.', 'Park']];
  for (const [text, place] of want) ok(inferPlace(text) === place, `infer "${text}": ${inferPlace(text)} != ${place}`);
  ok(shownPlace('A man waves.', 'Park', 'default') === undefined, 'a defaulted place is no place');
  ok(shownPlace('A man waves in the park.', 'Park', 'word') === 'Park', 'a place the words name stays');
  const plain = 'A person stands still.';
  ok(parse(plain).place.source === 'default' && shownPlace(plain, 'Park', parse(plain).place.source) === undefined, 'the parser default shows no chip');
  for (const text of ['A man waves.', 'blorp zingle', 'A person stands still.']) {
    const c = compile(text);
    ok(!c.matched.includes('Park') && !/^Park/.test(c.title), `no Park chip for "${text}": ${c.matched.join(', ')} / ${c.title}`);
  }
  ok(compile('A man sitting on a bench in the park.').matched.includes('Park'), 'the park when the words say it');

  // --- c08: traffic only from the user's own words ---
  const chase = { keys: [key([0, 0.5, -8], [0, 0.5, 0]), key([0, 0.5, 92], [0, 0.5, 100])], handheld: null };
  const away = { place: 'Highway', subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 8, in_view: true, doing: 'drive', speed_ms: 22, facing: 'away' }] };
  const quiet = motionPromptFor({ notes: 'A car on the highway. Low angle.', camera: chase, subjects: [] }, away);
  ok(!/traffic|chase/i.test(quiet), `no traffic words unless typed: ${quiet}`);
  const busy = motionPromptFor({ notes: 'A car chase through traffic on the highway. Low angle.', camera: chase, subjects: [] }, away);
  ok(/car chase through traffic/i.test(busy), `the user's own traffic words stay: ${busy}`);

  // --- Run 4 (c09 drift): no cast picked, the car keeps its own words from the idea or Look text, never invented ones ---
  const orbit = { keys: [key([0, 1.5, 8], [0, 0.5, 0]), key([8, 1.5, 0], [0, 0.5, 0])], handheld: null };
  const parkedCar = { subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 8, in_view: true }] };
  const c09 = { notes: 'A car parked on the outskirts.', camera: orbit, subjects: [] };
  // R&D's Run 3 c09 Look box, word for word
  const c09Look = 'A pale dirt track curves in from the lower left across dry ground with sparse grass toward a flat horizon. A short row of black wooden fence posts stands along the track behind the car at left, a small stone sits at far left, and at the right of the frame a small weathered farmhouse with pale clapboard walls and a dark shingle roof stands alone beside the track, with a dark door and one dark window. The car is about fourteen metres away and fills about a quarter of the frame width. Neutral late-morning light from frame left, about thirty degrees high, lights the car\'s front with a bright highlight along the hood and lays its shadow toward the farmhouse. Powder blue paint, white roof, chrome, weathered clapboard.';
  const ownWords = motionPromptFor(c09, parkedCar, { look: c09Look });
  ok(ownWords.startsWith('A car with powder blue paint, white roof and chrome parked on the outskirts'), `c09: the Look box's car words: ${ownWords}`);
  ok(!/1960s|sedan|four-door|steel wheels|clapboard|shingle|door|window/i.test(ownWords.split('. ')[0]), `c09: nothing the text doesn't say about the car: ${ownWords}`);
  const described = motionPromptFor(c09, parkedCar, { look: 'The car is the same 1960s four-door sedan with steel wheels. Dry ground.' });
  ok(described.startsWith('A 1960s four-door sedan with steel wheels parked on the outskirts.'), `c09: a described car: ${described}`);
  const plainCar = motionPromptFor(c09, parkedCar, { look: 'A dirt track and a farmhouse at the right. The car is about fourteen metres away.' });
  ok(plainCar === motionPromptFor(c09, parkedCar), `no car words: the line is unchanged: ${plainCar}`);
  ok(plainCar.startsWith('A car parked on the outskirts.'), `no car words: "A car": ${plainCar}`);
  ok(carWordsIn(['A camera car drives past the hero car.', 'The red car chase starts.']) === null, 'a camera car, the hero car or a car chase is no description');
  const sedanCast = { id: 's', name: 'Blue sedan', kind: 'prop', line: 'a powder blue 1960s four-door sedan with a bright white painted roof', url: '' };
  ok(motionPromptFor(c09, parkedCar, { look: c09Look, picked: [sedanCast] }).startsWith('A powder blue 1960s four-door sedan with a bright white painted roof parked'), 'a picked cast still wins');

  // --- Run 5 (c09, c07): the car keeps its first "with" part, the project's one car is named, and the form's line follows ---
  // R&D's c09/cast.json and c07/cast_added.json lines, word for word
  const c09Cast = { id: 'b', name: 'Blue sedan', kind: 'prop', url: '', line: 'a powder blue 1960s four-door sedan with a bright white painted roof clearly separate from the light blue body, a plain smooth hood, a wide chrome grille bar with four round headlights, a bare chrome bumper, and steel wheels with chrome hubcaps' };
  const c07Cast = { id: 'g', name: 'Green coupe car', kind: 'prop', url: '', line: 'a dark green 1970s two-door coupe with chrome bumpers, steel wheels with chrome trim rings and plain black tyres, a smooth, unbroken tail with a flat trunk lid and no spoiler, and clean flanks with no badges or emblems' };
  ok(vehicleSubject(c09Cast.line) === 'a powder blue 1960s four-door sedan with a bright white painted roof', `c09: the white roof stays: ${vehicleSubject(c09Cast.line)}`);
  ok(vehicleSubject(c07Cast.line) === 'a dark green 1970s two-door coupe with chrome bumpers', `c07: the first "with" part only: ${vehicleSubject(c07Cast.line)}`);
  ok(vehicleSubject('a red pickup truck') === 'a red pickup truck', 'no "with": nothing added');
  const c09Picked = motionPromptFor(c09, parkedCar, { look: c09Look, picked: [c09Cast] });
  ok(c09Picked.startsWith('A powder blue 1960s four-door sedan with a bright white painted roof parked on the outskirts'), `c09: picked cast with the roof: ${c09Picked}`);
  ok(!/hubcaps|grille|headlights|bumper/i.test(c09Picked), `c09: never past the first "with" part: ${c09Picked}`);
  // the project's one car, not picked (Make shot and the form add it): c09's variant D and c07
  const c09Lone = motionPromptFor(c09, parkedCar, { look: c09Look, picked: loneCar([maya, c09Cast]) });
  ok(c09Lone.startsWith('A powder blue 1960s four-door sedan with a bright white painted roof parked'), `c09: the one car cast, unpicked: ${c09Lone}`);
  const c07Spec = { notes: 'A car driving on the highway. Tracking shot.', camera: push, subjects: [] };
  const c07Framing = { place: 'Highway', subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 6, in_view: true, doing: 'drive', speed_ms: 22, facing: 'left' }] };
  const c07Line = motionPromptFor(c07Spec, c07Framing, { cast: [], picked: loneCar([c07Cast]) });
  ok(c07Line.startsWith('A dark green 1970s two-door coupe with chrome bumpers driving along an open highway'), `c07: the car words: ${c07Line}`);
  ok(loneCar([c07Cast, c09Cast]).length === 0 && loneCar([maya, marcus]).length === 0, 'two cars or no car: none added');
  const twoPeople = motionPromptFor(s04, lone, { cast: [maya, marcus, c07Cast], picked: loneCar([maya, marcus, c07Cast]) });
  ok(twoPeople === none, `a person shot ignores the lone car: ${twoPeople}`);
  // the form: Make shot's line is marked as the app's, rebuilt from the cast and Look until the user types in it
  const actions = readFileSync(new URL('../../src/actions.ts', import.meta.url), 'utf8');
  ok(/prompt_auto: "1"/.test(actions), 'openMadePreviz marks its line as the app\'s (prompt_auto)');
  ok(/pre\?\.prompt_auto/.test(form) && /motion, motionAuto: false/.test(form) && /if \(!dr\.data\.motionAuto \|\| !spec\) return;/.test(form),
    'the Previz form rebuilds an auto line and stops once the user types');

  // --- c07: a car the sentence drives, filmed tracking, rides with a camera car ---
  const opts = { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true };
  const c07 = parse('A car driving on the highway. Tracking shot.');
  const built = await shots().build(c07, opts);
  const drive = built.subject && built.subject.object.userData.drive;
  ok(drive && drive.follow === true && drive.speed === CAMERA_CAR_SPEED, `c07: the car drives with the camera: ${JSON.stringify(drive)}`);
  const framed = frameShot(built, c07, { aspect: 1920 / 1088 });
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  ok(framed.start && d(framed.start.pos, framed.end.pos) < 1e-6 && d(framed.start.target, framed.end.target) < 1e-6,
    `c07: the camera car holds its place beside the car (no swing past a pinned target): ${JSON.stringify(framed.start)} ${JSON.stringify(framed.end)}`);
  // the heading the car drives on is the way the camera travels: a drive along +yaw
  const yaw = built.subject.object.rotation.y;
  ok(Math.abs(drive.yaw - yaw) < 1e-6, 'c07: the drive is on the car\'s own heading');
  // Run 5 (R&D's c07): no parking garage and no post on the camera car's line (or between it and the car)
  const shown = (o) => { for (let n = o; n; n = n.parent) if (!n.visible) return false; return true; };
  const garage = () => { const g = []; built.kit.set.traverse((o) => { if (o.isMesh && /^garage_/.test(o.name) && shown(o)) g.push(o.name); }); return g; };
  const postsOn = () => {
    const out = [], m4 = new THREE.Matrix4(), p = new THREE.Vector3();
    built.kit.root.updateMatrixWorld(true);
    built.kit.dressing.traverse((o) => {
      if (!o.isInstancedMesh || !/post/.test(String(o.parent && o.parent.name))) return;
      for (let n = 0; n < o.count; n++) {
        o.getMatrixAt(n, m4);
        p.setFromMatrixPosition(new THREE.Matrix4().multiplyMatrices(o.matrixWorld, m4));
        const camX = framed.start.pos[0], carX = built.subject.object.position.x;
        if (p.y > -100 && (Math.abs(p.x - camX) < 1 || (p.x - carX) * (camX - p.x) > 0)) out.push(p.toArray().map((v) => v.toFixed(1)).join(' '));
      }
    });
    return out;
  };
  ok(built.layout.id === 'highway' && garage().length === 0, `c07: no garage geometry in the camera-car shot: ${built.layout.id} ${garage().join(', ')}`);
  ok(postsOn().length === 0, `c07: no post on the camera car's line: camera x ${framed.start.pos[0].toFixed(2)}, posts ${postsOn().join(' / ')}`);
  shots().teardown();
  ok(garage().length === 12,`c07: the garage comes back on the next build: ${garage().length}`);
  const parked = await shots().build(parse('A car parked by a farmhouse on the outskirts. The camera orbits.'), opts);
  ok(!(parked.subject && parked.subject.object.userData.drive), 'a parked car never drives');
  const held = await shots().build(parse('A car driving on the highway. The camera holds still.'), opts);
  ok(!(held.subject && held.subject.object.userData.drive && held.subject.object.userData.drive.follow), 'a locked-off camera never rides with the car');

  // --- R2-2: an existing (legacy) layout keeps its kit trees out unless the words place them ---
  // places.ts imports extensionless (as Vite resolves them): loaded once step 25's resolve hook is in
  const { dressDescribed } = await import('../../src/previz/places.ts');
  const count = (notes, place) => {
    let trees = 0;
    const mats = { gray: new THREE.MeshStandardMaterial(), dark: new THREE.MeshStandardMaterial(), light: new THREE.MeshStandardMaterial() };
    try {
      dressDescribed({ seed: 1, notes, duration_s: 4, subjects: [], environment: { props: [place] } }, mats, { add: (o) => { o.traverse?.((n) => { if (n.geometry?.type === 'ConeGeometry' || n.geometry?.type === 'SphereGeometry') trees++; }); }, actors: [] });
    } catch (e) { console.log('R2-2 count: ' + e); trees = -1; }
    return trees;
  };
  const bare = count('A man jogs in the park.', 'park');
  const leafy = count('A man jogs under the trees in the park.', 'park');
  ok(bare >= 0 && leafy > bare, `R2-2: a park's kit trees only when the words place them (${bare} vs ${leafy})`);

  return problems;
}
