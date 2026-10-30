// Step 15 (Fresh-9 user test): the words a made previz hands to Previz to final: the motion prompt names the camera
// move, what comes into view and where, and a group's line. (The start frame's words are the server's since step 17:
// tests/test_start_frame.py.)
import { cameraMismatch, cameraWords, facingMismatch, groupWords, keepWords, motionPromptFor, motionSpecOf, personSubject, revealWords, vehicleCameraWords, vehicleSubject } from '../../src/previz/motionPrompt.ts';
import { facingOf } from '../../src/previz/facing.ts';
import * as THREE from 'three';
import { readFileSync } from 'node:fs';
import { blankReason, frameView, wordFor } from '../../src/previz/inView.ts';

const key = (pos, look = [0, 1, 0]) => ({ t: 0, pos, look });
const spec = (keys, subjects = [{ id: 'a', kind: 'humanoid', pos: [0, 0, 0] }], notes = '') => ({ camera: { keys, handheld: null }, subjects, notes });

export function runHandoffWords() {
  const problems = [];
  const eq = (a, b, what) => { if (a !== b) problems.push(`${what}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
  const has = (a, b, what) => { if (!a.includes(b)) problems.push(`${what}: ${JSON.stringify(a)} lacks ${JSON.stringify(b)}`); };

  // Scene names to words: people win over places, the kit's own groups and the ground say nothing.
  eq(wordFor('lake_sitter'), 'people', 'a kit extra is a person');
  eq(wordFor('geo_water'), 'water', 'water geometry');
  eq(wordFor('water'), 'water', 'plain water');
  eq(wordFor('inst_tree_round'), 'trees', 'instanced trees');
  eq(wordFor('kit_lake'), null, 'the kit root is not water');
  eq(wordFor('geo_ground'), null, 'bare ground');
  eq(wordFor('lb'), null, 'an id with no word');

  // s03: the lake is behind the camera at frame 0; the motion prompt says where it comes in.
  const s03 = { place: 'Lake', view: { share: { grass: 0.5, 'a path': 0.1, trees: 0.2 }, outside: ['water'], actors: [0] }, reveals: [{ word: 'water', at_s: 2, side: 'left' }] };
  eq(revealWords(s03)[0], 'Water comes into view on the left at about 2 s.', 'reveal cue');
  eq(revealWords({ reveals: [{ word: 'trees', at_s: 1.5, side: 'middle' }] })[0], 'Trees come into view ahead at about 1.5 s.', 'plural reveal ahead');

  // Camera words from the keys: camera at +z looking at the subject; +x is its right.
  eq(cameraWords(spec([key([0, 1.5, 5]), key([0, 1.5, 3])])), 'The camera pushes in.', 'push in');
  eq(cameraWords(spec([key([0, 1.5, 3]), key([0, 1.5, 6])])), 'The camera pulls back.', 'pull back');
  eq(cameraWords(spec([key([0, 1.5, 5]), key([5, 1.5, 0])])), 'The camera orbits to the right around them.', 'orbit right');
  eq(cameraWords(spec([key([0, 1.5, 5]), key([-5, 1.5, 0])])), 'The camera orbits to the left around them.', 'orbit left');
  eq(cameraWords(spec([key([0, 1.5, 5]), key([0, 8, 5])])), 'The camera rises high above them.', 'drone up');
  eq(cameraWords(spec([key([0, 1.5, 5]), key([0, 1.5, 5])])), 'The camera holds still.', 'locked');

  // A stored spec as the server sends it (unset fields are null) still gives the motion words; a Blender clip gives none.
  const stored = { notes: 'Lena waves.', camera: { keys: [key([0, 1.5, 5]), key([0, 1.5, 3])], pick: null, frame: null, handheld: null }, subjects: [{ kind: 'humanoid', pos: [0, 0, 0], speed: null }] };
  eq(motionSpecOf(stored) && motionPromptFor(motionSpecOf(stored)), 'Lena waves. The camera pushes in. Keep the same person and camera as the driver.', 'a stored spec with nulls');
  eq(motionSpecOf(undefined), null, 'no previz spec');

  // A car as the subject with the camera travelling with it: the camera car and its speed (step 17).
  const car = { subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 8, in_view: true, doing: 'drive', speed_ms: 25 }] };
  const chase = { notes: 'A red sedan speeds down the highway.', camera: { keys: [key([0, 0.5, -8], [0, 0.5, 0]), key([0, 0.5, 92], [0, 0.5, 100])], handheld: null }, subjects: [], duration_s: 4 };
  eq(motionPromptFor(chase, car), 'A red sedan speeds along an open highway. Filmed from a camera car driving alongside at the same speed, the road and background streaming past. Keep the same car, road and camera as the driver.', 'a chase car');
  eq(vehicleCameraWords({ camera: { keys: [key([0, 1.5, 5]), key([0, 1.5, 5])] }, notes: 'A car parked on the street.' }, { subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 7, in_view: true }] }, 4), null, 'a parked car with a locked-off camera keeps its own words');
  eq(vehicleCameraWords({ camera: { keys: [key([0, 1.5, 5]), key([1, 1.5, 4])] }, notes: 'A red sedan drives fast down the highway.' }, { subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 7, in_view: true }] }, 4), 'Filmed from a camera car driving alongside at the same speed, the road and background streaming past.', 'the sentence says it drives: no speed known, none said');
  eq(vehicleCameraWords(chase, { subjects: [{ kind: 'person', x: 0.5, distance_m: 4, in_view: true }] }, 4), null, 'a person is not a car');

  // s06: three dancers in a line keep their order.
  const trio = [{ kind: 'humanoid', pos: [-1, 0, 0] }, { kind: 'humanoid', pos: [0, 0, 0.1] }, { kind: 'humanoid', pos: [1, 0, 0] }];
  has(groupWords({ subjects: trio }), 'side by side in a line', 'a line of three');
  eq(groupWords({ subjects: [trio[0]] }), '', 'one person has no group line');
  const full = motionPromptFor(spec([key([0, 3, 6]), key([0, 1, 6])], trio, 'Three friends dance on a rooftop. The camera cranes down slowly.'), undefined);
  // The Layout-to-Render card's short prompt: frame 1 and the action, the camera, then the Keep line; no negations.
  eq(full, 'Three friends dance on a city rooftop. The camera cranes down. Keep the same three people, rooftop and camera as the driver.', 'full motion prompt');
  eq(groupWords({ subjects: trio }), 'They stay side by side in a line, in the same order.', 'the group line has no negation');
  eq(keepWords({ subjects: [] }, { subjects: [{ kind: 'person', in_view: true }, { kind: 'vehicle', in_view: true }, { kind: 'vehicle', in_view: false }] }), 'person and car', 'keep: what is in view');
  eq(motionPromptFor({ notes: 'A woman waves. A boat drifts past. Warm light.', camera: { keys: [key([0, 1.5, 5]), key([0, 1.5, 5])] }, subjects: [{ kind: 'humanoid', pos: [0, 0, 0] }] }),
    'A woman waves in warm light. The camera holds still. Keep the same woman and camera as the driver.', 'one action sentence, never more; the light said');

  // Run 1b (the director's FACING_AND_VIDEO_TUNING.md 2): subject from the cast line, action, a 4 to 8 word place, the user's
  // light; the camera; "Keep the same <woman / man / two people / car>, <place> and camera as the driver."
  const still = { keys: [key([0, 1.5, 5]), key([0, 1.5, 3])], handheld: null };
  const lone = (word) => ({ subjects: [{ kind: 'person', word, x: 0.5, distance_m: 4, in_view: true }] });
  const maya = { id: 'm', name: 'Maya', kind: 'person', line: 'a woman in her 30s with short black hair, a red wool coat, grey trousers, white trainers', url: '' };
  const coupe = { id: 'c', name: 'Green coupe', kind: 'prop', line: 'Car reference sheet on a plain light gray studio background: the same dark green 1970s fastback muscle coupe with chrome bumpers shown three ways.', url: '' };
  eq(personSubject(maya.line), 'a woman in her 30s in a red wool coat and grey trousers', 'cast line cut to age and two garments under one in (Run 3)');
  eq(personSubject('a man in his 60s with a short gray beard, navy wool peacoat, gray flat cap'), 'a man in his 60s in a navy wool peacoat and gray flat cap', 'a real grey stays grey (Run 2, R&D #10)');
  eq(personSubject('a man in his 60s in a navy wool peacoat and grey flat cap'), 'a man in his 60s in a navy wool peacoat and grey flat cap', 'Run 3: the closed cast form, one in');
  eq(/\bin in\b/.test(personSubject('a woman in her 40s with a straight black bob, in a knee-length mustard yellow wool coat')), false, 'R2-9: never in in');
  eq(vehicleSubject(coupe.line), 'a dark green 1970s fastback muscle coupe with chrome bumpers', 'a car cast line cut to the car and its first "with" part');
  eq(motionPromptFor({ notes: 'Maya walks toward camera on the street. Warm evening light.', camera: still, subjects: [] }, lone('woman'), { cast: [maya] }),
    'A woman in her 30s in a red wool coat and grey trousers walks toward camera on a city street in warm evening light. The camera pushes in. Keep the same woman, street and camera as the driver.', 'one person: cast subject, place, light');
  eq(motionPromptFor({ notes: 'A man sits on a bench.', camera: still, subjects: [] }, { ...lone('man'), place: 'Park' }, { look: 'The grey figure is a man sitting on a bench along a gravel path in the park.' }),
    'A man sits on a bench along a gravel path in the park. The camera pushes in. Keep the same man, park and camera as the driver.', 'one person: the place from the Look box');
  eq(motionPromptFor({ notes: 'Two friends hug.', camera: still, subjects: [] }, { place: 'Lake', subjects: [{ kind: 'person', word: 'friends', in_view: true }, { kind: 'person', in_view: true }] }, { cast: [maya] }),
    'Two friends hug on the shore of a lake. The camera pushes in. Keep the same two people, lake and camera as the driver.', 'two people: their words, the place from the previz');
  const driving = { notes: 'A car chase on the highway at dusk.', camera: { keys: [key([0, 0.5, -8], [0, 0.5, 0]), key([0, 0.5, 92], [0, 0.5, 100])], handheld: null }, subjects: [], duration_s: 4 };
  const carLine = motionPromptFor(driving, car, { cast: [coupe], picked: [coupe] });
  eq(carLine, 'A dark green 1970s fastback muscle coupe with chrome bumpers in a car chase along an open highway at dusk. Filmed from a camera car driving alongside at the same speed, the road and background streaming past. Keep the same car, road and camera as the driver.', 'a car: cast subject, no km/h');
  eq(/km\/h|\d+ ?kph/.test(carLine), false, 'no speed number');
  const booth = motionPromptFor({ notes: 'A woman sits alone in a diner booth at night as headlights sweep across her. The light fades to dark.', camera: still, subjects: [] }, lone('woman'));
  eq(booth, 'A woman sits alone in a diner booth at night. The camera pushes in. Keep the same woman, booth and camera as the driver.', 'a light change is never promised');
  eq(motionPromptFor({ notes: 'A man jogs in the park.', camera: { ...still, handheld: true }, subjects: [] }, lone('man')).split(/(?<=[.!?])\s+/).length, 3, 'handheld folds into the camera sentence: two sentences, then Keep');
  // The page's line passes the server's word filter (prompt_words.py BANNED and NEGATION) untouched.
  const BANNED = /(?<![\w-])(previz|pre-viz|mannequins?|blockout|block-out|backdrop|studio|clay|3D|small vehicles)(?![\w-])|\b(no|not|never|without|nobody|nothing|none|neither|nor)\b|n't\b|\bgr[ae]y\b[^.]*\b(figures?|shapes?|surfaces?|layout)\b/i;
  for (const line of [carLine, booth, motionPromptFor({ notes: 'A grey figure waves by the lake.', camera: still, subjects: [] }, lone('woman'), { picked: [{ ...maya, line: 'a woman in her 30s with long gray hair, a gray wool coat' }] })]) {
    eq(BANNED.test(line), false, `word filter: ${line}`);
  }

  // Facing from the figure's rotation (local +z forward): the camera at +z looking down -z; its right is +x.
  const toCam = { x: 0, z: 1 }, right = { x: 1, z: 0 };
  eq(facingOf({ x: 0, z: 1 }, toCam, right), 'camera', 'facing the camera');
  eq(facingOf({ x: 0, z: -1 }, toCam, right), 'away', 'facing away');
  eq(facingOf({ x: 1, z: 0 }, toCam, right), 'right', 'facing frame right');
  eq(facingOf({ x: -1, z: 0.2 }, toCam, right), 'left', 'facing frame left');
  eq(facingOf({ x: 0.5, z: 0.866 }, toCam, right), 'camera', 'a three-quarter view toward the lens reads as toward the camera');
  const away = { subjects: [{ kind: 'person', x: 0.5, distance_m: 4, in_view: true, facing: 'away' }] };
  eq(facingMismatch('She walks toward the camera.', away), 'The previz has the person facing away; the previz wins.', 'the words disagree: the previz wins, flagged');
  const pair = { subjects: [{ kind: 'person', in_view: true, facing: 'away' }, { kind: 'person', in_view: true, facing: 'camera' }] };
  eq(facingMismatch('They walk toward the camera.', pair), null, 'someone in the previz faces that way: no flag');
  eq(facingMismatch('Both seen from behind.', { subjects: [{ kind: 'person', in_view: true, facing: 'left' }, { kind: 'person', in_view: true, facing: 'camera' }] }),
    'No one in the previz is facing away; the previz wins.', 'a group: nobody faces that way');
  eq(facingMismatch('Seen from behind, she walks away.', away), null, 'the words agree');
  // Run 2 (R&D #2): a bench turned toward camera or a shadow toward frame left is not a facing claim.
  eq(facingMismatch('At the right a park bench sits empty, turned toward camera, and beyond it a maple.', away), null, 'a bench is not a subject');
  eq(facingMismatch('Late sun from camera right lays her shadow toward frame left.', away), null, 'a light sentence is not a facing claim');
  eq(facingMismatch('Maya stands on the kerb, facing camera.', away, ['Maya']), 'The previz has the person facing away; the previz wins.', 'a named subject still counts');
  eq(facingMismatch('She walks toward the camera.', { subjects: [{ kind: 'person', x: 0.5, distance_m: 4, in_view: true }] }), null, 'no facing recorded: nothing to flag');

  // Step 20, R&D's eb0cb5f shots (tests/fixtures/step20): their stored keys are the template's, so the words came out
  // "tracks to the left and pushes in" for every person shot. The real move: the recorded one, else the sentence's.
  const shotData = (shot) => JSON.parse(readFileSync(new URL(`../../../tests/fixtures/step20/${shot}.json`, import.meta.url), 'utf8'));
  const motion = (shot) => { const d = shotData(shot); return motionPromptFor(motionSpecOf(d.spec), d.framing); };
  for (const shot of ['s04', 's06']) has(shotData(shot).before.motion, 'tracks to the left and pushes in', `${shot} before`);
  eq(shotData('s05').before.motion, 'Two friends hugging on a city sidewalk. The camera pushes in.', 's05 before');
  eq(motion('s06'), 'Three friends dancing on a city rooftop. The camera cranes down. Keep the same three people, rooftop and camera as the driver.', 's06 cranes down');
  eq(motion('s05'), 'Two friends hugging on a city sidewalk. The camera pulls back from close to wide. Keep the same two people, street and camera as the driver.', 's05 close to wide');
  eq(motion('s04'), 'A man sitting on a bench along a path in the park. The camera pushes in. Keep the same man, park and camera as the driver.', 's04 pushes in');
  has(motion('s03'), 'The camera orbits to the left around her and pushes in.', 's03 keeps its orbit (its keys are the orbit), one woman is her'); // Run 2, R&D #15
  const withCast = (shot) => { const d = shotData(shot); return motionPromptFor(motionSpecOf(d.spec), d.framing, { cast: d.cast, picked: d.cast, look: d.user_look }); };
  eq(withCast('s03'), 'A woman in her 40s in a cream chunky knit sweater and olive green linen trousers standing on the shore of a lake waving, tall dark pines along the far shore. The camera orbits to the left around her and pushes in. Keep the same woman, lake and camera as the driver.', 's03 with its cast: the real pines, her (Run 2)');
  eq(withCast('c07'), 'A dark green 1970s fastback muscle coupe with chrome bumpers driving along an open highway. Filmed from a camera car driving alongside at the same speed, the road and background streaming past. Keep the same car, road and camera as the driver.', 'c07 with its cast');
  has(motion('c08'), 'Filmed from a camera car', 'c08 keeps the camera-car line');

  // Run 2 (the director (e), R&D #6, #7, #15): a parked car on an orbit is never a camera car; a moving car's world streams
  // opposite its facing; an empty road said in positive words; one person is her or him; keyframes for orbits and cranes.
  const orbit = { keys: [key([0, 1.5, 8], [0, 0.5, 0]), key([8, 1.5, 0], [0, 0.5, 0])], handheld: null };
  const parkedCar = { place: 'Farm', subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 8, in_view: true }] };
  const c09 = motionPromptFor({ notes: 'A car parked by a farmhouse on the outskirts.', camera: orbit, subjects: [] }, parkedCar);
  has(c09, 'The camera circles slowly around the car, the farmhouse turning behind it.', 'c09: a parked car on an orbit');
  eq(/camera car|alongside/.test(c09), false, 'c09: never a camera car');
  const side = (facing) => ({ subjects: [{ kind: 'vehicle', x: 0.5, distance_m: 8, in_view: true, doing: 'drive', speed_ms: 22, facing }] });
  eq(vehicleCameraWords(chase, side('left'), 4), 'Filmed from a camera car driving alongside at the same speed, the car drives to frame left, the road and background streaming past to frame right.', 'c07: the travel direction matches the facing');
  eq(vehicleCameraWords(chase, side('right'), 4), 'Filmed from a camera car driving alongside at the same speed, the car drives to frame right, the road and background streaming past to frame left.', 'nose right: the world streams left');
  eq(vehicleCameraWords(chase, side('away'), 4), 'The camera follows low and close behind the car, the road streaming toward it.', 'chased from behind');
  const c08 = motionPromptFor({ notes: 'A car chase through traffic on the highway.', camera: chase.camera, subjects: [] }, side('away'), { look: 'The road ahead of the hero car is empty to the horizon.' });
  has(c08, 'on an empty freeway stretching to the horizon', 'c08: the empty road in positive words');
  eq(/traffic/.test(c08), false, 'c08: no traffic');
  has(motionPromptFor({ notes: 'A man waves.', camera: orbit, subjects: [] }, lone('man')), 'around him', 'one man is him');
  eq(/\bthem\b/.test(motionPromptFor({ notes: 'Someone waves.', camera: orbit, subjects: [] }, lone(undefined))), false, 'one person is never them');
  // A recorded move wins over the sentence: frame 0 at 4 m high, the end at eye level, the same distance.
  const s06 = shotData('s06');
  const recorded = { ...s06.framing, move: { start: { pos: [0, 4.3, 6], look: [0, 1, 0] }, end: { pos: [0, 1.5, 6], look: [0, 1, 0] } } };
  has(motionPromptFor(motionSpecOf({ ...s06.spec, notes: 'Three friends dancing on a rooftop.' }), recorded), 'The camera cranes down.', 'recorded crane down');
  // v2.4 9: the user's camera words first, then the path; a disagreement is flagged (the path's words), not written.
  const pushed = { ...recorded, move: { start: { pos: [0, 1.5, 6], look: [0, 1, 0] }, end: { pos: [0, 1.5, 3], look: [0, 1, 0] } } };
  has(motionPromptFor(motionSpecOf(s06.spec), pushed), 'The camera cranes down.', 'the user\'s words beat a recorded push');
  eq(cameraMismatch(motionSpecOf(s06.spec), pushed), 'pushes in', 'a recorded push against "cranes down" is a mismatch');
  eq(cameraMismatch(motionSpecOf(s06.spec), recorded), null, 'a recorded crane down agrees');
  eq(cameraMismatch(motionSpecOf(s06.spec), s06.framing), 'tracks to the left and pushes in', 's06: the stored path disagrees');
  eq(cameraMismatch(motionSpecOf(shotData('s04').spec), shotData('s04').framing), null, 's04: both push in');
  eq(cameraMismatch(motionSpecOf(shotData('s05').spec), shotData('s05').framing), 'pushes in', 's05: close to wide against a push');
  eq(cameraMismatch(motionSpecOf({ ...s06.spec, notes: 'Three friends dancing on a rooftop.' }), s06.framing), null, 'no camera words, nothing to disagree with');

  // A real scene, rays and all: a lake behind the camera, a tree and the mannequin ahead, bare ground under them.
  const scene = new THREE.Scene();
  const mat = new THREE.MeshBasicMaterial();
  const kit = new THREE.Group(); kit.name = 'kit_lake'; scene.add(kit);
  const set = new THREE.Group(); set.name = 'set'; kit.add(set);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200).rotateX(-Math.PI / 2), mat); ground.name = 'geo_ground'; set.add(ground);
  const water = new THREE.Mesh(new THREE.PlaneGeometry(20, 20).rotateX(-Math.PI / 2), mat); water.name = 'water'; water.position.set(0, 0.01, 25); set.add(water);
  const tree = new THREE.Group(); tree.name = 'tree'; kit.add(tree);
  const trunk = new THREE.Mesh(new THREE.BoxGeometry(1, 6, 1), mat); trunk.name = 'Mesh_3'; trunk.position.set(3, 3, -8); tree.add(trunk);
  const man = new THREE.Group(); scene.add(man);
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.8, 0.3), mat); body.position.set(0, 0.9, -5); man.add(body);
  const cam = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 500); cam.position.set(0, 1.5, 5); cam.lookAt(0, 1, -5);
  const v = frameView(scene, cam, [man]);
  if (!v.share.trees) problems.push(`the tree ahead is in view: ${JSON.stringify(v)}`);
  if (!v.outside.includes('water')) problems.push(`the lake behind the camera is out of view: ${JSON.stringify(v)}`);
  eq(JSON.stringify(v.actors), '[0]', 'the mannequin is in view');
  eq(v.inside, null, 'the camera is in the open');
  eq(blankReason(v, 9), null, 'a frame with the subject in it is never blank');
  const hedge = new THREE.Mesh(new THREE.BoxGeometry(4, 2, 4), mat); hedge.name = 'hedge'; hedge.position.set(0, 1, 5); scene.add(hedge);
  const inside = frameView(scene, cam, [man]);
  eq(inside.inside, 'hedges', 'the camera inside a hedge');
  eq(blankReason(inside, 2), 'Frame 2: the camera is inside the hedges. Move the camera out of it, then Make shot again.', 'hedge message');
  hedge.removeFromParent();
  // The subject itself passing the lens (the low-angle street shot): said as such.
  const close = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 500);
  close.position.set(0, 0.9, -4.6);
  close.lookAt(0, 0.9, -5);
  eq(blankReason(frameView(scene, close, [man]), 3), 'Frame 3: the subject comes right up against the lens. Move the camera back, then Make shot again.', 'the subject against the lens');
  cam.lookAt(0, 20, 5.1); // straight up: sky only
  eq(blankReason(frameView(scene, cam, [man]), 40), 'Frame 40 shows only empty ground or sky. Aim the camera at the subject, then Make shot again.', 'sky only');

  return { fail: problems.length, problems, line: `handoff words: ${problems.length ? 'FAIL' : 'pass'} (reveals, camera, groups, blank frames)` };
}
