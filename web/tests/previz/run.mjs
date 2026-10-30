// Offline parser tests: node tests/run.mjs   (exit code 1 on any failure)
//  1. every sentence in shot_specs.json parses to its expected spec (place, actors, props, relations, camera, unmatched)
//  2. the in-vehicle phrase table (data/in_vehicle_phrases.json): every inside phrase seats the person, outside cues never do
//  3. determinism: same sentence -> identical spec and seed; 4. speed: average parse time
import { ROOT, readJSON, parse, essential, lex } from './common.mjs';
import { runGaitChecks } from './gait.mjs';
import { runLoneFigure, runMotionAudit, runNoWeather, runShotGuard } from './shot_guard.mjs';
import { runDirector } from './director30.mjs';
import { runE3Checks } from './e3_ui.mjs';
import { runWindowSeat } from './window_seat.mjs';
import { runFinishChoice } from './finish_choice.mjs';
import { runHandoffWords } from './handoff_words.mjs';
import { runPausePov } from './pause_pov.mjs';
import { runFlowParams } from './flow_params.mjs';
import { runStep23Units } from './step23_units.mjs';
import { runStep25Units } from './step25_units.mjs';
import { runRun3Units } from './run3_units.mjs';
import { runRun6Delete } from './run6_delete.mjs';
import { runRun7CastDelete } from './run7_cast_delete.mjs';
import { runRun7Video } from './run7_video.mjs';
import { runPathEdit } from './path_edit.mjs';
import { runRun8Units } from './run8_units.mjs';
import { runRun10Units } from './run10_units.mjs';
import { runOrbitFacing } from './orbit_facing.mjs';
let fail = 0, pass = 0;
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
for (const c of readJSON('tests/shot_specs.json')) {
  const got = essential(parse(c.text));
  if (eq(got, c.expect)) { pass++; continue; }
  fail++;
  console.log('FAIL  ' + c.text);
  for (const k of Object.keys({ ...c.expect, ...got })) if (!eq(got[k], c.expect[k])) console.log('   ' + k + '\n     expected ' + JSON.stringify(c.expect[k]) + '\n     got      ' + JSON.stringify(got[k]));
}
// in-vehicle phrase table
const iv = readJSON('data/in_vehicle_phrases.json');
let ivPass = 0;
for (const [phrase, row] of Object.entries(iv.phrases)) {
  const text = /\b(driver|chauffeur|cabbie|trucker|passenger|kid|child)\b/.test(phrase) ? 'a ' + phrase : 'a man ' + phrase;
  const s = parse(text);
  const person = s.actors.find((a) => a.kind === 'person');
  const r = person && person.relations.find((x) => x.type === 'in');
  const v = r && s.props.find((p) => p.id === r.target);
  const wantVehicle = row.vehicle || null;
  const ok = r && (!wantVehicle || (v && v.asset === wantVehicle)) && (!row.seat || r.seats[0] === row.seat || (row.seat === 'passenger' && wantVehicle === 'bus') || (row.seat.startsWith('rear') && r.seats[0].startsWith('rear')));
  if (ok) ivPass++; else { fail++; console.log('FAIL  in-vehicle phrase "' + text + '" expected ' + JSON.stringify(row) + ' got ' + JSON.stringify(person && person.relations)); }
}
for (const text of ['a man next to a car', 'a woman leaning on the car', 'a man on the hood of a car', 'a man washing a car', 'a man standing behind the van', 'a dog beside the truck']) {
  const s = parse(text);
  if (s.actors.some((a) => a.relations.some((r) => r.type === 'in'))) { fail++; console.log('FAIL  outside cue seated someone: ' + text); } else ivPass++;
}
// determinism
for (const t of ['woman walking a dog in the park', 'two people sitting on a bench in the city', 'man sitting in a car']) {
  const a = parse(t), b = parse(t);
  if (!eq(a, b) || a.seed !== b.seed) { fail++; console.log('FAIL  not deterministic: ' + t); }
}
// speed
const all = readJSON('tests/shot_specs.json').map((c) => c.text);
const t0 = performance.now(); for (let k = 0; k < 20; k++) for (const t of all) parse(t);
const per = (performance.now() - t0) / (20 * all.length);
const gait = runGaitChecks();
fail += gait.fail;
if (gait.fail) for (const p of gait.problems) console.log('FAIL  gait ' + p);
const guard = await runShotGuard();
fail += guard.fail;
console.log(`shot specs: ${pass}/${pass + (fail - gait.fail - guard.fail - (Object.keys(iv.phrases).length + 6 - ivPass))} pass, in-vehicle table: ${ivPass}/${Object.keys(iv.phrases).length + 6}, lexicon ${Object.keys(lex.words).length} words + ${lex.phrases.length} phrases, parse ${per.toFixed(3)} ms per sentence`);
console.log(`gait: ${gait.line}`);
console.log(guard.line);
const weather = await runNoWeather();
fail += weather.fail;
for (const p of weather.problems) console.log('FAIL  ' + p);
console.log(weather.line);
const motion = await runMotionAudit();
fail += motion.fail;
for (const p of motion.problems) console.log('FAIL  motion ' + p);
console.log(motion.line);
const director = await runDirector();
fail += director.fail;
console.log(director.line);
const lone = await runLoneFigure();
fail += lone.fail;
for (const p of lone.problems) console.log('FAIL  ' + p);
console.log(lone.line);
const seat = await runWindowSeat();
fail += seat.fail;
for (const p of seat.problems) console.log('FAIL  window seat ' + p);
console.log(seat.line);
const fin = runFinishChoice();
fail += fin.fail;
for (const p of fin.problems) console.log('FAIL  ' + p);
console.log(fin.line);
const hand = runHandoffWords();
fail += hand.fail;
for (const p of hand.problems) console.log('FAIL  ' + p);
console.log(hand.line);
const pp = await runPausePov();
fail += pp.fail;
for (const p of pp.problems) console.log('FAIL  ' + p);
console.log(pp.line);
const e3 = runE3Checks();
fail += e3.fail;
for (const p of e3.problems) console.log('FAIL  ' + p);
console.log(e3.line);
const fp = runFlowParams();
fail += fp.fail;
for (const p of fp.problems) console.log('FAIL  ' + p);
console.log(fp.line);
const s23 = runStep23Units();
fail += s23.fail;
for (const p of s23.problems) console.log('FAIL  ' + p);
console.log(s23.line);
const s25 = await runStep25Units();
fail += s25.fail;
for (const p of s25.problems) console.log('FAIL  ' + p);
console.log(s25.line);
const r3 = await runRun3Units();
fail += r3.length;
for (const p of r3) console.log('FAIL  ' + p);
console.log(`run 3 units (shot cast only, no keyframe switch, empty place, traffic, camera car, trees): ${r3.length ? `${r3.length} failed` : 'all pass'}`);
const r6 = runRun6Delete();
fail += r6.length;
for (const p of r6) console.log('FAIL  ' + p);
console.log(`run 6 delete (next project, confirm line, gone references): ${r6.length ? `${r6.length} failed` : 'all pass'}`);
const r7c = runRun7CastDelete();
fail += r7c.length;
for (const p of r7c) console.log('FAIL  ' + p);
console.log(`run 7 cast delete (confirm words, gone character in a reopened form): ${r7c.length ? `${r7c.length} failed` : 'all pass'}`);
const r7v = runRun7Video();
fail += r7v.length;
for (const p of r7v) console.log('FAIL  ' + p);
console.log(`run 7 video (sizes, 8n+1 lengths, centre crop, picker placement): ${r7v.length ? `${r7v.length} failed` : 'all pass'}`);
const pe = runPathEdit();
fail += pe.fail;
for (const p of pe.problems) console.log('FAIL  ' + p);
console.log(pe.line);
const r8 = runRun8Units();
fail += r8.length;
for (const p of r8) console.log('FAIL  ' + p);
console.log(`run 8 units (lock seed, measured times, kinds): ${r8.length ? `${r8.length} failed` : 'all pass'}`);
const r10 = runRun10Units();
fail += r10.length;
for (const p of r10) console.log('FAIL  ' + p);
console.log(`run 10 units (the Look in plain English, the seed lock): ${r10.length ? `${r10.length} failed` : 'all pass'}`);
const of = await runOrbitFacing();
fail += of.fail;
console.log(of.line);
console.log(fail ? `FAILED (${fail})` : 'ALL PASS');
void ROOT;
process.exit(fail ? 1 : 0);
