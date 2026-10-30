// Fix E3 checks that run in Node: the "Try:" row, the Look prompt and the plain-error map.
import { grammar, parse } from './common.mjs';
import { insertFragment, tryFragments, tryPools } from '../../src/previz/tryRow.ts';
import { lookPromptFor, withoutLookLine } from '../../src/previz/lookPrompt.ts';
import { LIBRARY_FAILED, plainError } from '../../src/previz/errors.ts';

export function runE3Checks() {
  const problems = [];
  const pools = tryPools(grammar, parse);
  for (const kind of ['place', 'action', 'camera']) if (pools[kind].length < 4) problems.push(`try row: only ${pools[kind].length} ${kind} fragments`);
  let parsed = 0;
  let total = 0;
  for (const text of [...pools.place, ...pools.action, ...pools.camera]) {
    total++;
    const s = parse(`A man walking in the park, ${text}`);
    const alone = parse(text);
    if (s && alone && (alone.matched.length > 0 || alone.camera.move || alone.place.source === 'word')) parsed++;
    else problems.push(`try row: "${text}" doesn't parse`);
  }
  const a = tryFragments(grammar, 0, parse).map((f) => f.text);
  const b = tryFragments(grammar, 0, parse).map((f) => f.text);
  const c = tryFragments(grammar, 1, parse).map((f) => f.text);
  if (a.length < 4 || a.length > 6) problems.push(`try row: ${a.length} fragments`);
  if (a.join('|') !== b.join('|')) problems.push('try row: same seed, different fragments');
  if (a.join('|') === c.join('|')) problems.push('try row: the seed does not rotate the set');
  const ins = insertFragment('A man sitting in a booth', 24, 'slow push in');
  if (ins.text !== 'A man sitting in a booth, slow push in') problems.push(`insert: "${ins.text}"`);
  const mid = insertFragment('A woman. Camera orbits', 8, 'on a rooftop');
  if (!mid.text.startsWith('A woman. On a rooftop')) problems.push(`insert mid: "${mid.text}"`);
  const seat = 'A man sitting alone in a booth in a diner at night, looking out the window. The camera starts wide and slowly pushes in, and at the end he turns and looks at the camera.';
  const look = lookPromptFor(seat);
  // Run 9: the time of day and what they wear lead (they were lost when they came last).
  if (!look.startsWith('It is night. The grey figure is a man sitting alone in a booth in a diner at night, looking out the window.') || /camera/i.test(look)) problems.push(`look prompt: "${look}"`);
  const pier = lookPromptFor('A man in a long coat stands on a pier at dusk. The camera slowly pushes in.');
  if (!pier.startsWith('It is dusk. The man wears a long coat. The grey figure is a man in a long coat who stands on a pier at dusk.')) problems.push(`look lead: "${pier}"`);
  if (!lookPromptFor('Two women in red dresses walk along the beach at sunset.').startsWith('It is sunset. The two women wear red dresses.')) problems.push('look lead: plural wardrobe');
  if (/wears/.test(lookPromptFor('A man sitting alone in a diner.'))) problems.push('look lead: "in a diner" is no wardrobe');
  const cast = lookPromptFor('Maya walks down a street at dusk.', [{ id: 'c1', name: 'Maya', kind: 'person', line: 'a woman in her thirties with short black hair', url: '' }]);
  if (!cast.includes('Maya is a woman in her thirties with short black hair.')) problems.push(`look prompt cast: "${cast}"`);
  // The look line: night, then interior, then golden, else the default; whole words, any case; never "camera".
  const FILM = 'A real photograph in the look of a large-format 65mm commercial film frame, natural colour, sharp and clean.';
  const LINE = {
    night: `${FILM} Practical lights glow against deep blue shadows, soft highlight roll-off, fine grain.`,
    interior: `${FILM} Soft window light falls across the room with gentle falloff into the shadows, fine grain.`,
    golden: `${FILM} Low golden sun rakes across the scene with long soft shadows and warm highlights, fine grain.`,
    default: `${FILM} Soft directional light that suits the place and time, shallow depth of field, fine grain.`,
  };
  for (const [text, want] of [
    [seat, 'night'], ['A woman runs across a bridge.', 'default'], ['A man waits in the kitchen.', 'interior'],
    ['A couple walks the beach at Golden Hour.', 'golden'], ['A man in a diner at sunset.', 'interior'], ['A van under NEON signs indoors.', 'night'],
    ['A cyclist rides out After Dark.', 'night'], ['A man in his bathroom.', 'default'], ['Two kids play in the late afternoon.', 'golden'],
    ['A girl sings at a nightclub.', 'default'], ['A runner at dawn in the hallway.', 'interior'],
  ]) {
    const got = lookPromptFor(text);
    if (!got.endsWith(LINE[want])) problems.push(`look line ${want}: "${got}"`);
    if (/camera/i.test(got)) problems.push(`look line says camera: "${got}"`);
  }
  if (!lookPromptFor('A woman runs across a bridge.').startsWith('The grey figure is a woman who runs across a bridge.')) problems.push('look line: the scene sentence comes first');
  if (withoutLookLine(look) !== look.replace(` ${LINE.night}`, '')) problems.push(`look line strip: "${withoutLookLine(look)}"`);
  const saved = console.error;
  console.error = () => {};
  const raw = [new Error('kit rooftop 404'), new TypeError("Cannot read properties of undefined (reading 'x')"), Object.assign(new Error('POST /api/previz/x 500'), { status: 500 })];
  for (const [e, where] of [[raw[0], 'build'], [raw[1], 'build'], [raw[2], 'upload'], [raw[2], 'finish']]) {
    const said = plainError(e, where);
    if (/404|500|TypeError|undefined|\/api|rooftop|\.glb/.test(said)) problems.push(`plain error leaks: "${said}"`);
  }
  const server = plainError(Object.assign(new Error('Frame 41 came out blank. Check the preview, then try Make shot again.'), { status: 400 }), 'finish');
  console.error = saved;
  if (!server.startsWith('Frame 41')) problems.push(`plain error drops a plain server line: "${server}"`);
  if (!LIBRARY_FAILED.includes("didn't load")) problems.push('library notice');
  return { fail: problems.length, problems, line: `e3: try row ${parsed}/${total} fragments parse (${pools.place.length} places, ${pools.action.length} actions, ${pools.camera.length} moves), look prompt, plain errors: ${problems.length ? 'FAIL' : 'pass'}` };
}
