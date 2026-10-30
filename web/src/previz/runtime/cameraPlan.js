// Camera plan from the sentence. The same words always give the same segments.
// A segment is one move. The first move word in a segment wins. "reveal" only widens the end.

const PHRASES = [
  ['pull back from a close up', 'pull_out'],
  ['crane up to reveal', 'crane_reveal'],
  ['comes down to eye level', 'crane_down'],
  ['lifts over the shoulder', 'boom_up'],
  ['pulls up and away overhead', 'rise'],
  ['camera rises straight up', 'rise'],
  ['rises straight up', 'rise'],
  ['top down rising', 'rise'],
  ['overhead rise', 'rise'],
  ['rises up behind', 'boom_up'],
  ['follows close behind', 'fpv'],
  ['flies over', 'aerial'],
  ['flying over', 'aerial'],
  ['bird\'s eye flyover', 'aerial'],
  ['birds eye flyover', 'aerial'],
  ['documentary style', 'handheld'],
  ['no camera move', 'static'],
  ['still camera', 'static'],
  ['fixed camera', 'static'],
  ['locked off', 'static'],
  ['locked-off', 'static'],
  ['hand held', 'handheld'],
  ['handheld', 'handheld'],
  ['dutch angle', 'dutch'],
  ['tilted frame', 'dutch'],
  ['body mount', 'body_mount'],
  ['body cam', 'body_mount'],
  ['snorricam', 'body_mount'],
  ['dolly zoom', 'dolly_zoom'],
  ['vertigo shot', 'dolly_zoom'],
  ['contra zoom', 'dolly_zoom'],
  ['drone shot', 'aerial'],
  ['drone chase', 'fpv'],
  ['aerial shot', 'aerial'],
  ['fpv follow', 'fpv'],
  ['fpv chase', 'fpv'],
  ['chase cam', 'fpv'],
  ['crane shot', 'crane_up'],
  ['crane reveal', 'crane_reveal'],
  ['crane down', 'crane_down'],
  ['crane up', 'crane_up'],
  ['cranes down', 'crane_down'],
  ['cranes up', 'crane_up'],
  ['boom down', 'boom_down'],
  ['boom up', 'boom_up'],
  ['booms up', 'boom_up'],
  ['pedestal up', 'boom_up'],
  ['drone up', 'rise'],
  ['tilt down', 'tilt_down'],
  ['tilts down', 'tilt_down'],
  ['tilt up', 'tilt_up'],
  ['tilts up', 'tilt_up'],
  ['tilts up to', 'tilt_up'],
  ['tilts with', 'tilt_up'],
  ['whip pan', 'pan'],
  ['pan left', 'pan_left'],
  ['pan right', 'pan_right'],
  ['pans across', 'pan'],
  ['pans with', 'pan'],
  ['truck left', 'truck_left'],
  ['truck right', 'truck_right'],
  ['trucks left', 'truck_left'],
  ['trucks right', 'truck_right'],
  ['trucks past', 'truck_right'],
  ['crab left', 'truck_left'],
  ['crab right', 'truck_right'],
  ['dolly along', 'dolly_along'],
  ['lateral dolly', 'dolly_along'],
  ['low dolly', 'dolly_along'],
  ['dolly out', 'pull_out'],
  ['dolly in', 'push_in'],
  ['tracks alongside', 'track'],
  ['track alongside', 'track'],
  ['pushes in', 'push_in'],
  ['pushing in', 'push_in'],
  ['rotates around', 'orbit'],
  ['orbits around', 'orbit'],
  ['orbiting around', 'orbit'],
  ['pulls back', 'pull_out'],
  ['tracking shot', 'track'],
  ['runs alongside', 'track'],
  ['travels with', 'track'],
  ['side by side', 'track'],
  ['follow shot', 'follow'],
  ['circles around', 'orbit'],
  ['rotate around', 'orbit'],
  ['arc around', 'orbit'],
  ['go around', 'orbit'],
  ['full circle', 'orbit'],
  ['all the way around', 'orbit'],
  ['half orbit', 'orbit'],
  ['quarter orbit', 'orbit'],
  ['orbit 360', 'orbit'],
  ['circle around', 'orbit'],
  ['push in', 'push_in'],
  ['move in', 'push_in'],
  ['creep in', 'push_in'],
  ['close in', 'push_in'],
  ['slow push', 'push_in'],
  ['pull back', 'pull_out'],
  ['pull out', 'pull_out'],
  ['pull away', 'pull_out'],
  ['pulls wide', 'pull_out'],
  ['back away', 'pull_out'],
  ['zoom in', 'push_in'],
  ['zoom out', 'pull_out'],
  ['steady cam', 'follow'],
  ['steadicam', 'follow'],
  ['behind them', 'follow'],
  ['comes down', 'crane_down'],
  ['descends', 'crane_down'],
  ['lowers', 'crane_down'],
  ['camera rises', 'boom_up'],
  ['rises', 'boom_up'],
  ['tracking', 'track'],
  ['parallel', 'track'],
  ['orbit', 'orbit'],
  ['orbits', 'orbit'],
  ['orbiting', 'orbit'],
  ['circles', 'orbit'],
  ['circle', 'orbit'],
  ['360', 'orbit'],
  ['crane', 'crane_up'],
  ['dolly', 'push_in'],
  ['follow', 'follow'],
  ['follows', 'follow'],
  ['aerial', 'aerial'],
  ['drone', 'aerial'],
  ['fpv', 'fpv'],
  ['static', 'static'],
  ['tripod', 'static'],
  ['locked', 'static'],
  ['handheld', 'handheld'],
  ['shaky', 'handheld'],
  ['dutch', 'dutch'],
  ['canted', 'dutch'],
  ['pan', 'pan'],
  ['zolly', 'dolly_zoom'],
  ['swoops', 'swoop'],
  ['swoop', 'swoop'],
];

const MODS = new Set(['handheld', 'dutch']);
const SPLIT = /\b(?:and then|after that|at the end|opens on|begins|starts|finally|before|then|next|lands on|settles|ends)\b/;

function boundary(text, index, length) {
  const before = index === 0 || /[^a-z0-9]/.test(text[index - 1]);
  const after = index + length >= text.length || /[^a-z0-9]/.test(text[index + length]);
  return before && after;
}

function findHits(low) {
  const hits = [];
  for (let i = 0; i < low.length; i++) {
    if (i > 0 && /[a-z0-9]/.test(low[i - 1])) continue;
    for (const [phrase, id] of PHRASES) {
      if (!low.startsWith(phrase, i) || !boundary(low, i, phrase.length)) continue;
      if (id === 'track' && /(?:on a |running )$/.test(low.slice(Math.max(0, i - 12), i))) break;
      if ((id === 'aerial' && phrase === 'drone') && /\ba drone flying\b/.test(low) && !/\bdrone shot\b/.test(low)) break;
      if ((phrase === 'flying over' || phrase === 'flies over') && /\b(bird|birds|plane|airplane|seagull|hawk)\b/.test(low.slice(0, i))) break;
      // "orbits" is a planned move only with its sides named (Run 14); otherwise the shot's own orbit framing stays
      if (/^orbit(?:s|ing)/.test(phrase) && !Object.keys(orbitSides(low)).length) continue;
      if (phrase === 'rises' &&/rises straight up/.test(low.slice(i, i + 20))) continue;
      if (phrase === 'crane' && /crane (?:up|down|shot|reveal)/.test(low.slice(i, i + 16))) continue;
      if (phrase === 'dolly' && /dolly (?:along|in|out|zoom)/.test(low.slice(i, i + 14))) continue;
      if (phrase === 'follow' && /follow(?:s| shot)/.test(low.slice(i, i + 12))) continue;
      hits.push({ index: i, length: phrase.length, id, phrase });
      i += phrase.length - 1;
      break;
    }
  }
  return hits;
}

function paramsOf(low, index, length, id) {
  const window = low.slice(Math.max(0, index - 24), Math.min(low.length, index + length + 28));
  const params = {};
  const deg = window.match(/(\d+)\s*(?:degree|deg)/);
  if (deg) params.angle = Math.min(360, parseInt(deg[1], 10));
  else if (id === 'orbit' && /\b360\b|full circle|all the way/.test(window)) params.angle = 360;
  else if (id === 'orbit' && /half orbit/.test(window)) params.angle = 180;
  else if (id === 'orbit' && /quarter orbit/.test(window)) params.angle = 90;
  else if (id === 'orbit') params.angle = 90;
  if (/\b(left to right)\b/.test(window)) params.direction = 'left_to_right';
  else if (/\b(right to left)\b/.test(window)) params.direction = 'right_to_left';
  else if (/\bclockwise\b/.test(window)) params.direction = 'clockwise';
  else if (/\bcounterclockwise\b/.test(window)) params.direction = 'counterclockwise';
  else if (/\bleft\b/.test(window)) params.direction = 'left';
  else if (/\bright\b/.test(window)) params.direction = 'right';
  if (/\b(slowly|slow)\b/.test(window)) params.speed = 0.6;
  if (/\b(fast|quick|whip)\b/.test(window)) params.speed = 1.6;
  if (id === 'orbit') Object.assign(params, orbitSides(low));
  if (/from high/.test(window)) params.from_height = 'high';
  if (/to eye level|eye level/.test(window)) params.to_height = 'eye';
  if (/from the ground|on the ground/.test(window)) params.from_height = 'ground';
  return params;
}

// Run 14 (bug 6): where an orbit starts and ends around the subject, in degrees (0 in front of them, 180 behind,
// 90 at their side): "starts from behind and orbits around to the front" goes 180 -> 0.
const SIDE = '(behind|in front|the front|front|the back|back|the side|side)';
const SIDE_DEG = (w) => (/behind|back/.test(w) ? 180 : /side/.test(w) ? 90 : 0);
function orbitSides(low) {
  const out = {};
  const from = low.match(new RegExp('\\b(?:starts?|starting|begins?|beginning|opens?|from)\\s+(?:out\\s+)?(?:from\\s+)?(?:directly\\s+|right\\s+)?' + SIDE + '\\b'));
  const to = low.match(new RegExp('\\b(?:around|round|to|ends?|ending|finish(?:es)?|lands?|settles?)\\s+(?:up\\s+)?(?:to\\s+|at\\s+|in\\s+)?(?:directly\\s+|right\\s+)?' + SIDE + '\\b'));
  if (from) out.from_deg = SIDE_DEG(from[1]);
  if (to && (!from || to.index !== from.index)) out.to_deg = SIDE_DEG(to[1]);
  if (out.from_deg != null && out.to_deg === out.from_deg) delete out.to_deg;
  return out;
}

function tracks(id, low, index) {
  if (['track', 'follow', 'fpv', 'body_mount'].includes(id)) return 1;
  if (id === 'aerial' && /\b(car|skate|walk|jog|run|driv|rid|bike|sprint|board)\b/.test(low)) return 1;
  if (id === 'push_in' || id === 'pull_out') {
    const around = low.slice(Math.max(0, index - 40), index + 48);
    if (/reveal|skates off|into the distance|off into/.test(around)) return 0;
    if (/walk|jog|run|skat|rid/.test(low)) return 1;
  }
  return 0;
}

const TIMES = { 1: [0, 1], 2: [0, 0.45, 1], 3: [0, 0.35, 0.7, 1] };

/** planCamera(text) -> { move, segments, handheld, dutch, lensMm, facing, reveal, chips } */
export function planCamera(text) {
  const low = String(text || '').toLowerCase().replace(/-/g, ' ');
  const hits = findHits(low);
  const moves = [];
  for (const hit of hits) {
    if (MODS.has(hit.id)) continue;
    const prev = moves[moves.length - 1];
    const between = prev ? low.slice(prev.index + prev.length, hit.index) : '';
    const split = SPLIT.test(between) || /\band\b/.test(between);
    if (prev && !split) continue;
    moves.push(hit);
    if (moves.length === 3) break;
  }
  const n = moves.length;
  const cuts = n ? TIMES[n] || TIMES[3] : [];
  const ramp = low.match(/(\d+)\s*(?:mm\s*)?to\s+(\d+)\s*mm/);
  const segments = moves.map((hit, i) => {
    const params = paramsOf(low, hit.index, hit.length, hit.id);
    if (hit.id === 'dolly_zoom' && ramp) {
      params.focal_from = Math.max(8, Math.min(200, parseInt(ramp[1], 10)));
      params.focal_to = Math.max(8, Math.min(200, parseInt(ramp[2], 10)));
    }
    return {
      move: hit.id,
      from_t: cuts[i],
      to_t: cuts[i + 1],
      params,
      tracking: tracks(hit.id, low, hit.index),
    };
  });
  const handheld = /\b(handheld|hand held|shaky|documentary style)\b/.test(low)
    ? { seed: 1, pos_m: 0, rot_deg: 0, hz: 0 }
    : null;
  const dutch = /\b(dutch|dutch angle|canted|tilted frame)\b/.test(low);
  let lensMm = null;
  const mm = low.match(/(\d+)\s*(?:mm|millimeter)\b/);
  if (mm) lensMm = Math.max(8, Math.min(200, parseInt(mm[1], 10)));
  else if (/\blong lens\b/.test(low)) lensMm = 85;
  else if (/\bwide lens\b/.test(low)) lensMm = 24;
  let facing = null;
  if (/\b(toward|towards)\s+(?:the\s+)?(?:camera|lens|us)\b|\b(?:toward|towards)\s+us\b/.test(low)) facing = 'toward';
  else if (/\baway from\s+(?:the\s+)?(?:camera|lens)\b|\baway\b/.test(low)) facing = 'away';
  else if (/\bleft to right\b/.test(low)) facing = 'left_to_right';
  else if (/\bacross\b/.test(low) && !/\bacross from\b/.test(low)) facing = 'across';
  const chips = [];
  if (/\brack focus\b/.test(low)) chips.push('Rack focus');
  const reveal = /\breveal/.test(low);
  return {
    move: segments.length ? segments[0].move : null,
    segments,
    handheld,
    dutch,
    lensMm,
    facing,
    reveal,
    chips,
    speed: /\b(slow motion|slo-mo|in slow motion)\b/.test(low) ? 0.5 : null,
  };
}
