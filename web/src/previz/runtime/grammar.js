// Shot-spec parser: a small rule-based scene grammar. No AI, no network, deterministic.
//   text -> tokens -> lemmas -> phrase matching (longest first) -> tagged items -> subject / verb / preposition / object
//   frame -> post rules (vehicle seats, bench slots, leashes, implied props, place, camera subject) -> shot spec (JSON)
// The shot spec is the contract between this parser and the builder (runtime/builder.js). See SHOT_SPEC.md.
import { tokenize, CATEGORY, SEATING, SURFACES } from './lexicon.js';
import { planCamera } from './cameraPlan.js';

export const SPEC_VERSION = 1;
const ANIMAL_POSE = { stand: 'stand', walk: 'walk', jog: 'run', run: 'run', sit: 'sit', lie_down: 'lie', crouch: 'sit', kneel: 'sit', sip: 'drink', drink: 'drink', eat: 'eat', fly: 'fly', fly_kite: 'fly' }; // v1.4: drink/eat/fly resolve per animal (poser.js ANIMAL_FALLBACK)
const SEATS = {
  car: ['driver', 'passenger', 'rear_R', 'rear_L'], suv: ['driver', 'passenger', 'rear_R', 'rear_L'], van: ['driver', 'passenger', 'rear_R', 'rear_L'],
  truck: ['driver', 'passenger'],
  bus: ['driver', 'passenger', 'row1_L_win', 'row1_R_win', 'row2_L_win', 'row2_R_win', 'row3_L_win', 'row3_R_win', 'row4_L_win', 'row4_R_win', 'rear_L', 'rear_R'],
};
const RIDER_POSE = { bicycle: 'ride_bike', skateboard: 'ride_skate', boat: 'sit_boat' };

/** FNV-1a 32-bit hash of the normalized sentence: the scene seed. */
export function hashSeed(text) {
  const s = tokenize(text).join(' ');
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

/** Tag the sentence: phrase matches first (on lemmas), then single words. */
export function tagTokens(text, lex) {
  const raw = tokenize(text);
  const lem = raw.map((w) => lex.lemma(w));
  const out = [];
  for (let i = 0; i < raw.length;) {
    let hit = null;
    for (const p of lex.phrases) {
      const n = p.tokens.length;
      if (n < 2 || i + n > lem.length) continue;
      let ok = true;
      for (let k = 0; k < n; k++) if (p.tokens[k] !== lem[i + k] && p.tokens[k] !== raw[i + k]) { ok = false; break; }
      if (ok) { hit = p; break; }
    }
    if (hit) { out.push({ text: raw.slice(i, i + hit.tokens.length).join(' '), lemma: hit.tokens.join(' '), entry: hit.entry, phrase: true }); i += hit.tokens.length; continue; }
    const w = raw[i], l = lem[i];
    let e = lex.words[l] || lex.words[w] || null;
    const v = lex.verbs[l];
    const prev = out.length ? out[out.length - 1].entry : null;
    const verbSlot = prev && (prev.kind === 'entity' || prev.kind === 'group' || prev.kind === 'role' || prev.kind === 'pron' || prev.kind === 'conj' || (prev.kind === 'stop' && /^(is|are|was|were)$/.test(out[out.length - 1].text)));
    if (v && e && e.kind !== 'verb' && (w.endsWith('ing') || (verbSlot && w !== l && /s$/.test(w) && e.kind === 'entity') || (verbSlot && e.kind === 'entity' && w === l && !/^(phone|boat|kayak|canoe)$/.test(w)))) e = v;
    if (!e && w.endsWith('ing') && !v) e = null;
    else if (!e && v) e = v;
    if (!e && /^\d+$/.test(w)) e = { kind: 'num', count: parseInt(w, 10) };
    out.push({ text: w, lemma: l, entry: e, plural: !!(e && e.kind === 'entity' && (w !== l && !w.endsWith("'s") || w === 'people')) });
    i++;
  }
  return out;
}

const CAMERA_STEM = {
  push: 'push_in', pull: 'pull_out', pan: 'pan', track: 'track', follow: 'follow',
  crane: 'crane_up', orbit: 'orbit', circle: 'orbit', rotate: 'orbit',
};
function wordStem(word) {
  if (!word) return '';
  if (word.endsWith('ies')) return word.slice(0, -3) + 'y';
  if (word.endsWith('ing')) {
    const s = word.slice(0, -3);
    return /(.)\1$/.test(s) ? s.slice(0, -1) : s;
  }
  if (word.endsWith('es')) return word.slice(0, -2);
  if (word.endsWith('s') && !word.endsWith('ss')) return word.slice(0, -1);
  return word;
}
/** A camera-clause verb. eat means the next token is the direction (in / out) and is not a preposition. */
function cameraMoveOf(entry, word, nextWord) {
  const stems = [wordStem(entry && entry.verb), wordStem(word), wordStem(entry && entry.pose)].filter(Boolean);
  const has = (s) => stems.includes(s);
  const out = nextWord === 'out' || nextWord === 'back';
  const inn = nextWord === 'in';
  if ((has('move') || has('zoom') || has('dolly') || has('push')) && inn) return { move: 'push_in', eat: true };
  if ((has('move') || has('zoom') || has('dolly') || has('pull')) && out) return { move: 'pull_out', eat: true };
  if (has('go') && (nextWord === 'around' || nextWord === 'round')) return { move: 'orbit', eat: true };
  if (has('crane') && (nextWord === 'down' || nextWord === 'descends')) return { move: 'crane_down', eat: nextWord === 'down' };
  if (has('dolly') && (nextWord === 'along' || nextWord === 'past')) return { move: 'dolly_along', eat: false };
  if (has('dolly')) return { move: 'push_in', eat: false };
  if (has('zoom')) return { move: 'push_in', eat: false };
  if (has('move')) return { move: null, eat: false };
  for (const s of stems) if (CAMERA_STEM[s]) return { move: CAMERA_STEM[s], eat: false };
  return null;
}

/**
 * parseShot(text, lex, grammar) -> shot spec
 * lex = buildLexicon(manifest, grammar). Pure function: the same text always gives the same spec.
 */
/** Where an "at the end" action starts, as a share of the shot (frame 76 of 97), so it lands in the last second. */
const END_FROM = 0.79;

/** A person pausing: "pauses", "stops for a second", "waits", "stands still". */
export const PAUSE = /\b(paus(e|es|ed|ing)|stops?|stopped|stopping|waits?|waited|waiting|halts?|halted|freezes|froze|stands still|holds still|catches (his|her|their) breath)\b/i;

/**
 * The shot-time spans of the pauses between moves. The action's clauses (split at commas, "then", "before"; camera
 * sentences left out) share the shot evenly, and each pausing clause is a hold over its share, 12% to 40% of the shot.
 * A pause with no move before or after it ("a man waits at the bus stop") holds nothing: it's a pose, not a beat.
 */
export function holdSpans(text) {
  const action = String(text || '').split(/(?<=[.!?])\s+/).filter((s) => !/\b(camera|lens|shot)\b/i.test(s)).join(' ');
  const clauses = action.split(/[,;.!?]|\bthen\b|\bbefore\b|\band\b(?=\s+(?:paus|stop|wait|halt))/i).map((c) => c.trim()).filter(Boolean);
  const n = clauses.length;
  if (n < 2) return [];
  const out = [];
  clauses.forEach((c, i) => {
    if (!PAUSE.test(c)) return;
    if (/\b(car|suv|van|truck|bus|taxi|train|bike|boat|traffic|light)s?\b/i.test(c)) return; // the car stops, not the person
    if (!clauses.some((d, j) => j !== i && !PAUSE.test(d))) return;
    const mid = (i + 0.5) / n;
    const half = Math.min(0.2, Math.max(0.06, 0.5 / n));
    out.push([+Math.max(0, mid - half).toFixed(3), +Math.min(1, mid + half).toFixed(3)]);
  });
  return out;
}

export function parseShot(text, lex, grammar) {
  const items = tagTokens(text, lex);
  const ents = [];            // mention order
  const matched = [], unmatched = [];
  const camera = { subject: null, shot: null, angle: null, move: null };
  const world = { time: null, weather: null };
  let place = null, placeWord = null;
  let count = null, size = 1, sawDet = false;
  let subjects = [], clauseVerb = false, conj = false, prep = null, lastVerb = null, outside = false, cameraClause = false;
  const notes = [];
  const beats = [];
  let pairHint = null, heldChain = null, placeByPrep = false, placeStandin = false; // v1.4: two-person pair id from the sentence; last 'holds' verb for 'a phone and a cup'

  const newEnt = (fields) => { const e = Object.assign({ relations: [], size: 1, pose: null, implied: false }, fields); ents.push(e); return e; };
  const isActor = (e) => e.cat === 'person' || e.cat === 'animal';
  const lastPerson = (g) => { for (let i = ents.length - 1; i >= 0; i--) if (ents[i].cat === 'person' && (!g || !ents[i].gender || ents[i].gender === g)) return ents[i]; return null; };
  const rel = (s, type, target, extra = {}) => { if (s && target && s !== target && !s.relations.some((r) => r.type === type && r.target === target)) s.relations.push(Object.assign({ type, target }, extra)); };
  const subjectsOrImplied = () => {
    if (!subjects.length) { const p = newEnt({ cat: 'person', asset: 'person', word: '(implied)', count: 1, preset: 'adult', implied: true }); subjects = [p]; }
    return subjects;
  };
  const implyProp = (asset, owner, word) => newEnt({ cat: CATEGORY[asset], asset, word: word || '(implied)', count: owner ? owner.count : 1, implied: true, scale: 1 });

  function applyPrep(r, obj) {
    const subs = subjects.length ? subjects : [];
    for (const s of subs) {
      if (s === obj) continue;
      const c = obj.cat;
      if (c === 'vehicle') {
        if (s.cat !== 'person') { rel(s, 'near', obj, { side: 'beside' }); if (r === 'in') notes.push(`a ${s.asset} cannot sit in a vehicle yet: placed beside it`); continue; }
        if (outside) s.outsideVehicle = true;
        if (outside || ['near', 'behind', 'front', 'across', 'facing', 'against', 'around', 'outside', 'past', 'along', 'toward', 'under', 'from'].includes(r)) rel(s, 'near', obj, { side: sideOf(r) });
        else if (r === 'in' || r === 'of' || r === 'at' && s.seatPref) rel(s, 'in', obj);
        else if (r === 'on') rel(s, obj.asset === 'bus' ? 'in' : 'near', obj, obj.asset === 'bus' ? {} : { side: 'beside' });
        else if (r === 'with') rel(s, 'near', obj, { side: 'beside' });
        else rel(s, 'near', obj, { side: sideOf(r) });
      } else if (c === 'furniture') {
        if ((r === 'on' || r === 'in' || r === 'at' && s.pose === 'sit') && SEATING.includes(obj.asset) && s.cat !== 'handheld') rel(s, 'on', obj);
        else if (r === 'on' && SURFACES.includes(obj.asset) && s.cat !== 'person' && s.cat !== 'animal') rel(s, 'on', obj);
        else if ((r === 'on' || r === 'at' || r === 'behind') && SURFACES.includes(obj.asset)) rel(s, 'near', obj, { side: 'front' });
        else rel(s, 'near', obj, { side: sideOf(r) });
      } else if (c === 'rideable') {
        if ((r === 'on' || r === 'in') && s.cat === 'person') rel(s, 'rides', obj);
        else rel(s, 'near', obj, { side: sideOf(r) });
      } else if (c === 'handheld') {
        if ((r === 'with' || r === 'in' || (r === 'on' && obj.asset === 'phone')) && s.cat === 'person') rel(s, 'holds', obj);
        else if (r === 'on' && s.cat !== 'person') rel(s, 'near', obj, { side: 'beside' });
        else rel(s, 'near', obj, { side: sideOf(r) });
      } else if (c === 'animal' || c === 'person') {
        if (r === 'with' && s.cat === 'person' && c === 'animal') rel(s, 'with', obj);
        else if (r === 'with') rel(s, 'near', obj, { side: 'beside' });
        else if (r === 'facing' || r === 'toward' || r === 'across' || r === 'at') rel(s, 'facing', obj);
        else if (r === 'on' && s.cat === 'animal' && c === 'person') rel(s, 'near', obj, { side: 'beside' });
        else rel(s, 'near', obj, { side: sideOf(r) });
      } else { // set dressing
        rel(s, 'near', obj, { side: sideOf(r) });
      }
    }
  }
  // "at the end", "in the end", "finally", "by the end": the action belongs to the last quarter of the shot, not the 70 percent mark.
  const END_WORDS = /^(end|finally|last|final|ending)$/;
  const atEnd = (idx) => { for (let k = idx; k >= 0; k--) { const t = items[k].text; if (t === '.' || t === ';') break; if (END_WORDS.test(t)) return true; } return false; };
  const lookBeat = (idx, ahead) => {
    const who = subjects[0] || lastPerson(null);
    const end = atEnd(idx);
    const out = /\b(?:out|through)\b.*\bwindows?\b|\bwindows?\b/.test(ahead) && !/camera|lens/.test(ahead);
    const type = /camera|lens/.test(ahead) ? 'look_at_camera' : out ? 'look_out' : /\bup\b/.test(ahead) ? 'look_up' : 'look';
    // looking out of a window is where the eyes are from the first frame; it only ends when something else turns the head
    if (type === 'look_out') { if (!beats.some((b) => b.type === 'look_out' && b.who === who)) beats.push({ who, type, from_t: 0, to_t: 1, params: { target: 'window' } }); return; }
    // "turns and looks at the camera" is one beat
    if (type === 'look_at_camera' && beats.some((b) => b.type === 'look_at_camera' && b.who === who)) return;
    beats.push({ who, type, from_t: end ? END_FROM : 0.7, to_t: 1, params: {} });
  };
  const sideOf = (r) => ({ behind: 'behind', front: 'front', across: 'across', facing: 'front', under: 'under', against: 'against', around: 'around', past: 'beside', along: 'beside', toward: 'front' }[r] || 'beside');

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    if (it.text === '.' || it.text === ';') {
      subjects = []; prep = null; conj = false; clauseVerb = false; lastVerb = null;
      cameraClause = false; outside = false; heldChain = null; count = null;
      continue;
    }
    const e = it.entry;
    if (!e) {
      // "turns (and looks) at the camera": a head turn to the lens. "the end" of "at the end" is a time word, not an unknown noun.
      if (/^turn(?:s|ed|ing)?$/.test(it.text)) {
        const ahead = items.slice(i, i + 6).map((x) => x.text).join(' ');
        if (/camera|lens/.test(ahead)) lookBeat(i, ahead);
        else unmatched.push(it.text);
        continue;
      }
      if (END_WORDS.test(it.text) && i > 0 && /^(?:the|very)$/.test(items[i - 1].text)) continue;
      if (!/^\d+$/.test(it.text)) unmatched.push(it.text);
      // an unknown verb ('washing a car') still links its subject to the object: placed near it
      if (/ing$/.test(it.text) && subjects.length) { lastVerb = { verb: it.text, obj: { any: 'near' } }; clauseVerb = true; prep = null; }
      continue;
    }
    if (e.kind !== 'stop' && e.kind !== 'det') matched.push({ text: it.text, as: e.kind === 'entity' ? e.asset : e.kind === 'verb' ? (e.pose || e.rides || e.verb) : e.kind === 'place' ? e.place : e.kind });
    switch (e.kind) {
      case 'stop':
        if (/^(while|as|then|before|next|finally)$/.test(it.text)) {
          subjects = []; prep = null; conj = false; clauseVerb = false; lastVerb = null;
          cameraClause = false; outside = false; heldChain = null;
        }
        break;
      case 'det': sawDet = true; break;
      case 'num': case 'quant':
        count = e.count;
        if (ents.length && ents[ents.length - 1].group) ents[ents.length - 1].count = e.count;
        break;
      case 'size': size *= e.factor; break;
      case 'conj':
        if (/^(as|while|then)$/.test(it.text)) cameraClause = false;
        conj = true; prep = null; outside = false; break;
      case 'camera':
        cameraClause = true;
        if (e.shot) camera.shot = e.shot;
        if (e.angle && !camera.angle) camera.angle = e.angle;
        if (e.from) camera.from = e.from;
        if (e.move === 'crane_reveal') camera.move = 'crane_reveal';
        else if (e.move && !camera.move) camera.move = e.move;
        if (e.chip) notes.push(e.chip);
        if (e.from === 'doorway' && !place) { place = 'interior'; placeWord = 'doorway'; placeStandin = true; placeByPrep = true; }
        break;
      case 'world': for (const k of ['time', 'weather']) if (e[k] && !world[k]) world[k] = e[k]; break;
      case 'place':
        // first place wins, except a place named after a preposition ('in the field') beats an earlier bare object word ('eating grass')
        if (!place || (prep && !placeByPrep)) { place = e.place; placeWord = it.text; placeByPrep = !!prep; placeStandin = !!e.standin; }
        prep = null; heldChain = null; break;
      case 'prep':
        if (e.rel === 'of' && prep) break; // 'in the back of the car'
        prep = e.rel; break;
      case 'outside': prep = e.rel; outside = true; break;
      case 'seat': case 'invehicle': {
        // seat words belong to the person mentioned last ('a man in the driver's seat with a woman in the passenger seat')
        subjectsOrImplied();
        const x = lastPerson(null);
        if (x) { x.seatPref = e.seat || x.seatPref || null; x.wantsVehicle = true; }
        prep = 'in'; break;
      }
      case 'lean': { subjectsOrImplied(); const x = lastPerson(null); if (x) { x.lean = true; x.wantsVehicle = true; } prep = null; break; }
      case 'pron': {
        const p = lastPerson(e.gender);
        if (p) { if (conj && !clauseVerb) subjects.push(p); else { subjects = [p]; clauseVerb = false; } }
        conj = false; break;
      }
      case 'verb': {
        if (e.parked) break;
        if (cameraClause) {
          const next = items[i + 1];
          const hit = cameraMoveOf(e, it.text, next && next.text);
          if (hit) {
            if (hit.move === 'crane_down' || hit.move === 'crane_reveal') camera.move = hit.move;
            else if (hit.move && !camera.move) camera.move = hit.move;
            if (hit.eat) i += 1;
            lastVerb = null; clauseVerb = true; conj = false; prep = null; outside = false; heldChain = null;
            break;
          }
          const stem = wordStem(it.text);
          if (stem === 'look' || /^look/.test(it.text)) {
            lookBeat(i, items.slice(i, i + 6).map((x) => x.text).join(' '));
            lastVerb = null; clauseVerb = true; conj = false; prep = null; break;
          }
          if (['pass', 'hand', 'pull', 'push', 'follow', 'look', 'face', 'board', 'hold', 'lens', 'pan', 'tilt', 'rise', 'boom', 'reveal'].includes(stem)) {
            lastVerb = null; clauseVerb = true; conj = false; prep = null; break;
          }
        }
        if (/^looks?$/.test(it.lemma) || /^look/.test(it.text)) {
          lookBeat(i, items.slice(i, i + 6).map((x) => x.text).join(' '));
          lastVerb = null; clauseVerb = true; conj = false; prep = null; break;
        }
        if (e.pair) { if (!pairHint) pairHint = e.pair; if (!subjects.length && !e.pose) { lastVerb = e; clauseVerb = true; prep = null; break; } }
        // "FPV follow." is a camera move. A bare camera verb with nobody named and no object should not invent a walker.
        if (!subjects.length) {
          const nxt = items[i + 1];
          const noun = nxt && nxt.entry && (nxt.entry.kind === 'entity' || nxt.entry.kind === 'role' || nxt.entry.kind === 'group');
          const bare = !noun && cameraMoveOf(e, it.text, nxt && nxt.text);
          if (bare && bare.move) {
            if (!camera.move) camera.move = bare.move;
            if (bare.eat) i += 1;
            lastVerb = null; clauseVerb = true; conj = false; prep = null; outside = false; heldChain = null;
            break;
          }
        }
        const subs = subjectsOrImplied(); let chain = null;
        // 'drinking coffee': the object noun fills the implied prop instead of making a second one
        for (const s of subs) {
          if (e.pose) {
            const seated = s.pose === 'sit' || s.pose === 'drive' || s.pose === 'sit_passenger';
            if (s.cat === 'person' && !(s.pose && e.pose === 'stand') && !(seated && ['sip', 'hold_phone', 'wave'].includes(e.pose))) s.pose = e.pose;
            if (s.cat === 'animal' && ANIMAL_POSE[e.pose]) s.pose = ANIMAL_POSE[e.pose];
          }
          if (s.cat !== 'person') continue;
          if (e.rides) { const b = implyProp(e.rides, s); rel(s, 'rides', b); s.impliedHeld = b; }
          if (e.implies) {
            const pr = implyProp(e.implies, s); rel(s, 'holds', pr, e.hand && e.hand !== 'R' ? { hand: e.hand } : {}); s.impliedHeld = pr;
            if (it.phrase) { pr.implied = false; pr.word = it.text; } // 'drinking coffee' as one phrase names the thing
            if (!e.hand || e.hand === 'R') chain = subs; // 'holding an umbrella and a cup': the cup goes in the other hand
          }
          if (e.stands_at) s.standsAt = e.stands_at;
          if (e.on_table) s.onTable = e.on_table;
          if (e.implies_vehicle) { s.wantsVehicle = true; s.drives = true; s.seatPref = s.seatPref || e.seat || 'driver'; }
          if (e.mutual) s.mutual = e.mutual;
        }
        lastVerb = e; clauseVerb = true; conj = false; prep = null; outside = false; heldChain = chain ? { subs: chain.slice() } : null;
        break;
      }
      case 'entity': case 'group': case 'role': {
        const w = e.kind === 'group' ? e : null;
        let ent;
        if (e.kind === 'role') {
          ent = newEnt({ cat: 'person', asset: 'person', word: it.text, count: count || 1, preset: 'adult', role: { seat: e.seat, vehicle: e.vehicle || null } });
          ent.seatPref = e.seat; ent.wantsVehicle = true; if (e.seat === 'driver') ent.drives = true;
        } else if (w) {
          ent = newEnt({ cat: 'person', asset: 'person', word: it.text, count: count || w.count, preset: 'adult', group: it.text });
          if (w.presets && (!count || count === w.count)) { ent.presets = w.presets.slice(); ent.genders = w.genders ? w.genders.slice() : null; }
        } else {
          // object noun that satisfies a verb's implied prop ('drinking coffee', 'texting on her phone')
          const holder = subjects.find((s) => s.impliedHeld && s.impliedHeld.asset === e.asset && s.impliedHeld.implied);
          if (holder && (lastVerb || prep === 'on' || prep === 'with')) { holder.impliedHeld.implied = false; holder.impliedHeld.word = it.text; count = null; size = 1; sawDet = false; prep = null; break; }
          const plural = it.plural;
          const cat = e.cat;
          const pd = (grammar.plural_default || {})[cat === 'person' ? 'person' : 'other'] || 2;
          ent = newEnt({ cat, asset: e.asset, word: it.text, count: count || (plural ? pd : 1), scale: e.scale ?? 1 });
          if (cat === 'person') { ent.preset = e.preset || 'adult'; if (e.gender) ent.gender = e.gender; if (e.pose) ent.pose = e.pose; if (e.statue) ent.statue = true; }
          if (cat === 'person' && e.rides) { const b = implyProp(e.rides, ent); rel(ent, 'rides', b); }
        }
        if (size !== 1) ent.size = size;
        const maxc = (grammar.max_count || {})[ent.cat === 'person' ? 'person' : ent.cat === 'animal' ? 'animal' : 'other'] || 12;
        if (ent.count > maxc) { notes.push(`count ${ent.count} capped at ${maxc}`); ent.count = maxc; }
        if (prep === 'in' && ent.asset === 'house' && !place) {
          place = 'interior'; placeWord = ent.word; placeByPrep = true; ent.dropped = true;
          prep = null; conj = false; count = null; size = 1; sawDet = false; outside = false;
          break;
        }
        if (prep) {
          if (isActor(ent) && !subjects.length) { subjects = [ent]; clauseVerb = false; }
          else { applyPrep(prep, ent); if (prep === 'with' && ent.cat === 'person') ent.companionOf = subjects[0]; }
          prep = null;
        } else if (lastVerb && lastVerb.obj && subjects.length && !conj) {
          const map = lastVerb.obj, type = map[ent.cat] || map.any;
          if (type) {
            for (const s of subjects) if (s !== ent) {
              if (type === 'in') { rel(s, 'in', ent); if (lastVerb.verb === 'drive') { s.seatPref = 'driver'; s.drives = true; } }
              else if (type === 'on' && SURFACES.includes(ent.asset) && !SEATING.includes(ent.asset)) rel(s, 'near', ent, { side: 'front' });
              else if (type === 'on') rel(s, 'on', ent);
              else if (type === 'rides') rel(s, ent.cat === 'rideable' ? 'rides' : 'near', ent);
              else if (type === 'walks') { rel(s, 'walks', ent); if (!ent.pose) ent.pose = ANIMAL_POSE[s.pose] || 'walk'; }
              else if (type === 'holds') { rel(s, 'holds', ent); heldChain = { subs: subjects.slice() }; }
              else if (type === 'pair') { /* the pair rule below places them */ }
              else if (type === 'with') rel(s, 'near', ent, { side: 'beside' });
              else if (type === 'facing' || type === 'toward') rel(s, 'facing', ent);
              else if (type === 'against') rel(s, 'near', ent, { side: 'against' });
              else rel(s, 'near', ent, { side: 'beside' });
            }
            if (lastVerb.mutual && ent.cat === 'person') ent.mutual = lastVerb.mutual;
          } else if (isActor(ent)) { subjects = [ent]; clauseVerb = false; }
          lastVerb = null;
        } else if (isActor(ent) || !subjects.length) {
          if (conj && !clauseVerb) subjects.push(ent); else { subjects = [ent]; clauseVerb = false; lastVerb = null; }
        } else if (conj && heldChain && ent.cat === 'handheld') {
          // 'holding a phone and a cup': the second thing goes in the other hand
          for (const s of heldChain.subs) if (s.cat === 'person' && s.relations.filter((r) => r.type === 'holds').length === 1) rel(s, 'holds', ent, { hand: 'L' });
          heldChain = null;
        } else if (conj && subjects.length && !clauseVerb) subjects.push(ent);
        else if (!isActor(ent)) { /* a thing mentioned on its own, e.g. 'a man, a tree' */ }
        conj = false; count = null; size = 1; sawDet = false; outside = false;
        break;
      }
      default: break;
    }
  }
  void sawDet;
  if (/\b(hurdles the camera|jumps over the camera)\b/i.test(text)) {
    for (const e of ents) if (e.asset === 'fence' || /hurdle/i.test(e.word || '')) e.dropped = true;
    beats.push({ who: null, type: 'jump', from_t: 0.45, to_t: 0.7, params: {} });
  }
  // A pause between moves ("walks to the bench, pauses, then sits") is a hold of its own, not part of the move before it:
  // the clauses share the shot, and the pause's clause is the person's hold (gait.js stops travel and eases to a stand).
  const mover = ents.find((e) => e.cat === 'person' && !e.dropped);
  for (const h of mover ? holdSpans(text) : []) beats.push({ who: mover, type: 'hold', from_t: h[0], to_t: h[1], params: {} });
  if (beats.some((b) => b.type === 'hold')) for (let k = unmatched.length - 1; k >= 0; k--) if (PAUSE.test(unmatched[k])) unmatched.splice(k, 1);
  if (/\ba drone flying\b/i.test(text) && !/\bdrone shot\b/i.test(text) && !ents.some((e) => e.asset === 'airplane')) {
    newEnt({ cat: 'set', asset: 'airplane', word: 'drone', count: 1, scale: 0.02 });
  }
  // ---------------- post rules ----------------
  const live = ents.filter((e) => !e.dropped);
  const plan = planCamera(text);
  if (/\b(bird|birds)\b/i.test(text) && /\bfly/i.test(text)) for (const a of live) if (a.asset === 'bird' && (!a.pose || a.pose === 'stand')) a.pose = 'fly';
  if (/\b(bird|birds)\b/i.test(text) && /\bflying over\b/i.test(text) && !/\bdrone\b/i.test(text)) {
    if (camera.move === 'aerial') camera.move = null;
    if (plan.move === 'aerial') plan.move = null;
    plan.segments = (plan.segments || []).filter((s) => s.move !== 'aerial');
  }
  if (plan.move) camera.move = plan.move;
  if (plan.facing) camera.facing = plan.facing;
  if (plan.lensMm) camera.lensMm = plan.lensMm;
  if (plan.handheld) camera.handheld = plan.handheld;
  if (plan.dutch) camera.dutch = 16;
  if (!camera.move && camera.dutch && /\b(runs|running|walks|walking)\b/i.test(text)) camera.move = 'follow';
  if (plan.segments.length) camera.segments = plan.segments;
  if (plan.reveal) camera.reveal = true;
  if (/\bhero shot\b/i.test(text)) { if (!camera.angle) camera.angle = 'low'; if (!camera.move) camera.move = 'push_in'; }
  if (/\bextreme wide\b/i.test(text)) camera.shot = 'wide';
  if (/\binsert shot\b|\binsert\b/i.test(text)) camera.shot = 'extreme_close';
  if (/\btwo-shot\b|\btwo shot\b/i.test(text) && !camera.shot) camera.shot = 'medium';
  if (/\b(camera low|low and fast|from the ground)\b/i.test(text) && !camera.angle) camera.angle = 'low';
  if ((camera.move === 'truck_left' || camera.move === 'truck_right')) {
    for (let i = live.length - 1; i >= 0; i--) {
      const e = live[i];
      if (e.cat === 'vehicle' && e.asset === 'truck') live.splice(i, 1);
    }
  }
  if (/\bbooth\b/i.test(text) && !live.some((e) => e.asset === 'window_panel')) {
    live.push(newEnt({ cat: 'set', asset: 'window_panel', word: 'window', count: 1, implied: true, scale: 1 }));
  }
  if (/\ba couple\b/i.test(text) && /\b(danc|hug|argu|talk)/i.test(text)) {
    const who = live.find((e) => e.cat === 'person');
    const pr = lex.pairs && lex.pairs.couple;
    if (who && pr) {
      const pose = /\bdanc/.test(text) ? 'dance' : (who.pose && who.pose !== 'stand' ? who.pose : 'talk');
      who.pair = { id: 'couple', poses: [pose, pose], offset: pr.b.offset, facing: pr.b.facing };
      who.pose = pose;
    }
  }
  if (plan.speed) camera.subjectSpeed = plan.speed;
  for (const chip of plan.chips) if (!notes.includes(chip)) notes.push(chip);
  for (const e of live) if (/^booth$/.test(e.word || '')) { e.asset = 'bench'; e.cat = 'furniture'; }
  for (const p of live) if (p.cat === 'person' && (!p.pose || p.pose === 'stand')) {
    const w = p.word || '';
    if (/dancer/.test(w)) p.pose = 'dance';
    else if (/boxer/.test(w)) p.pose = 'fight_guard';
    else if (/waiter/.test(w)) p.pose = 'walk';
    else if (/hiker|runner/.test(w)) p.pose = 'walk';
    else if (/jogger/.test(w)) p.pose = 'jog';
  }
  for (const p of live) if (p.cat === 'person') for (const r of p.relations) if (r.target && /^booth$/.test(r.target.word || '')) { p.pose = p.pose && p.pose !== 'stand' ? p.pose : 'sit'; r.type = 'on'; delete r.side; }
  if (camera.move === 'follow' && !/\b(stand|sitting|sits|sat)\b/i.test(text)) {
    for (const p of live) if (p.cat === 'person' && !p.pose) p.pose = 'walk';
  }
  if (/\bacross from\b/i.test(text)) {
    const sitters = live.filter((e) => e.cat === 'person');
    if (sitters.length >= 2) rel(sitters[0], 'facing', sitters[1]);
  }
  // 1. people who want a vehicle get one ('driver', 'in the back seat', 'driving' with no vehicle word)
  for (const p of live.filter((e) => e.cat === 'person' && e.wantsVehicle && !e.relations.some((r) => r.type === 'in'))) {
    const want = p.role && p.role.vehicle;
    let v = live.find((e) => e.cat === 'vehicle' && (!want || e.asset === want) && !p.relations.some((r) => r.target === e && r.type === 'near'));
    if (!v) { v = newEnt({ cat: 'vehicle', asset: want || 'car', word: '(implied)', count: 1, implied: true, scale: 1 }); live.push(v); }
    p.relations = p.relations.filter((r) => r.target !== v);
    rel(p, 'in', v);
  }
  // 2. seats: explicit preferences first, then adults to the front, children to the back
  for (const v of live.filter((e) => e.cat === 'vehicle')) {
    const occ = [];
    for (const p of live) for (const r of p.relations) if (r.type === 'in' && r.target === v) for (let i = 0; i < p.count; i++) occ.push({ p, r, i, preset: p.presets ? p.presets[i] || 'adult' : p.preset || 'adult' });
    if (!occ.length) continue;
    const free = (SEATS[v.asset] || SEATS.car).slice();
    const take = (s) => { const k = free.indexOf(s); if (k < 0) return null; free.splice(k, 1); return s; };
    const want = (o) => {
      if (o.p.seatPref === 'driver' && o.i === 0) return ['driver'];
      if (o.p.seatPref === 'passenger') return v.asset === 'bus' ? ['passenger', 'row1_L_win', 'row1_R_win'] : ['passenger'];
      if (o.p.seatPref === 'rear') return v.asset === 'bus' ? ['rear_R', 'rear_L'] : ['rear_R', 'rear_L'];
      if (v.asset === 'bus' && !o.p.drives) return [];
      if (o.preset === 'child' && !o.p.drives) return ['rear_R', 'rear_L', 'passenger'];
      return ['driver', 'passenger'];
    };
    const seatOf = new Map();
    for (const o of occ) { if (!o.p.seatPref && !(o.preset === 'child' && !o.p.drives)) continue; for (const s of want(o)) if (take(s)) { seatOf.set(o, s); break; } }
    for (const o of occ) { if (seatOf.has(o)) continue; let got = null; for (const s of want(o)) if ((got = take(s))) break; if (!got) got = free.find((s) => s !== 'driver' || !v.asset.startsWith('bus') && false) ? take(free.find((s) => s !== 'driver')) : null; seatOf.set(o, got); }
    for (const o of occ) {
      const s = seatOf.get(o);
      o.r.seats = o.r.seats || []; o.r.poses = o.r.poses || [];
      o.r.seats[o.i] = s || null;
      o.r.poses[o.i] = !s ? 'stand' : o.p.lean ? 'lean_window' : s === 'driver' ? 'drive' : 'sit_passenger';
      if (!s) notes.push(`no free seat in the ${v.asset}: one ${o.p.word} stands beside it`);
    }
    for (const o of occ) if (o.i === 0) o.p.pose = o.r.poses[0];
  }
  // 3. bench slots (3 per bench, more benches when needed) and the implied bench for sitting people
  for (const p of live.filter((e) => e.cat === 'person' && e.pose === 'sit' && e.outsideVehicle)) { p.pose = 'stand'; notes.push('sitting on a vehicle is not supported: the person stands beside it'); }
  // 'cooking', 'washing dishes': stand at a counter (the one mentioned, else an implied one)
  for (const p of live.filter((e) => e.cat === 'person' && e.standsAt && !e.relations.some((r) => ['in', 'on', 'rides'].includes(r.type)))) {
    if (p.relations.some((r) => r.type === 'near' && !isActor(r.target))) continue; // 'washing a car': at the car, no counter
    let c = live.find((e) => e.asset === p.standsAt);
    if (!c) { c = newEnt({ cat: CATEGORY[p.standsAt] || 'furniture', asset: p.standsAt, word: '(implied)', count: 1, implied: true, scale: 1 }); live.push(c); }
    rel(p, 'near', c, { side: 'front' });
  }
  // 'typing', 'drawing': the laptop goes on a table (mentioned or implied) in front of the person
  for (const p of live.filter((e) => e.cat === 'person' && e.onTable)) {
    let lap = live.find((e) => e.asset === p.onTable);
    if (!lap) { lap = newEnt({ cat: CATEGORY[p.onTable] || 'handheld', asset: p.onTable, word: '(implied)', count: 1, implied: true, scale: 1 }); live.push(lap); }
    let t = live.find((e) => SURFACES.includes(e.asset));
    if (!t) { t = newEnt({ cat: 'furniture', asset: 'table', word: '(implied)', count: 1, implied: true, scale: 1 }); live.push(t); }
    p.relations = p.relations.filter((r) => !(r.type === 'near' && r.target === lap));
    if (!lap.relations.some((r) => r.type === 'on')) rel(lap, 'on', t);
    if (!p.relations.some((r) => r.target === t)) rel(p, 'near', t, { side: 'front' });
  }
  for (const p of live.filter((e) => e.cat === 'person' && e.pose === 'sit' && !e.relations.some((r) => ['in', 'on', 'rides'].includes(r.type)))) {
    let b = live.find((e) => SEATING.includes(e.asset));
    if (!b) { b = newEnt({ cat: 'furniture', asset: 'bench', word: '(implied)', count: 1, implied: true, scale: 1 }); live.push(b); }
    rel(p, 'on', b);
  }
  for (const b of live.filter((e) => SEATING.includes(e.asset))) {
    let n = 0; const sitters = [];
    for (const p of live) for (const r of p.relations) if (r.type === 'on' && r.target === b) { sitters.push([p, r]); n += p.count; }
    if (!n) continue;
    const order = n === 1 ? [1] : n === 2 ? [0, 2] : Array.from({ length: n }, (_, i) => i);
    let k = 0;
    for (const [p, r] of sitters) { r.slots = order.slice(k, k + p.count); k += p.count; if (p.pose === 'lie_down') { r.type = 'near'; r.side = 'front'; delete r.slots; } else if (p.cat === 'person') p.pose = 'sit'; }
    b.count = Math.max(b.count, Math.ceil(n / 3));
  }
  // 4. a person with an animal while walking or running: leash
  for (const p of live.filter((e) => e.cat === 'person')) {
    for (const r of p.relations) if (r.type === 'with' && r.target.cat === 'animal') {
      if (['walk', 'jog', 'run', null, 'stand'].includes(p.pose) && r.target.asset === 'dog') { r.type = 'walks'; if (!p.pose) p.pose = 'walk'; }
      else { r.type = 'near'; r.side = 'beside'; }
    }
  }
  // person and dog as joint subjects of a walk ('a man and his dog walking')
  for (const p of live.filter((e) => e.cat === 'person' && ['walk', 'jog', 'run'].includes(e.pose))) {
    const d = live.find((e) => e.asset === 'dog' && ['walk', 'run'].includes(e.pose) && !live.some((q) => q.relations.some((r) => r.type === 'walks' && r.target === e)));
    if (d && !p.relations.some((r) => r.type === 'walks')) rel(p, 'walks', d);
  }
  for (const p of live) for (const r of p.relations) if (r.type === 'rides' && r.target.count < p.count) r.target.count = p.count;
  for (const p of live) for (const r of p.relations) if (r.type === 'walks' && !r.target.pose) r.target.pose = p.pose === 'run' || p.pose === 'jog' ? 'run' : 'walk';
  // 5. held things set the pose; riders get the rider pose; defaults
  for (const p of live.filter((e) => e.cat === 'person')) {
    const held = p.relations.filter((r) => r.type === 'holds').map((r) => r.target.asset);
    if (held.includes('phone') && (!p.pose || p.pose === 'stand' || p.pose === 'walk' || p.pose === 'hold')) p.pose = p.pose === 'walk' ? 'walk' : 'hold_phone';
    const ride = p.relations.find((r) => r.type === 'rides');
    if (ride) p.pose = RIDER_POSE[ride.target.asset] || p.pose;
    if (!p.pose) p.pose = 'stand';
    if (p.mutual) {
      const q = live.find((x) => x !== p && x.cat === 'person' && !p.relations.some((r) => r.target === x) && !x.relations.some((r) => r.target === p));
      if (q) rel(p, 'facing', q); else if (p.count > 1) p.formation = 'facing';
    }
    const lim = (grammar.size_limits || {}).person || [0.55, 1.4];
    if (p.size !== 1) p.size = Math.min(lim[1], Math.max(lim[0], p.size));
  }
  for (const a of live.filter((e) => e.cat === 'animal')) if (!a.pose) a.pose = 'stand';
  // 5. lying or sitting next to a bed, sofa or bench means on it ('sleeping in bed')
  for (const e of live) if (e.cat === 'person' && ['lie_down', 'sit'].includes(e.pose)) for (const r of e.relations) if (r.type === 'near' && SEATING.includes(r.target.asset) && !e.relations.some((x) => x.type === 'on')) { r.type = 'on'; delete r.side; }
  // 5a. pairs (library v1.4): hug, kiss, shake_hands, fight, couple. Two people, poses a / b, b placed at the pair offset in
  // a's frame and turned by facing. Too few people: the partner (or both) is implied. Not for people seated in or riding something.
  const pairDef = pairHint && lex.pairs && lex.pairs[pairHint];
  if (pairDef) {
    const busy = (e) => e.relations.some((r) => ['in', 'rides', 'on'].includes(r.type));
    let people = live.filter((e) => e.cat === 'person' && !busy(e));
    const total = people.reduce((n, e) => n + e.count, 0);
    const mk = (count, like) => { const p = newEnt({ cat: 'person', asset: 'person', word: '(implied)', count, preset: (like && like.preset) || 'adult', implied: true }); live.push(p); return p; };
    if (!people.length && !live.some((e) => e.cat === 'person')) people = [mk(2)];
    else if (total === 1) people.push(mk(1, people[0]));
    const P = { id: pairHint, poses: [pairDef.a.pose, pairDef.b.pose], offset: pairDef.b.offset, facing: pairDef.b.facing };
    // "walking together" is one pair phrase, so the walk verb never lands. A couple who were said to walk still walk.
    const said = /\brunning\b|\bruns\b/.test(text) ? 'run' : /\bjogging\b|\bjogs\b/.test(text) ? 'jog' : /\bwalking\b|\bwalks\b|\bwalked\b/.test(text) ? 'walk' : null;
    if (pairHint === 'couple' && said && P.poses[0] === 'stand') P.poses = [said, said];
    // Side by side, not the stand-in 0.55 m. Same direction, so a walk does not turn into a face-off.
    if (pairHint === 'couple' && said) P.offset = [0.72, 0, 0];
    const keep = (e) => pairHint === 'couple' && e.pose && e.pose !== 'stand';
    if (people.length && people[0].count >= 2) {
      const a = people[0];
      a.pair = keep(a) ? { ...P, poses: [a.pose, a.pose] } : P;
      if (!keep(a)) a.pose = P.poses[0];
      if (a.formation === 'facing') delete a.formation;
    } else if (people.length >= 2) {
      const [a, b] = people;
      a.relations = a.relations.filter((r) => !(r.target === b && (r.type === 'facing' || r.type === 'near')));
      b.relations = b.relations.filter((r) => !(r.target === a && (r.type === 'facing' || r.type === 'near')));
      const poses = keep(a) ? [a.pose, keep(b) ? b.pose : a.pose] : P.poses;
      a.pose = poses[0]; b.pose = poses[1];
      rel(a, 'pair', b, { pair: { ...P, poses } });
    }
  }
  // 5b. held-prop rules (library v1.3.1): a guitar makes 'play_guitar', a book makes 'read'; the pose picks the slot (kick -> foot, carry -> both hands, guitar -> strap)
  const propPoses = grammar.prop_poses || {}, slots = grammar.action_slots || {};
  for (const e of [...live]) {
    const pa = e.cat === 'person' && lex.personActions && (lex.personActions[e.word] || lex.personActions[String(e.word).replace(/s$/, '')]);
    if (!pa || (e.pose && e.pose !== 'stand')) continue;
    e.pose = pa.pose;
    if (pa.implies && !e.relations.some((r) => r.type === 'holds')) { const pr = implyProp(pa.implies, e); live.push(pr); rel(e, 'holds', pr, pa.hand && pa.hand !== 'R' ? { hand: pa.hand } : {}); }
  }
  for (const e of live) {
    if (e.cat !== 'person') continue;
    for (const r of e.relations) {
      if (r.type !== 'holds') continue;
      const pp = propPoses[r.target.asset];
      if (pp && (!e.pose || e.pose === 'stand' || e.pose === 'hold')) e.pose = pp;
    }
    for (const r of e.relations) if (r.type === 'holds' && !r.hand && typeof slots[e.pose] === 'string') r.hand = slots[e.pose];
  }
  // 6. place: explicit word, else implied by a thing (house, building, boat, bus), else in-car scenes get the road kit
  if (/\ba crowd\b/i.test(text)) {
    for (const e of live) if (e.cat === 'person' && /^crowd$/i.test(e.word || '') && e.count < 8) e.count = 12;
  }
  if (/\b(car chase|through traffic)\b/i.test(text)) { place = 'highway'; placeWord = 'highway'; }
  else if (/\bkitchen\b/i.test(text)) { place = 'kitchen'; placeWord = 'kitchen'; }
  else if (/\bcafe\b/i.test(text)) { place = 'restaurant'; placeWord = 'cafe'; }
  else if (/\bdesk\b/i.test(text)) { place = 'interior'; placeWord = 'desk'; }
  if (/\bargu/i.test(text) && !/\bdanc/.test(text)) {
    const folks = live.filter((e) => e.cat === 'person');
    const face = { id: 'argue', poses: ['talk', 'talk'], offset: [0, 0, 1.35], facing: 180 };
    if (folks.length === 1 && folks[0].count >= 2) { folks[0].pair = face; folks[0].pose = 'talk'; }
  }
  if (/\bslow[- ]?danc/i.test(text)) {
    const folks = live.filter((e) => e.cat === 'person');
    const hold = { id: 'slow_dance', poses: ['hug', 'hug'], offset: [0, 0, 0.42], facing: 180 };
    if (folks.length === 1 && folks[0].count >= 2) { folks[0].pair = hold; folks[0].pose = 'hug'; delete folks[0].formation; }
  }
  if (/\bbus pulls\b/i.test(text)) for (const p of live) if (p.cat === 'person' && p.pose === 'pull') p.pose = 'sit';
  if (/\bjumps? over\b/i.test(text)) for (const p of live) if (p.cat === 'person' && (p.pose === 'give' || p.pose === 'stand')) p.pose = 'run';
  if (!/\bkeys?\b/i.test(text)) for (let i = live.length - 1; i >= 0; i--) if (live[i].asset === 'key_small') live.splice(i, 1);
  let placeSource = place ? 'word' : null;
  if (!place) for (const e of live) { const ip = (grammar.implied_places || {})[e.asset]; if (ip && !e.implied) { place = ip; placeSource = 'implied'; placeWord = e.word; break; } }
  if (!place) for (const e of live) { const ip = (grammar.implied_places || {})[e.asset]; if (ip && (e.cat === 'vehicle' || e.cat === 'furniture')) { place = ip; placeSource = 'implied'; placeWord = e.word; break; } }
  if (!place && live.some((e) => e.relations.some((r) => r.type === 'in' && r.target.asset !== 'bus' && (r.seats || []).some(Boolean)))) { place = 'car_interior'; placeSource = 'implied'; placeWord = null; }
  if (!place) { place = grammar.default_place || 'park'; placeSource = 'default'; }
  // ---------------- ids and output ----------------
  const actors = live.filter(isActor), props = live.filter((e) => !isActor(e));
  actors.forEach((e, i) => { e.id = 'a' + (i + 1); });
  props.forEach((e, i) => { e.id = 'p' + (i + 1); });
  const outRel = (r) => { const o = { type: r.type, target: r.target.id }; for (const k of ['side', 'seats', 'poses', 'slots', 'hand', 'pair']) if (r[k] != null) o[k] = r[k]; return o; };
  const subj = actors.find((a) => a.cat === 'person') || actors[0] || props[0] || null;
  camera.subject = subj ? subj.id : null;
  return {
    version: SPEC_VERSION,
    text: String(text),
    seed: hashSeed(text),
    place: { id: place, word: placeWord, source: placeSource, ...(placeStandin && placeWord ? { standin: true } : {}) }, // v1.4: standin = no own kit yet, nearest kit shown
    world,
    actors: actors.map((a) => {
      const o = { id: a.id, kind: a.cat, asset: a.asset, word: a.word, count: a.count, pose: a.pose };
      if (a.formation) o.formation = a.formation;
      if (a.pair) o.pair = a.pair;
      if (a.cat === 'person') { o.preset = a.preset || 'adult'; if (a.presets) o.presets = a.presets; if (a.gender) o.gender = a.gender; if (a.genders) o.genders = a.genders; if (a.statue) o.statue = true; if (a.role) o.role = a.role.seat; }
      if (a.size !== 1) o.size = +a.size.toFixed(3);
      if (a.implied) o.implied = true;
      o.relations = a.relations.filter((r) => r.target.id).map(outRel);
      return o;
    }),
    props: props.map((p) => {
      const o = { id: p.id, kind: p.cat, asset: p.asset, word: p.word, count: p.count };
      if (p.scale != null && p.scale !== 1) o.scale = p.scale;
      if (p.size !== 1) o.size = +p.size.toFixed(3);
      if (p.implied) o.implied = true;
      const rs = p.relations.filter((r) => r.target.id).map(outRel);
      if (rs.length) o.relations = rs;
      return o;
    }),
    camera,
    beats: beats.map((b) => ({ actor: b.who && b.who.id ? b.who.id : (b.actor || null), type: b.type, from_t: b.from_t, to_t: b.to_t, params: b.params || {} })),
    matched, unmatched: [...new Set(unmatched)], notes,
  };
}
