// Lexicon builder: library_manifest.json + grammar.json -> word map and phrase list for the shot-spec parser.
// Pure data, no three.js: runs in the browser and in Node. Deterministic (sorted keys, longest phrase first).

export const CATEGORY = {
  person: 'person', dog: 'animal', cat: 'animal',
  car: 'vehicle', suv: 'vehicle', truck: 'vehicle', van: 'vehicle', bus: 'vehicle',
  bicycle: 'rideable', skateboard: 'rideable', boat: 'rideable',
  bench: 'furniture', table: 'furniture', cup: 'handheld', phone: 'handheld',
  tree: 'set', house: 'set', building: 'set', fountain: 'set', post: 'set', rock: 'set',
  // library v1.3 additions
  guitar: 'handheld', laptop: 'handheld', book: 'handheld', ball: 'handheld', plate_food: 'handheld', produce: 'handheld',
  bag_box: 'handheld', key_small: 'handheld', board: 'handheld',
  sofa: 'furniture', counter: 'furniture',
  train: 'set', airplane: 'set', horse: 'set', sign: 'set', window_panel: 'set',
  // library v1.4 additions
  bed: 'furniture', cow: 'animal', giraffe: 'animal', elephant: 'animal', bird: 'animal', skis: 'rideable',
  umbrella: 'handheld', bat_stick: 'handheld', racket: 'handheld', kite: 'handheld', cart: 'handheld', toy: 'handheld',
  vase: 'handheld', microphone: 'handheld', candle: 'handheld',
  cabinet: 'set', fence: 'set', stairs: 'set', gym_set: 'set', grill: 'set',
};
/** Furniture people sit on (3 slots: attach_seat_0..2) and surfaces things stand on or people stand at. */
export const SEATING = ['bench', 'sofa', 'bed'];
export const SURFACES = ['table', 'counter'];
// Unknown future assets fall back to 'set' (placed near the stage). Add them here to give them a role.

/** Verb entry from a manifest action: pose, the held prop and hand, a surface to stand at, a prop on the lap or a table. */
export function manifestVerb(k, a) {
  const e = { kind: 'verb', verb: k, pose: a.pose, obj: { any: 'near' } };
  const pr = a.props || {};
  if (pr.attach_hand_R) { e.implies = pr.attach_hand_R; e.hand = 'R'; }
  else if (pr.attach_hand_L) { e.implies = pr.attach_hand_L; e.hand = 'L'; }
  else if (pr.at_foot_R) { e.implies = pr.at_foot_R; e.hand = 'foot_R'; }
  else if (pr.strap) { e.implies = pr.strap; e.hand = 'strap'; }               // library v1.3.1: guitar on the strap
  else if (pr.between_hands) { e.implies = pr.between_hands; e.hand = 'hands'; } // v1.3.1: two-handed carry
  else if (pr.push) { e.implies = pr.push; e.hand = 'push'; }                     // v1.4: cart at the person's origin
  else if (pr.airborne) { e.implies = pr.airborne; e.hand = 'airborne'; }         // v1.4: kite in the air on a line
  if (a.pair) e.pair = a.pair;                                                     // v1.4: two-person pose (hug, kiss, ...)
  if (pr.stand_at) e.stands_at = pr.stand_at;
  if (pr.on_lap_or_table) e.on_table = pr.on_lap_or_table;
  return e;
}

export function tokenize(text) {
  return String(text).toLowerCase()
    .replace(/[\u2018\u2019\u02bc]/g, "'").replace(/[\u2013\u2014]/g, ' ')
    .replace(/(\d)\s*x\s*4/g, '$1x4')
    .replace(/[.;]/g, ' $& ')
    .replace(/[^a-z0-9'&\-.; ]+/g, ' ')
    .split(/\s+/).flatMap((t) => {
      const cleaned = t.replace(/^[']+|[']+$/g, '');
      if (!cleaned) return [];
      if (cleaned === 'two-shot' || cleaned === 'close-up' || cleaned === 'slo-mo') return [cleaned];
      if (/\d/.test(cleaned)) return [cleaned.replace(/^-+|-+$/g, '')];
      return cleaned.split('-').map((p) => p.replace(/^[']+|[']+$/g, '')).filter(Boolean);
    });
}

/** Vocabulary-guided lemmatizer: tries a short list of English inflection rules and keeps the first candidate the lexicon knows. */
export function makeLemmatizer(vocab, exceptions = {}) {
  const cache = new Map();
  return function lemma(w) {
    if (cache.has(w)) return cache.get(w);
    let out = exceptions[w];
    if (!out) {
      if (vocab.has(w)) out = w;
      else {
        const c = [];
        if (w.endsWith("'s")) c.push(w.slice(0, -2));
        if (w.endsWith("s'")) c.push(w.slice(0, -1));
        if (w.endsWith('ies')) c.push(w.slice(0, -3) + 'y');
        if (w.endsWith('es')) c.push(w.slice(0, -2));
        if (w.endsWith('s') && !w.endsWith('ss')) c.push(w.slice(0, -1));
        if (w.endsWith('ing')) { const s = w.slice(0, -3); c.push(s, s + 'e'); if (/(.)\1$/.test(s)) c.push(s.slice(0, -1)); if (s.endsWith('y')) c.push(s.slice(0, -1) + 'ie'); }
        if (w.endsWith('ed')) { const s = w.slice(0, -2); c.push(s, s + 'e', w.slice(0, -1)); if (/(.)\1$/.test(s)) c.push(s.slice(0, -1)); if (s.endsWith('i')) c.push(s.slice(0, -1) + 'y'); }
        out = c.find((x) => vocab.has(x)) || w;
      }
    }
    cache.set(w, out);
    return out;
  };
}

function inflectionOfGrammarVerb(w, verbs, exc) {
  if (verbs[w]) return false;
  if (exc[w] && verbs[exc[w]]) return true;
  const c = [];
  if (w.endsWith('ing')) { const s = w.slice(0, -3); c.push(s, s + 'e'); if (/(.)\1$/.test(s)) c.push(s.slice(0, -1)); }
  if (w.endsWith('ed')) { const s = w.slice(0, -2); c.push(s, s + 'e'); }
  if (w.endsWith('s')) c.push(w.slice(0, -1));
  return c.some((x) => verbs[x]);
}

/**
 * buildLexicon(manifest, grammar) -> { words: { lemma: entry }, phrases: [{ tokens, entry }], lemma, vocab }
 * entry.kind: entity | verb | place | camera | world | num | quant | group | role | seat | lean | outside | prep | det | conj | pron | size | stop
 */
export function buildLexicon(manifest, grammar) {
  const multi = [];   // [phraseText, entry]
  const single = {};
  const put = (key, entry) => { if (key.includes(' ')) multi.push([key, entry]); else if (!single[key]) single[key] = entry; };
  // order = precedence for single words (first writer wins)
  for (const [k, v] of Object.entries(grammar.camera || {})) if (k.includes(' ') || ['closeup', 'close-up', 'overhead', 'pov', 'orbit', 'orbiting', 'pan', 'tracking', '360', 'static', 'circles', 'rotates', 'zooms', 'moves', 'drone', 'aerial', 'fpv', 'dutch', 'dolly', 'crane', 'follow', 'handheld', 'rise', 'boom'].includes(k)) put(k, { kind: 'camera', ...v });
  for (const [k, v] of Object.entries(grammar.world_words || {})) put(k, { kind: 'world', ...v }); // time of day, weather
  for (const [k, v] of Object.entries(grammar.roles || {})) put(k, { kind: 'role', ...v });
  for (const [k, v] of Object.entries(grammar.seat_phrases || {})) put(k, { kind: 'seat', seat: v });
  for (const k of grammar.lean_phrases || []) put(k, { kind: 'lean' });
  for (const [k, v] of Object.entries(grammar.outside_phrases || {})) put(k, { kind: 'outside', rel: v });
  for (const [k, v] of Object.entries(grammar.quantifiers || {})) put(k, { kind: 'quant', count: v });
  for (const [k, v] of Object.entries(grammar.group_nouns || {})) put(k, { kind: 'group', ...v });
  for (const [k, v] of Object.entries(grammar.places || {})) put(k, { kind: 'place', place: v });
  // the manifest's place words (library v1.3: 160+ words onto the kits; grammar.json wins on a clash)
  // (a word that is also a thing, 'house', stays a thing; a preposition or stopword, 'inside', stays what it is;
  //  a phrase naming a thing, 'inside a car', is left to the grammar so the car is still made)
  const reserved = new Set([...Object.keys(manifest.words || {}), ...Object.keys(grammar.prepositions || {}), ...(grammar.stopwords || []), ...(grammar.determiners || [])]);
  for (const k of Object.keys(manifest.places || {}).sort()) { const v = manifest.places[k]; if (v && v.kit && !reserved.has(k) && !(k.includes(' ') && k.split(' ').some((t) => manifest.words && manifest.words[t]))) put(k, { kind: 'place', place: v.kit, ...(v.standin ? { standin: true } : {}) }); }
  // set dressing named in the library (doorway, windowsill, curb, courtyard, deck) stands in for the nearest kit
  for (const k of Object.keys(manifest.set_words || {}).sort()) {
    const v = manifest.set_words[k];
    if (!v || !v.kit || v.kit === 'any' || reserved.has(k)) continue;
    if (k.includes(' ') && k.split(' ').some((t) => manifest.words && manifest.words[t])) continue;
    put(k, { kind: 'place', place: v.kit, standin: true });
  }
  // entities from the manifest (scale stand-ins kept), person words enriched with gender / preset
  for (const k of Object.keys(manifest.words || {}).sort()) {
    const w = manifest.words[k];
    const e = { kind: 'entity', asset: w.asset, cat: CATEGORY[w.asset] || 'set', scale: w.scale ?? 1, word: k };
    if (w.note) e.note = w.note;
    if (w.asset === 'person') Object.assign(e, grammar.person_words[k] || {});
    if (k === 'statue') e.statue = true;
    put(k, e);
  }
  for (const [k, v] of Object.entries(grammar.person_words || {})) put(k, { kind: 'entity', asset: 'person', cat: 'person', scale: 1, word: k, ...v });
  // library v1.4 pairs: two-person phrases (hug, kiss, shake hands, fight, couple words). They win over plain verbs; the
  // 'couple' phrases only fill words nothing else claims (a couple, two people and bride and groom keep their meaning)
  const pairs = manifest.pairs || {};
  const numberWords = new Set(Object.keys(grammar.numbers || {}));
  for (const k of Object.keys(manifest.pair_phrases || {}).sort()) {
    const id = manifest.pair_phrases[k], pr = pairs[id];
    if (!pr) continue;
    if (id === 'couple' && (single[k] || multi.some(([m]) => m === k) || k.split(' ').some((t) => numberWords.has(t) || (manifest.words || {})[t]))) continue;
    put(k, { kind: 'verb', verb: k, pose: id === 'couple' ? null : pr.a.pose, pair: id, obj: { person: 'pair', any: 'near' } });
  }
  for (const [k, v] of Object.entries(grammar.numbers || {})) put(k, { kind: 'num', count: v });
  for (const [k, v] of Object.entries(grammar.prepositions || {})) put(k, { kind: 'prep', rel: v });
  for (const [k, v] of Object.entries(grammar.verbs || {})) put(k, { kind: 'verb', verb: k, ...v });
  // manifest action words not already covered: verbs by their pose
  for (const k of Object.keys(manifest.actions || {}).sort()) {
    const a = manifest.actions[k];
    if (single[k] && !k.includes(' ')) continue;
    if (a.helper === 'seatInVehicle') { if (k.includes(' ')) put(k, a.pose === 'lean_window' ? { kind: 'lean' } : { kind: 'invehicle', seat: a.seat || null }); continue; }
    if (k.includes('_')) continue;
    if (inflectionOfGrammarVerb(k, grammar.verbs || {}, grammar.lemma_exceptions || {})) continue; // 'walking' -> walk keeps the grammar verb's object rules
    put(k, manifestVerb(k, a));
  }
  for (const [k, v] of Object.entries(grammar.camera || {})) if (!k.includes(' ')) put(k, { kind: 'camera', ...v });
  for (const [k, v] of Object.entries(grammar.sizes || {})) put(k, { kind: 'size', factor: v });
  for (const k of grammar.determiners || []) put(k, { kind: 'det' });
  for (const k of grammar.conjunctions || []) put(k, { kind: 'conj' });
  for (const [k, v] of Object.entries(grammar.pronouns || {})) put(k, { kind: 'pron', gender: v });
  for (const k of grammar.stopwords || []) put(k, { kind: 'stop' });
  // every verb, even where a noun of the same spelling won the single-word slot ('drink', 'phone', 'skateboard', 'boat')
  const verbs = {};
  for (const [k, v] of Object.entries(grammar.verbs || {})) if (!k.includes(' ')) verbs[k] = { kind: 'verb', verb: k, ...v };
  for (const k of Object.keys(manifest.actions || {}).sort()) {
    const a = manifest.actions[k];
    if (inflectionOfGrammarVerb(k, grammar.verbs || {}, grammar.lemma_exceptions || {})) continue;
    if (verbs[k] || k.includes(' ') || k.includes('_') || a.helper) continue;
    verbs[k] = manifestVerb(k, a);
  }
  const vocab = new Set([...Object.keys(single), ...Object.keys(verbs)]);
  // phrase words join the vocabulary, except inflected grammar verbs ('drinking' in 'drinking water' must still lemmatize to 'drink');
  // phrase matching compares raw tokens too, so those phrases still match
  for (const [k] of multi) for (const t of k.split(' ')) if (!inflectionOfGrammarVerb(t, grammar.verbs || {}, grammar.lemma_exceptions || {})) vocab.add(t);
  const lemma = makeLemmatizer(vocab, grammar.lemma_exceptions || {});
  // phrases are matched on lemmas, so 'sits in the back seat' and 'sitting in the back seat' both hit
  const seen = new Set();
  const phrases = [];
  for (const [k, e] of multi) {
    const tokens = tokenize(k).map(lemma);
    const key = tokens.join(' ');
    if (seen.has(key)) continue;
    seen.add(key);
    phrases.push({ tokens, text: k, entry: e });
  }
  phrases.sort((a, b) => b.tokens.length - a.tokens.length || (a.text < b.text ? -1 : 1));
  // person words that also name an action (guitarist, bassist): the person gets that pose and its prop (library v1.3.1)
  const personActions = {};
  for (const k of Object.keys(manifest.actions || {}).sort()) {
    const w = single[k];
    if (!k.includes(' ') && w && w.kind === 'entity' && w.asset === 'person' && manifest.actions[k].pose) personActions[k] = manifestVerb(k, manifest.actions[k]);
  }
  return { words: single, verbs, phrases, lemma, vocab, personActions, pairs, maxPhrase: phrases.length ? phrases[0].tokens.length : 1 };
}
