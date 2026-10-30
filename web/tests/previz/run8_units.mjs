// Run 8: Lock seed (lib/seed.ts) and measured render times (lib/timing.ts). Locked, the next run sends the last seed;
// off, a fresh one each run; a reopened result comes back locked only with a seed. Elapsed and finished times come
// from the job's own clock; "usually" is the median of the last five measured runs; the kind matches the server's.
import { freshSeed, nextSeed, SEED_MAX, seedFrom, usedSeed } from '../../src/lib/seed.ts';
import { elapsedOf, epochS, kindOf, medianOf, tookOf } from '../../src/lib/timing.ts';

export function runRun8Units() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };

  // seeds
  ok(freshSeed(() => 0) === 1 && freshSeed(() => 0.999999999) <= SEED_MAX, 'fresh seeds stay in 1..SEED_MAX');
  const many = new Set(Array.from({ length: 20 }, () => nextSeed({ seed: 5, locked: false })));
  ok(many.size > 1 && !many.has(5) || many.size > 15, 'unlocked: a fresh seed each run');
  ok(nextSeed({ seed: 4242, locked: true }) === 4242 && nextSeed({ seed: 4242, locked: true }) === 4242, 'locked: the same seed every run');
  ok(nextSeed({ seed: null, locked: true }, () => 0.5) === freshSeed(() => 0.5), 'locked with no seed yet: a fresh one');
  const s = usedSeed({ seed: null, locked: true }, 77);
  ok(s.seed === 77 && s.locked === true && usedSeed(s, undefined).seed === 77, 'a run keeps the seed it used');
  ok(JSON.stringify(seedFrom(9, true)) === '{"seed":9,"locked":true}', 'a saved locked seed reopens locked');
  ok(JSON.stringify(seedFrom('12', false)) === '{"seed":12,"locked":false}', 'a saved seed as text');
  ok(JSON.stringify(seedFrom(undefined, true)) === '{"seed":null,"locked":false}', 'no seed: never locked');

  // times
  ok(epochS(1700000000) === 1700000000 && epochS(1700000000000) === 1700000000 && epochS(null) === null, 'epoch seconds from seconds or ms');
  ok(elapsedOf({ status: 'running', started_at: 100 }, 142_500) === 42.5, 'elapsed is now minus started');
  ok(elapsedOf({ status: 'queued', started_at: 100 }, 142_500) === null, 'a queued job has no elapsed time');
  ok(tookOf({ status: 'done', took_s: 52 }) === 52, 'took_s when the server measured it');
  ok(tookOf({ status: 'done', started_at: 10, finished_at: 62 }) === 52, 'else finished minus started');
  ok(tookOf({ status: 'canceled', started_at: 10, finished_at: 62 }) === null, 'a cancelled run has no made-in time');
  ok(medianOf([]) === null, 'no measured run: no "usually"');
  ok(medianOf([52]) === 52 && medianOf([50, 12, 11, 9, 13]) === 12 && medianOf([90, 10, 20, 30, 40, 50]) === 30, 'median of the last five');
  ok(medianOf([0.2]) === 1, 'at least 1 s');

  // the kind matches pickups/app/timings.py kind_of
  ok(kindOf('image', { model: 'z-image-turbo', aspect: '16:9', resolution: '1K', count: 1 }) === 'image|z-image-turbo|16:9|1K|1', 'image kind');
  ok(kindOf('video', { mode: 'text', width: 1280, height: 704, frames: 97 }) === 'video|text|1280x704|97', 'video kind');
  ok(kindOf('start_frame', { method: 'zbuf' }) === 'start_frame|zbuf' && kindOf('recast', { engine: 'scail' }) === 'recast|scail', 'start frame and take kinds');
  ok(kindOf('image', { model: 'z', resolution: '2K' }) !== kindOf('image', { model: 'z', resolution: '1K' }), 'another size is another kind');
  return problems;
}
