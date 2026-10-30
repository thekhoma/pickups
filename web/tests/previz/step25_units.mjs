// Step 25: the previz layout the render gets is clean. (a) no parked vehicles, signs or background extras the sentence
// never asked for; (b) no path, knots or move arrows in an exported frame, whatever the editor shows; (c) a sleek car
// ("a sports car", "a coupe") is a fastback, not the sedan's notchback, and SUVs, pickups and vans are their own
// models; (d) everything that shows is lit (shaded); (e) the ground, roads and paths carry a soft irregular mottle,
// not a grid, so motion reads; (f) trees: the kit's stay out unless the sentence places them (or the place is a forest),
// and a tree is a faceted pine (cones on a trunk), never a ball on a pole; (g) water's edges and lines are a soft tone
// darker than the water, never a light stripe.
import './node_dom.mjs';
import { registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import * as THREE from 'three';
import { parse } from './common.mjs';
import { shots } from './shot_guard.mjs';
import { bodyTypeOf, roofProfile } from '../../src/previz/runtime/bodyshape.js';
import { FILLER_ASSETS, WANTS_TREES } from '../../src/previz/runtime/builder.js';
import { GROUND_MATERIALS, GROUND_NOISE_SIZE, groundNoise, groundTexture } from '../../src/previz/runtime/groundtex.js';
import { setMaterials } from '../../src/previz/runtime/kit.js';

// the app's own TS (extensionless imports, as Vite resolves them)
registerHooks({
  resolve(spec, ctx, next) {
    try { return next(spec, ctx); } catch (e) {
      if (/^\.\.?\//.test(spec) && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});

const LIT = (m) => m && (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial || m.isMeshLambertMaterial || m.isMeshPhongMaterial || m.isMeshToonMaterial);
const idOf = (c) => String(c.userData.asset || c.userData.libraryId || String(c.name || '').replace(/^inst_/, '')).split('__')[0];
const build = (text) => shots().build(parse(text), { subjectTier: 'lite', liteCount: 1, crowdCount: 12, instancing: true });
const shownKit = (b) => b.kit.dressing.children.filter((c) => c.visible);
const shownIds = (b) => shownKit(b).map(idOf);

export async function runStep25Units() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };

  // --- (a) no filler ---
  for (const text of ['A car driving on the highway. Tracking shot.', 'Two friends hugging on a city sidewalk. Close to wide.', 'A man walking in the park.', 'A woman walking down a suburban street.']) {
    const b = await build(text);
    const extra = shownKit(b).filter((c) => [...FILLER_ASSETS, 'person', 'dog', 'cat'].includes(idOf(c)) && !c.userData.drive);
    ok(!extra.length, `(a) ${text}: filler still shown: ${extra.map((c) => c.name).join(', ')}`);
    for (const p of b.props) ok(p.object && p.object.visible, `(a) ${text}: the sentence's own ${p.object && p.object.name} is hidden`);
    for (const a of b.actors) ok(a.object && a.object.visible, `(a) ${text}: an actor is hidden`);
  }
  {
    const b = await build('A car driving on the highway. Tracking shot.');
    ok(b.props.some((p) => p.object.visible && idOf(p.object) === 'car'), '(a) the hero car is not shown');
    // what the sentence asks for stays: a parking lot, a busy street, a chase's traffic, the car you sit in
    const lot = shownIds(await build('A man walking through a parking lot.'));
    ok(lot.filter((id) => ['car', 'suv', 'truck', 'van'].includes(id)).length >= 2, `(a) a parking lot lost its cars: ${lot.join(',')}`);
    const busy = shownIds(await build('A woman crossing a busy city street.'));
    ok(busy.some((id) => ['car', 'bus', 'van'].includes(id)), `(a) a busy street lost its traffic: ${busy.join(',')}`);
    const chase = await build('A car chase through traffic on the highway.');
    ok(shownKit(chase).filter((c) => c.userData.drive).length >= 2, '(a) a chase lost its moving traffic');
    const inside = await build('A man sitting in a car.');
    ok(inside.layout.id === 'car_interior' && inside.props.some((p) => p.object.visible && idOf(p.object) === 'car'), '(a) car_interior lost the car you sit in');
    // the same sentence, the same layout: the editor shows what the render gets (one build path for both)
    const again = shownIds(await build('A car driving on the highway. Tracking shot.'));
    ok(JSON.stringify(again) === JSON.stringify(shownIds(b)), '(a) two builds of one sentence differ');
  }

  // --- (c) body types ---
  ok(bodyTypeOf('sports car', 'car') === 'fastback' && bodyTypeOf('coupe', 'car') === 'fastback' && bodyTypeOf('hatchback', 'car') === 'fastback', '(c) sleek words are not fastbacks');
  ok(bodyTypeOf('car', 'car') === null && bodyTypeOf('sedan', 'car') === null && bodyTypeOf('taxi', 'car') === null, '(c) a plain car is not the notchback sedan');
  ok(bodyTypeOf('sports car', 'suv') === null, '(c) only the car model is reshaped');
  {
    const sedan = (await build('A car driving on the highway.')).props.find((p) => idOf(p.object) === 'car').object;
    const sport = (await build('A sports car driving on the highway.')).props.find((p) => idOf(p.object) === 'car').object;
    ok(sport.userData.body === 'fastback' && !sedan.userData.body, '(c) the sports car is not marked fastback');
    const f = [0.08, 0.16, 0.24, 0.32, 0.5, 0.62];
    const a = roofProfile(sedan, f), s = roofProfile(sport, f);
    // notchback: a flat deck, then a step up to the rear window. fastback: one slope from the tail to the roof, lower roof
    const deckRise = a[1] - a[0], slopeRise = s[2] - s[0];
    ok(Math.max(...s) < Math.max(...a) - 0.03, `(c) the fastback roof is not lower: ${s} vs ${a}`);
    ok(s[1] > s[0] && s[2] > s[1] && s[3] > s[2], `(c) the fastback tail is not one rising slope: ${s}`);
    ok(Math.abs(s[0] - a[0]) < 0.25 && slopeRise > deckRise, `(c) the fastback tail does not slope: ${s} vs ${a}`);
    const suv = (await build('An SUV driving on the highway.')).props.map((p) => idOf(p.object));
    const truck = (await build('A pickup truck driving through the desert.')).props.map((p) => idOf(p.object));
    ok(suv.includes('suv'), `(c) an SUV is not the suv model: ${suv}`);
    ok(truck.includes('truck'), `(c) a pickup is not the truck model: ${truck}`);
  }

  // --- (d) shaded ---
  for (const text of ['A car driving on the highway. Tracking shot.', 'Two friends hugging on a city sidewalk. Close to wide.', 'A sports car driving on the highway.', 'A woman sitting at a table in a restaurant.']) {
    const b = await build(text);
    const flat = [];
    b.root.traverseVisible((n) => {
      if (!n.isMesh || n.isSprite) return;
      for (const m of [].concat(n.material)) {
        if (!m || m.visible === false || m.colorWrite === false || (m.transparent && m.opacity === 0)) continue; // invisible pick proxies
        if (!LIT(m)) flat.push(`${n.name || n.parent?.name}:${m.type}`);
      }
    });
    ok(!flat.length, `(d) ${text}: unlit meshes: ${[...new Set(flat)].slice(0, 6).join(', ')}`);
  }

  // --- (e) ground texture ---
  {
    const N = GROUND_NOISE_SIZE, n = groundNoise();
    let mean = 0; for (const v of n) mean += v; mean /= n.length;
    let sd = 0; for (const v of n) sd += (v - mean) ** 2; sd = Math.sqrt(sd / n.length);
    ok(Math.abs(mean - 0.5) < 0.02 && sd > 0.1 && sd < 0.2, `(e) noise is not soft and centred: mean ${mean.toFixed(3)} sd ${sd.toFixed(3)}`);
    // not a grid: take out the soft swells (a 9 px box blur) and no whole row or column of what is left stands out,
    // as a grid's lines would
    const blur = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let t = 0; for (let j = -4; j <= 4; j++) for (let i = -4; i <= 4; i++) t += n[((y + j + N) % N) * N + ((x + i + N) % N)];
      blur[y * N + x] = n[y * N + x] - t / 81;
    }
    let hsd = 0; for (const v of blur) hsd += v * v; hsd = Math.sqrt(hsd / blur.length);
    const rows = new Float64Array(N), cols = new Float64Array(N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { rows[y] += blur[y * N + x] / N; cols[x] += blur[y * N + x] / N; }
    const lineDev = Math.max(...[...rows, ...cols].map(Math.abs));
    ok(lineDev < hsd * 0.5, `(e) a row or column stands out like a grid line: ${lineDev.toFixed(4)} vs ${hsd.toFixed(4)}`);
    // soft: neighbours differ little
    let step = 0; for (let y = 0; y < N; y++) for (let x = 0; x < N - 1; x++) step += Math.abs(n[y * N + x + 1] - n[y * N + x]);
    step /= N * (N - 1);
    ok(step < sd * 0.25, `(e) the texture is not soft: neighbour step ${step.toFixed(3)}`);
    // irregular: the likeness falls off with distance and never comes back (a pattern or grid repeats at its period)
    const ac = (lag) => { let s = 0; for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) s += (n[y * N + x] - mean) * (n[y * N + (x + lag) % N] - mean) + (n[y * N + x] - mean) * (n[((y + lag) % N) * N + x] - mean); return s / (2 * N * N * sd * sd); };
    const lags = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128].map((l) => [l, ac(l)]);
    let low = 1, back = 0;
    for (const [, a] of lags) { back = Math.max(back, a - low); low = Math.min(low, a); }
    ok(back < 0.15, `(e) the texture repeats inside its tile: ${lags.map(([l, a]) => l + ':' + a.toFixed(2)).join(' ')}`);
    // seamless tile edges (no seam line every tile)
    let edge = 0; for (let y = 0; y < N; y++) edge += Math.abs(n[y * N] - n[y * N + N - 1]);
    ok(edge / N < step * 3, `(e) the tile has a seam: ${(edge / N).toFixed(3)}`);
    const tex = groundTexture();
    ok(tex.isDataTexture && tex.wrapS === THREE.RepeatWrapping && tex.wrapT === THREE.RepeatWrapping, '(e) the ground texture does not repeat-wrap');
    const mats = setMaterials();
    for (const k of GROUND_MATERIALS) ok(mats[k].userData.groundNoise === tex, `(e) t3_${k} has no ground texture`);
    for (const k of ['set', 'post', 'water']) ok(!mats[k].userData.groundNoise, `(e) t3_${k} should not be mottled`);
    const sh = { uniforms: {}, vertexShader: THREE.ShaderLib.standard.vertexShader, fragmentShader: THREE.ShaderLib.standard.fragmentShader };
    mats.ground.onBeforeCompile(sh);
    ok(sh.uniforms.pzNoise && sh.uniforms.pzNoise.value === tex, '(e) the shader has no noise uniform');
    ok(/vPzWorld = \(modelMatrix \* pzP\)/.test(sh.vertexShader) && /diffuseColor\.rgb \*= pzGround\(vPzWorld\.xz\)/.test(sh.fragmentShader), '(e) the shader does not sample the noise in world space');
    ok((sh.fragmentShader.match(/texture2D\(pzNoise/g) || []).length === 3 && !/fract\(|step\(/.test(sh.fragmentShader.slice(sh.fragmentShader.indexOf('float pzGround'), sh.fragmentShader.indexOf('float pzGround') + 600)), '(e) the ground shader is not three soft samples');
    // the built shot's floor and road carry it
    const b = await build('A car driving on the highway. Tracking shot.');
    const floors = [];
    b.root.traverseVisible((o) => { if (o.isMesh && [].concat(o.material).some((m) => /^t3_(ground|road|path)$/.test(m.name))) floors.push(o); });
    ok(floors.length >= 2 && floors.every((o) => [].concat(o.material).some((m) => m.userData.groundNoise)), `(e) the highway floor is not textured: ${floors.map((o) => o.name).join(',')}`);
  }

  // --- (b) exported frames have no editor graphics ---
  {
    const { buildScene } = await import('../../src/previz/scene.ts');
    const { PrevizRuntime } = await import('../../src/previz/runtime.ts');
    const { compile } = await import('../../src/previz/compile.ts');
    const spec = compile('A man walking in the park.').spec;
    const hadDoc = 'document' in globalThis;
    // the sky gradient's canvas and the debug line's element: nothing to draw into in Node
    if (!hadDoc) globalThis.document = { createElement: () => ({ getContext: () => null }), getElementById: () => null };
    try {
    const built = buildScene(spec, null);
    const overlays = [];
    built.scene.traverse((o) => { if (o.userData.overlay) overlays.push(o); });
    // Run 7: Top view's camera marker, path, people rings and grid sit in one overlay group, each part named
    const names = overlays.map((o) => o.name);
    ok(['top_overlay', 'camera_marker', 'camera_path', 'footprints', 'handles'].every((n) => names.includes(n)), `(b) the path, camera marker, rings or arrows are not marked overlay: ${overlays.map((o) => o.type + ':' + o.name)}`);    const seen = [];
    const renderer = {
      shadowMap: {}, getSize: (v) => v.set(1920, 1088), setScissorTest() {}, setViewport() {}, setScissor() {},
      render: (scene, cam) => { const on = []; scene.traverseVisible((o) => { if (o.userData.overlay) on.push(o.name || o.type); }); seen.push({ cam: cam === built.camera ? 'eye' : 'top', on }); },
    };
    const rt = Object.create(PrevizRuntime.prototype);
    Object.assign(rt, { renderer, spec, built, view: 'split', shotMode: false, clean: false, shown: 0 });
    built.actors.forEach((a) => { if (a.handles) a.handles.visible = true; }); // a subject picked in the editor
    rt.renderFrame(0);
    ok(seen.some((s) => s.on.length), '(b) the editor split view shows no path or arrows (the test is not looking)');
    seen.length = 0;
    rt.setClean(true);
    for (const i of [0, 48, 96]) rt.renderFrame(i);
    rt.setClean(false);
    ok(seen.length === 3 && seen.every((s) => s.cam === 'eye' && !s.on.length), `(b) an exported frame shows editor graphics: ${JSON.stringify(seen)}`);
    ok(built.actors.every((a) => !a.handles || a.handles.visible), '(b) the editor selection is not put back after export');
    // export.ts turns it on around the loop and off in finally
    const src = (await import('node:fs')).readFileSync(new URL('../../src/previz/export.ts', import.meta.url), 'utf8');
    ok(/rt\.setClean\(true\);\s*try \{/.test(src) && /finally \{\s*rt\.setClean\(false\);/.test(src), '(b) exportPreviz does not wrap its loop in clean mode');
    } finally { if (!hadDoc) delete globalThis.document; }
  }

  // --- (f) trees: placed or not at all, and pines, not balls on poles ---
  {
    for (const text of ['A woman standing by a lake.', 'A man walking in the park.', 'A woman walking down a suburban street.', 'Two friends hugging on a city sidewalk. Close to wide.']) {
      const b = await build(text);
      ok(!shownIds(b).includes('tree'), `(f) ${text}: kit trees shown the sentence never placed`);
    }
    for (const text of ['A man walking under the trees in the park.', 'A hiker walking through the woods.', 'A woman walking in the forest.']) {
      ok(shownIds(await build(text)).includes('tree'), `(f) ${text}: the trees it asks for are gone`);
    }
    ok(!WANTS_TREES.test('A man walking down the street.') && WANTS_TREES.test('a tree-lined avenue'), '(f) WANTS_TREES reads the wrong words');
    const b = await build('A hiker walking through the woods.');
    // Run 2 (R&D #4): the layout export leaves the tree proxies out (past the start still, LTX drew them as topiary).
    {
      const { PrevizRuntime } = await import('../../src/previz/runtime.ts');
      const rt = Object.create(PrevizRuntime.prototype);
      Object.assign(rt, { built: { scene: b.kit.root } });
      ok(rt.treesShowing().length > 0, '(f) Run 2: treesShowing finds no tree in the woods');
      const src = (await import('node:fs')).readFileSync(new URL('../../src/previz/runtime.ts', import.meta.url), 'utf8');
      ok(/if \(this\.clean\) \{\s*const hidden = \[\.\.\.this\.overlaysShowing\(\), \.\.\.this\.treesShowing\(\)\];/.test(src), '(f) Run 2: the clean export keeps the tree proxies');
    }
    const inst = b.kit.dressing.children.find((c) => idOf(c) === 'tree' && c.visible);
    ok(inst, '(f) no tree to measure');
    if (inst) {
      inst.updateMatrixWorld(true);
      const pts = [], mi = new THREE.Matrix4(), v = new THREE.Vector3(), base = new THREE.Vector3();
      let flatOK = true;
      inst.traverse((m) => {
        if (!m.isMesh) return;
        if (m.isInstancedMesh) m.getMatrixAt(0, mi); else mi.identity();
        const w = new THREE.Matrix4().multiplyMatrices(m.matrixWorld, mi);
        const pos = m.geometry.attributes.position, nrm = m.geometry.attributes.normal;
        for (let i = 0; i < pos.count; i++) pts.push(v.fromBufferAttribute(pos, i).applyMatrix4(w).clone());
        // faceted: each triangle's three normals are one (flat shading reads the cones' facets in the export)
        for (let i = 0; !m.geometry.index && i + 2 < nrm.count; i += 3) if (Math.abs(nrm.getX(i) - nrm.getX(i + 1)) + Math.abs(nrm.getY(i) - nrm.getY(i + 2)) > 1e-3) flatOK = false;
        if (m.geometry.index) flatOK = false;
      });
      ok(inst.children.length && inst.children.every((m) => m.material && LIT(m.material)), '(f) the pine is not lit');
      ok(flatOK, '(f) the pine is smooth-shaded, not faceted');
      const y0 = Math.min(...pts.map((p) => p.y)), h = Math.max(...pts.map((p) => p.y)) - y0;
      base.copy(pts.reduce((a, p) => (p.y < a.y ? p : a), pts[0]));
      const ax = pts.filter((p) => p.y - y0 < 0.02 * h), cx = ax.reduce((s2, p) => s2 + p.x, 0) / ax.length, cz = ax.reduce((s2, p) => s2 + p.z, 0) / ax.length;
      // the widest the tree gets at or above each height: a stack of cones narrows to its tip; a ball is widest half way up
      const rAbove = (f) => Math.max(0, ...pts.filter((p) => (p.y - y0) / h >= f - 1e-6).map((p) => Math.hypot(p.x - cx, p.z - cz))) / h;
      const prof = [0.2, 0.3, 0.5, 0.7, 0.9].map(rAbove);
      ok(prof.every((r, i) => !i || r <= prof[i - 1] + 1e-6) && prof[2] < prof[0] * 0.85, `(f) the canopy does not narrow to the top: ${prof.map((r) => r.toFixed(2))}`);
      ok(prof[3] < 0.05 && prof[0] > 0.12 && prof[0] < 0.3, `(f) not a pine's proportions (a ball on a pole?): ${prof.map((r) => r.toFixed(2))}`);
      ok(h > 3, `(f) the pine lost its height: ${h.toFixed(2)} m`);
    }
  }

  // --- (g) water edges: darker than the water, never a light stripe ---
  {
    const lum = (c) => 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    for (const text of ['A woman standing by a lake.', 'A man walking along the beach.', 'A woman sitting by the river.']) {
      const b = await build(text);
      let water = null;
      const edges = [];
      b.root.traverse((o) => {
        if (!o.isMesh || !o.material || !o.material.color) return; // hidden edges too: a later toggle must not bring back a light one
        const mn = o.material.name || '';
        if (mn === 't3_water') water = o;
        if (mn === 't3_foam' || mn === 't3_waterLine') edges.push(o);
      });
      ok(water, `(g) ${text}: no water to compare (${b.layout.id})`);
      if (!water) continue;
      const wl = lum(water.material.color);
      ok(edges.length > 0, `(g) ${text}: no shore edge or water lines found`);
      for (const e of edges) {
        const l = lum(e.material.color);
        ok(l < wl && l > wl * 0.6, `(g) ${text}: ${e.name} (${e.material.name}) is not a soft darker tone than the water (${l.toFixed(3)} vs ${wl.toFixed(3)})`);
      }
    }
  }

  return { fail: problems.length, problems, line: `step25 units (filler, overlays, body types, lit, ground, trees, water edges): ${problems.length ? problems.length + ' failing' : 'all pass'}` };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const r = await runStep25Units();
  for (const p of r.problems) console.log('  ' + p);
  console.log(r.line);
  process.exit(r.fail ? 1 : 0);
}
