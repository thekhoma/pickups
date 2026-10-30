// Run 7: Create's Text to video and Image to video (LTX-2.5). The sizes the pickers offer (multiples of 64, 1080p at 16:9
// only), lengths as 8n+1 frames at 24 fps never over 97, the start picture's centre crop, and where a picker's list opens
// (below when it fits, else above, else the roomier side, capped with a scroll).
import { coverCrop, framesFor, hasSize, resolutionFor, VIDEO_ASPECTS, VIDEO_DURATIONS, VIDEO_SIZES, videoSize } from '../../src/lib/video.ts';
import { placeMenu } from '../../src/lib/placeMenu.ts';

export function runRun7Video() {
  const problems = [];
  const ok = (cond, what) => { if (!cond) problems.push(what); };
  const near = (a, b) => Math.abs(a - b) < 1e-9;

  for (const [res, row] of Object.entries(VIDEO_SIZES)) {
    for (const [aspect, [w, h]] of Object.entries(row)) {
      ok(w % 64 === 0 && h % 64 === 0, `${res} ${aspect}: ${w}x${h} not multiples of 64`);
      const [aw, ah] = aspect.split(':').map(Number);
      ok(Math.abs(w / h - aw / ah) / (aw / ah) < 0.1, `${res} ${aspect}: ${w}x${h} is off its aspect`);
    }
  }
  ok(VIDEO_ASPECTS.every((a) => hasSize(a, '720p')), 'every aspect has 720p');
  ok(hasSize('16:9', '1080p') && !hasSize('9:16', '1080p') && !hasSize('21:9', '1080p'), '1080p is 16:9 only');
  ok(resolutionFor('9:16', '1080p') === '720p' && resolutionFor('16:9', '1080p') === '1080p', 'an aspect without 1080p falls back to 720p');
  ok(videoSize('16:9', '1080p').join('x') === '1920x1088' && videoSize('9:16', '1080p').join('x') === '704x1280', 'sizes');

  ok(JSON.stringify(VIDEO_DURATIONS.map((d) => d.frames)) === '[25,49,73,97]', 'durations are 25, 49, 73, 97 frames');
  for (const d of VIDEO_DURATIONS) ok((d.frames - 1) % 8 === 0 && d.frames === d.seconds * 24 + 1, `${d.seconds} s is ${d.frames}`);
  ok(framesFor(4) === 97 && framesFor(10) === 97 && framesFor(0) === 9 && (framesFor(2.5) - 1) % 8 === 0, 'framesFor stays 8n+1 within 9..97');

  // A 16:9 picture: at 16:9 all of it; at 9:16 a centred vertical strip; at 21:9 a centred band (top and bottom trimmed).
  const full = coverCrop(1920, 1080, '16:9');
  ok(near(full.x, 0) && near(full.w, 1) && near(full.h, 1), `16:9 at 16:9: ${JSON.stringify(full)}`);
  const tall = coverCrop(1920, 1080, '9:16');
  ok(near(tall.h, 1) && near(tall.x, (1 - tall.w) / 2) && near(tall.w, (9 / 16) / (16 / 9)), `16:9 at 9:16: ${JSON.stringify(tall)}`);
  const wide = coverCrop(1920, 1080, '21:9');
  ok(near(wide.w, 1) && near(wide.y, (1 - wide.h) / 2) && wide.h < 1, `16:9 at 21:9: ${JSON.stringify(wide)}`);

  // The image-model list (7 rows, about 260 px) from the docked composer near the bottom of a 900 px window: it opens up.
  const view = { width: 1440, height: 900 };
  const dock = placeMenu({ top: 820, bottom: 856, left: 300, width: 180 }, { height: 260, width: 240 }, view);
  ok(dock.up && dock.maxH === 260 && dock.bottom === 900 - 820 + 8, `dock: ${JSON.stringify(dock)}`);
  // From high on the page it opens down.
  const hero = placeMenu({ top: 300, bottom: 336, left: 300, width: 180 }, { height: 260, width: 240 }, view);
  ok(!hero.up && hero.top === 344 && hero.maxH === 260, `hero: ${JSON.stringify(hero)}`);
  // No room either way at full height: the roomier side, capped, so it scrolls and never leaves the window.
  const cramped = placeMenu({ top: 340, bottom: 376, left: 1400, width: 180 }, { height: 600, width: 240 }, { width: 1440, height: 640 });
  ok(cramped.up && cramped.maxH === 340 - 8 - 12 && cramped.left === 1440 - 240 - 12, `cramped: ${JSON.stringify(cramped)}`);
  return problems;
}
