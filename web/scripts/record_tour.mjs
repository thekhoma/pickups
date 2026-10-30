// Record the demo autopilot (?tour=1) to an MP4 for the README, the launch page and the contest video.
//
// Usage (from web/): node scripts/record_tour.mjs [url] [out.mp4]
//   url defaults to http://127.0.0.1:7788/?tour=1 ; the page fires a window "tour:done" event at the end.
// Needs the `playwright` dev dependency and ffmpeg on PATH. Records at 1920x1080 (device scale 1), 30 fps WebM, then
// converts to H.264 MP4 plus a 1280-wide loop.
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const url = process.argv[2] ?? "http://127.0.0.1:7788/?tour=1";
const out = resolve(process.argv[3] ?? "pickups_tour.mp4");
const dir = mkdtempSync(join(tmpdir(), "pickups-tour-"));

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 1,
  colorScheme: "dark",
  recordVideo: { dir, size: { width: 1920, height: 1080 } },
});
const page = await context.newPage();
await page.goto(url, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.evaluate(
  () => new Promise((done) => window.addEventListener("tour:done", () => done(), { once: true })),
  { timeout: 5 * 60_000 },
);
await page.waitForTimeout(800); // hold the end card
await context.close();
await browser.close();

const webm = readdirSync(dir).find((f) => f.endsWith(".webm"));
renameSync(join(dir, webm), join(dir, "tour.webm"));
execFileSync("ffmpeg", ["-v", "error", "-y", "-i", join(dir, "tour.webm"), "-c:v", "libx264", "-crf", "16", "-preset", "slow",
  "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]);
execFileSync("ffmpeg", ["-v", "error", "-y", "-i", out, "-vf", "scale=1280:-2:flags=lanczos", "-c:v", "libx264", "-crf", "22",
  "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out.replace(/\.mp4$/, "_1280.mp4")]);
console.log("wrote", out);
