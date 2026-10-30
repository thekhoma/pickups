// The demo autopilot (?tour=1, docs/demo-autopilot.md): the real UI plays itself in mock mode.
// A drawn cursor glides to each control and clicks it; typing is human-paced; renders are the mock's accelerated
// queue. One mono caption per beat: on a shot it sits inside the picture (bottom-left), clear of the transport, the
// take strip and the rail; elsewhere bottom-left of the page. It follows layout changes and steps aside from whatever
// the cursor is working on. Esc ends the tour; ?loop=1 starts again; "tour:done" fires at the end.
// On the end card a click or Esc plays the tour again from the top.
// Names are generic demo names (a "Stunt previz" project with the stunt driver's sheet and the set, the K03 hood push).
import { navigate, setCurrentProject, setUi } from "../actions";
import { DEFAULT_TOUR_COPY, loadTourCopy, mentionScript, type TourCopy } from "../lib/tourCopy";
import { getState } from "../store";

const params = new URLSearchParams(window.location.search);
const PACE = Math.min(4, Math.max(0.25, Number(params.get("pace")) || 1));
const LOOP = params.has("loop");

class Stopped extends Error {}
let stopped = false;
let running = false;
/** The end card is up: a click or Esc replays instead of stopping. */
let ended = false;

/** Deterministic jitter for typing (same video every time). */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(7);

function wait(ms: number): Promise<void> {
  return new Promise((resolve, reject) => {
    window.setTimeout(() => (stopped ? reject(new Stopped()) : resolve()), ms / PACE);
  });
}

async function waitFor<T>(find: () => T | null | undefined | false, timeout = 15000): Promise<T> {
  const t0 = performance.now();
  for (;;) {
    if (stopped) throw new Stopped();
    const v = find();
    if (v) return v;
    if (performance.now() - t0 > timeout) throw new Error("tour: timed out waiting");
    await new Promise((r) => window.setTimeout(r, 80));
  }
}

function shown(el: Element | null): el is HTMLElement {
  if (!(el instanceof HTMLElement)) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}

/** The first visible element matching `sel` whose text (or aria-label) matches. */
function find(sel: string, text?: string | RegExp, root: ParentNode = document): HTMLElement | null {
  for (const el of Array.from(root.querySelectorAll(sel))) {
    if (!shown(el)) continue;
    if (text == null) return el;
    const label = `${el.getAttribute("aria-label") ?? ""}\n${el.textContent ?? ""}`.trim();
    if (typeof text === "string" ? label.split("\n").some((l) => l.trim().startsWith(text)) : text.test(label)) return el;
  }
  return null;
}

// --- the overlay: cursor, caption, key hint, end card ------------------------------------------------------------

let layer: HTMLDivElement | null = null;
let cursor: HTMLDivElement | null = null;
let caption: HTMLDivElement | null = null;
let keyHint: HTMLDivElement | null = null;
let pos = { x: window.innerWidth * 0.6, y: window.innerHeight * 0.45 };
/** What the cursor is working on: the caption keeps clear of it. */
let aim: HTMLElement | null = null;
let captionFrame = 0;

function overlay() {
  if (layer) return;
  layer = document.createElement("div");
  layer.setAttribute("aria-hidden", "true");
  layer.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483000;";
  cursor = document.createElement("div");
  cursor.style.cssText =
    "position:absolute;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;border:2px solid rgba(238,233,224,.96);" +
    "box-shadow:0 0 0 1px rgba(0,0,0,.35),0 2px 12px rgba(0,0,0,.5);will-change:transform;transition:transform 0ms;";
  const dot = document.createElement("div");
  dot.style.cssText = "position:absolute;left:50%;top:50%;width:4px;height:4px;margin:-2px 0 0 -2px;border-radius:50%;background:#EEE9E0;";
  cursor.appendChild(dot);
  caption = document.createElement("div");
  caption.style.cssText =
    "position:absolute;left:24px;top:0;display:none;align-items:center;gap:8px;padding:7px 11px;border-radius:3px;white-space:nowrap;" +
    "background:rgba(13,12,11,.86);border:1px solid #3A3631;color:#EEE9E0;font:500 11px/1 'Geist Mono',ui-monospace,monospace;" +
    "letter-spacing:.08em;text-transform:uppercase;transition:opacity 200ms ease;";
  keyHint = document.createElement("div");
  keyHint.style.cssText =
    "position:absolute;left:0;top:0;display:none;padding:4px 8px;border-radius:4px;background:#EEE9E0;color:#0D0C0B;" +
    "font:600 13px/1 'Geist Mono',ui-monospace,monospace;box-shadow:0 2px 10px rgba(0,0,0,.45);";
  layer.append(caption, keyHint, cursor);
  document.body.appendChild(layer);
  place(pos.x, pos.y, 0);
  // Layout moves under the caption (a shot loads, the rail opens a form and the picture shrinks): keep it placed every
  // frame, so a page change never leaves it over the take strip for a moment.
  const follow = () => {
    placeCaption();
    captionFrame = window.requestAnimationFrame(follow);
  };
  captionFrame = window.requestAnimationFrame(follow);
}

function place(x: number, y: number, ms: number) {
  pos = { x, y };
  if (!cursor) return;
  cursor.style.transition = `transform ${Math.round(ms)}ms cubic-bezier(.45,.05,.25,1)`;
  cursor.style.transform = `translate(${x}px, ${y}px)`;
}

function say(text: string | null) {
  if (!caption) return;
  if (!text) {
    caption.style.display = "none";
    return;
  }
  caption.innerHTML = "";
  const dot = document.createElement("span");
  dot.style.cssText = "width:6px;height:6px;border-radius:50%;background:#F5301E;display:inline-block;";
  caption.append(dot, document.createTextNode(text));
  caption.style.display = "flex";
  placeCaption();
}

/**
 * Where the caption may sit, best first. On a shot: inside the picture, so it can't cover the transport, the take
 * strip or the rail's buttons (none while the shot is still loading). On the Image page: the results canvas first
 * (empty until the images land). Elsewhere: bottom-left, top-right under the bar, top-left, bottom-right.
 */
function captionSpots(w: number, h: number): Array<{ x: number; y: number }> {
  const route = getState().route.name;
  if (route === "shot") {
    const r = document.querySelector('[aria-label="Viewer"] video')?.getBoundingClientRect();
    if (!r || r.width < w + 32 || r.height < h + 32) return [];
    return [
      { x: r.left + 16, y: r.bottom - 16 - h },
      { x: r.right - 16 - w, y: r.bottom - 16 - h },
    ];
  }
  const spots: Array<{ x: number; y: number }> = [];
  if (route === "create") {
    const r = document.querySelector('section[aria-label="Results"]')?.getBoundingClientRect();
    if (r && r.width >= w + 48 && r.height >= h + 120) spots.push({ x: r.left + (r.width - w) / 2, y: r.top + 96 }, { x: r.right - 24 - w, y: r.bottom - 24 - h });
  }
  const W = window.innerWidth;
  const H = window.innerHeight;
  return [...spots, { x: 24, y: H - 60 - h }, { x: W - 24 - w, y: 72 }, { x: 24, y: 72 }, { x: W - 24 - w, y: H - 24 - h }];
}

/** What the caption must not sit on: every visible control on the page (buttons, links, fields, radios, cards), read at
 * most every 200 ms. */
const CONTROLS = 'button, a[href], input, textarea, select, [role="button"], [role="radio"], [role="tab"], [role="menuitem"], [role="slider"]';
let controls: { at: number; rects: DOMRect[] } = { at: 0, rects: [] };
function controlRects(): DOMRect[] {
  const now = performance.now();
  if (now - controls.at < 200) return controls.rects;
  const rects: DOMRect[] = [];
  for (const el of Array.from(document.querySelectorAll(CONTROLS))) {
    if (layer?.contains(el)) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.top > window.innerHeight || r.right < 0 || r.left > window.innerWidth) continue;
    rects.push(r);
  }
  controls = { at: now, rects };
  return rects;
}

function placeCaption() {
  if (!caption || caption.style.display === "none") return;
  const w = caption.offsetWidth;
  const h = caption.offsetHeight;
  const spots = captionSpots(w, h);
  if (!spots.length) {
    if (caption.style.visibility !== "hidden") caption.style.visibility = "hidden";
    return;
  }
  const a = aim?.isConnected ? aim.getBoundingClientRect() : null;
  const overlap = (s: { x: number; y: number }, r: DOMRect) =>
    Math.max(0, Math.min(s.x + w, r.right) - Math.max(s.x, r.left)) * Math.max(0, Math.min(s.y + h, r.bottom) - Math.max(s.y, r.top));
  // Inside the picture nothing else is in the way; elsewhere the first spot that covers no control wins, else the one
  // that covers least. What the cursor is working on counts most.
  const others = getState().route.name === "shot" ? [] : controlRects();
  const cost = (s: { x: number; y: number }) => (a && a.width ? overlap(s, a) * 100 : 0) + others.reduce((t, r) => t + overlap(s, r), 0);
  let spot = spots[0];
  let best = Infinity;
  for (const s of spots) {
    const c = cost(s);
    if (c < best) {
      best = c;
      spot = s;
    }
    if (c === 0) break;
  }
  const left = `${Math.round(spot.x)}px`;
  const top = `${Math.round(spot.y)}px`;
  if (caption.style.left !== left) caption.style.left = left;
  if (caption.style.top !== top) caption.style.top = top;
  if (caption.style.visibility !== "visible") caption.style.visibility = "visible";
}

async function pulse() {
  if (!cursor || !layer) return;
  const ring = document.createElement("div");
  ring.style.cssText = `position:absolute;left:${pos.x}px;top:${pos.y}px;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;border:2px solid rgba(238,233,224,.8);transition:transform 320ms ease-out,opacity 320ms ease-out;`;
  layer.appendChild(ring);
  cursor.style.transition = "transform 160ms ease";
  cursor.style.transform = `translate(${pos.x}px, ${pos.y}px) scale(.82)`;
  requestAnimationFrame(() => {
    ring.style.transform = "scale(2)";
    ring.style.opacity = "0";
  });
  window.setTimeout(() => ring.remove(), 400);
  await wait(160);
  if (cursor) cursor.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
}

async function showKey(text: string) {
  if (!keyHint) return;
  keyHint.textContent = text;
  keyHint.style.display = "block";
  keyHint.style.transform = `translate(${pos.x + 16}px, ${pos.y + 14}px)`;
  await wait(900);
  keyHint.style.display = "none";
}

// --- moving, clicking, typing --------------------------------------------------------------------------------------

async function moveTo(target: HTMLElement | { x: number; y: number }, fx = 0.5, fy = 0.5) {
  let x: number;
  let y: number;
  if (target instanceof HTMLElement) {
    aim = target;
    let r = target.getBoundingClientRect();
    if (r.top < 64 || r.bottom > window.innerHeight - 56) {
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      await wait(450);
      r = target.getBoundingClientRect();
    }
    x = r.left + r.width * fx;
    y = r.top + r.height * fy;
  } else {
    aim = null;
    ({ x, y } = target);
  }
  placeCaption();
  const d = Math.hypot(x - pos.x, y - pos.y);
  const ms = Math.min(820, Math.max(240, d * 0.85)) / PACE;
  place(x, y, ms);
  await wait(ms * PACE + 60);
}

function fire(el: HTMLElement, type: string, init: Record<string, unknown> = {}) {
  const base = { bubbles: true, cancelable: true, clientX: pos.x, clientY: pos.y, view: window, ...init };
  const ev = type.startsWith("pointer") ? new PointerEvent(type, { pointerId: 1, pointerType: "mouse", isPrimary: true, ...base }) : new MouseEvent(type, base);
  el.dispatchEvent(ev);
}

async function click(el: HTMLElement, hold = 260) {
  await moveTo(el);
  await pulse();
  fire(el, "pointerdown");
  fire(el, "mousedown");
  fire(el, "pointerup");
  fire(el, "mouseup");
  el.click();
  await wait(hold);
}

async function clickWhen(sel: string, text?: string | RegExp, hold?: number, root?: ParentNode) {
  const el = await waitFor(() => find(sel, text, root));
  await click(el, hold);
  return el;
}

const setValue = (el: HTMLInputElement | HTMLTextAreaElement, v: string) => {
  const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(el, v);
  el.dispatchEvent(new Event("input", { bubbles: true }));
};

/** Types text at a human pace. `{Name}` picks a cast member from the @-mention list (chips appear). */
async function typeInto(el: HTMLInputElement | HTMLTextAreaElement, script: string, ms = 35) {
  await click(el, 120);
  el.focus();
  const parts = script.split(/(\{[^}]+\})/).filter(Boolean);
  let afterMention = false;
  for (const part of parts) {
    const m = /^\{(.+)\}$/.exec(part);
    if (m) {
      const name = m[1];
      for (const ch of `@${name.slice(0, 2)}`) {
        setValue(el, el.value + ch);
        await wait(40 + rand() * 25);
      }
      const option = await waitFor(() => find('[role="listbox"] [role="option"]', name), 4000).catch(() => null);
      if (option) {
        await wait(200);
        await click(option, 140);
        await waitFor(() => el.value.includes(`@${name}`), 2000).catch(() => null);
        await moveTo(el, 0.85, 0.7);
      } else {
        setValue(el, el.value + name.slice(2) + " ");
      }
      el.focus();
      afterMention = true;
      continue;
    }
    if (afterMention && /^[,.;:!?]/.test(part) && el.value.endsWith(" ")) setValue(el, el.value.slice(0, -1));
    const text = afterMention && part.startsWith(" ") ? part.slice(1) : part;
    afterMention = false;
    for (const ch of text) {
      setValue(el, el.value + ch);
      await wait(ms + rand() * ms * 0.7 + (/[,.;]/.test(ch) ? 80 : 0));
    }
  }
}

/** Glide across an element, sending pointer moves (scrubs a hover-wipe card). */
async function scrub(el: HTMLElement, from = 0.2, to = 0.85, ms = 1600) {
  await moveTo(el, from, 0.5);
  const under = () => (document.elementFromPoint(pos.x, pos.y) as HTMLElement | null) ?? el;
  let target = under();
  fire(target, "pointerover");
  const r = el.getBoundingClientRect();
  const steps = Math.round(ms / 40);
  for (let i = 0; i <= steps; i++) {
    const k = i / steps;
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    const x = r.left + r.width * (from + (to - from) * e);
    place(x, r.top + r.height * 0.5, 40);
    target = under();
    fire(target, "pointermove");
    await wait(40);
  }
  await wait(300);
  fire(target, "pointerout", { relatedTarget: document.body });
}

/** Sweep the shot viewer's wipe (the take landed: playblast → final). */
async function sweepWipe(shotId: string, path = [50, 18, 82, 50]) {
  const stage = find('[aria-label="Viewer"] video') ?? find('[aria-label="Viewer"]');
  const r = stage?.getBoundingClientRect();
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1];
    const b = path[i];
    const steps = 30;
    for (let s = 0; s <= steps; s++) {
      const k = s / steps;
      const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      const p = a + (b - a) * e;
      setUi(shotId, { wipePos: p, wipe: true, view: "after" });
      if (r) place(r.left + (r.width * p) / 100, r.top + r.height * 0.55, 40);
      await wait(36);
    }
    await wait(180);
  }
}

function key(k: string, opts: KeyboardEventInit = {}) {
  window.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...opts }));
}

// --- the end card -----------------------------------------------------------------------------------------------

let replayHint: HTMLDivElement | null = null;

async function endCard() {
  say(null);
  if (cursor) cursor.style.display = "none";
  const card = document.createElement("div");
  card.style.cssText =
    "position:fixed;inset:0;z-index:2147483001;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;" +
    "background:#0D0C0B;opacity:0;transition:opacity 600ms ease;text-align:center;padding:0 8vw;cursor:pointer;";
  card.innerHTML =
    '<div style="font:700 48px/1 \'Instrument Sans\',sans-serif;color:#EEE9E0;letter-spacing:-.04em;text-transform:lowercase">pickups</div>' +
    '<div style="font:400 clamp(18px,2.1vw,28px)/1.2 \'Instrument Sans\',sans-serif;color:#EEE9E0;letter-spacing:.14em;text-transform:uppercase;max-width:1100px">Power your creativity, locally.</div>' +
    '<div style="font:500 12px/1 \'Geist Mono\',ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:#A39D93">Open source · runs on ComfyUI</div>';
  replayHint = document.createElement("div");
  replayHint.textContent = "Click or press Esc to play again";
  replayHint.style.cssText =
    "position:absolute;left:0;right:0;bottom:56px;font:500 11px/1 'Geist Mono',ui-monospace,monospace;letter-spacing:.1em;" +
    "text-transform:uppercase;color:#857F77;opacity:0;transition:opacity 600ms ease;";
  card.appendChild(replayHint);
  card.addEventListener("click", replay);
  document.body.appendChild(card);
  ended = true;
  requestAnimationFrame(() => (card.style.opacity = "1"));
  await wait(3400);
}

/** After "tour:done" the replay hint fades in. A recording (webdriver) keeps the card clean. */
function offerReplay() {
  if (replayHint && !navigator.webdriver) replayHint.style.opacity = "1";
}

/** Plays the whole tour again on a fresh page, so the mock starts clean (same as ?loop=1). */
function replay() {
  const url = new URL(window.location.href);
  url.searchParams.set("tour", "1");
  url.hash = "";
  window.history.replaceState(null, "", url);
  window.location.reload();
}

// --- the beats -----------------------------------------------------------------------------------------------------

const SHOT = "shot_hood_push";
/** Links in the left panel (Home, Image, Projects and each project; expanded or as the icon rail). */
const PANEL_LINK = 'aside[aria-label="Panel"] a';
const WEATHER_SHOT = "shot_castle_dusk";
const WEATHER_PROJECT = "prj_highlands";
/** The demo project (the previz shot, its cast, the image beat). */
const DEMO_PROJECT = "prj_previz";
/** The demo's words (tour.json on this machine, or the generic built-ins). */
let copy: TourCopy = DEFAULT_TOUR_COPY;
const names = () => copy.cast.map((c) => c.name);

async function beatHome() {
  say("01 · Remove, with real pixels");
  const hero = await waitFor(() => find("section video"), 12000);
  await wait(500);
  await moveTo(hero, 0.62, 0.5);
  await wait(1100);
  const card = await waitFor(() => find('section[aria-labelledby="make-title"] button .hairline-frame') ?? find("button .hairline-frame"), 6000).catch(() => null);
  if (card) await scrub(card, 0.15, 0.9, 1300);
  await wait(200);
}

async function beatProject() {
  say("02 · Projects and cast");
  window.scrollTo({ top: 0, behavior: "smooth" });
  await wait(300);
  // Projects live in the left panel: open the demo project there.
  await clickWhen(PANEL_LINK, copy.project, 600);
  await waitFor(() => find(`button[aria-label^="${CSS.escape(copy.previz_shot)}"]`));
  await wait(800);
  await clickWhen('nav[aria-label="Project"] a', "Cast", 400);
  await waitFor(() => Array.from(document.querySelectorAll("p.mono")).some((p) => p.textContent?.includes(`@${names()[0]}`)));
  const last = names()[names().length - 1];
  const card = Array.from(document.querySelectorAll<HTMLElement>("div.group")).find((d) => shown(d) && d.textContent?.includes(`@${last}`)) ?? null;
  if (card) await moveTo(card, 0.5, 0.35);
  await wait(1100);
}

async function beatMention() {
  say("03 · @-mention your cast");
  await clickWhen('nav[aria-label="Project"] a', "Shots", 350);
  await clickWhen(`button[aria-label^="${CSS.escape(copy.previz_shot)}"]`, undefined, 500);
  await waitFor(() => getState().route.name === "shot");
  await clickWhen("aside button, button", "Previz to final", 450);
  const look = await waitFor(() => find('textarea[aria-label="Look"]') as HTMLTextAreaElement | null);
  await typeInto(look, mentionScript(copy.prompt, names()), 24);
  await wait(400);
  const receipt = find("div", /^What each model gets/i);
  if (receipt) await moveTo(receipt, 0.5, 0.5);
  await wait(1200);
}

async function beatDirector() {
  say("04 · Prompt director");
  const look = await waitFor(() => find('textarea[aria-label="Look"]'));
  let box: HTMLElement | null = look;
  while (box && !find("button", "Enhance", box)) box = box.parentElement;
  const enhance = box ? find("button", "Enhance", box) : find("button", "Enhance");
  if (!enhance) return;
  await click(enhance, 200);
  const use = await waitFor(() => find("button", /^Use$/), 12000);
  await wait(500);
  const panel = use.closest("div.rounded-\\[4px\\]") as HTMLElement | null;
  if (panel) await moveTo(panel, 0.5, 0.45);
  await wait(copy.director_text ? 1900 : 1400);
  await click(use, 400);
}

async function beatRender() {
  say("05 · Render locally");
  const go = await waitFor(() => Array.from(document.querySelectorAll<HTMLElement>("button")).find((b) => shown(b) && b.textContent?.trim() === "Previz to final" && b.className.includes("bg-ivory")) ?? null);
  await click(go, 400);
  const clock = await waitFor(() => find("div.mono", /\d\d:\d\d:\d\d/), 6000).catch(() => null);
  if (clock) await moveTo(clock, 0.7, 0.5);
  await waitFor(() => find("div", /^What changed$/), 30000);
  await wait(400);
  placeCaption();
  key(" ");
  await wait(250);
  await sweepWipe(SHOT);
  await wait(300);
}

async function beatDeliver() {
  say("06 · Approve and export");
  const approve = find("button", /^Approve$/);
  if (approve) await moveTo(approve);
  await pulse();
  key("A", { shiftKey: true });
  await showKey("⇧A");
  await waitFor(() => find("button", /^Approved$/), 4000).catch(() => null);
  await wait(300);
  await clickWhen(`nav[aria-label="Breadcrumb"] a`, copy.project, 450);
  await clickWhen('nav[aria-label="Project"] a', "Deliveries", 500);
  await waitFor(() => find("li button .hairline-frame"));
  await wait(700);
  await clickWhen("button", "Export timeline", 400);
  await clickWhen('[role="menuitem"]', "Resolve / Premiere (XML)", 300);
  await waitFor(() => find("p", /^Saved /), 6000).catch(() => null);
  await wait(1000);
}

async function beatShare() {
  say("07 · Share a cut-down");
  const row = await waitFor(() => Array.from(document.querySelectorAll<HTMLElement>("li button")).find((b) => shown(b) && !!b.querySelector(".hairline-frame")) ?? null);
  await click(row, 700);
  await waitFor(() => find("div", /^What changed$/));
  await clickWhen("button", "Share", 400);
  await clickWhen('[role="menuitem"]', "9:16 stacked", 300);
  const busy = find('[role="status"]', /Rendering/);
  if (busy) await moveTo(busy, 0.97, 1.6);
  await waitFor(() => find('video[aria-label="9:16 stacked preview"]'), 12000);
  const prev = find('video[aria-label="9:16 stacked preview"]');
  if (prev) await moveTo(prev, 1.6, 0.5);
  await wait(1800);
}

async function beatWeather() {
  say("08 · Weather");
  // The weather shot lives in the mock's other project (with the castle footage): switch to it, then open the shot.
  const other = getState().projects?.find((p) => p.id === WEATHER_PROJECT)?.name;
  if (other) {
    await clickWhen(PANEL_LINK, other, 600);
  } else {
    window.location.hash = "#/projects";
  }
  const tile = await waitFor(() => find(`button[aria-label^="${CSS.escape(copy.weather_shot)}"]`), 8000);
  await click(tile, 500);
  await waitFor(() => {
    const r = getState().route;
    return r.name === "shot" && r.id === WEATHER_SHOT;
  });
  await clickWhen("aside button, button", "Change the weather", 400);
  await clickWhen('[role="radiogroup"][aria-label="Change"] [role="radio"], [role="radiogroup"][aria-label="Change"] button', "Snow", 300);
  const go = await waitFor(() => Array.from(document.querySelectorAll<HTMLElement>("button")).find((b) => shown(b) && b.textContent?.trim() === "Change the weather" && b.className.includes("bg-ivory")) ?? null);
  await click(go, 300);
  const clock = await waitFor(() => find("div.mono", /\d\d:\d\d:\d\d/), 6000).catch(() => null);
  if (clock) await moveTo(clock, 0.7, 0.5);
  await waitFor(() => find("div", /^What changed$/), 30000);
  await wait(300);
  placeCaption();
  key(" ");
  await wait(200);
  await sweepWipe(WEATHER_SHOT, [50, 10, 50]);
  await wait(200);
}

async function beatImage() {
  say("09 · Images");
  // Back to the demo project (the weather beat switched away), so the new image joins its cast.
  if (getState().currentProjectId !== DEMO_PROJECT) {
    await clickWhen(PANEL_LINK, copy.project, 500);
  }
  await clickWhen(PANEL_LINK, "Image", 600);
  await waitFor(() => find("h1", "Image"));
  const change = await waitFor(() => find("button", "Change"), 8000).catch(() => null);
  if (change) {
    await click(change, 350);
    await clickWhen('[role="radio"]', /FLUX 3/, 300);
  }
  await clickWhen('[role="radiogroup"][aria-label="Aspect"] [role="radio"]', "16:9", 150);
  await clickWhen('[role="radiogroup"][aria-label="Resolution"] [role="radio"]', "2K", 150);
  const prompt = await waitFor(() => find('textarea[aria-label="Prompt"]') as HTMLTextAreaElement | null);
  setValue(prompt, "");
  await typeInto(prompt, mentionScript(copy.image_prompt, names()), 24);
  await clickWhen('[role="radiogroup"][aria-label="Count"] [role="radio"]', "4", 250);
  await clickWhen("button", "Generate", 300);
  await waitFor(() => document.querySelectorAll('button[aria-label="Open image"]').length >= 4, 20000);
  await wait(900);
  await clickWhen('button[aria-label="Open image"]', undefined, 600);
  const dialog = await waitFor(() => find('[role="dialog"]'));
  await clickWhen("button", "Save as character", 350, dialog);
  const name = await waitFor(() => find('input[aria-label="Name"]', undefined, dialog) as HTMLInputElement | null);
  await typeInto(name, copy.image_cast_name, 30);
  await clickWhen("button", new RegExp(`^${copy.image_cast_kind}$`, "i"), 200, dialog);
  const add = await waitFor(() => Array.from(dialog.querySelectorAll("button")).find((b) => b.textContent?.trim() === "Save as character" && b.className.includes("bg-ivory")) ?? null);
  await click(add, 1000);
  await wait(700);
}

const BEATS: Array<[string, () => Promise<unknown>]> = [
  ["home", beatHome],
  ["project", beatProject],
  ["mention", beatMention],
  ["director", () => beatDirector()],
  ["render", beatRender],
  ["deliver", beatDeliver],
  ["share", beatShare],
  ["weather", beatWeather],
  ["image", beatImage],
];

async function run() {
  copy = (await loadTourCopy()) ?? DEFAULT_TOUR_COPY;
  overlay();
  await waitFor(() => getState().projects, 15000);
  // ?tour=<beat> starts there (for checking one beat); ?tour=1 plays everything from the hero.
  const from = Math.max(0, BEATS.findIndex(([n]) => n === params.get("tour")));
  if (from === 0) {
    // Start clean: an empty project, so Home opens on the hero.
    setCurrentProject("unsorted");
    navigate({ name: "home" });
  } else {
    setCurrentProject(DEMO_PROJECT);
    navigate(from >= BEATS.findIndex(([n]) => n === "image") ? { name: "create", panel: "image" } : { name: "shot", id: SHOT });
  }
  await wait(1200);
  for (const [name, beat] of BEATS.slice(from)) await safely(name, beat);
  await endCard();
}

/** A beat that fails (a selector moved) is skipped, so a recording always reaches the end card. */
async function safely(name: string, fn: () => Promise<unknown>) {
  try {
    await fn();
  } catch (e) {
    if (e instanceof Stopped) throw e;
    console.warn(`[tour] skipped the ${name} beat:`, e);
    (window as unknown as { __tourSkipped?: string[] }).__tourSkipped = [...((window as unknown as { __tourSkipped?: string[] }).__tourSkipped ?? []), `${name}: ${String(e)}`];
  }
}

function stop() {
  stopped = true;
  window.cancelAnimationFrame(captionFrame);
  layer?.remove();
  layer = cursor = caption = keyHint = aim = null;
  window.removeEventListener("keydown", onKey, true);
}

function onKey(e: KeyboardEvent) {
  if (e.key !== "Escape" || !e.isTrusted) return;
  if (ended) replay();
  else stop();
}

export async function startTour(): Promise<void> {
  if (running) return;
  running = true;
  window.addEventListener("keydown", onKey, true);
  try {
    await document.fonts?.ready;
    await run();
    window.dispatchEvent(new Event("tour:done"));
    if (LOOP && !stopped) window.location.reload();
    else offerReplay();
  } catch (e) {
    if (!(e instanceof Stopped)) {
      console.warn("[tour] stopped:", e);
      window.dispatchEvent(new Event("tour:done"));
    }
  } finally {
    running = false;
  }
}
