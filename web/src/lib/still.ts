// Recast's "Make one": a character still FLUX 3 klein makes from a few words, before the render. It runs as an
// ordinary image job (task "image", the Images path), lands in the project's Images and goes to Recast as `who_ref`.
// The framing mirrors tasks/recast.py STILL (whole body, facing camera, arms a little away, plain backdrop), the still
// Recast makes by itself when it only has words.
import type { ImageGenerateRequest } from "../types";

export const STILL_PREFIX = "Studio photograph, full-length shot of";

const FRAMING =
  "standing centered and facing the camera in a relaxed neutral stance, arms slightly away from the body, feet " +
  "shoulder-width apart, the whole figure from the top of the head to the shoes inside the frame, hands empty. Soft, " +
  "even light from the front, a gentle shadow under the chin. Plain light-grey seamless backdrop and floor, a faint " +
  "contact shadow under the feet.";

/** The full caption FLUX 3 klein gets for "a friendly green alien in a silver flight suit". */
export function stillPrompt(who: string): string {
  return `${STILL_PREFIX} ${who.trim().replace(/[.\s]+$/, "")}, ${FRAMING}`;
}

/** Whether an image was made by stillPrompt (a project image Recast made from words). */
export function isStillPrompt(prompt: string | null | undefined): boolean {
  return !!prompt && prompt.startsWith(STILL_PREFIX);
}

/** The words a still was made from ("" when the prompt isn't one of ours). */
export function stillWords(prompt: string | null | undefined): string {
  if (!isStillPrompt(prompt)) return "";
  const rest = prompt!.slice(STILL_PREFIX.length).trim();
  const end = rest.lastIndexOf(`, ${FRAMING}`);
  return (end >= 0 ? rest.slice(0, end) : rest).trim();
}

/** Portrait, about 1 MP (832×1280 on FLUX 3 klein), one image. */
export function stillRequest(who: string, seed: number): ImageGenerateRequest {
  return { model: "flux3-klein", prompt: stillPrompt(who), aspect: "2:3", resolution: "1K", count: 1, seed, refs: [] };
}

export const newSeed = (): number => 1 + Math.floor(Math.random() * 2_000_000_000);

// "Describe them" is often written as the change ("turn him into a knight in shining armour"). FLUX 3 klein's caption
// and SCAIL-2's prompt both want the person ("a knight in shining armour"), so a lead-in like that comes off first.
const WHO = String.raw`(?:him|her|them|it|me|us|(?:this|that)\s+[\w'-]+|the(?:\s+[\w'-]+){1,4}?)`;
const LEAD_INS = [
  new RegExp(String.raw`^(?:please\s+)?(?:turn|change|transform|convert|morph)\s+(?:${WHO}\s+)?(?:in)?to\s+(?:be\s+)?`, "i"),
  new RegExp(String.raw`^(?:please\s+)?(?:replace|swap)\s+${WHO}\s+(?:with|for|by)\s+`, "i"),
  // "make him a knight", "make the driver look like a robot" (not "make him older": no person follows).
  new RegExp(String.raw`^(?:please\s+)?make\s+(?:him|her|them|it|(?:the|this|that)\s+[\w'-]+)\s+(?:into\s+|look\s+like\s+|be\s+)?(?=(?:an?|the|some|@)\s?)`, "i"),
  new RegExp(String.raw`^(?:please\s+)?(?:re-?cast|cast)\s+${WHO}\s+as\s+`, "i"),
  new RegExp(String.raw`^i\s*(?:want|would\s+like|'d\s+like)\s+${WHO}\s+to\s+(?:be|look\s+like)\s+`, "i"),
  /^(?:as|into|to\s+be|be|now)\s+/i,
];

/** The person in "Describe them", without an instruction's lead-in: "turn him into a knight" → "a knight". Words that
 * already describe someone come back as they are (trimmed, no full stop at the end). */
export function personWords(text: string): string {
  let w = text.trim().replace(/\s+/g, " ");
  for (const re of LEAD_INS) {
    const m = w.match(re);
    if (m && m[0].length < w.length) {
      w = w.slice(m[0].length);
      break;
    }
  }
  return w.replace(/[.\s]+$/, "").replace(/^please\s+/i, "");
}

/** LTX-2.5's prompt for a Recast, as tasks/recast_ltx.py builds it (keep the two in step). What they do ("walks
 * towards the camera") is read from the shot when it runs, so the receipt shows where it goes. */
export function ltxRecastPrompt(who: string, scene = ""): string {
  const w = who.trim().replace(/\.+$/, "") || "the new character";
  return [
    `${w[0].toUpperCase()}${w.slice(1)} [does what the person in your shot does].`,
    scene.trim(),
    "They are lit by the scene's own natural light, with soft, realistic shadows and no glow around them; the camera and the place stay exactly as in the original shot.",
  ]
    .filter(Boolean)
    .join(" ");
}

/** SCAIL-2's prompt for a Recast, as tasks/recast.py builds it when the take has no kept prompt (keep the two in step):
 * the person, then the take's scene words, then the framing line. */
export function scailPrompt(who: string, scene = ""): string {
  const w = who.trim().replace(/\.+$/, "");
  return [
    w ? `${w[0].toUpperCase()}${w.slice(1)}, moves through the scene with the original performer's exact pose, timing, movement and position in the frame.` : "",
    scene.trim(),
    "The same place, light, camera angle and shot scale as the original shot; a normal-sized head in proportion to the body.",
  ]
    .filter(Boolean)
    .join(" ");
}
