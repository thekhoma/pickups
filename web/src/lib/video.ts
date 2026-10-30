// New video on LTX-2.5 (Create: Text to video, Image to video): the sizes and lengths the pickers offer.
// The same table as src/pickups/video.py (tests/test_video.py keeps the two in step). Width and height are multiples of
// 64 (pass 1 runs at half size); the length is 8n+1 frames at 24 fps, at most 97. Only sizes that ran on a 24 GB card
// are offered: 720p at every aspect, 1080p at 16:9.

export type VideoMode = "text" | "image";
export type VideoAspect = "16:9" | "9:16" | "1:1" | "4:3" | "21:9";
export type VideoResolution = "720p" | "1080p";

export const VIDEO_FPS = 24;
export const VIDEO_MAX_FRAMES = 97;
export const VIDEO_ASPECTS: VideoAspect[] = ["16:9", "9:16", "1:1", "4:3", "21:9"];
export const VIDEO_RESOLUTIONS: VideoResolution[] = ["720p", "1080p"];

export const VIDEO_SIZES: Record<VideoResolution, Partial<Record<VideoAspect, [number, number]>>> = {
  "720p": { "16:9": [1280, 704], "9:16": [704, 1280], "1:1": [960, 960], "4:3": [1024, 768], "21:9": [1344, 576] },
  "1080p": { "16:9": [1920, 1088] },
};

export const VIDEO_DURATIONS: Array<{ seconds: number; frames: number }> = [
  { seconds: 1, frames: 25 },
  { seconds: 2, frames: 49 },
  { seconds: 3, frames: 73 },
  { seconds: 4, frames: 97 },
];

export const VIDEO_DEFAULTS = { aspect: "16:9" as VideoAspect, resolution: "1080p" as VideoResolution, duration: 4 };

/** Does this resolution come in this aspect (1080p is 16:9 only)? */
export function hasSize(aspect: VideoAspect, resolution: VideoResolution): boolean {
  return !!VIDEO_SIZES[resolution][aspect];
}

/** The resolution actually used: the one asked for when it comes in this aspect, else 720p (every aspect has it). */
export function resolutionFor(aspect: VideoAspect, resolution: VideoResolution): VideoResolution {
  return hasSize(aspect, resolution) ? resolution : "720p";
}

/** Final width and height for an aspect and resolution (falling back to 720p). */
export function videoSize(aspect: VideoAspect, resolution: VideoResolution): [number, number] {
  return VIDEO_SIZES[resolutionFor(aspect, resolution)][aspect] ?? [1280, 704];
}

/** Frames for a length in seconds (8n+1 at 24 fps), never over 97. */
export function framesFor(seconds: number): number {
  const hit = VIDEO_DURATIONS.find((d) => d.seconds === seconds);
  if (hit) return hit.frames;
  const n = Math.round(Math.max(0, seconds) * VIDEO_FPS / 8) * 8 + 1;
  return Math.max(9, Math.min(VIDEO_MAX_FRAMES, n));
}

/** The middle of a picture that fills the frame (ImageScale crop "center"): the visible box as fractions of the picture.
 * A wide picture at 9:16 keeps a vertical strip; a 16:9 one at 21:9 loses some top and bottom. */
export function coverCrop(imageW: number, imageH: number, aspect: VideoAspect | [number, number]): { x: number; y: number; w: number; h: number } {
  const [fw, fh] = Array.isArray(aspect) ? aspect : (aspect.split(":").map(Number) as [number, number]);
  if (!(imageW > 0 && imageH > 0 && fw > 0 && fh > 0)) return { x: 0, y: 0, w: 1, h: 1 };
  const frame = fw / fh;
  const pic = imageW / imageH;
  if (pic > frame) {
    const w = frame / pic;
    return { x: (1 - w) / 2, y: 0, w, h: 1 };
  }
  const h = pic / frame;
  return { x: 0, y: (1 - h) / 2, w: 1, h };
}
