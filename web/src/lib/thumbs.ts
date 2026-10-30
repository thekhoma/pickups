// Thumbnail warm-up: take stacks and the takes strip show their posters at once instead of fading in from black.

/** Thumbnails the browser has decoded this session. */
export const thumbSeen = new Set<string>();

/** The clip behind a shot card's picture (latest_thumb_url, the newest take's): what plays when the card is hovered.
 * The backend's project cover_video_url is the same rule. */
export function latestVideo(shot: { video_url: string; thumb_url: string; latest_thumb_url?: string; versions: { video_url: string; thumb_url: string }[] }): string {
  const still = shot.latest_thumb_url || shot.thumb_url;
  const v = [...shot.versions].reverse().find((x) => x.thumb_url === still);
  return v?.video_url || shot.video_url;
}

export function preloadThumbs(urls: Array<string | null | undefined>): void {
  for (const u of urls) {
    if (!u || thumbSeen.has(u)) continue;
    const img = new Image();
    img.decoding = "async";
    img.onload = () => thumbSeen.add(u);
    img.src = u;
  }
}
