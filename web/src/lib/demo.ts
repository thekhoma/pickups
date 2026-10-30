// Demo mode (?mock=1) is picked when the page loads (api.ts createApi), so going in and out reloads the page. Leaving
// it opens the first screen (Where does ComfyUI run?) even after the welcome was seen.
const LEFT = "pickups.leftDemo";

/** The first screen's "Try the demo": the whole app on example shots, no ComfyUI needed. */
export function enterDemo(): void {
  window.location.assign(`${window.location.pathname}?mock=1`);
}

/** The top bar's "Exit demo": back to the real app, on the first screen. */
export function leaveDemo(): void {
  try {
    window.sessionStorage.setItem(LEFT, "1");
  } catch {
    // no session storage: the real app still loads, on Home
  }
  window.location.assign(window.location.pathname);
}

/** True once, on the load right after Exit demo. */
export function leftDemo(): boolean {
  try {
    const left = window.sessionStorage.getItem(LEFT) === "1";
    window.sessionStorage.removeItem(LEFT);
    return left;
  } catch {
    return false;
  }
}
