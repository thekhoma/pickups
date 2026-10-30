// The demo autopilot (?tour=1, docs/demo-autopilot.md) runs in mock mode and drives the real UI.

/** True while the page was opened with ?tour=1. */
export const TOUR = typeof window !== "undefined" && /[?&]tour(=|&|$)/.test(window.location.search);
