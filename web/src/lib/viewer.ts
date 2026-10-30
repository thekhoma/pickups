// The full-screen character viewer's position and arrows (components/CharacterViewer.tsx), as plain functions.

export interface ViewerItem {
  key: string;
  label: string;
  /** null: still being made (a spinner), or never made when `failed`. */
  url: string | null;
  /** The sheet failed before this view was made: a quiet "Failed" placeholder, never a spinner. */
  failed?: boolean;
}

/** "Front · 2 of 4" (just the label when there's one): the dialog's name for screen readers; the tabs show it on screen. */
export function viewerPosition(items: ViewerItem[], index: number): string {
  const it = items[index];
  if (!it) return "";
  return items.length > 1 ? `${it.label} · ${index + 1} of ${items.length}` : it.label;
}

/** Left and Right go round: from the last back to the first. */
export function stepIndex(index: number, by: number, count: number): number {
  return count ? (((index + by) % count) + count) % count : 0;
}
