// Where a portaled list opens against its button (components/Picker.tsx and the start-picture chooser): below when it
// fits, else above when that fits, else the roomier side, capped to the room (it scrolls), and never past an edge.

const GAP = 8; // between the button and the list
const EDGE = 12; // the list never comes closer than this to the window's edge
const MIN_H = 120;

export interface MenuPlace {
  left: number;
  top?: number;
  bottom?: number;
  maxH: number;
  up: boolean;
}

export function placeMenu(
  button: { top: number; bottom: number; left: number; width: number },
  menu: { height: number; width: number },
  view: { width: number; height: number },
): MenuPlace {
  const below = view.height - button.bottom - GAP - EDGE;
  const above = button.top - GAP - EDGE;
  const up = !(menu.height <= below || (menu.height > above && below >= above));
  const maxH = Math.max(MIN_H, Math.min(menu.height, up ? above : below));
  const w = Math.max(menu.width, button.width);
  const left = Math.max(EDGE, Math.min(button.left, view.width - w - EDGE));
  return up ? { left, bottom: view.height - button.top + GAP, maxH, up } : { left, top: button.bottom + GAP, maxH, up };
}
