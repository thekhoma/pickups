// Facing at frame 0 from a figure's rotation, for the start still's words (start_frame.py FACING_PERSON and FACING_VEHICLE).

/**
 * Which way a figure or vehicle faces, from its rotation (the scene's figures look along their local +z, builder.js):
 * toward the camera or away when it points within about 53 degrees of the line to the lens, otherwise the side of the
 * frame it points to.
 */
export function facingOf(forward: { x: number; z: number }, toCamera: { x: number; z: number }, camRight: { x: number; z: number }): "camera" | "away" | "left" | "right" {
  const n = (v: { x: number; z: number }) => Math.hypot(v.x, v.z) || 1;
  const cos = (forward.x * toCamera.x + forward.z * toCamera.z) / (n(forward) * n(toCamera));
  if (cos >= 0.6) return "camera";
  if (cos <= -0.6) return "away";
  return forward.x * camRight.x + forward.z * camRight.z >= 0 ? "right" : "left";
}
