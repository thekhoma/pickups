// The shot spec is the only source of truth for a previz. The page, the export and the server all read this shape.
import { z } from "zod";

export const TEMPLATE_IDS = ["giant_orbit", "street_run", "tunnel", "car_track", "product_orbit"] as const;
export type TemplateId = (typeof TEMPLATE_IDS)[number];

/** Templates, plus a shot compiled from a sentence. */
export const SPEC_TEMPLATE_IDS = [...TEMPLATE_IDS, "described"] as const;

export const Vec3 = z.tuple([z.number(), z.number(), z.number()]);
export type Vec3 = [number, number, number];

const Key = z
  .object({
    t: z.number().min(0).max(1),
    pos: Vec3,
    look: Vec3,
    focal_mm: z.number().min(8).max(200).optional(),
    roll: z.number().min(-180).max(180).optional(),
  })
  .strict();

export const PrevizSpecSchema = z
  .object({
    version: z.literal(1),
    template: z.enum(SPEC_TEMPLATE_IDS),
    duration_s: z.number().positive().max(60),
    fps: z.literal(24),
    size: z.tuple([z.literal(1920), z.literal(1088)]),
    lens: z
      .object({
        focal_mm: z.number().min(8).max(200),
        film_gauge_mm: z.literal(36),
      })
      .strict(),
    environment: z
      .object({
        ground: z.enum(["plane", "none"]),
        sky: z.literal("gradient"),
        fog: z.tuple([z.number(), z.number()]),
        props: z.array(z.string()).max(24),
      })
      .strict(),
    subjects: z
      .array(
        z
          .object({
            id: z.string().min(1).max(40),
            kind: z.enum(["humanoid", "car", "product", "dog", "cat", "bike", "skate", "boat"]),
            height_m: z.number().positive().max(2000),
            pos: Vec3,
            anim: z.string().max(40),
            speed: z.number().positive().max(4).optional(),
          })
          .strict(),
      )
      .max(6),
    camera: z
      .object({
        interp: z.literal("catmull_rom"),
        /** How round the path is between keys: 0 straight lines, 1 smooth (default), 1.5 rounder. */
        curve: z.number().min(0).max(1.5).optional(),
        ease: z.enum(["in_out", "linear"]),
        speed: z.number().min(0.25).max(3).optional(),
        taper: z.number().min(0).max(1).optional(),
        distance: z.number().min(0.35).max(4).optional(),
        frame: z.tuple([z.number().min(-1.5).max(1.5), z.number().min(-1.5).max(1.5)]).optional(),
        free: z.boolean().optional(),
        /** With free: the camera still rides with the subject (a bent tracking path keeps following). */
        follow: z.boolean().optional(),
        pick: z.enum(["close", "wide", "close-wide", "dolly-in", "dolly-out", "orbit", "drone", "fpv", "track", "parallel", "locked"]).optional(),
        keys: z.array(Key).min(2).max(24),
        segments: z.array(z.object({
          move: z.string().min(1).max(40),
          from_t: z.number().min(0).max(1),
          to_t: z.number().min(0).max(1),
          params: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).optional(),
          tracking: z.number().min(0).max(1),
        }).strict()).max(3).optional(),
        handheld: z.object({
          seed: z.number().int(),
          pos_m: z.number().min(0).max(0.2),
          rot_deg: z.number().min(0).max(5),
          hz: z.number().min(0).max(4),
        }).strict().nullable().optional(),
      })
      .strict(),
    beats: z.array(z.object({
      actor: z.string().max(40).nullable(),
      type: z.string().min(1).max(40),
      from_t: z.number().min(0).max(1),
      to_t: z.number().min(0).max(1),
      params: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).optional(),
    }).strict()).max(8).optional(),
    seed: z.number().int(),
    speed: z.number().positive().max(20).optional().default(1),
    notes: z.string().max(2000).optional().default(""),
  })
  .strict();

export type PrevizSpec = z.infer<typeof PrevizSpecSchema>;
export type CameraKey = PrevizSpec["camera"]["keys"][number];

export const PREVIZ_W = 1920;
export const PREVIZ_H = 1088;
export const PREVIZ_FPS = 24;
/** A new gray shot: 97 frames, so the render keeps the last beat. The slider can run longer than this. */
export const PREVIZ_DEFAULT_S = 97 / 24;
/** Longest gray shot the page will play. A later render still stops at the GPU's own frame cap. */
export const PREVIZ_MAX_S = 8;

/** Frames the page renders. The bias matches the server, so 97/24 does not round down to 96. */
export function shotFrames(durationS: number, fps: number, cap: number): number {
  const n = Math.max(1, Math.round(durationS * fps + 1e-6));
  return Math.min(Math.max(1, cap), n);
}

export const TEMPLATE_LABEL: Record<TemplateId, string> = {
  giant_orbit: "Giant orbit",
  street_run: "Street run",
  tunnel: "Tunnel",
  car_track: "Car tracking",
  product_orbit: "Product orbit",
};
