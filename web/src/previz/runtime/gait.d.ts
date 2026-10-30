import type { Object3D } from 'three';

export function applyGait(obj: Object3D, t: number, speedScale: number, duration: number): boolean;
export function bindGait(obj: Object3D, actor: { pose?: string; kind?: string; id?: string }, index: number, spec: { seed?: number; camera?: { move?: string | null } }, kit: unknown): void;
export function gaitSkipped(): string[];
