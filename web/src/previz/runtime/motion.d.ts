import type { Object3D, Vector3 } from 'three';

export function bindMotion(person: Object3D, kind: 'argue' | 'sway' | 'type' | 'chat' | 'dance' | 'gaze', index?: number): Object3D;
export function applyMotion(person: Object3D, t: number, duration?: number): void;
export function applyDrive(obj: Object3D, t: number, duration?: number): { along: number; side: number; dx: number; dz: number } | null;
export function driveSide(drive: { lanes?: [number, number, number][] }, t: number): number;
export function bindGaze(person: Object3D, beats: { type: string; from_t: number; to_t: number }[], outPoint?: Vector3 | null): boolean;
export function aimGaze(built: { actors?: { object?: Object3D }[] }, camPos: number[] | Vector3): number;
