import type { BufferGeometry, Object3D } from 'three';
export const PINE_TIERS: [number, number, number][];
export function isTreeId(id: unknown): boolean;
export function pineCanopy(h: number): BufferGeometry;
export function pineTrunk(h: number): BufferGeometry;
export function pineTemplate<T extends Object3D>(root: T): T;
