import type * as THREE from 'three';
export function bodyTypeOf(word: string | null | undefined, asset: string): 'fastback' | null;
export function shapeBody(car: THREE.Object3D, type: string | null): THREE.BufferGeometry[];
export function roofProfile(car: THREE.Object3D, fractions?: number[]): number[];
