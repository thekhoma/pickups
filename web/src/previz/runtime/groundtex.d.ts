import type * as THREE from 'three';
export const GROUND_NOISE_SIZE: number;
export const GROUND_SCALES: [number, number][];
export const GROUND_MATERIALS: string[];
export function groundNoise(size?: number, seed?: number): Float32Array;
export function groundTexture(): THREE.DataTexture;
export function texturedGround<M extends THREE.Material>(mat: M): M;
