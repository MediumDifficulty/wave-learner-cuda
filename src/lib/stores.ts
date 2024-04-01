import { writable } from "svelte/store";

export const waveData = writable<Float32Array>(new Float32Array(0))