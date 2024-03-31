import { writable } from "svelte/store";
import { WAVE_RES } from "./consts";

export const waveData = writable<Float32Array>(new Float32Array(WAVE_RES))