export function lerp(a: number, b: number, t: number) {
    return a + (b - a) * t
}

export class Vector {
    constructor(public x: number, public y: number) {}
}