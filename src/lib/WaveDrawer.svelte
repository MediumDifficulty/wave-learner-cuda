<script lang="ts">
    import { onMount } from "svelte";
    import { waveData } from "./stores";
    import { Vector, lerp } from "./maths";
    import { WAVE_RES } from "./consts";

    export let bestLine: Float32Array

    let canvas: HTMLCanvasElement
    const canvasRes = new Vector(0, 0)

    let mouseIsDown = false
    let prevMousePos: {x: number, y: number} | null = null
    let ctx: CanvasRenderingContext2D

    $: if (ctx && bestLine) redrawCanvas(ctx)

    onMount(() => {
        ctx = canvas.getContext('2d')!

        const obs = new ResizeObserver(e => {
            const rect = e[0].contentRect
            canvas.width = rect.width
            canvas.height = rect.height

            canvasRes.x = canvas.clientWidth
            canvasRes.y = canvas.clientHeight

            redrawCanvas(ctx)
        })
        
        obs.observe(canvas)
    })

    function mouseMove(e: MouseEvent) {
        if (mouseIsDown) {
            if (prevMousePos) {
                drawLine(new Vector(prevMousePos.x, prevMousePos.y), new Vector(e.offsetX, e.offsetY))
            }
            
            prevMousePos = {
                x: e.offsetX,
                y: e.offsetY
            }

            redrawCanvas(ctx)
        }
    }

    function drawLine(from: Vector, to: Vector) {
        const fromDat = screenToData(from)
        const toDat = screenToData(to)

        const diffX = Math.abs(fromDat.x - toDat.x)
        const diffY = Math.abs(fromDat.y - toDat.y)

        const minX = Math.min(fromDat.x, toDat.x)
        const maxX = Math.max(fromDat.x, toDat.x)
        const minY = Math.min(fromDat.y, toDat.y)
        const maxY = Math.max(fromDat.y, toDat.y)

        waveData.update(data => {
            let j = 0;
            for (let i = minX; i <= maxX; i++) {
                let ratioX;
                if (fromDat.x < toDat.x) {
                    ratioX = (i - minX) / diffX;
                } else {
                    ratioX = (maxX - i) / diffX;
                }
                const interpolatedY = minY + ratioX * diffY * Math.sign(toDat.y - fromDat.y);
                data[i] = interpolatedY;
            }
            return data;
        })
    }

    function redrawCanvas(ctx: CanvasRenderingContext2D) {
        const startTime = performance.now()

        ctx.fillStyle = "black"
        ctx.fillRect(0, 0, canvasRes.x, canvasRes.y)

        drawGraph($waveData, "white")
        drawGraph(bestLine, "green")

        console.debug(`redrew in ${Math.round((performance.now() - startTime) * 1000)}μs`)
    }

    function drawGraph(data: Float32Array, style: string) {
        ctx.beginPath()
        const beginPos = dataToScreen(0, data[0])
        ctx.moveTo(beginPos.x, beginPos.y)
        data.forEach((n, i) => {
            if (i > 1) {
                const pos = dataToScreen(i, n)
                ctx.lineTo(pos.x, pos.y)
            }
        })

        ctx.strokeStyle = style
        ctx.stroke()
    }

    function dataToScreen(index: number, value: number): Vector {
        return new Vector(
            (index / WAVE_RES) * canvasRes.x,
            canvasRes.y - ((value + 1) / 2) * canvasRes.y
        )
    }

    function screenToData(pos: Vector): Vector {
        return new Vector(
            Math.floor((pos.x / canvasRes.x) * WAVE_RES),
            (((canvasRes.y -  pos.y) / canvasRes.y)) * 2 - 1
        )
    }
</script>

<canvas
    bind:this={canvas}
    on:mousedown={() => mouseIsDown = true}
    on:mouseup={() => {mouseIsDown = false; prevMousePos = null}}
    on:mouseleave={() => {mouseIsDown = false; prevMousePos = null}}
    on:mousemove={mouseMove}
>This browser doesn't support the canvas element!</canvas>

<style>
    canvas {
        width: 100%;
        height: 100%;
    }
</style>