<script lang="ts">
    import { invoke } from "@tauri-apps/api";
    import WaveDrawer from "./lib/WaveDrawer.svelte";
    import { WAVE_RES } from "./lib/consts";
    import { waveData } from "./lib/stores";
    // import type { HyperParameters } from "$lib/wasm-helpers";
    // import * as wasm from "$lib/wasm/trainer"
    import { onMount } from "svelte";

    let bestLine = new Float32Array(WAVE_RES)
    let bestFormula = ""
    let bestFitness = 0

    let stepCount = 1

    let training = false

    const params = {
        starting_functions: 1,
        selection_fraction: 0.2,
        mutation_probability: 0.1,
        mutation_strength: 0.2,
        function_addition_probability: 0.1,
        function_subtraction_probability: 0.1,
    }

    onMount(() => {
        // wasm.init()
        console.log($waveData)
        // wasm.greet()

        // setInterval(() => {
        //     if (training) {
        //         // wasm.step_training()
        //         // bestLine = wasm.get_best_output()
        //         // bestFormula = wasm.get_best_formula()
        //     }
        // }, 1)
    })

    async function step() {
        // for (let i = 0; i < stepCount; i++)
        //     wasm.step_training()

        // bestLine = wasm.get_best_output()
        // bestFormula = wasm.get_best_formula()

        // training = !training
        await invoke('step_training', { quantity: stepCount })
        bestFormula = await invoke('best_formula')

        bestFitness = await invoke('best_fitness')
        bestLine = await invoke('output', { index: 0 })
    }

    async function initTraining() {
        await invoke('init_training', { params, goal: Array.from($waveData), seed: 1234 })
        bestFitness = await invoke('best_fitness')
    }
</script>

<main>    
    <div class="drawer-container">
        <WaveDrawer bestLine={bestLine} />
        <div>{bestFormula}</div>
    </div>
    <button on:click={initTraining}>Start Training</button>
    <input type="number" bind:value={stepCount} />
    <button on:click={step}>Step Training</button>
    <div>{bestFitness}</div>
</main>

<style>
    :global(body) {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }

    .drawer-container {
        width: 70%;
        margin-left: auto;
        margin-right: auto;
        margin-top: 5rem;
        aspect-ratio: 4/3;
    }

    main {
        margin-left: auto;
        margin-right: auto;
        width: 75%;
    }
</style>