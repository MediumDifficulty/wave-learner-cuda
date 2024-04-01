<script lang="ts">
    import { invoke } from "@tauri-apps/api";
    import WaveDrawer from "./lib/WaveDrawer.svelte";
    import { waveData } from "./lib/stores";
    import { type HyperParameters } from "../src-tauri/bindings/HyperParameters"

    invoke('wave_res')
        .then(v => {
            const value = v as number
            $waveData = new Float32Array(value)
        })

    $: bestLine = new Float32Array($waveData.length)
    let bestFormula = ""
    let bestFitness = 0

    let stepCount = 1

    const params: HyperParameters = {
        starting_functions: 1,
        selection_fraction: 0.2,
        mutation_probability: 0.1,
        mutation_strength: 0.2,
        function_addition_probability: 0.1,
        function_subtraction_probability: 0.1,
    }

    async function step() {
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
        {#if $waveData.length > 0}
            <WaveDrawer bestLine={bestLine} />
        {/if}
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