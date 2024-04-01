// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{mem, sync::{Arc, Mutex}, time::Instant};

use cudarc::{driver::{CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits}, nvrtc::Ptx};
use tauri::State;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl ValidAsZeroBits for Agent {}
unsafe impl DeviceRepr for Agent {}

unsafe impl DeviceRepr for HyperParameters {}

impl Default for Agent {
    fn default() -> Self {
        Self { functions: [FunctionCoefficients::default(); MAX_FUNCTIONS as usize], functions_len: Default::default(), fitness: Default::default() }
    }
}

impl ToString for Agent {
    fn to_string(&self) -> String {
        self.functions
            .iter()
            .take(self.functions_len as usize)
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

impl ToString for FunctionCoefficients {
    fn to_string(&self) -> String {
        format!("{}*{}(x + {})", self.scale, match self.function_type as WaveFunction {
            WaveFunction::Sine => "sin",
            WaveFunction::SawTooth => "frac",
            WaveFunction::Count => unreachable!(),
        }, self.x_translation)
    }
}


impl Default for FunctionCoefficients {
    fn default() -> Self {
        Self { function_type: WaveFunction::Sine, scale: Default::default(), x_translation: Default::default() }
    }
}

const POPULATION_SIZE: usize = 1 << 26;
const WAVE_RES: usize = 512;

const GPU_ALLOCATED: usize =
    POPULATION_SIZE * (mem::size_of::<Agent>() + mem::size_of::<CurandState>()) +
    WAVE_RES * (mem::size_of::<f32>() * 2);

const TRAINER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/trainer.ptx"));

#[allow(dead_code)]
struct CurandState {
    d: u32,
    v: [u32; 5],
    boxmuller_flag: i32,
    boxmuller_flag_double: i32,
    boxmuller_extra: f32,
    boxmuller_extra_double: f64,
}

unsafe impl ValidAsZeroBits for CurandState {}
unsafe impl DeviceRepr for CurandState {}

struct CudaState {
    cuda_device: Arc<CudaDevice>,
    agents: CudaSlice<Agent>,
    curand_states: CudaSlice<CurandState>,
    goal: CudaSlice<f32>,
    output_buffer: CudaSlice<f32>,
    params: HyperParameters,
    init_kernel: CudaFunction,
    step_kernel: CudaFunction,
    step_sort_kernel: CudaFunction,
    output_kernel: CudaFunction,
    best: Option<Agent>,
}

impl CudaState {
    fn new() -> Self {
        let params = HyperParameters {
            starting_functions: 1,
            selection_fraction: 0.2,
            mutation_probability: 0.1,
            mutation_strength: 0.2,
            function_addition_probability: 0.1,
            function_subtraction_probability: 0.1,
        };

        let cuda_device = CudaDevice::new(0).unwrap();

        // TODO: Precompile it and use `from_file` instead
        cuda_device.load_ptx(Ptx::from_src(TRAINER_PTX), "trainer", &["init_kernel", "step_kernel", "step_sort_kernel", "output_kernel"]).unwrap();

        println!("Allocating {}Gb on GPU...", (GPU_ALLOCATED as f64 / 10_000_000.).round() / 100.);

        let agents = cuda_device.alloc_zeros::<Agent>(POPULATION_SIZE).unwrap();
        let curand_states = cuda_device.alloc_zeros::<CurandState>(POPULATION_SIZE).unwrap();
        let goal = cuda_device.alloc_zeros::<f32>(WAVE_RES).unwrap();
        let output_buffer = cuda_device.alloc_zeros::<f32>(WAVE_RES).unwrap();

        let init_kernel = cuda_device.get_func("trainer", "init_kernel").unwrap();
        let step_kernel = cuda_device.get_func("trainer", "step_kernel").unwrap();
        let step_sort_kernel = cuda_device.get_func("trainer", "step_sort_kernel").unwrap();
        let output_kernel = cuda_device.get_func("trainer", "output_kernel").unwrap();
        
        Self {
            cuda_device,
            agents,
            curand_states,
            goal,
            output_buffer,
            params,
            init_kernel,
            step_kernel,
            step_sort_kernel,
            output_kernel,
            best: None,
        }
    }

    pub fn init(&mut self, seed: u64, goal: &[f32]) {
        self.cuda_device.htod_sync_copy_into(goal, &mut self.goal).unwrap();
        unsafe {
            self.init_kernel.clone().launch(
                LaunchConfig::for_num_elems(POPULATION_SIZE as u32),
                (
                    &self.curand_states,
                    &self.agents,
                    self.params,
                    &self.goal,
                    WAVE_RES as i32,
                    seed,
                    POPULATION_SIZE as i32
                )).unwrap();
        }
        self.cuda_device.synchronize().unwrap();

        self.best = None;
    }

    pub fn step(&mut self, quantity: usize) {
        let top = (POPULATION_SIZE as f32 * self.params.selection_fraction) as u32;
        let bottom = POPULATION_SIZE as u32 - top;

        for _ in 0..quantity {
            unsafe {
                self.step_kernel.clone().launch(
                    LaunchConfig::for_num_elems(bottom),
                    (
                        &self.curand_states,
                        &self.agents,
                        self.params,
                        &self.goal,
                        WAVE_RES as i32,
                        POPULATION_SIZE as i32,
                        bottom as i32,
                    )).unwrap();
            }
            
            self.sort_agents();
        }

        self.cuda_device.synchronize().unwrap();

        // Invalidate the best agent so it will be re-calculated
        self.best = None;
    }

    fn sort_agents(&mut self) {
        let mut k: i32 = 2;
        while k <= POPULATION_SIZE as i32 {
            let mut j: i32 = k >> 1;
            while j > 0 {
                unsafe {
                    self.step_sort_kernel.clone().launch(
                        LaunchConfig::for_num_elems(POPULATION_SIZE as u32),
                        (
                            &self.agents,
                            j,
                            k,
                            POPULATION_SIZE as i32,
                        )
                    ).unwrap();
                }

                j >>= 1;
            }

            k <<= 1;
        }
    }

    pub fn best(&mut self) -> Agent {
        if let Some(best) = self.best {
            return best;
        }

        let best = self.cuda_device.dtoh_sync_copy(
            &self.agents.slice_mut(0..1),
        ).unwrap();

        self.best = Some(best[0]);
        best[0]
    }

    pub fn output(&mut self, agent_index: i32) -> Vec<f32> {
        unsafe {
            self.output_kernel.clone().launch(
                LaunchConfig::for_num_elems(WAVE_RES as u32),
                (
                    &self.agents,
                    &self.output_buffer,
                    WAVE_RES as i32,
                    agent_index,
                    WAVE_RES as i32,
                )
            ).unwrap();
        }

        self.cuda_device.dtoh_sync_copy(&self.output_buffer).unwrap()
    }
}

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn init_training(params: HyperParameters, goal: Vec<f32>, seed: u64, cuda: State<Mutex<CudaState>>) {
    let mut cuda_lock = cuda.lock().unwrap();
    cuda_lock.params = params;
    cuda_lock.init(seed, goal.as_slice());
    println!("Initialised training");
    println!("{params:?}");
}

#[tauri::command]
fn step_training(quantity: usize, cuda: State<Mutex<CudaState>>) {
    let start_time = Instant::now();
    let mut cuda_lock = cuda.lock().unwrap();
    cuda_lock.step(quantity);
    println!("{quantity} steps took {}ms", start_time.elapsed().as_millis());
}

#[tauri::command]
fn best_fitness(cuda: State<Mutex<CudaState>>) -> f32 {
    let mut cuda_lock = cuda.lock().unwrap();
    let r = cuda_lock.best();
    r.fitness
}

#[tauri::command]
fn best_formula(cuda: State<Mutex<CudaState>>) -> String {
    let mut cuda_lock = cuda.lock().unwrap();
    cuda_lock.best().to_string()
}

#[tauri::command]
fn output(index: i32, cuda: State<Mutex<CudaState>>) -> Vec<f32> {
    let mut cuda_lock = cuda.lock().unwrap();
    cuda_lock.output(index)
}

fn main() {
    tauri::Builder::default()
        .manage(Mutex::new(CudaState::new()))
        .invoke_handler(tauri::generate_handler![
            init_training,
            step_training,
            best_fitness,
            best_formula,
            output,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
