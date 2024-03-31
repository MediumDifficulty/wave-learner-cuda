use std::{env, fs, path::PathBuf, process::Command};

use regex::Regex;

fn main() {
    tauri_build::build();

    println!("cargo:rerun-if-changed=cuda");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_src = PathBuf::from("src/cuda/trainer.cu");
    let ptx_file = out_dir.join("trainer.ptx");

    // Maybe use `cc` crate?
    let nvcc_status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg(&ptx_file)
        .arg(&cuda_src)
        .status()
        .unwrap();

    assert!(nvcc_status.success(), "CUDA failed to compile.");

    let bindings = bindgen::Builder::default()
        .header("src/cuda/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .no_copy("*")
        .no_debug("*")
        .rustified_enum("WaveFunction")
        .generate()
        .expect("Unable to generate bindings");

    let generated_bindings = bindings.to_string();

    let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");
    let modified_bindings = format!(
        "use serde::Deserialize;\n{}",
        Regex::new(r"#\s*\[\s*derive\s*\((?P<d>[^)]+)\)\s*\]\s*pub\s*struct HyperParameters").unwrap()
            .replace_all(&modified_bindings, "#[derive($d, Deserialize)]\npub struct HyperParameters")
    );

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_path.join("bindings.rs"), modified_bindings.as_bytes())
        .expect("Couldn't write bindings!");
}
