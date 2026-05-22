//! Build script for zer-compute.
//!
//! When the `cuda` feature is enabled: compiles CUDA kernels (.cu → .ptx) via nvcc.
//! When the `vulkan` feature is enabled: compiles Slang shaders (.slang → .spv) via slangc.
//! Output is embedded into OUT_DIR so the Rust code can include it via `include_bytes!`.
//!
//! CUDA notes:
//!   - Requires CUDA Toolkit 13.1 or later (enforced below).
//!   - Targets SM 8.6 (Ampere) as the minimum compute capability.
//!   - Release: `-O3 --use_fast_math --restrict` for maximum throughput.
//!   - Debug (`debug-shaders` feature): `-g -G -O0` for cuda-gdb.
//!
//! Vulkan / Slang notes:
//!   - Requires `slangc` on PATH (https://github.com/shader-slang/slang/releases).
//!   - Compiles to SPIR-V 1.5 targeting Vulkan 1.3 compute.
//!   - Release: `-O3 -matrix-layout-column-major`.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda_enabled          = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    let vulkan_enabled        = std::env::var("CARGO_FEATURE_VULKAN").is_ok();
    let debug_shaders_enabled = std::env::var("CARGO_FEATURE_DEBUG_SHADERS").is_ok();

    if cuda_enabled {
        compile_cuda_kernels(debug_shaders_enabled);
    }

    if vulkan_enabled {
        compile_slang_shaders();
    }
}

// ── CUDA version gate ─────────────────────────────────────────────────────────

fn check_cuda_version() {
    let output = match Command::new("nvcc").arg("--version").output() {
        Ok(o)  => o,
        Err(e) => panic!(
            "nvcc not found ({e}). Install the CUDA toolkit to use the `cuda` feature."
        ),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);

    let (major, minor) = stdout
        .lines()
        .find_map(|line| {
            let idx  = line.find("release ")?;
            let rest = &line[idx + 8..];
            let end  = rest.find(',')?;
            let ver  = &rest[..end];
            let mut it = ver.splitn(2, '.');
            let maj: u32 = it.next()?.parse().ok()?;
            let min: u32 = it.next()?.parse().ok()?;
            Some((maj, min))
        })
        .unwrap_or_else(|| panic!("Could not parse CUDA version from nvcc --version:\n{stdout}"));

    const REQ_MAJOR: u32 = 13;
    const REQ_MINOR: u32 = 1;

    if (major, minor) < (REQ_MAJOR, REQ_MINOR) {
        panic!(
            "CUDA Toolkit {major}.{minor} is below the required {REQ_MAJOR}.{REQ_MINOR}.\n\
             Update to CUDA Toolkit 13.1 or later: https://developer.nvidia.com/cuda-downloads"
        );
    }
}

// ── CUDA ─────────────────────────────────────────────────────────────────────

fn compile_cuda_kernels(debug: bool) {
    check_cuda_version();

    let out_dir    = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernel_dir = PathBuf::from("src/backend/cuda/kernels");
    let kernels    = ["em_reduce", "hello_backend"];
    let n          = kernels.len();

    let opt_flags: &[&str] = if debug {
        &["-g", "-G", "-O0"]
    } else {
        &["-O3", "--use_fast_math", "--restrict"]
    };

    if debug {
        println!("   [debug-shaders] CUDA kernels compiled with -g -G -O0");
    }

    for (i, name) in kernels.iter().enumerate() {
        let cu_path  = kernel_dir.join(format!("{name}.cu"));
        let ptx_path = out_dir.join(format!("{name}.ptx"));

        println!("cargo:rerun-if-changed={}", cu_path.display());

        println!("   Compiling CUDA [{}/{n}] {name}.cu", i + 1);

        let mut cmd = Command::new("nvcc");
        cmd.args(["-ptx", "-arch=sm_86"]);
        cmd.args(opt_flags);
        cmd.args([
            "-I", kernel_dir.to_str().unwrap(),
            "-o", ptx_path.to_str().unwrap(),
            cu_path.to_str().unwrap(),
        ]);

        let output = cmd.output();

        match output {
            Ok(o) if o.status.success() => {}
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                panic!("nvcc exited with status {} while compiling {name}.cu\n{stderr}", o.status);
            }
            Err(e) => {
                panic!(
                    "nvcc not found ({e}). Install the CUDA toolkit to use the `cuda` feature."
                );
            }
        }
    }
}

// ── Vulkan / Slang ────────────────────────────────────────────────────────────

fn compile_slang_shaders() {
    // Verify slangc is on PATH with a helpful error.
    let version_check = Command::new("slangc").arg("-v").output();
    if version_check.is_err() || !version_check.unwrap().status.success() {
        panic!(
            "slangc not found on PATH. Install the Slang shader compiler to use the `vulkan` feature.\n\
             Download from: https://github.com/shader-slang/slang/releases\n\
             Then add the bin/ directory to PATH."
        );
    }

    let out_dir    = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from("src/backend/vulkan/shaders");
    // Each tuple: (slang source stem, entry point name, output spv stem).
    // em_reduce has three entry points compiled to three separate SPIR-V modules.
    let shaders: &[(&str, &str, &str)] = &[
        ("hello_backend",   "hello_backend_main", "hello_backend"),
        ("em_reduce",       "em_estep",            "em_estep"),
        ("em_reduce",       "em_mstep_partial",    "em_mstep_partial"),
        ("em_reduce",       "em_mstep_final",      "em_mstep_final"),
    ];

    let n = shaders.len();
    for (i, (src_stem, entry, out_stem)) in shaders.iter().enumerate() {
        let slang_path = shader_dir.join(format!("{src_stem}.slang"));
        let spv_path   = out_dir.join(format!("{out_stem}.spv"));

        println!("cargo:rerun-if-changed={}", slang_path.display());

        println!("   Compiling Slang [{}/{n}] {src_stem}.slang [{entry}] → {out_stem}.spv", i + 1);

        let status = Command::new("slangc")
            .args([
                slang_path.to_str().unwrap(),
                "-target",  "spirv",
                "-profile", "spirv_1_5",
                "-entry",   entry,
                "-stage",   "compute",
                "-O3",
                "-matrix-layout-column-major",
                "-o", spv_path.to_str().unwrap(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => panic!(
                "slangc failed (exit {s}) compiling {src_stem}.slang entry={entry}"
            ),
            Err(e) => panic!("slangc not found ({e})"),
        }
    }
}
