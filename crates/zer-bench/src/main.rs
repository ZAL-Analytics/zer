//! `zer-bench`, unified benchmark harness for zer and competitor libraries.
//!
//! # Subcommands
//!
//! | Subcommand    | Purpose                                                        |
//! |---------------|----------------------------------------------------------------|
//! | `throughput`  | Measure raw compare/EM/score throughput (replaces old crates)  |
//! | `accuracy`    | Run zer against a labeled dataset; write shared CSV summary    |
//! | `library`     | Run a competitor library script and collect its CSV            |
//! | `library-all` | Run all configured libraries for a given mode and dataset      |
//! | `compare`     | Read multiple CSV summaries; print side-by-side table          |
//!
//! # Examples
//!
//! ```bash
//! # Throughput (BRP-style, CUDA)
//! cargo run --release -p zer-bench --features=cuda -- \
//!     throughput --preset brp --target cuda
//!
//! # Accuracy, run a named preset (datasets/GT wired up automatically)
//! cargo run --release -p zer-bench -- \
//!     accuracy --preset brp-dedupe-small --out bench_results/
//!
//! # Accuracy, manual dataset override
//! cargo run --release -p zer-bench -- \
//!     accuracy --dataset data/benchmarks/brp_small/brp_persons.csv --source brp \
//!              --mode deduplicate \
//!              --ground-truth data/benchmarks/brp_small/ground_truth_pairs.csv \
//!              --out bench_results/
//!
//! # Library benchmark, uses data/benchmarks/brp_small/ by default (no --dataset needed)
//! cargo run --release -p zer-bench -- \
//!     library --library splink --mode dedupe --out bench_results/
//!
//! # Library benchmark, explicit dataset override
//! cargo run --release -p zer-bench -- \
//!     library --library splink --mode dedupe \
//!             --dataset /path/to/custom.csv --out bench_results/
//!
//! # Cross-library comparison table
//! cargo run --release -p zer-bench -- \
//!     compare --results bench_results/ --mode dedupe --dataset brp_persons
//! ```

use clap::{Parser, Subcommand};

mod cmd;
mod nvtx_layer;

#[derive(Parser)]
#[command(name = "zer-bench", about = "zer benchmark harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure raw compare/EM/score throughput.
    Throughput(cmd::throughput::ThroughputArgs),

    /// Run zer against a labeled dataset and write a shared CSV summary.
    Accuracy(cmd::accuracy::AccuracyArgs),

    /// Run a competitor library benchmark script and collect its summary CSV.
    Library(cmd::library::LibraryArgs),

    /// Run all configured competitor libraries for a given mode and dataset.
    #[command(name = "library-all")]
    LibraryAll(cmd::library::LibraryArgs),

    /// Read multiple summary CSVs and print a side-by-side comparison table.
    Compare(cmd::compare::CompareArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Throughput(args) => cmd::throughput::run(args),
        Command::Accuracy(args)   => cmd::accuracy::run(args).await,
        Command::Library(args)    => cmd::library::run(args),
        Command::LibraryAll(args) => cmd::library::run_all(args),
        Command::Compare(args)    => cmd::compare::run(args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
