//! `zer-bench`, unified benchmark harness for zer and competitor libraries.
//!
//! # Subcommands
//!
//! | Subcommand   | Purpose                                                                |
//! |--------------|------------------------------------------------------------------------|
//! | `throughput` | Measure raw compare/EM/score throughput; optionally benchmark libs too |
//! | `accuracy`   | Run zer against a labeled dataset; optionally benchmark libs too       |
//! | `compare`    | Read multiple CSV summaries; print side-by-side table                  |
//! | `plot`       | Delegate to benchmarks/utils/plot_results.py                           |
//!
//! # Examples
//!
//! ```bash
//! # Throughput (BRP-style, CUDA)
//! cargo run -p zer-bench --features=cuda -- \
//!     throughput --scenario brp/dedupe --target cuda
//!
//! # Throughput + Splink comparison in one command
//! cargo run -p zer-bench -- \
//!     throughput --scenario brp/dedupe --compare-libs splink --out bench_results/
//!
//! # Accuracy, run a named scenario (datasets/GT wired up automatically)
//! cargo run -p zer-bench -- \
//!     accuracy --scenario brp/dedupe --out bench_results/
//!
//! # Accuracy + Splink comparison in one command
//! cargo run -p zer-bench -- \
//!     accuracy --scenario brp/dedupe --compare-libs splink --out bench_results/
//!
//! # Cross-library comparison table from existing CSV files
//! cargo run -p zer-bench -- \
//!     compare --results bench_results/ --mode dedupe --dataset brp_persons
//!
//! # Plot results
//! cargo run -p zer-bench -- \
//!     plot --input bench_results/ --output results.png
//! ```

use clap::{Parser, Subcommand};

mod cmd;

#[derive(Parser)]
#[command(name = "zer-bench", about = "zer benchmark harness")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure raw compare/EM/score throughput.
    /// Use --compare-libs to also run competitor libraries and print an inline table.
    Throughput(cmd::throughput::ThroughputArgs),

    /// Run zer against a labeled dataset and write a shared CSV summary.
    /// Use --compare-libs to also run competitor libraries and print an inline table.
    Accuracy(cmd::accuracy::AccuracyArgs),

    /// Read multiple summary CSVs and print a side-by-side comparison table.
    Compare(cmd::compare::CompareArgs),

    /// Generate plots from benchmark summary CSVs via plot_results.py.
    Plot(cmd::plot::PlotArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Throughput(args) => cmd::throughput::run(args),
        Command::Accuracy(args) => cmd::accuracy::run(args).await,
        Command::Compare(args) => cmd::compare::run(args),
        Command::Plot(args) => cmd::plot::run(args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
