//! Show how `CalibrationTable` adjusts Fellegi-Sunter match probabilities.
//!
//! Prints a table of input probabilities and the resulting posteriors after
//! applying each of the three judge decisions.
//!
//! Run with:
//!   cargo run -p zer-judge --example calibration_table

use zer_judge::CalibrationTable;

fn main() {
    let table = CalibrationTable::default();

    println!("CalibrationTable (defaults):");
    println!("  lr_increase  = {:.4}  → applied when judge says 'match'", table.lr_increase);
    println!("  lr_decrease  = {:.4}  → applied when judge says 'non-match'", table.lr_decrease);
    println!("  lr_no_change = {:.4}  → applied when judge abstains", table.lr_no_change);
    println!();

    println!(
        "{:<8}  {:>12}  {:>12}  {:>12}",
        "p_in", "→ increase", "→ no_change", "→ decrease",
    );
    println!("{}", "─".repeat(50));

    for &p in &[0.10_f32, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95] {
        println!(
            "{:<8.2}  {:>12.4}  {:>12.4}  {:>12.4}",
            p,
            table.update_increase(p),
            table.update_no_change(p),
            table.update_decrease(p),
        );
    }

    println!();
    println!("JSON: {}", serde_json::to_string(&table).unwrap());
}
