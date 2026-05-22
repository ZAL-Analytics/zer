//! Show the `AuditLog` open → append → inspect pattern.
//!
//! Writes 5 judge decisions to a temporary JSONL file, then reads them back
//! to show the recorded format.
//!
//! Run with:
//!   cargo run -p zer-judge --example audit_log

use std::{io::BufRead, path::PathBuf, sync::Arc};
use zer_judge::audit::{AuditEntry, AuditLog};

fn main() -> std::io::Result<()> {
    // ── 1. Open log (temp file or user-specified path) ────────────────────────
    let path: PathBuf = std::env::args()
        .find_map(|a| a.strip_prefix("--out=").map(PathBuf::from))
        .unwrap_or_else(|| {
            let mut p = std::env::temp_dir();
            p.push("zer_audit_example.jsonl");
            p
        });

    println!("Writing audit log to: {}", path.display());
    let log = Arc::new(AuditLog::open(&path)?);

    // ── 2. Simulate 5 judge decisions ─────────────────────────────────────────
    let decisions = [
        (1u64, 2u64, 0.55_f32, 0.78_f32, "increase"),
        (3,    4,    0.48,     0.31,      "decrease"),
        (5,    6,    0.52,     0.51,      "no_change"),
        (7,    8,    0.61,     0.82,      "increase"),
        (9,   10,    0.40,     0.19,      "decrease"),
    ];

    for (a, b, match_prob, entail, verdict) in decisions {
        let pair_text = format!(
            "[CLS] COL:naam VAL:person_{a} [SEP] COL:naam VAL:person_{b} [SEP]"
        );
        log.append(&AuditEntry {
            record_a:          a,
            record_b:          b,
            pair_text,
            match_probability: match_prob,
            entailment_score:  entail,
            verdict,
        });
    }

    println!("Wrote 5 entries.");
    println!();

    // ── 3. Read back and display ──────────────────────────────────────────────
    drop(log); // flush and close

    println!("Recorded JSONL entries:");
    println!("{}", "─".repeat(80));

    let file   = std::fs::File::open(&path)?;
    let reader = std::io::BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        // Pretty-print via serde_json
        let v: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
        println!(
            "  [{i}] record_a={:<4} record_b={:<4} prob={:<6.3} entail={:<6.3} verdict={}",
            v["record_a"], v["record_b"],
            v["match_probability"].as_f64().unwrap_or(0.0),
            v["entailment_score"].as_f64().unwrap_or(0.0),
            v["verdict"].as_str().unwrap_or("?"),
        );
    }

    println!("{}", "─".repeat(80));
    println!();
    println!("Each line is a self-contained JSON object suitable for offline analysis.");
    println!("To open the log in Python:");
    println!("  import json; entries = [json.loads(l) for l in open('{}')]", path.display());

    Ok(())
}
