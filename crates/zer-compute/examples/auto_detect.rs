//! Example: backend auto-detection and batch size reporting.
//!
//! Demonstrates how to:
//!   1. Let `GpuBackend::auto_detect()` pick the best available backend
//!      (CUDA → AVX2 → CPU).
//!   2. Inspect the backend's VRAM and auto-tuned batch size.
//!   3. Build `DeviceComparator` and `DeviceScorer` for a typical BRP schema.
//!
//! Run with:
//!   cargo run -p zer-compute --example auto_detect
//!
//! To force the CUDA path:
//!   cargo run -p zer-compute --features cuda --example auto_detect

use std::sync::Arc;

use zer_compute::{BatchSizer, DeviceComparator, DeviceScorer, GpuBackend};
use zer_core::schema::{FieldKind, SchemaBuilder};

fn main() {
    // ── 1. Auto-detect backend ───────────────────────────────────────────────
    let backend = Arc::new(GpuBackend::auto_detect());
    println!("Selected backend : {}", backend.name());

    // ── 2. VRAM information ──────────────────────────────────────────────────
    match backend.total_vram_bytes() {
        Some(total) => println!(
            "Total VRAM       : {:.1} GiB",
            total as f64 / (1024.0 * 1024.0 * 1024.0)
        ),
        None => println!("Total VRAM       : N/A (CPU backend)"),
    }
    match backend.available_vram_bytes() {
        Some(avail) => println!(
            "Available VRAM   : {:.1} GiB",
            avail as f64 / (1024.0 * 1024.0 * 1024.0)
        ),
        None => println!("Available VRAM   : N/A (CPU backend)"),
    }

    // ── 3. Batch size auto-tuning ────────────────────────────────────────────
    let schema = SchemaBuilder::new()
        .field("voornamen", FieldKind::Name)
        .field("achternaam", FieldKind::Name)
        .field("tussenvoegsel", FieldKind::Categorical)
        .field("geboortedatum", FieldKind::Date)
        .field("geboorteland", FieldKind::Categorical)
        .field("nationaliteit", FieldKind::Categorical)
        .field("straatnaam", FieldKind::Address)
        .field("huisnummer", FieldKind::Address)
        .field("postcode", FieldKind::Id)
        .field("woonplaats", FieldKind::Address)
        .build()
        .unwrap();

    let available = backend
        .available_vram_bytes()
        .unwrap_or(4 * 1024 * 1024 * 1024);
    let sizer = BatchSizer::new();
    let max_batch = sizer.max_batch_size(available, schema.fields.len());
    println!("Max GPU batch    : {} pairs", max_batch);

    // ── 4. Construct comparator and scorer ───────────────────────────────────
    let _comparator = DeviceComparator::new(Arc::clone(&backend), &schema).unwrap();
    let _scorer = DeviceScorer::new(Arc::clone(&backend));
    println!("DeviceComparator and DeviceScorer constructed successfully.");

    println!("\nDone, ready to process pairs.");
}
