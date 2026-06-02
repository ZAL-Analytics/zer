/// Integration tests for `zer_judge::audit::AuditLog`.
///
/// Verifies file creation, JSONL formatting, multi-threaded safety, and
/// append behaviour across multiple open/close cycles.

use std::{io::BufRead, sync::Arc};
use zer_judge::audit::{AuditEntry, AuditLog};

fn make_entry(id_a: u64, id_b: u64, verdict: &'static str) -> AuditEntry {
    AuditEntry {
        record_a:          id_a,
        record_b:          id_b,
        pair_text:         format!("pair {id_a}/{id_b}"),
        match_probability: 0.70,
        entailment_score:  0.85,
        verdict,
    }
}

// ── File lifecycle ────────────────────────────────────────────────────────────

#[test]
fn open_creates_file_if_not_exists() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.jsonl");
    assert!(!path.exists());
    let _log = AuditLog::open(&path).unwrap();
    assert!(path.exists());
}

#[test]
fn open_missing_parent_returns_io_error() {
    let path = std::path::Path::new("/nonexistent/deep/audit.jsonl");
    assert!(AuditLog::open(path).is_err());
}

// ── Single-writer correctness ─────────────────────────────────────────────────

#[test]
fn single_append_produces_valid_jsonl() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.jsonl");
    let log  = AuditLog::open(&path).unwrap();
    log.append(&make_entry(1, 2, "increase"));
    drop(log);

    let file  = std::fs::File::open(&path).unwrap();
    let lines: Vec<String> = std::io::BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .collect();

    assert_eq!(lines.len(), 1);
    let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(v["record_a"],          1);
    assert_eq!(v["record_b"],          2);
    assert_eq!(v["verdict"],           "increase");
    assert!((v["match_probability"].as_f64().unwrap() - 0.70).abs() < 1e-4);
    assert!((v["entailment_score"].as_f64().unwrap()  - 0.85).abs() < 1e-4);
}

#[test]
fn one_line_per_entry() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.jsonl");
    let log  = AuditLog::open(&path).unwrap();
    for i in 0..10_u64 {
        log.append(&make_entry(i, i + 100, "no_change"));
    }
    drop(log);

    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content.lines().count(), 10);
    for line in content.lines() {
        serde_json::from_str::<serde_json::Value>(line)
            .expect("each line must be valid JSON");
    }
}

// ── Persistence across open/close ─────────────────────────────────────────────

#[test]
fn appends_accumulate_across_sessions() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.jsonl");
    for session in 0..3_u64 {
        let log = AuditLog::open(&path).unwrap();
        log.append(&make_entry(session, session + 10, "decrease"));
    }
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content.lines().count(), 3, "3 sessions  times  1 entry each = 3 lines");
}

// ── Thread safety ─────────────────────────────────────────────────────────────

#[test]
fn concurrent_writes_all_appear_in_file() {
    let dir  = tempfile::tempdir().unwrap();
    let path = dir.path().join("concurrent.jsonl");
    let log  = Arc::new(AuditLog::open(&path).unwrap());

    let handles: Vec<_> = (0..8_u64).map(|i| {
        let log_clone = Arc::clone(&log);
        std::thread::spawn(move || {
            for j in 0..10_u64 {
                log_clone.append(&make_entry(i, j, "increase"));
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }
    drop(log);

    let content = std::fs::read_to_string(&path).unwrap();
    let line_count = content.lines().count();
    assert_eq!(line_count, 80, "8 threads  times  10 entries each = 80 lines");

    // Every line must be valid JSON
    for line in content.lines() {
        serde_json::from_str::<serde_json::Value>(line)
            .expect("concurrent write must not corrupt JSON");
    }
}
