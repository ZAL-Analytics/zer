/// Structured audit log: records every judge decision to a JSONL file for
/// offline inspection and model improvement.
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
    sync::Mutex,
};

use zer_core::record::RecordId;

/// One log entry, serialized as a single JSON line.
#[derive(Debug, serde::Serialize)]
pub struct AuditEntry {
    pub record_a:          RecordId,
    pub record_b:          RecordId,
    pub pair_text:         String,
    pub match_probability: f32,
    pub entailment_score:  f32,
    pub verdict:           &'static str,
}

/// Thread-safe append-only audit log.
///
/// Writes one JSON line per judge decision.  Wraps the writer in a `Mutex`
/// so the judge's worker thread can write without ownership conflicts.
pub struct AuditLog {
    writer: Mutex<BufWriter<File>>,
}

impl AuditLog {
    /// Open or create the audit log at `path`, appending if it already exists.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { writer: Mutex::new(BufWriter::new(file)) })
    }

    /// Append one entry.  Silently drops write errors to avoid panicking the
    /// judge thread, logging is best-effort.
    pub fn append(&self, entry: &AuditEntry) {
        if let Ok(mut w) = self.writer.lock() {
            if let Ok(json) = serde_json::to_string(entry) {
                let _ = writeln!(w, "{json}");
                let _ = w.flush();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    fn make_entry(verdict: &'static str) -> AuditEntry {
        AuditEntry {
            record_a:          1,
            record_b:          2,
            pair_text:         "test pair".into(),
            match_probability: 0.75,
            entailment_score:  0.82,
            verdict,
        }
    }

    #[test]
    fn open_creates_file() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let _log = AuditLog::open(&path).expect("open failed");
        assert!(path.exists(), "file should be created");
    }

    #[test]
    fn append_writes_valid_json_line() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let log  = AuditLog::open(&path).unwrap();
        log.append(&make_entry("increase"));
        drop(log);

        let file    = std::fs::File::open(&path).unwrap();
        let reader  = std::io::BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
        assert_eq!(lines.len(), 1);

        let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(v["record_a"], 1);
        assert_eq!(v["record_b"], 2);
        assert_eq!(v["verdict"], "increase");
    }

    #[test]
    fn append_multiple_entries_each_on_own_line() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let log  = AuditLog::open(&path).unwrap();
        for verdict in &["increase", "decrease", "no_change"] {
            log.append(&make_entry(verdict));
        }
        drop(log);

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in &lines {
            serde_json::from_str::<serde_json::Value>(line)
                .expect("each line should be valid JSON");
        }
    }

    #[test]
    fn open_missing_parent_returns_error() {
        let path = Path::new("/nonexistent/deep/path/audit.jsonl");
        assert!(AuditLog::open(path).is_err());
    }

    #[test]
    fn open_appends_to_existing_file() {
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        {
            let log = AuditLog::open(&path).unwrap();
            log.append(&make_entry("increase"));
        }
        {
            let log = AuditLog::open(&path).unwrap();
            log.append(&make_entry("decrease"));
        }
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content.lines().count(), 2, "should have 2 lines from two sessions");
    }
}
