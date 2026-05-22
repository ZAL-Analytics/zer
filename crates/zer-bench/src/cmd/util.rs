use std::path::PathBuf;

/// Log whether TensorRT engine cache is warm (engines present) or cold (first-run compile).
pub fn log_trt_cache_status() {
    let cache_dir = format!(
        "{}/.cache/zer-judge/trt-engines",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_owned())
    );
    let engine_count = std::fs::read_dir(&cache_dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().map_or(false, |x| x == "engine"))
                .count()
        })
        .unwrap_or(0);
    if engine_count > 0 {
        println!("TRT warm: cached engines found  engine_count={engine_count}  cache_dir={cache_dir}");
    } else {
        println!("TRT cold: no cached engines, TRT will compile now (takes 2-5 min, cached after)  cache_dir={cache_dir}");
    }
}

/// Walk up from the current directory to find the workspace root (the
/// `Cargo.toml` that contains `[workspace]`).  Falls back to `.` if not found.
pub fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_default();
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = std::fs::read_to_string(&candidate) {
                if content.contains("[workspace]") {
                    return dir;
                }
            }
        }
        if !dir.pop() {
            return PathBuf::from(".");
        }
    }
}

/// Resolve an `--out` / `--results` path relative to the workspace root when
/// the supplied path is relative.  Absolute paths are returned unchanged.
pub fn resolve_out_dir(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        workspace_root().join(p)
    }
}
