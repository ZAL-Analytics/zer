/// Integration tests for `JudgeBackend` and `JudgeTarget`.
///
/// These exercise the public API from outside the crate, no access to private
/// fields, no test helpers shared with the module.
use zer_judge::{backend::JudgeTarget, JudgeBackend};

// ── JudgeBackend ──────────────────────────────────────────────────────────────

#[test]
fn auto_detect_defaults_to_cpu_without_flag() {
    // Under `cargo test` the process args don't include --judge-target=,
    // so auto_detect() must fall back to CPU.
    let backend = JudgeBackend::auto_detect();
    assert_eq!(backend.name(), "cpu");
    assert_eq!(backend.target(), JudgeTarget::Cpu);
}

#[test]
fn cpu_constructor_returns_cpu() {
    let backend = JudgeBackend::cpu();
    assert_eq!(backend.name(), "cpu");
    assert_eq!(backend.target(), JudgeTarget::Cpu);
}

#[test]
fn from_target_cpu_string_returns_cpu_backend() {
    let backend = JudgeBackend::from_target("cpu");
    assert_eq!(backend.name(), "cpu");
    assert_eq!(backend.target(), JudgeTarget::Cpu);
}

#[test]
fn execution_providers_always_has_at_least_one_entry() {
    // CPU is always appended as the final fallback.
    let backend = JudgeBackend::cpu();
    let eps = backend.execution_providers();
    assert!(
        !eps.is_empty(),
        "execution_providers must have at least the CPU fallback"
    );
}

#[test]
fn display_shows_judge_backend_and_name() {
    let backend = JudgeBackend::cpu();
    assert_eq!(backend.to_string(), "JudgeBackend(cpu)");
}

// ── JudgeTarget ───────────────────────────────────────────────────────────────

#[test]
fn judge_target_from_name_accepts_all_valid_names() {
    let cases = [
        ("cpu", JudgeTarget::Cpu),
        ("cuda", JudgeTarget::Cuda),
        ("rocm", JudgeTarget::Rocm),
        ("directml", JudgeTarget::DirectMl),
        ("openvino", JudgeTarget::OpenVino),
    ];
    for (name, expected) in cases {
        let t = JudgeTarget::from_name(name)
            .unwrap_or_else(|| panic!("from_name({name:?}) returned None"));
        assert_eq!(t, expected, "wrong variant for {name:?}");
    }
}

#[test]
fn judge_target_as_str_roundtrips_with_from_name() {
    let targets = [
        JudgeTarget::Cpu,
        JudgeTarget::Cuda,
        JudgeTarget::Rocm,
        JudgeTarget::DirectMl,
        JudgeTarget::OpenVino,
    ];
    for t in targets {
        let name = t.as_str();
        let back = JudgeTarget::from_name(name)
            .unwrap_or_else(|| panic!("from_name({name:?}) failed for {t:?}"));
        assert_eq!(back, t);
    }
}

#[test]
fn judge_target_from_name_rejects_unknown_inputs() {
    assert!(JudgeTarget::from_name("").is_none());
    assert!(JudgeTarget::from_name("tpu").is_none());
    assert!(
        JudgeTarget::from_name("CPU").is_none(),
        "must be case-sensitive"
    );
    assert!(
        JudgeTarget::from_name("Cuda").is_none(),
        "must be case-sensitive"
    );
}

#[test]
fn judge_target_default_is_cpu() {
    assert_eq!(JudgeTarget::default(), JudgeTarget::Cpu);
}
