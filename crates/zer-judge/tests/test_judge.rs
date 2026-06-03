/// Integration tests for `zer_judge::judge::DebertaJudge`.
///
/// These tests focus on structural and behavioural properties that do NOT
/// require a real ONNX model file, model-loading tests live in test_inference.rs.
use zer_core::traits::JudgeVerdict;
use zer_judge::{
    judge::{DebertaJudge, DebertaJudgeConfig},
    CalibrationTable,
};

// ── DebertaJudgeConfig ────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = DebertaJudgeConfig::default();
    assert_eq!(cfg.promote_threshold, 0.6);
    assert_eq!(cfg.demote_threshold, 0.35);
    assert!(cfg.audit_log.is_none());
}

#[test]
fn config_clone_is_independent() {
    let cfg = DebertaJudgeConfig::default();
    let mut c2 = cfg.clone();
    c2.promote_threshold = 0.9;
    assert_eq!(
        cfg.promote_threshold, 0.6,
        "original must not be affected by clone mutation"
    );
}

#[test]
fn config_with_custom_calibration() {
    let cal = CalibrationTable::new(10.0, 0.01, 1.0);
    let cfg = DebertaJudgeConfig {
        promote_threshold: 0.7,
        demote_threshold: 0.3,
        calibration: cal,
        audit_log: None,
        batch_size: 64,
    };
    assert_eq!(cfg.promote_threshold, 0.7);
    assert_eq!(cfg.demote_threshold, 0.3);
}

// ── Type properties ───────────────────────────────────────────────────────────

#[test]
fn deberta_judge_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DebertaJudge>();
}

#[test]
fn deberta_judge_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<DebertaJudge>();
}

#[test]
fn deberta_judge_config_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DebertaJudgeConfig>();
}

// ── adjudicate() contract ─────────────────────────────────────────────────────

// NOTE: tests that call DebertaJudge::new() require a real model file and live
// in test_inference.rs (marked #[ignore]).  Here we test the adjudication
// contract at the verdict-threshold level, using the CalibrationTable only.

#[test]
fn calibration_verdict_dispatching() {
    let table = CalibrationTable::default();
    let p = 0.6_f32;

    let after_increase = table.update_probability(p, &JudgeVerdict::IncreaseConfidence);
    let after_decrease = table.update_probability(p, &JudgeVerdict::DecreaseConfidence);
    let after_nochange = table.update_probability(p, &JudgeVerdict::NoChange);

    assert!(
        after_increase > p,
        "IncreaseConfidence must raise probability"
    );
    assert!(
        after_decrease < p,
        "DecreaseConfidence must lower probability"
    );
    assert!(
        (after_nochange - p).abs() < 1e-4,
        "NoChange must leave probability unchanged"
    );
}

// ── Clone behaviour ───────────────────────────────────────────────────────────

// Cloning a DebertaJudge is only meaningful with a real model (shares the
// SyncSender handle), so we verify the Clone bound compiles at the type level.
#[test]
fn deberta_judge_config_implements_clone() {
    fn requires_clone<T: Clone>(_: T) {}
    requires_clone(DebertaJudgeConfig::default());
}
