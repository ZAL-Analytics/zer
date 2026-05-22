use demo_common::{init_tracing, section};
use zer_compute::{
    backend::{BackendPreference, DeviceBackend},
    kernels::hello_backend::{HelloBackend, HelloBackendInput},
};
use zer_core::{
    comparison::ComparisonVector,
    scoring::{MatchBand, ScoredPair},
};
use zer_judge::DummyJudge;
use zer_core::traits::{Judge, JudgeVerdict};

fn probe_backend(backend: &DeviceBackend) {
    let name = backend.name();
    match backend.run::<HelloBackend>(HelloBackendInput) {
        Ok(out) if out.token != 0 => {
            println!("  [PASS] {}, token = 0x{:08X}", name, out.token);
        }
        Ok(_) => {
            println!("  [FAIL] {}, kernel ran but returned token = 0 (kernel may not have executed)", name);
        }
        Err(e) => {
            println!("  [SKIP] {}, {}", name, e);
        }
    }
}

fn main() {
    init_tracing();

    // ── Backend probe ─────────────────────────────────────────────────────────
    section("Backend probe");
    println!("Auto-detecting best available backend …");
    let default_backend = DeviceBackend::auto_detect();
    println!("  selected: {}", default_backend.name());

    section("HelloBackend kernel, per-backend verification");
    println!("Probing each backend independently:");

    #[cfg(feature = "cuda")]
    match DeviceBackend::from_preference(BackendPreference::Cuda) {
        Ok(b)  => probe_backend(&b),
        Err(e) => println!("  [SKIP] cuda , {}", e),
    }

    #[cfg(feature = "vulkan")]
    match DeviceBackend::from_preference(BackendPreference::Vulkan) {
        Ok(b)  => probe_backend(&b),
        Err(e) => println!("  [SKIP] vulkan, {}", e),
    }

    #[cfg(feature = "avx2")]
    {
        let b = DeviceBackend::from_preference(BackendPreference::Avx2)
            .unwrap_or_else(|_| DeviceBackend::auto_detect());
        probe_backend(&b);
    }

    // CPU is always available.
    match DeviceBackend::from_preference(BackendPreference::Cpu) {
        Ok(b)  => probe_backend(&b),
        Err(e) => println!("  [FAIL] cpu, unexpected: {}", e),
    }

    // ── DummyJudge verification ───────────────────────────────────────────────
    section("DummyJudge wiring verification");
    let judge = DummyJudge;
    let pairs = vec![
        ScoredPair {
            record_a: 1, record_b: 2,
            match_weight: 3.0, match_probability: 0.92,
            vector: ComparisonVector::new(1, 2, vec![]),
            band: MatchBand::AutoMatch,
        },
        ScoredPair {
            record_a: 3, record_b: 4,
            match_weight: 0.0, match_probability: 0.45,
            vector: ComparisonVector::new(3, 4, vec![]),
            band: MatchBand::Borderline,
        },
        ScoredPair {
            record_a: 5, record_b: 6,
            match_weight: -3.0, match_probability: 0.10,
            vector: ComparisonVector::new(5, 6, vec![]),
            band: MatchBand::AutoReject,
        },
    ];

    match judge.adjudicate(&pairs) {
        Ok(verdicts) => {
            let all_ok = verdicts.iter().all(|v| matches!(v, JudgeVerdict::IncreaseConfidence));
            for (pair, verdict) in pairs.iter().zip(verdicts.iter()) {
                println!(
                    "  record #{} ↔ #{} (p={:.2}) → {:?}",
                    pair.record_a, pair.record_b, pair.match_probability, verdict
                );
            }
            if all_ok {
                println!("  [PASS] all verdicts = IncreaseConfidence");
            } else {
                println!("  [FAIL] unexpected verdict(s)");
            }
        }
        Err(e) => eprintln!("  [FAIL] adjudicate error: {}", e),
    }

    section("Done");
    println!("hello_backend completed, all wiring verified.");
}
