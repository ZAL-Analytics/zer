use std::collections::HashSet;
use std::path::Path;

use regex::Regex;
use zer_core::{error::ZerError, schema::FieldKind};

const DEFAULT_NAME_HEURISTICS: &str = include_str!("../heuristics_name.toml");
const DEFAULT_VALUE_PATTERNS: &str = include_str!("../heuristics_values.toml");

// ── Name heuristics ───────────────────────────────────────────────────────────

/// A single name-matching rule mapping one or more column-name patterns to a [`FieldKind`].
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NameRule {
    pub kind: FieldKind,
    #[serde(default)]
    pub contains: Vec<String>,
    #[serde(default)]
    pub exact: Vec<String>,
    #[serde(default)]
    pub starts_with: Vec<String>,
    #[serde(default)]
    pub ends_with: Vec<String>,
}

/// Ordered list of name-matching rules loaded from `heuristics_name.toml`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NameHeuristics {
    pub rules: Vec<NameRule>,
}

impl NameHeuristics {
    /// Parse from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, ZerError> {
        toml::from_str(s).map_err(|e| ZerError::Config(e.to_string()))
    }

    /// Load from a TOML file on disk.
    pub fn from_file(path: &Path) -> Result<Self, ZerError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_toml_str(&content)
    }

    /// Load the default heuristics.
    ///
    /// Checks `ZER_NAME_HEURISTICS` env var first; if set and loadable, uses
    /// that file. Otherwise falls back to the embedded `heuristics_name.toml`.
    pub fn load_default() -> Self {
        if let Ok(path) = std::env::var("ZER_NAME_HEURISTICS") {
            match Self::from_file(Path::new(&path)) {
                Ok(h) => return h,
                Err(e) => tracing::warn!(
                    "ZER_NAME_HEURISTICS={path:?}: failed to load ({e}), using embedded default"
                ),
            }
        }
        Self::from_toml_str(DEFAULT_NAME_HEURISTICS)
            .expect("embedded heuristics_name.toml is always valid")
    }

    /// Try to match a column name against the rules. Returns `None` when no
    /// rule matches, signalling the caller to fall back to value sampling.
    pub fn infer_kind(&self, name: &str) -> Option<FieldKind> {
        let n = name.to_ascii_lowercase();
        for rule in &self.rules {
            if rule.exact.iter().any(|p| n == p.as_str())
                || rule.contains.iter().any(|p| n.contains(p.as_str()))
                || rule.starts_with.iter().any(|p| n.starts_with(p.as_str()))
                || rule.ends_with.iter().any(|p| n.ends_with(p.as_str()))
            {
                return Some(rule.kind);
            }
        }
        None
    }
}

// ── Value patterns ────────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct RawValuePattern {
    kind: FieldKind,
    regex: String,
    #[serde(default)]
    threshold: f32,
    unique_rate_min: Option<f32>,
    unique_rate_max: Option<f32>,
    avg_len_min: Option<f32>,
    avg_len_max: Option<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct RawFallback {
    default_kind: FieldKind,
}

#[derive(Debug, serde::Deserialize)]
struct RawValuePatterns {
    patterns: Vec<RawValuePattern>,
    fallback: RawFallback,
}

/// A value-sampling pattern with its regex pre-compiled.
#[derive(Debug)]
pub struct CompiledValuePattern {
    pub kind: FieldKind,
    /// `None` when the pattern has no regex (purely statistical conditions).
    pub regex: Option<Regex>,
    pub threshold: f32,
    pub unique_rate_min: Option<f32>,
    pub unique_rate_max: Option<f32>,
    pub avg_len_min: Option<f32>,
    pub avg_len_max: Option<f32>,
}

/// Ordered list of value-sampling patterns loaded from `heuristics_values.toml`.
#[derive(Debug)]
pub struct ValuePatterns {
    pub patterns: Vec<CompiledValuePattern>,
    pub fallback_kind: FieldKind,
}

impl ValuePatterns {
    fn from_raw(raw: RawValuePatterns) -> Result<Self, ZerError> {
        let mut patterns = Vec::with_capacity(raw.patterns.len());
        for p in raw.patterns {
            let regex = if p.regex.is_empty() {
                None
            } else {
                Some(Regex::new(&p.regex).map_err(|e| {
                    ZerError::Config(format!("invalid regex {:?}: {e}", p.regex))
                })?)
            };
            patterns.push(CompiledValuePattern {
                kind: p.kind,
                regex,
                threshold: p.threshold,
                unique_rate_min: p.unique_rate_min,
                unique_rate_max: p.unique_rate_max,
                avg_len_min: p.avg_len_min,
                avg_len_max: p.avg_len_max,
            });
        }
        Ok(Self { patterns, fallback_kind: raw.fallback.default_kind })
    }

    /// Parse from a TOML string. Returns `Err` if any regex is invalid.
    pub fn from_toml_str(s: &str) -> Result<Self, ZerError> {
        let raw: RawValuePatterns =
            toml::from_str(s).map_err(|e| ZerError::Config(e.to_string()))?;
        Self::from_raw(raw)
    }

    /// Load from a TOML file on disk.
    pub fn from_file(path: &Path) -> Result<Self, ZerError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_toml_str(&content)
    }

    /// Load the default patterns.
    ///
    /// Checks `ZER_VALUE_PATTERNS` env var first; if set and loadable, uses
    /// that file. Otherwise falls back to the embedded `heuristics_values.toml`.
    pub fn load_default() -> Self {
        if let Ok(path) = std::env::var("ZER_VALUE_PATTERNS") {
            match Self::from_file(Path::new(&path)) {
                Ok(p) => return p,
                Err(e) => tracing::warn!(
                    "ZER_VALUE_PATTERNS={path:?}: failed to load ({e}), using embedded default"
                ),
            }
        }
        Self::from_toml_str(DEFAULT_VALUE_PATTERNS)
            .expect("embedded heuristics_values.toml is always valid")
    }

    /// Infer a [`FieldKind`] from a slice of sampled text values.
    ///
    /// Evaluates patterns in order; returns the first match. Falls back to
    /// `fallback_kind` (typically `FreeText`) when nothing matches.
    pub fn infer_kind(&self, samples: &[&str]) -> FieldKind {
        if samples.is_empty() {
            return self.fallback_kind;
        }
        let total = samples.len() as f32;
        let unique_rate = samples.iter().collect::<HashSet<_>>().len() as f32 / total;
        let avg_len = samples.iter().map(|s| s.len() as f32).sum::<f32>() / total;

        for pat in &self.patterns {
            let match_frac = match &pat.regex {
                Some(re) => samples.iter().filter(|s| re.is_match(s)).count() as f32 / total,
                None => 1.0,
            };
            if match_frac >= pat.threshold
                && pat.unique_rate_min.map_or(true, |min| unique_rate >= min)
                && pat.unique_rate_max.map_or(true, |max| unique_rate <= max)
                && pat.avg_len_max.map_or(true, |max| avg_len <= max)
                && pat.avg_len_min.map_or(true, |min| avg_len >= min)
            {
                return pat.kind;
            }
        }
        self.fallback_kind
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_heuristics_embedded_default_loads() {
        let h = NameHeuristics::load_default();
        assert!(!h.rules.is_empty());
    }

    #[test]
    fn name_heuristics_matches_known_patterns() {
        let h = NameHeuristics::load_default();
        assert_eq!(h.infer_kind("first_name"), Some(FieldKind::Name));
        assert_eq!(h.infer_kind("geboortedatum"), Some(FieldKind::Date));
        assert_eq!(h.infer_kind("msisdn"), Some(FieldKind::Phone));
        assert_eq!(h.infer_kind("postcode"), Some(FieldKind::Address));
        assert_eq!(h.infer_kind("bsn"), Some(FieldKind::Id));
    }

    #[test]
    fn name_heuristics_returns_none_for_unknown() {
        let h = NameHeuristics::load_default();
        assert_eq!(h.infer_kind("xyzzy_col"), None);
    }

    #[test]
    fn value_patterns_embedded_default_loads() {
        let p = ValuePatterns::load_default();
        assert!(!p.patterns.is_empty());
    }

    #[test]
    fn value_patterns_date_detection() {
        let p = ValuePatterns::load_default();
        let samples: Vec<&str> = (0..20).map(|_| "2024-03-15").collect();
        assert_eq!(p.infer_kind(&samples), FieldKind::Date);
    }

    #[test]
    fn value_patterns_fallback_on_empty() {
        let p = ValuePatterns::load_default();
        assert_eq!(p.infer_kind(&[]), FieldKind::FreeText);
    }

    #[test]
    fn custom_name_heuristics_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom_name.toml");
        std::fs::write(
            &path,
            r#"
[[rules]]
kind  = "Id"
exact = ["mijnkolom"]
"#,
        )
        .unwrap();

        let h = NameHeuristics::from_file(&path).unwrap();
        assert_eq!(h.infer_kind("mijnkolom"), Some(FieldKind::Id));
        assert_eq!(h.infer_kind("other"), None);
    }

    #[test]
    fn custom_value_patterns_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom_values.toml");
        std::fs::write(
            &path,
            r#"
[[patterns]]
kind      = "Phone"
regex     = '^\+31\d{9}$'
threshold = 0.8

[fallback]
default_kind = "FreeText"
"#,
        )
        .unwrap();

        let p = ValuePatterns::from_file(&path).unwrap();
        let samples: Vec<&str> = (0..20).map(|_| "+31612345678").collect();
        assert_eq!(p.infer_kind(&samples), FieldKind::Phone);
    }

    #[test]
    fn invalid_toml_returns_error() {
        let result = NameHeuristics::from_toml_str("this is not toml ][");
        assert!(matches!(result, Err(ZerError::Config(_))));
    }

    #[test]
    fn invalid_regex_returns_error() {
        let result = ValuePatterns::from_toml_str(
            r#"
[[patterns]]
kind      = "Date"
regex     = '[invalid'
threshold = 0.8

[fallback]
default_kind = "FreeText"
"#,
        );
        assert!(matches!(result, Err(ZerError::Config(_))));
    }
}
