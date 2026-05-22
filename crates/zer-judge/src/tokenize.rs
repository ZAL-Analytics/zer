/// Tokenizer wrapper for NLI judge models.
///
/// Wraps `tokenizers::Tokenizer` and produces padded `ndarray` tensors
/// suitable for direct ORT input.
use std::path::Path;

use ndarray::Array2;
use tokenizers::{Encoding, Tokenizer};

use crate::spec::{JudgeModelSpec, TokenizerSource};

pub struct JudgeTokenizer {
    inner:      Tokenizer,
    max_length: usize,
}

impl JudgeTokenizer {
    /// Load the tokenizer described by `spec`.
    pub fn from_spec(spec: &dyn JudgeModelSpec) -> Result<Self, crate::error::JudgeError> {
        let inner = match spec.tokenizer_source() {
            TokenizerSource::File(p) => load_from_file(p)?,
            TokenizerSource::HuggingFace(_) => {
                return Err(crate::error::JudgeError::Tokenizer(
                    "HuggingFace tokenizer source not supported in offline mode; \
                     supply a local tokenizer.json via TokenizerSource::File instead"
                        .into(),
                ));
            }
        };
        Ok(Self { inner, max_length: spec.max_length() })
    }

    /// Encode a batch of texts into padded ORT-ready tensors.
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids)` each of shape
    /// `[batch_size, max_length]` with dtype `i64`.
    pub fn encode_batch(
        &self,
        texts: &[String],
    ) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>), crate::error::JudgeError> {
        let n = texts.len();
        let encodings: Vec<Encoding> = self.inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| crate::error::JudgeError::Tokenizer(e.to_string()))?;

        // Dynamic padding: allocate only as wide as the longest sequence in this
        // batch (capped by max_length). Avoids O(seq²) attention waste when real
        // token counts are much shorter than the model's hard maximum.
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .min(self.max_length);

        let mut input_ids      = Array2::<i64>::zeros((n, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((n, seq_len));
        let mut token_type_ids = Array2::<i64>::zeros((n, seq_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids   = enc.get_ids();
            let mask  = enc.get_attention_mask();
            let types = enc.get_type_ids();
            // Count only real (non-padding) positions via the attention mask.
            // encode_batch pads shorter sequences with the tokenizer's [PAD] token,
            // which may have id≠0, so copying all ids[] would put non-zero values
            // at padding positions. Stopping at real_len keeps those positions at 0.
            let real_len = mask.iter().filter(|&&m| m != 0).count().min(seq_len);

            for j in 0..real_len {
                input_ids     [[i, j]] = ids[j]   as i64;
                attention_mask[[i, j]] = mask[j]  as i64;
                token_type_ids[[i, j]] = types[j] as i64;
            }
        }

        Ok((input_ids, attention_mask, token_type_ids))
    }
}

fn load_from_file(path: &Path) -> Result<Tokenizer, crate::error::JudgeError> {
    Tokenizer::from_file(path)
        .map_err(|e| crate::error::JudgeError::Tokenizer(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::MiniLmSpec;

    const MINILM_DIR: &str = "../../models/nli-base/fp16_fused/nli-minilm-onnx";

    fn try_minilm_tokenizer() -> Option<JudgeTokenizer> {
        let spec = MiniLmSpec::from_dir(MINILM_DIR);
        JudgeTokenizer::from_spec(&spec).ok()
    }

    #[test]
    fn encode_batch_shape_matches_max_length() {
        let tok = match try_minilm_tokenizer() { Some(t) => t, None => return };
        let texts = vec!["hello world".to_string(), "foo bar baz".to_string()];
        let (ids, mask, types) = tok.encode_batch(&texts).unwrap();
        assert_eq!(ids.nrows(),   2);
        assert_eq!(mask.nrows(),  2);
        assert_eq!(types.nrows(), 2);
        // Dynamic padding: seq_len ≤ max_length and > 0
        assert!(ids.ncols() > 0 && ids.ncols() <= 512);
        assert_eq!(mask.ncols(),  ids.ncols());
        assert_eq!(types.ncols(), ids.ncols());
    }

    #[test]
    fn encode_batch_attention_mask_nonzero_for_tokens() {
        let tok = match try_minilm_tokenizer() { Some(t) => t, None => return };
        let texts = vec!["alice".to_string()];
        let (_ids, mask, _types) = tok.encode_batch(&texts).unwrap();
        // At least the first few positions should be masked (=1)
        let nonzero_count: usize = mask.iter().filter(|&&v| v != 0).count();
        assert!(nonzero_count > 0, "attention mask should have non-zero entries");
    }

    #[test]
    fn encode_batch_padding_positions_are_zero() {
        let tok = match try_minilm_tokenizer() { Some(t) => t, None => return };
        // Encode a short and a long text so seq_len > actual tokens for the short one.
        let texts = vec!["hi".to_string(), "a b c d e f g h i j k l m n o p q r s t".to_string()];
        let (ids, _mask, _types) = tok.encode_batch(&texts).unwrap();
        // "hi" occupies very few positions; the tail of row 0 should be padding zeros.
        let ncols = ids.ncols();
        if ncols > 4 {
            assert_eq!(ids[[0, ncols - 1]], 0, "padding positions should be 0");
        }
    }

    #[test]
    fn encode_batch_single_text_succeeds() {
        let tok = match try_minilm_tokenizer() { Some(t) => t, None => return };
        let result = tok.encode_batch(&["single sentence".to_string()]);
        assert!(result.is_ok());
        let (ids, _, _) = result.unwrap();
        assert_eq!(ids.nrows(), 1);
    }

    #[test]
    fn huggingface_source_returns_error() {
        use crate::spec::TokenizerSource;
        struct HubSpec { source: TokenizerSource }
        impl crate::spec::JudgeModelSpec for HubSpec {
            fn name(&self)             -> &str            { "test" }
            fn model_path(&self)       -> &Path           { Path::new("/nonexistent") }
            fn tokenizer_source(&self) -> &TokenizerSource { &self.source }
            fn max_length(&self)       -> usize            { 128 }
            fn entailment_idx(&self)   -> usize            { 0 }
            fn vram_bytes(&self)       -> u64              { 0 }
        }
        let spec = HubSpec { source: TokenizerSource::HuggingFace("test/model".into()) };
        // HuggingFace source is intentionally unsupported in offline mode
        let result = JudgeTokenizer::from_spec(&spec);
        assert!(result.is_err(), "HuggingFace source should return an error");
    }
}
