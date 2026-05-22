/// ORT session wrapper: loads the ONNX model and runs batched inference.
use ndarray::Array2;
use ort::{session::Session, value::TensorRef};

use crate::{backend::{JudgeBackend, JudgeTarget}, error::JudgeError, spec::JudgeModelSpec};

pub struct OnnxSession {
    session:             Session,
    entailment_idx:      usize,
    has_token_type_ids:  bool,
}

impl OnnxSession {
    /// Load the ONNX model described by `spec`, configured for `backend`.
    ///
    /// If TensorRT is selected but the model contains ORT-fused ops
    /// (`com.microsoft` domain), TRT cannot parse them and would emit hundreds
    /// of error-level log lines before falling back to the CUDA EP anyway.  In
    /// that case we silently downgrade to a CUDA-only session to suppress the
    /// noise, TRT provides no benefit for these fused graphs.
    pub fn from_spec(
        spec:    &dyn JudgeModelSpec,
        backend: &JudgeBackend,
    ) -> Result<Self, JudgeError> {
        let builder = Session::builder()?;

        let mut builder = if backend.target() == JudgeTarget::TensorRt
            && model_has_ort_fused_ops(spec.model_path())
        {
            tracing::warn!(
                model = %spec.model_path().display(),
                "TRT selected but model contains ORT-fused ops (com.microsoft domain); \
                 TRT cannot parse these, falling back to CUDA EP. \
                 Use a 'base' (non-fused) ONNX export for genuine TRT acceleration."
            );
            // Build with CUDA EP only (no TRT) to avoid parse-error spam.
            let cuda_backend = JudgeBackend::cuda_or_cpu();
            cuda_backend.configure_session(builder)?
        } else {
            backend.configure_session(builder)?
        };

        let session = builder.commit_from_file(spec.model_path())?;

        let has_token_type_ids = session.inputs()
            .iter()
            .any(|inp| inp.name() == "token_type_ids");

        Ok(Self {
            session,
            entailment_idx: spec.entailment_idx(),
            has_token_type_ids,
        })
    }

    /// Run inference on a pre-tokenized batch.
    ///
    /// Returns the softmax-normalised entailment probability for each pair.
    /// Shape of each input array: `[batch_size, seq_len]`.
    pub fn run_batch(
        &mut self,
        input_ids:      &Array2<i64>,
        attention_mask: &Array2<i64>,
        token_type_ids: &Array2<i64>,
    ) -> Result<Vec<f32>, JudgeError> {
        let batch_size = input_ids.nrows();

        let id_ref   = TensorRef::from_array_view(input_ids.view())
            .map_err(|e| JudgeError::Inference(e.to_string()))?;
        let mask_ref = TensorRef::from_array_view(attention_mask.view())
            .map_err(|e| JudgeError::Inference(e.to_string()))?;

        let inputs = if self.has_token_type_ids {
            let type_ref = TensorRef::from_array_view(token_type_ids.view())
                .map_err(|e| JudgeError::Inference(e.to_string()))?;
            ort::inputs![
                "input_ids"      => id_ref,
                "attention_mask" => mask_ref,
                "token_type_ids" => type_ref,
            ]
        } else {
            ort::inputs![
                "input_ids"      => id_ref,
                "attention_mask" => mask_ref,
            ]
        };

        let outputs = self.session.run(inputs)
            .map_err(|e| JudgeError::Inference(e.to_string()))?;

        // ORT output name "logits", shape [batch, num_labels].
        let logits_view = outputs["logits"]
            .try_extract_array::<f32>()
            .map_err(|e| JudgeError::Inference(e.to_string()))?;

        let shape      = logits_view.shape();
        let num_labels = shape[1];
        let mut probs  = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let row: Vec<f32> = (0..num_labels)
                .map(|j| logits_view[[i, j]])
                .collect();
            probs.push(softmax_at(&row, self.entailment_idx));
        }

        Ok(probs)
    }
}

/// Returns true if the ONNX model file at `path` contains ORT-fused ops
/// (nodes in the `com.microsoft` domain). TensorRT cannot parse these and
/// falls back noisily; callers use this to skip TRT for fused-graph exports.
fn model_has_ort_fused_ops(path: &std::path::Path) -> bool {
    // Read raw bytes and look for the domain string without a full ONNX parse.
    std::fs::read(path)
        .map(|bytes| bytes.windows(13).any(|w| w == b"com.microsoft"))
        .unwrap_or(false)
}

/// Numerically stable softmax evaluated at a single index.
fn softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max  = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps[idx] / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_at_highest_index_gives_max_prob() {
        let logits = vec![1.0_f32, 5.0, 2.0];
        let p = softmax_at(&logits, 1);
        assert!(p > 0.9, "highest logit should dominate: {p}");
    }

    #[test]
    fn softmax_at_sums_correctly() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let sum: f32 = (0..3).map(|i| softmax_at(&logits, i)).sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax probs should sum to 1: {sum}");
    }
}
