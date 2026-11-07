use std::{fs, path::Path, sync::{Arc, Mutex}};

use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use tokenizers::{EncodeInput, Tokenizer};

use crate::{AegisError, EMBEDDING_DIMENSION};

pub trait EmbeddingBackend: Send + Sync + 'static {
    fn embed_blocking(&self, normalized_text: &str) -> Result<Vec<f32>, AegisError>;
}

#[derive(Clone)]
pub struct SemanticEmbedder { backend: Arc<dyn EmbeddingBackend> }

impl SemanticEmbedder {
    pub fn new(backend: Arc<dyn EmbeddingBackend>) -> Self { Self { backend } }
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, AegisError> {
        let input = text.trim().to_owned();
        if input.is_empty() { return Err(AegisError::InvalidRequest("empty prompt".into())); }
        let backend = Arc::clone(&self.backend);
        let vector = tokio::task::spawn_blocking(move || backend.embed_blocking(&input)).await.map_err(|e| AegisError::Worker(e.to_string()))??;
        if vector.len() != EMBEDDING_DIMENSION || vector.iter().any(|v| !v.is_finite()) { return Err(AegisError::Embedding("embedding output has invalid dimension or values".into())); }
        Ok(vector)
    }
}

/// ONNX adapter for BERT-style sentence embedding exports with `input_ids`,
/// `attention_mask`, and optional `token_type_ids`, producing
/// `[batch, sequence, hidden]` or `[batch, hidden]` float output. Token-level
/// outputs are mean pooled with the attention mask and L2-normalized.
pub struct OrtEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_tokens: usize,
}

impl OrtEmbedder {
    pub fn load_verified(model_path: &str, model_sha256: &str, tokenizer_path: &str, tokenizer_sha256: &str, max_tokens: usize) -> Result<Self, AegisError> {
        validate_artifact(model_path, model_sha256, "ONNX model")?;
        validate_artifact(tokenizer_path, tokenizer_sha256, "tokenizer")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| AegisError::Embedding(format!("tokenizer load failed: {e}")))?;
        let mut session = Session::builder().map_err(|e| AegisError::Embedding(format!("ONNX session init failed: {e}")))?
            .commit_from_file(model_path).map_err(|e| AegisError::Embedding(format!("ONNX model load failed: {e}")))?;
        // Force graph metadata validation on startup, rather than discovering
        // incompatible tensor names or outputs on a live request.
        let input_names: Vec<String> = session.inputs.iter().map(|input| input.name.clone()).collect();
        if !input_names.iter().any(|n| n == "input_ids") || !input_names.iter().any(|n| n == "attention_mask") {
            return Err(AegisError::Embedding(format!("model inputs must include input_ids and attention_mask; found {input_names:?}")));
        }
        if max_tokens == 0 { return Err(AegisError::Config("embedding_max_tokens must be > 0".into())); }
        Ok(Self { session: Mutex::new(session), tokenizer, max_tokens })
    }
}

impl EmbeddingBackend for OrtEmbedder {
    fn embed_blocking(&self, text: &str) -> Result<Vec<f32>, AegisError> {
        let encoding = self.tokenizer.encode(EncodeInput::Single(text.into()), true).map_err(|e| AegisError::Embedding(format!("tokenization failed: {e}")))?;
        let ids = encoding.get_ids().iter().take(self.max_tokens).map(|id| i64::from(*id)).collect::<Vec<_>>();
        let mask = encoding.get_attention_mask().iter().take(self.max_tokens).map(|v| i64::from(*v)).collect::<Vec<_>>();
        if ids.is_empty() { return Err(AegisError::InvalidRequest("prompt produced no tokens".into())); }
        let ids_tensor = Tensor::from_array(([1usize, ids.len()], ids)).map_err(|e| AegisError::Embedding(format!("input_ids tensor: {e}")))?;
        let mask_tensor = Tensor::from_array(([1usize, mask.len()], mask)).map_err(|e| AegisError::Embedding(format!("attention_mask tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AegisError::Embedding("ONNX session mutex poisoned".into()))?;
        let mut inputs = ort::inputs! { "input_ids" => ids_tensor, "attention_mask" => mask_tensor };
        if session.inputs.iter().any(|input| input.name == "token_type_ids") {
            let types = vec![0_i64; mask.len()];
            let tensor = Tensor::from_array(([1usize, types.len()], types)).map_err(|e| AegisError::Embedding(format!("token_type_ids tensor: {e}")))?;
            inputs.push(("token_type_ids".into(), tensor.into_dyn()));
        }
        let outputs = session.run(inputs).map_err(|e| AegisError::Embedding(format!("ONNX inference failed: {e}")))?;
        let output = outputs.iter().next().ok_or_else(|| AegisError::Embedding("model returned no output".into()))?;
        let (shape, values) = output.1.try_extract_tensor::<f32>().map_err(|e| AegisError::Embedding(format!("output tensor is not f32: {e}")))?;
        let vector = match shape {
            [1, hidden] if *hidden == EMBEDDING_DIMENSION => values.to_vec(),
            [1, seq, hidden] if *hidden == EMBEDDING_DIMENSION && *seq == ids_len(&mask) => {
                let mut pooled = vec![0.0_f32; EMBEDDING_DIMENSION];
                let mut count = 0.0_f32;
                for token in 0..*seq {
                    if mask[token] == 0 { continue; }
                    count += 1.0;
                    for dim in 0..EMBEDDING_DIMENSION { pooled[dim] += values[token * EMBEDDING_DIMENSION + dim]; }
                }
                if count > 0.0 { for value in &mut pooled { *value /= count; } }
                pooled
            }
            _ => return Err(AegisError::Embedding(format!("unexpected embedding output shape: {shape:?}"))),
        };
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= f32::EPSILON || !norm.is_finite() { return Err(AegisError::Embedding("model returned zero/invalid embedding".into())); }
        Ok(vector.into_iter().map(|v| v / norm).collect())
    }
}

fn ids_len(mask: &[i64]) -> usize { mask.len() }

fn validate_artifact(path: &str, expected: &str, label: &str) -> Result<(), AegisError> {
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) { return Err(AegisError::Config(format!("{label} SHA-256 must be 64 hex characters"))); }
    let metadata = fs::metadata(Path::new(path)).map_err(|e| AegisError::Config(format!("cannot access {label} at {path}: {e}")))?;
    if !metadata.is_file() || metadata.len() == 0 { return Err(AegisError::Config(format!("{label} is not a non-empty file"))); }
    let data = fs::read(path).map_err(|e| AegisError::Config(format!("cannot read {label}: {e}")))?;
    let actual = format!("{:x}", Sha256::digest(data));
    if !actual.eq_ignore_ascii_case(expected) { return Err(AegisError::Config(format!("{label} SHA-256 mismatch (actual {actual})"))); }
    Ok(())
}
