//! AegisLLM library: local embeddings, semantic cache, fault policies,
//! OpenAI-compatible proxy, and asynchronous observability interfaces.
pub mod fault;
pub mod observability;
pub mod proxy;
pub mod semantic;
pub mod vector_db;

use std::{sync::Arc, time::Duration};
use reqwest::Client;
use tokio::sync::{broadcast, watch};
use thiserror::Error;

pub use fault::{FaultConfig, FaultKind};
pub use observability::{Metrics, MetricsSnapshot, OperationalEvent};
pub use proxy::{router, ChatCompletionRequest, ChatMessage};
pub use semantic::{EmbeddingBackend, OrtEmbedder, SemanticEmbedder};
pub use vector_db::{CacheHit, VectorStore, calculate_cosine_similarity};
pub const EMBEDDING_DIMENSION: usize = 384;

#[derive(Debug, Error)]
pub enum AegisError {
    #[error("configuration error: {0}")] Config(String),
    #[error("invalid request: {0}")] InvalidRequest(String),
    #[error("embedding failed: {0}")] Embedding(String),
    #[error("upstream request failed: {0}")] Upstream(#[from] reqwest::Error),
    #[error("serialization failed: {0}")] Serialization(#[from] serde_json::Error),
    #[error("background task failed: {0}")] Worker(String),
    #[error("I/O error: {0}")] Io(#[from] std::io::Error),
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct AegisConfig {
    pub bind_addr: String,
    pub upstream_base_url: String,
    pub upstream_bearer_token: Option<String>,
    pub semantic_cache_enabled: bool,
    pub similarity_threshold: f32,
    pub max_cache_entries: usize,
    pub cache_ttl_seconds: u64,
    pub max_request_bytes: usize,
    pub request_timeout_seconds: u64,
    pub fault_injection_enabled: bool,
    pub fault_injection_secret: Option<String>,
    pub embedding_model_path: String,
    pub tokenizer_path: String,
    pub embedding_model_sha256: String,
    pub tokenizer_sha256: String,
    pub embedding_max_tokens: usize,
    pub dashboard_enabled: bool,
}

impl Default for AegisConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".into(),
            upstream_base_url: "https://api.openai.com".into(),
            upstream_bearer_token: None,
            semantic_cache_enabled: false,
            similarity_threshold: 0.96,
            max_cache_entries: 10_000,
            cache_ttl_seconds: 3600,
            max_request_bytes: 2 * 1024 * 1024,
            request_timeout_seconds: 120,
            fault_injection_enabled: false,
            fault_injection_secret: None,
            embedding_model_path: "models/model.onnx".into(),
            tokenizer_path: "models/tokenizer.json".into(),
            embedding_model_sha256: String::new(),
            tokenizer_sha256: String::new(),
            embedding_max_tokens: 256,
            dashboard_enabled: false,
        }
    }
}

impl AegisConfig {
    pub fn validate(&self) -> Result<(), AegisError> {
        let url = reqwest::Url::parse(&self.upstream_base_url)
            .map_err(|e| AegisError::Config(format!("invalid upstream URL: {e}")))?;
        if !matches!(url.scheme(), "http" | "https") { return Err(AegisError::Config("upstream URL must use HTTP(S)".into())); }
        if self.bind_addr.parse::<std::net::SocketAddr>().is_err() { return Err(AegisError::Config("bind_addr must be a socket address".into())); }
        if !self.similarity_threshold.is_finite() || !(0.0..=1.0).contains(&self.similarity_threshold) { return Err(AegisError::Config("similarity threshold must be in [0,1]".into())); }
        if self.max_cache_entries == 0 || self.max_request_bytes == 0 || self.embedding_max_tokens == 0 || self.cache_ttl_seconds == 0 || self.request_timeout_seconds == 0 { return Err(AegisError::Config("cache, body, token, and timeout limits must be nonzero".into())); }
        if self.fault_injection_enabled && self.fault_injection_secret.as_deref().is_none_or(str::is_empty) { return Err(AegisError::Config("enabled fault injection requires a secret".into())); }
        Ok(())
    }
}

pub struct AppState {
    pub config: Arc<AegisConfig>,
    pub embedder: Arc<SemanticEmbedder>,
    pub vectors: Arc<VectorStore>,
    pub upstream: Client,
    pub metrics: Arc<Metrics>,
    pub faults: FaultConfig,
}

impl AppState {
    pub fn new(config: AegisConfig, embedder: SemanticEmbedder) -> Result<Self, AegisError> {
        config.validate()?;
        let upstream = Client::builder()
            .connect_timeout(Duration::from_secs(config.request_timeout_seconds.min(10)))
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .pool_max_idle_per_host(32)
            .build()?;
        let vectors = VectorStore::new(config.max_cache_entries, Duration::from_secs(config.cache_ttl_seconds), config.similarity_threshold)?;
        let faults = FaultConfig { enabled: config.fault_injection_enabled, secret: config.fault_injection_secret.clone() };
        Ok(Self { config: Arc::new(config), embedder: Arc::new(embedder), vectors: Arc::new(vectors), upstream, metrics: Arc::new(Metrics::new()), faults })
    }
}

/// Shared metric channel types are exposed here for external monitor consumers.
pub type SnapshotReceiver = watch::Receiver<MetricsSnapshot>;
pub type EventReceiver = broadcast::Receiver<OperationalEvent>;
