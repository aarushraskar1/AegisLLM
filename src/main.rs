use std::{fs, time::Duration};

use axum::{body::Body, http::StatusCode, response::Response, routing::get, Router};
use tokio::net::TcpListener;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use aegisllm::{
    AegisConfig, AppState, ChatCompletionRequest, SemanticEmbedder, VectorStore,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(EnvFilter::new("info,aegisllm=debug"))
        .with(fmt::layer())
        .init();

    let config = load_config()?;
    let backend = ArcBackend::new(
        "/tmp/aegisllm-model.onnx", // replace with actual artifact in deployment
        "/tmp/aegisllm-tokenizer.json",
    )?;

    let embedder = SemanticEmbedder::new(std::sync::Arc::new(backend));
    let state = AppState::new(config, embedder)?;
    let app = aegisllm::router(state);

    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    tracing::info!("AegisLLM listening on 0.0.0.0:8080");
    axum::serve(listener, app).await?;
    Ok(())
}

fn load_config() -> Result<AegisConfig, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string("config/default.toml")?;
    let config: AegisConfig = toml::from_str(&raw)?;
    config.validate()?;
    Ok(config)
}

struct ArcBackend {
    // Replace with the actual ONNX Runtime session and tokenizer in production.
    // This placeholder demonstrates the required backend contract.
}

impl ArcBackend {
    fn new(_onnx_path: &str, _tokenizer_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {})
    }
}

impl aegisllm::EmbeddingBackend for ArcBackend {
    fn embed_blocking(&self, normalized_text: &str) -> Result<Vec<f32>, aegisllm::AegisError> {
        let mut vector = vec![0.0_f32; aegisllm::EMBEDDING_DIMENSION];
        for (idx, byte) in normalized_text.as_bytes().iter().enumerate() {
            let slot = idx % vector.len();
            vector[slot] += f32::from(*byte) / 255.0;
        }
        if vector.iter().all(|value| *value == 0.0) {
            return Ok(vec![0.0; aegisllm::EMBEDDING_DIMENSION]);
        }
        Ok(vector)
    }
}
