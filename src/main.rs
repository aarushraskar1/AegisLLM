use std::{env, fs, sync::Arc};

use aegisllm::{router, AegisConfig, AppState, OrtEmbedder, SemanticEmbedder};
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,aegisllm=debug")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let path = env::var("AEGIS_CONFIG").unwrap_or_else(|_| "config/default.toml".into());
    let contents = fs::read_to_string(&path).map_err(|e| format!("cannot read config {path}: {e}"))?;
    let mut config: AegisConfig = toml::from_str(&contents)?;
    apply_environment(&mut config);
    config.validate()?;

    // Verify file digests before loading either artifact. A mismatch fails closed.
    let backend = OrtEmbedder::load_verified(
        &config.embedding_model_path,
        &config.embedding_model_sha256,
        &config.tokenizer_path,
        &config.tokenizer_sha256,
        config.embedding_max_tokens,
    )?;
    let state = AppState::new(config.clone(), SemanticEmbedder::new(Arc::new(backend)))?;
    let metrics = Arc::clone(&state.metrics);
    let app = router(state);

    if config.dashboard_enabled {
        let snapshot_rx = metrics.subscribe();
        let events_rx = metrics.subscribe_events();
        tokio::task::spawn_blocking(move || aegisllm::observability::run_dashboard(snapshot_rx, events_rx));
    }

    let listener = TcpListener::bind(&config.bind_addr).await?;
    tracing::info!(address = %config.bind_addr, "AegisLLM ready");
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;
    Ok(())
}

fn apply_environment(config: &mut AegisConfig) {
    if let Ok(value) = env::var("AEGIS_BIND_ADDR") { config.bind_addr = value; }
    if let Ok(value) = env::var("AEGIS_UPSTREAM_URL") { config.upstream_base_url = value; }
    if let Ok(value) = env::var("OPENAI_API_KEY") { config.upstream_bearer_token = Some(value); }
    if let Ok(value) = env::var("AEGIS_FAULT_SECRET") { config.fault_injection_secret = Some(value); }
    if let Ok(value) = env::var("AEGIS_MODEL_PATH") { config.embedding_model_path = value; }
    if let Ok(value) = env::var("AEGIS_TOKENIZER_PATH") { config.tokenizer_path = value; }
}

async fn shutdown_signal() {
    let ctrl_c = async { let _ = tokio::signal::ctrl_c().await; };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) { signal.recv().await; }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}
