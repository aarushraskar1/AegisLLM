# AegisLLM

A local-first defensive reverse proxy library for LLM APIs.

## Current implementation

The initial library provides:

- An Axum OpenAI-compatible `POST /v1/chat/completions` route and health/readiness endpoints.
- A typed embedding-backend boundary that runs inference on Tokio's blocking pool. Implement this boundary with the exact ONNX model/tokenizer input, pooling, and output contract you deploy.
- A concurrent, bounded, TTL-based semantic response cache. It is disabled by default; cache eligibility is restricted to non-streaming, zero-temperature requests, and a policy fingerprint prevents reuse across generation settings.
- Upstream HTTP forwarding with timeouts, an optional configured bearer credential, safe response-header copying, and streaming passthrough.
- Opt-in fault responses gated by a test secret. `disconnect-10` deliberately truncates a stream after ten upstream chunks; it tests incomplete-body handling, and does not promise a TCP reset.
- Watch/broadcast metrics interfaces suitable for a separate dashboard task.

## Integration

Construct `AegisConfig`, implement `EmbeddingBackend` for your model runtime, create a `SemanticEmbedder`, then build `AppState` and serve `aegisllm::router(state)` with `axum::serve`. The core intentionally does not silently substitute a fake embedding model: semantic caching must remain disabled until a real compatible model is loaded and validated.

```rust,ignore
let config = aegisllm::AegisConfig::default();
let embedder = aegisllm::SemanticEmbedder::new(onnx_backend);
let state = aegisllm::AppState::new(config, embedder)?;
let app = aegisllm::router(state);
```

Set the upstream credential in the application's secret/configuration provider; do not commit credentials. Keep fault injection disabled outside isolated test environments.

## Validation

Run `cargo fmt --check`, `cargo test`, and `cargo clippy --all-targets -- -D warnings` in CI. The ONNX adapter should have model-artifact checksum validation and integration tests against the exact model and tokenizer files before semantic caching is enabled.
