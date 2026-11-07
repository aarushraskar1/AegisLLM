use std::time::Duration;

use aegisllm::{AegisConfig, SemantricEmbedder};

fn main() {
    let cfg = AegisConfig {
        upstream_base_url: "https://api.openai.com".to_string(),
        upstream_bearer_token: Some("test".to_string()),
        semantic_cache_enabled: true,
        similarity_threshold: 0.96,
        max_cache_entries: 1000,
        cache_ttl: Duration::from_secs(3600),
        max_request_bytes: 2 * 1024 * 1024,
        request_timeout: Duration::from_secs(30),
        fault_injection_enabled: true,
        fault_injection_secret: Some("aegis-dev".to_string()),
    };

    let _ = cfg.validate();
}
