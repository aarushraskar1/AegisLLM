use axum::{body::Body, http::{header, HeaderMap, HeaderValue, Response, StatusCode}};
use subtle::ConstantTimeEq;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind { RateLimit, ContextLength, DisconnectAfterTenChunks }
#[derive(Clone)]
pub struct FaultConfig { pub enabled: bool, pub secret: Option<String> }
impl FaultConfig {
    pub fn detect(&self, headers: &HeaderMap) -> Option<FaultKind> {
        if !self.enabled { return None; }
        let expected = self.secret.as_deref()?.as_bytes();
        let supplied = headers.get("x-aegis-test-secret")?.as_bytes();
        if !bool::from(expected.ct_eq(supplied)) { return None; }
        match headers.get("x-aegis-fault-type")?.to_str().ok()? {
            "429" => Some(FaultKind::RateLimit),
            "400-context" => Some(FaultKind::ContextLength),
            "disconnect-10" => Some(FaultKind::DisconnectAfterTenChunks),
            _ => None,
        }
    }
}
pub fn response(fault: FaultKind) -> Response<Body> {
    let (status, body, retry) = match fault {
        FaultKind::RateLimit => (StatusCode::TOO_MANY_REQUESTS, r#"{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"Rate limit exceeded (AegisLLM test fault)."}}"#, Some("2")),
        FaultKind::ContextLength => (StatusCode::BAD_REQUEST, r#"{"error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Request exceeds configured context length."}}"#, None),
        FaultKind::DisconnectAfterTenChunks => (StatusCode::BAD_REQUEST, r#"{"error":{"message":"disconnect fault is only valid for streaming requests"}}"#, None),
    };
    let mut result = Response::new(Body::from(body));
    *result.status_mut() = status;
    result.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Some(value) = retry { result.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static(value)); }
    result
}
