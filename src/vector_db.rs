use std::{sync::Arc, time::{Duration, Instant, SystemTime}};
use bytes::Bytes;
use dashmap::DashMap;
use crate::AegisError;

#[derive(Clone)]
struct Entry { vector: Arc<[f32]>, response: Bytes, model: String, fingerprint: [u8; 32], expiry: Instant, created: SystemTime }
#[derive(Clone, Debug)]
pub struct CacheHit { pub response: Bytes, pub similarity: f32 }
pub struct VectorStore { entries: DashMap<String, Entry>, max_entries: usize, ttl: Duration, threshold: f32 }

impl VectorStore {
    pub fn new(max_entries: usize, ttl: Duration, threshold: f32) -> Result<Self, AegisError> {
        if max_entries == 0 || ttl.is_zero() || !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) { return Err(AegisError::Config("invalid vector store bounds".into())); }
        Ok(Self { entries: DashMap::new(), max_entries, ttl, threshold })
    }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn insert(&self, key: String, vector: Vec<f32>, response: Bytes, model: String, fingerprint: [u8; 32]) {
        self.entries.insert(key, Entry { vector: Arc::from(vector), response, model, fingerprint, expiry: Instant::now() + self.ttl, created: SystemTime::now() });
        self.entries.retain(|_, entry| entry.expiry > Instant::now());
        while self.entries.len() > self.max_entries {
            let oldest = self.entries.iter().min_by_key(|e| e.created).map(|e| e.key().clone());
            if let Some(key) = oldest { self.entries.remove(&key); } else { break; }
        }
    }
    pub fn query(&self, query: &[f32], model: &str, fingerprint: &[u8; 32]) -> Option<CacheHit> {
        if query.is_empty() || query.iter().any(|x| !x.is_finite()) { return None; }
        let now = Instant::now();
        self.entries.iter().filter_map(|entry| {
            let item = entry.value();
            if item.expiry <= now || item.model != model || item.fingerprint != *fingerprint { return None; }
            let score = calculate_cosine_similarity(query, &item.vector);
            (score >= self.threshold).then(|| CacheHit { response: item.response.clone(), similarity: score })
        }).max_by(|a,b| a.similarity.total_cmp(&b.similarity))
    }
}

pub fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let (mut dot, mut aa, mut bb) = (0.0f32, 0.0f32, 0.0f32);
    for (&x,&y) in a.iter().zip(b) { dot += x*y; aa += x*x; bb += y*y; }
    let denom = aa.sqrt()*bb.sqrt();
    if denom <= f32::EPSILON || !denom.is_finite() { 0.0 } else { dot/denom }
}
