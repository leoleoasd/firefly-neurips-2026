use std::{
    collections::HashSet,
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::{RwLock, OnceCell};

use anyhow::Context;
use arc_swap::ArcSwap;
use bytes::Bytes;
use dashmap::DashMap;
use http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Response, StatusCode, Uri};
use http_body_util::BodyExt;
use hyper::body::Incoming;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use moka::future::Cache;
use reqwest::Client as ReqClient;
use serde::{Deserialize, Serialize};
use hyper::service::Service;
use tracing::{debug, error, info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use url::Url;
use uuid::Uuid;

// ============================== CONFIG ==============================

const CHUNK_SIZE: usize = 1024 * 1024; // 1MB
const ENABLE_CACHE: bool = true;
const CACHE_MAX_AGE_SECS: u64 = 300;
const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 100;
const DEFAULT_CLIENT_SESSION_ID: &str = "__default__";
const CLIENT_SESSION_HEADER: &str = "x-client-session-id";

const TARGET_HOST_REWRITES: &[(&str, &str)] = &[
    // ("old.example.com:443", "new.example.net:443"),
];

fn get_client_listen_port() -> u16 {
    std::env::var("CLIENT_LISTEN_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080)
}

fn get_api_gateway_url() -> String {
    std::env::var("API_GATEWAY_URL")
        .unwrap_or_else(|_| "https://3he3rx88gl.execute-api.us-east-1.amazonaws.com".to_string())
}

fn get_aws_region() -> String {
    std::env::var("AWS_REGION")
        .unwrap_or_else(|_| "us-east-1".to_string())
}

// ============================== AWS SIGV4 ==============================

use aws_credential_types::provider::ProvideCredentials;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, SigningParams, sign};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;

struct CachedCredentials {
    credentials: Credentials,
}

struct SigV4Signer {
    conf: aws_config::SdkConfig,
    cached_creds: RwLock<Option<CachedCredentials>>,
    region: String,
}

impl SigV4Signer {
    async fn new(region: String) -> anyhow::Result<Self> {
        let conf = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Ok(Self {
            conf,
            cached_creds: RwLock::new(None),
            region,
        })
    }

    async fn get_credentials(&self) -> anyhow::Result<Credentials> {
        // Try to read from cache first
        {
            let cache = self.cached_creds.read().await;
            if let Some(cached) = cache.as_ref() {
                if let Some(expiry) = cached.credentials.expiry() {
                    let now = SystemTime::now();
                    if expiry > now {
                        // Check if expiring within 5 minutes
                        if let Ok(time_until_expiry) = expiry.duration_since(now) {
                            if time_until_expiry < Duration::from_secs(5 * 60) {
                                warn!("AWS credentials will expire in {:?}, refreshing...", time_until_expiry);
                            } else {
                                // Cache is still valid
                                return Ok(cached.credentials.clone());
                            }
                        } else {
                            // Cache is still valid
                            return Ok(cached.credentials.clone());
                        }
                    }
                } else {
                    // No expiry means credentials don't expire
                    return Ok(cached.credentials.clone());
                }
            }
        }

        // Cache miss or expired - acquire write lock before fetching credentials
        let mut cache = self.cached_creds.write().await;

        // Double-check: another thread might have fetched credentials while we waited for the lock
        if let Some(cached) = cache.as_ref() {
            if let Some(expiry) = cached.credentials.expiry() {
                let now = SystemTime::now();
                if expiry > now {
                    // Check if expiring within 5 minutes
                    if let Ok(time_until_expiry) = expiry.duration_since(now) {
                        if time_until_expiry < Duration::from_secs(5 * 60) {
                            warn!("AWS credentials will expire in {:?}, refreshing...", time_until_expiry);
                        } else {
                            // Cache is now valid (another thread fetched it)
                            return Ok(cached.credentials.clone());
                        }
                    } else {
                        // Cache is now valid (another thread fetched it)
                        return Ok(cached.credentials.clone());
                    }
                }
            } else {
                // No expiry means credentials don't expire
                return Ok(cached.credentials.clone());
            }
        }

        // Fetch new credentials while holding write lock
        let creds = self
            .conf
            .credentials_provider()
            .ok_or_else(|| anyhow::anyhow!("No AWS credentials provider"))?
            .provide_credentials()
            .await?;

        *cache = Some(CachedCredentials {
            credentials: creds.clone(),
        });

        Ok(creds)
    }

    async fn sign(
        &self,
        method: &Method,
        url: &Url,
        mut headers: HeaderMap,
        body_bytes: &[u8],
    ) -> anyhow::Result<HeaderMap> {
        // Host header must match URL host:port
        if headers.get("host").is_none() {
            if let Some(host) = url.host_str() {
                let mut host_val = host.to_string();
                if let Some(port) = url.port() {
                    host_val.push(':');
                    host_val.push_str(&port.to_string());
                }
                headers.insert(HeaderName::from_static("host"), HeaderValue::from_str(&host_val)?);
            }
        }

        // Build path + query
        let mut path_q = url.path().to_string();
        if let Some(q) = url.query() {
            path_q.push('?');
            path_q.push_str(q);
        }
        if path_q.is_empty() {
            path_q.push('/');
        }

        // Prepare header iterator
        let mut owned_headers: Vec<(String, String)> = Vec::new();
        for (k, v) in headers.iter() {
            owned_headers.push((
                k.as_str().to_string(),
                v.to_str().unwrap_or("").to_string(),
            ));
        }

        let sign_req = SignableRequest::new(
            method.as_str(),
            &path_q,
            owned_headers.iter().map(|(k, v)| (k.as_str(), v.as_str())),
            SignableBody::Bytes(body_bytes),
        )?;

        // Get credentials (cached for 30 minutes)
        let creds = self.get_credentials().await?;

        // Build SigningParams
        let settings = SigningSettings::default();
        let identity = Identity::new(creds, None);
        let v4_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("execute-api")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| anyhow::anyhow!("SigningParams build error: {e}"))?;
        let params = SigningParams::from(v4_params);

        // Sign
        let signed = sign(sign_req, &params)
            .map_err(|e| anyhow::anyhow!("SigV4 sign error: {e}"))?;

        // Apply auth headers
        for (name, value) in signed.into_parts().0.headers() {
            headers.insert(
                HeaderName::from_str(name)?,
                HeaderValue::from_str(value)?,
            );
        }

        Ok(headers)
    }
}

// ============================== CACHE ==============================

#[derive(Clone)]
struct CacheEntry {
    status: u16,
    headers: Vec<(String, String)>,
    body: Bytes,
}

fn cache_key(method: &Method, full_url: &str) -> String {
    let mut k = String::with_capacity(method.as_str().len() + 1 + full_url.len());
    k.push_str(method.as_str());
    k.push(' ');
    k.push_str(full_url);
    k
}

fn is_static_asset(path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    if let Some(ext) = path.rsplit('.').next() {
        match ext.to_ascii_lowercase().as_str() {
            "js" | "mjs" | "jsx" |
            "css" | "scss" | "sass" | "less" |
            "jpg" | "jpeg" | "png" | "gif" | "webp" | "svg" | "ico" | "bmp" | "tiff" | "avif" |
            "woff" | "woff2" | "ttf" | "eot" | "otf" |
            "pdf" | "zip" | "tar" | "gz" => return true,
            _ => {}
        }
    }
    false
}

fn has_auth_like_headers(h: &HeaderMap) -> bool {
    let auth_keys: HashSet<&'static str> = ["authorization", "x-api-key", "x-auth-token"].into_iter().collect();
    for (k, _) in h.iter() {
        if auth_keys.contains(k.as_str().to_ascii_lowercase().as_str()) {
            return true
        }
    }
    false
}

// ---- "only cache when upstream allows" helpers ----
fn parse_cache_control(cc: &str) -> (bool /*no_store*/, bool /*is_private*/, Option<u64> /*max_age*/, bool /*public*/) {
    let mut no_store = false;
    let mut is_private = false;
    let mut max_age = None;
    let mut is_public = false;

    for part in cc.split(',').map(|s| s.trim().to_ascii_lowercase()) {
        match part.as_str() {
            "no-store" => no_store = true,
            "private" => is_private = true,
            "public" => is_public = true,
            s if s.starts_with("max-age=") => {
                if let Some(v) = s.get(8..).and_then(|n| n.parse::<u64>().ok()) {
                    max_age = Some(v);
                }
            }
            _ => {}
        }
    }
    (no_store, is_private, max_age, is_public)
}

fn expires_in_future(expires: &str) -> bool {
    if let Ok(dt) = httpdate::parse_http_date(expires) {
        if let Ok(now) = SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            let exp = dt.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            return exp > now;
        }
    }
    false
}

fn allow_cache_from_headers(hdrs: &[(String, String)]) -> bool {
    if hdrs.iter().any(|(k, v)| k.eq_ignore_ascii_case("x-allow-cache") && v.eq_ignore_ascii_case("true")) {
        return true;
    }

    let mut cc_val: Option<String> = None;
    let mut expires_val: Option<String> = None;

    for (k, v) in hdrs {
        if k.eq_ignore_ascii_case("cache-control") {
            cc_val = Some(v.clone());
        } else if k.eq_ignore_ascii_case("expires") {
            expires_val = Some(v.clone());
        }
    }

    if let Some(cc) = cc_val.as_deref() {
        let (no_store, is_private, max_age, is_public) = parse_cache_control(cc);
        if no_store || is_private {
            return false;
        }
        if is_public && max_age.unwrap_or(0) > 0 {
            return true;
        }
    }

    if let Some(cc) = cc_val.as_deref() {
        let (no_store, is_private, _max_age, _is_public) = parse_cache_control(cc);
        if no_store || is_private {
            return false;
        }
    }
    if let Some(ex) = expires_val.as_deref() {
        if expires_in_future(ex) {
            return true;
        }
    }

    false
}

// ============================== SESSION STICKINESS ==============================

/// Stores stickiness cookies for a client session.
/// Uses ArcSwap for lock-free reads and atomic updates.
#[derive(Debug, Default)]
struct StickySession {
    /// Cookies from responses (cookie_name -> full Set-Cookie value)
    cookies: ArcSwap<Vec<(String, String)>>,
}

impl StickySession {
    fn new() -> Self {
        Self {
            cookies: ArcSwap::new(Arc::new(Vec::new())),
        }
    }

    /// Get current cookies (lock-free read)
    fn get_cookies(&self) -> Arc<Vec<(String, String)>> {
        self.cookies.load_full()
    }

    /// Update cookies from response headers (atomic swap)
    fn update_cookies(&self, response_headers: &[(String, String)]) {
        let mut new_cookies: Vec<(String, String)> = Vec::new();

        // Extract all cookies from Set-Cookie headers
        for (name, value) in response_headers {
            if name.eq_ignore_ascii_case("set-cookie") {
                // Parse cookie name from "NAME=value; ..." format
                if let Some(cookie_part) = value.split(';').next() {
                    if let Some((cookie_name, _)) = cookie_part.split_once('=') {
                        new_cookies.push((cookie_name.to_string(), value.clone()));
                    }
                }
            }
        }

        if !new_cookies.is_empty() {
            // Merge with existing cookies (replace by name, keep others)
            let old_cookies = self.cookies.load();
            let mut merged: Vec<(String, String)> = old_cookies
                .iter()
                .filter(|(name, _)| !new_cookies.iter().any(|(n, _)| n == name))
                .cloned()
                .collect();
            merged.extend(new_cookies);
            self.cookies.store(Arc::new(merged));
        }
    }

    /// Build Cookie header value from stored cookies
    fn build_cookie_header(&self) -> Option<String> {
        let cookies = self.cookies.load();
        if cookies.is_empty() {
            return None;
        }

        // Extract "NAME=value" from each stored Set-Cookie value
        let cookie_pairs: Vec<String> = cookies
            .iter()
            .filter_map(|(_, set_cookie_value)| {
                set_cookie_value.split(';').next().map(|s| s.to_string())
            })
            .collect();

        if cookie_pairs.is_empty() {
            None
        } else {
            Some(cookie_pairs.join("; "))
        }
    }
}

/// Wrapper that handles lazy initialization of StickySession
struct SessionEntry {
    session: OnceCell<StickySession>,
}

impl SessionEntry {
    fn new() -> Self {
        Self {
            session: OnceCell::new(),
        }
    }
}

// ============================== WIRE TYPES ==============================

#[derive(Serialize)]
struct CreateConnReq {
    connection_id: String,
    target_host: String,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body_size: usize,
}

#[derive(Deserialize)]
struct CreateConnResp {
    status: String,
    connection_id: String,
}

#[derive(Deserialize)]
struct MetaResp {
    status: u16,
    headers: Vec<(String, String)>,
    body_size: usize,
    has_body: bool,
}

// ============================== CLIENT ==============================

#[derive(Clone)]
struct ProxyClient {
    http: ReqClient,
    signer: Arc<SigV4Signer>,
    api_base: Url,
    cache: Option<Cache<String, CacheEntry>>,
    /// Session stickiness: client_session_id -> StickySession
    /// Uses DashMap for sharded access, OnceCell for lazy init, ArcSwap for lock-free cookie access
    session_jars: Arc<DashMap<String, Arc<SessionEntry>>>,
}

impl ProxyClient {
    async fn new() -> anyhow::Result<Self> {
        let http = ReqClient::builder()
            .pool_max_idle_per_host(64)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_adaptive_window(true)
            .build()?;

        let region = get_aws_region();
        let signer = Arc::new(SigV4Signer::new(region).await?);
        let api_base = Url::parse(&get_api_gateway_url())?;
        let cache = if ENABLE_CACHE {
            Some(Cache::builder()
                .time_to_live(Duration::from_secs(CACHE_MAX_AGE_SECS))
                .max_capacity(20_000)
                .build())
        } else { None };

        Ok(Self {
            http,
            signer,
            api_base,
            cache,
            session_jars: Arc::new(DashMap::new()),
        })
    }

    /// Get or create a session entry for the given client session ID.
    /// If this is a new session, calls /info to establish ALB stickiness.
    async fn get_or_init_session(&self, client_session_id: &str) -> anyhow::Result<Arc<SessionEntry>> {
        // Get or insert the session entry (DashMap handles sharding)
        let entry = self.session_jars
            .entry(client_session_id.to_string())
            .or_insert_with(|| Arc::new(SessionEntry::new()))
            .clone();

        // Initialize the session if not already done (OnceCell handles the race)
        entry.session.get_or_try_init(|| async {
            debug!("Initializing sticky session for client_session_id={}", client_session_id);

            // Call /info to get ALB stickiness cookies
            let url = self.api_base.join("/info")?;
            let mut hdrs = HeaderMap::new();
            hdrs.insert("accept", HeaderValue::from_static("application/json"));

            let resp = self.signed_req_raw(Method::GET, url, hdrs, None).await?;

            let session = StickySession::new();

            // Extract cookies from response headers
            let response_headers: Vec<(String, String)> = resp
                .headers()
                .iter()
                .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();

            session.update_cookies(&response_headers);

            let cookie_count = session.get_cookies().len();
            info!("Initialized sticky session for client_session_id={}, got {} ALB cookies",
                  client_session_id, cookie_count);

            Ok::<StickySession, anyhow::Error>(session)
        }).await?;

        Ok(entry)
    }

    fn apply_rewrite(&self, original_host: &str, headers: &HeaderMap) -> String {
        if let Some(v) = headers.get("x-target-host-rewrite") {
            if let Ok(s) = v.to_str() {
                if let Some((orig, newv)) = s.split_once('=') {
                    if orig.trim() == original_host {
                        info!("Dynamic host rewrite: {original_host} -> {}", newv.trim());
                        return newv.trim().to_string();
                    }
                }
            }
        }
        for (k, v) in TARGET_HOST_REWRITES.iter() {
            if k.eq_ignore_ascii_case(&original_host) {
                info!("Static host rewrite: {original_host} -> {v}");
                return v.to_string();
            }
        }
        original_host.to_string()
    }

    async fn signed_req_raw(
        &self,
        method: Method,
        url: Url,
        headers: HeaderMap,
        body: Option<Bytes>,
    ) -> anyhow::Result<reqwest::Response> {
        let buf = body.unwrap_or_else(Bytes::new);
        let signed = self.signer.sign(&method, &url, headers, &buf).await?;
        let mut req = self.http.request(reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap(), url);
        req = req.headers(signed);
        if !buf.is_empty() {
            req = req.body(buf);
        }
        Ok(req.send().await?)
    }

    async fn signed_req_with_session(
        &self,
        method: Method,
        url: Url,
        mut headers: HeaderMap,
        body: Option<Bytes>,
        session: &StickySession,
    ) -> anyhow::Result<reqwest::Response> {
        // Add ALB stickiness cookies to request
        if let Some(cookie_header) = session.build_cookie_header() {
            if let Ok(hv) = HeaderValue::from_str(&cookie_header) {
                headers.insert("cookie", hv);
            }
        }

        let resp = self.signed_req_raw(method, url, headers, body).await?;

        // Update session cookies from response (lock-free atomic swap)
        let response_headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        session.update_cookies(&response_headers);

        Ok(resp)
    }

    async fn signed_req(
        &self,
        method: Method,
        url: Url,
        headers: HeaderMap,
        body: Option<Bytes>,
    ) -> anyhow::Result<reqwest::Response> {
        self.signed_req_raw(method, url, headers, body).await
    }

    async fn send_connection(
        &self,
        target_host: String,
        method: &Method,
        path: &str,
        headers: &HeaderMap,
        body_size: usize,
        session: &StickySession,
    ) -> anyhow::Result<String> {
        let connection_id = Uuid::new_v4().to_string();

        let header_list: Vec<(String, String)> = headers
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.as_str().to_string(), s.to_string())))
            .collect();

        let payload = CreateConnReq {
            connection_id: connection_id.clone(),
            target_host,
            method: method.as_str().to_string(),
            path: path.to_string(),
            headers: header_list,
            body_size,
        };

        let url = self.api_base.join("/proxy/connection")?;
        let mut hdrs = HeaderMap::new();
        hdrs.insert("content-type", HeaderValue::from_static("application/json"));

        let body = Bytes::from(serde_json::to_vec(&payload)?);
        let resp = self.signed_req_with_session(Method::POST, url, hdrs, Some(body), session).await?;
        let code = resp.status();
        let val = resp.bytes().await?;

        if code == StatusCode::OK {
            let _parsed: CreateConnResp = serde_json::from_slice(&val)?;
            info!("Created connection {}", connection_id);
            Ok(connection_id)
        } else {
            let txt = String::from_utf8_lossy(&val);
            anyhow::bail!("create_connection {}: {}", code, txt);
        }
    }

    async fn send_chunk(&self, connection_id: &str, chunk: Bytes, final_chunk: bool, session: &StickySession) -> anyhow::Result<()> {
        let url = self.api_base.join("/proxy/chunk")?;
        let mut hdrs = HeaderMap::new();
        hdrs.insert("content-type", HeaderValue::from_static("application/octet-stream"));
        hdrs.insert("x-connection-id", HeaderValue::from_str(connection_id)?);
        hdrs.insert("x-chunk-final", HeaderValue::from_static(if final_chunk { "true" } else { "false" }));

        let resp = self.signed_req_with_session(Method::POST, url, hdrs, Some(chunk), session).await?;
        if resp.status() != StatusCode::OK {
            let code = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            anyhow::bail!("send_chunk {}: {}", code, txt);
        }
        Ok(())
    }

    async fn get_meta(&self, connection_id: &str, session: &StickySession) -> anyhow::Result<MetaResp> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let url = self.api_base.join("/proxy/response")?;
            let mut hdrs = HeaderMap::new();
            hdrs.insert("x-connection-id", HeaderValue::from_str(connection_id)?);
            hdrs.insert("x-chunk-index", HeaderValue::from_static("0"));

            let resp = self.signed_req_with_session(Method::GET, url, hdrs, None, session).await?;
            let code = resp.status();
            let bytes = resp.bytes().await?;

            if code == StatusCode::OK {
                let meta: MetaResp = serde_json::from_slice(&bytes)?;
                return Ok(meta);
            }

            // Retry on 5xx errors
            if code.is_server_error() && attempt < MAX_RETRIES {
                let txt = String::from_utf8_lossy(&bytes);
                warn!("get_meta attempt {}/{} failed with {}: {}, retrying...", attempt, MAX_RETRIES, code, txt);
                tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS * attempt as u64)).await;
                continue;
            }

            // Non-retryable error or max retries reached
            let txt = String::from_utf8_lossy(&bytes);
            anyhow::bail!("get_meta {}: {}", code, txt);
        }
    }

    async fn get_body_chunk(&self, connection_id: &str, idx: usize, session: &StickySession) -> anyhow::Result<(Bytes, bool)> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let url = self.api_base.join("/proxy/response")?;
            let mut hdrs = HeaderMap::new();
            hdrs.insert("x-connection-id", HeaderValue::from_str(connection_id)?);
            hdrs.insert("x-chunk-index", HeaderValue::from_str(&idx.to_string())?);

            let resp = self.signed_req_with_session(Method::GET, url, hdrs, None, session).await?;
            let code = resp.status();

            if code == StatusCode::OK {
                let more = resp
                    .headers()
                    .get("X-More-Chunks")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let bytes = resp.bytes().await?;
                return Ok((bytes, more));
            }

            // Retry on 5xx errors
            if code.is_server_error() && attempt < MAX_RETRIES {
                let bytes = resp.bytes().await?;
                let txt = String::from_utf8_lossy(&bytes);
                warn!("get_body_chunk (idx={}) attempt {}/{} failed with {}: {}, retrying...", idx, attempt, MAX_RETRIES, code, txt);
                tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS * attempt as u64)).await;
                continue;
            }

            // Non-retryable error or max retries reached
            let bytes = resp.bytes().await?;
            let txt = String::from_utf8_lossy(&bytes);
            anyhow::bail!("get_body_chunk (idx={}): {} {}", idx, code, txt);
        }
    }

    async fn handle_http_proxy(&self, req: Request<Incoming>) -> anyhow::Result<Response<Full<Bytes>>> {
        // Reject CONNECT locally to match server behavior
        if req.method() == Method::CONNECT {
            return Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header("content-type", "text/plain")
                .header("connection", "close")
                .body(Full::new(Bytes::from("CONNECT not supported by this proxy")))?);
        }

        // split parts
        let (method, uri, headers, body_incoming) = {
            let (parts, body) = req.into_parts();
            (parts.method, parts.uri, parts.headers, body)
        };

        // Extract client session ID for ALB stickiness
        let client_session_id = headers
            .get(CLIENT_SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| DEFAULT_CLIENT_SESSION_ID.to_string());

        // Get or initialize sticky session (lock-free after init)
        let session_entry = self.get_or_init_session(&client_session_id).await?;
        let session = session_entry.session.get()
            .ok_or_else(|| anyhow::anyhow!("Session not initialized"))?;

        // Collect body bytes (hyper 1.x)
        let body_vec = body_incoming.collect().await?.to_bytes().to_vec();

        // Full request URL for cache key
        let full_req_url = if uri.scheme().is_some() && uri.authority().is_some() {
            uri.to_string()
        } else {
            let host = headers.get("host").and_then(|v| v.to_str().ok()).unwrap_or_default();
            let path_q = {
                let p = uri.path();
                if let Some(q) = uri.query() { format!("{p}?{q}") } else { p.to_string() }
            };
            format!("http://{host}{path_q}")
        };

        info!("=== New {} request: {} (session={}) ===", method, full_req_url, client_session_id);

        // Target host + path
        let (orig_host, path_only) = parse_target_host_and_path(&uri, &headers)
            .context("parse target host/path")?;
        let rewritten_host = self.apply_rewrite(&orig_host, &headers);

        // Prepare forward headers (strip proxy-only)
        let mut fwd_headers = HeaderMap::new();
        for (k, v) in headers.iter() {
            let key = k.as_str().to_ascii_lowercase();
            if key == "x-target-host-rewrite" || key == "remote-addr" || key == CLIENT_SESSION_HEADER {
                continue;
            }
            fwd_headers.append(k.clone(), v.clone());
        }
        // Preserve original Host
        if !fwd_headers.contains_key("host") {
            if let Ok(hv) = HeaderValue::from_str(&orig_host) {
                fwd_headers.insert("host", hv);
            }
        }

        // Cache read pre-check
        let mut use_cache = false;
        let cache_key_str = cache_key(&method, &full_req_url);
        if ENABLE_CACHE && method == Method::GET && is_static_asset(&path_only) && !has_auth_like_headers(&fwd_headers) {
            use_cache = true;
            if let Some(cache) = &self.cache {
                if let Some(entry) = cache.get(&cache_key_str).await {
                    info!("🎯 CACHE HIT: {} {}", method, full_req_url);
                    return Ok(build_response(entry.status, &entry.headers, entry.body));
                } else {
                    debug!("❌ CACHE MISS: {} {}", method, full_req_url);
                }
            }
        }

        // Create connection
        let conn_id = self
            .send_connection(
                rewritten_host.clone(),
                &method,
                &path_only,
                &fwd_headers,
                body_vec.len(),
                session,
            )
            .await?;

        // Send chunks
        if body_vec.is_empty() {
            self.send_chunk(&conn_id, Bytes::new(), true, session).await?;
        } else {
            let mut offset = 0usize;
            let total = body_vec.len();
            while offset < total {
                let end = (offset + CHUNK_SIZE).min(total);
                let final_chunk = end >= total;
                self.send_chunk(&conn_id, Bytes::copy_from_slice(&body_vec[offset..end]), final_chunk, session).await?;
                offset = end;
            }
        }

        // Wait for meta
        let meta = self.get_meta(&conn_id, session).await?;
        let status = meta.status;
        let resp_headers = meta.headers;

        // Filter hop-by-hop
        let filtered_headers: Vec<(String, String)> = resp_headers.into_iter().filter(|(k, _)| {
            let lk = k.to_ascii_lowercase();
            lk != "content-length" && lk != "transfer-encoding" && lk != "x-more-chunks" && lk != "x-chunk-index"
        }).collect();

        // Body
        let body_bytes = if meta.has_body {
            let mut buf = Vec::with_capacity(meta.body_size);
            let mut idx = 1usize;
            loop {
                let (chunk, more) = self.get_body_chunk(&conn_id, idx, session).await?;
                buf.extend_from_slice(&chunk);
                if !more { break; }
                idx += 1;
            }
            Bytes::from(buf)
        } else {
            Bytes::new()
        };

        // Cache store only if upstream allows
        if use_cache {
            let allow_cache = allow_cache_from_headers(&filtered_headers);
            if allow_cache {
                if let Some(cache) = &self.cache {
                    cache.insert(
                        cache_key_str,
                        CacheEntry { status, headers: filtered_headers.clone(), body: body_bytes.clone() }
                    ).await;
                    info!("💾 CACHED (allowed by upstream): {} {}", method, full_req_url);
                }
            } else {
                debug!("🚫 NOT CACHED: upstream did not allow caching for {}", full_req_url);
            }
        }

        Ok(build_response(status, &filtered_headers, body_bytes))
    }
}

// Build a hyper::Response from parts
fn build_response(status: u16, hdrs: &[(String, String)], body: Bytes) -> Response<Full<Bytes>> {
    let mut builder = Response::builder().status(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY));
    {
        let headers = builder.headers_mut().unwrap();
        for (k, v) in hdrs {
            if let (Ok(hn), Ok(hv)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v)) {
                headers.append(hn, hv);
            }
        }
    }
    builder.body(Full::new(body)).unwrap()
}

fn parse_target_host_and_path(uri: &Uri, headers: &HeaderMap) -> anyhow::Result<(String, String)> {
    if uri.scheme().is_some() && uri.authority().is_some() {
        let host = uri.authority().unwrap().as_str().to_string();
        let mut path = uri.path().to_string();
        if let Some(q) = uri.query() { path.push('?'); path.push_str(q); }
        Ok((host, path))
    } else {
        let host = headers.get("host").and_then(|v| v.to_str().ok())
            .ok_or_else(|| anyhow::anyhow!("Missing Host header"))?.to_string();
        let mut path = uri.path().to_string();
        if let Some(q) = uri.query() { path.push('?'); path.push_str(q); }
        Ok((host, path))
    }
}

// ============================== TOWER SERVICE ==============================

#[derive(Clone)]
struct ProxySvc {
    client: Arc<ProxyClient>,
}

impl Service<Request<Incoming>> for ProxySvc {
    type Response = Response<Full<Bytes>>;
    type Error = hyper::Error;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, req: Request<Incoming>) -> Self::Future {
        let client = self.client.clone();
        Box::pin(async move {
            match client.handle_http_proxy(req).await {
                Ok(resp) => Ok(resp),
                Err(e) => {
                    error!("Proxy error: {}", e);
                    let resp = Response::builder()
                        .status(StatusCode::BAD_GATEWAY)
                        .header("content-type", "text/plain")
                        .header("connection", "close")
                        .body(Full::new(Bytes::from("Proxy client error")))
                        .unwrap();
                    Ok(resp)
                }
            }
        })
    }
}

// ============================== MAIN ==============================

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,proxy_client_sigv4=info,hyper=info,aws_config=warn,aws_credential_types=warn,aws_smithy_runtime=warn"));
    fmt().with_env_filter(filter).with_target(false).compact().init();

    let client = Arc::new(ProxyClient::new().await?);

    let listen_port = get_client_listen_port();
    let api_gateway_url = get_api_gateway_url();
    let aws_region = get_aws_region();

    let addr = SocketAddr::from(([0, 0, 0, 0], listen_port));
    info!("Proxy client listening on http://{}", addr);
    info!("Configure your app to use http://127.0.0.1:{} as HTTP proxy", listen_port);
    info!("AWS SigV4: enabled; service=execute-api, region={}", aws_region);
    info!("API Gateway URL: {}", api_gateway_url);
    info!(
        "Cache: {} (TTL={}s; key=request URL)",
        if ENABLE_CACHE { "enabled" } else { "disabled" },
        CACHE_MAX_AGE_SECS
    );
    if !TARGET_HOST_REWRITES.is_empty() {
        info!("Static host rewrites: {:?}", TARGET_HOST_REWRITES);
    }
    info!("Dynamic rewrite header: X-Target-Host-Rewrite (format 'orig=new')");
    info!("Session stickiness header: {} (default: {})", CLIENT_SESSION_HEADER, DEFAULT_CLIENT_SESSION_ID);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    loop {
        let (stream, peer) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let svc = ProxySvc { client: client.clone() };
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .preserve_header_case(true)
                .title_case_headers(true)
                .serve_connection(io, svc)
                .await
            {
                warn!("Connection from {} closed with error: {}", peer, e);
            }
        });
    }
}
