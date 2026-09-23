use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use dashmap::DashMap;
use http::Method;
use rand::Rng;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::{
    str::FromStr,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{sync::Notify, time::interval};
use tracing::{debug, error, info};
use tracing_subscriber::{fmt, EnvFilter};

const SESSION_COOKIE_NAME: &str = "session_id";
const SESSION_COOKIE_MAX_AGE_SECS: u64 = 7 * 24 * 60 * 60; // 1 week

const SERVER_LISTEN_PORT: u16 = 8000;
const CHUNK_SIZE: usize = 1024 * 1024; // 1MB
const CONNECTION_TTL_SECS: u64 = 300; // 5 minutes

// -------------------- Types --------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreateConnReq {
    connection_id: String,
    target_host: String,
    method: String,
    path: String,
    headers: Vec<(String, String)>, // preserve duplicates
    body_size: usize,
}

#[derive(Debug, Clone)]
struct Connection {
    connection_id: String,
    target_host: String,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body_size: usize,
    body_chunks: Vec<Bytes>,
    body_complete: bool,

    response_data: Option<Bytes>,
    response_headers: Option<Vec<(String, String)>>,
    response_status: Option<u16>,
    response_complete: bool,

    // NEW: notify waiters when response becomes ready
    notify_ready: Arc<Notify>,

    created_at: SystemTime,
}

impl Connection {
    fn new(req: &CreateConnReq) -> Self {
        Self {
            connection_id: req.connection_id.clone(),
            target_host: req.target_host.clone(),
            method: req.method.clone(),
            path: req.path.clone(),
            headers: req.headers.clone(),
            body_size: req.body_size,
            body_chunks: Vec::new(),
            body_complete: false,
            response_data: None,
            response_headers: None,
            response_status: None,
            response_complete: false,
            notify_ready: Arc::new(Notify::new()),
            created_at: SystemTime::now(),
        }
    }

    fn total_received(&self) -> usize {
        self.body_chunks.iter().map(|b| b.len()).sum()
    }
}

#[derive(Clone)]
struct AppState {
    conns: Arc<DashMap<String, Connection>>,
    http: Client,
}

// -------------------- Helpers --------------------

fn json_error(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({ "error": msg });
    (status, Json(body)).into_response()
}

fn generate_session_id() -> String {
    let mut rng = rand::rng();
    let bytes: [u8; 32] = rng.random();
    hex::encode(bytes)
}

fn extract_session_id_from_headermap(headers: &HeaderMap) -> Option<String> {
    if let Some(cookie_header) = headers.get("cookie") {
        if let Ok(cookie_str) = cookie_header.to_str() {
            for cookie in cookie_str.split(';') {
                let cookie = cookie.trim();
                if let Some(rest) = cookie.strip_prefix(SESSION_COOKIE_NAME) {
                    if let Some(val) = rest.strip_prefix('=') {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}

fn build_session_cookie(session_id: &str) -> String {
    format!(
        "{}={}; Max-Age={}; Path=/; HttpOnly; SameSite=Lax",
        SESSION_COOKIE_NAME, session_id, SESSION_COOKIE_MAX_AGE_SECS
    )
}

// -------------------- Session Middleware --------------------

async fn session_middleware(request: Request, next: Next) -> Response {
    // Extract existing session_id or generate a new one
    let session_id = extract_session_id_from_headermap(request.headers())
        .unwrap_or_else(|| {
            let new_id = generate_session_id();
            debug!("Generated new session_id: {}", new_id);
            new_id
        });

    // Process the request
    let mut response = next.run(request).await;

    // Add/refresh session cookie in response
    let cookie_value = build_session_cookie(&session_id);
    if let Ok(header_value) = HeaderValue::from_str(&cookie_value) {
        response.headers_mut().append("set-cookie", header_value);
    }

    response
}

fn sanitize_forward_headers(orig: &[(String, String)]) -> (HeaderMap, Vec<String>) {
    // Exclude hop-by-hop & content-length, similar to Python version
    const EXCLUDE: &[&str] = &[
        "connection",
        "proxy-connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    ];
    let mut excluded = Vec::new();
    let mut map = HeaderMap::new();
    let mut has_accept_encoding = false;

    for (k, v) in orig {
        let kl = k.to_ascii_lowercase();
        if EXCLUDE.contains(&kl.as_str()) {
            excluded.push(k.clone());
            continue;
        }
        if kl == "accept-encoding" {
            has_accept_encoding = true;
        }
        if let (Ok(hn), Ok(hv)) = (HeaderName::from_str(k), HeaderValue::from_str(v)) {
            map.append(hn, hv);
        }
    }

    // Force identity to avoid auto-decompression ambiguity & keep raw bytes
    if !has_accept_encoding {
        map.append(
            HeaderName::from_static("accept-encoding"),
            HeaderValue::from_static("identity"),
        );
    }

    (map, excluded)
}

fn build_target_url(target_host: &str, path: &str) -> Result<Url, String> {
    if path.starts_with("http://") || path.starts_with("https://") {
        Url::parse(path).map_err(|e| e.to_string())
    } else {
        Url::parse(&format!("http://{}{}", target_host, path)).map_err(|e| e.to_string())
    }
}

// -------------------- Routes --------------------

async fn startup_log() {
    info!("=== Starting Async Proxy Server (Rust) ===");
    info!("Listening on port {}", SERVER_LISTEN_PORT);
    info!("Chunk size: {} bytes", CHUNK_SIZE);
    info!("=== Server ready to accept connections ===");
}

fn get_hostname() -> String {
    hostname::get()
        .map(|h| h.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn get_local_ip() -> String {
    local_ip_address::local_ip()
        .map(|ip| ip.to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

async fn info() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "hostname": get_hostname(),
        "ip_address": get_local_ip(),
        "port": SERVER_LISTEN_PORT,
    }))
}

async fn create_connection(
    State(state): State<AppState>,
    Json(payload): Json<CreateConnReq>,
) -> impl IntoResponse {
    info!(
        "Creating connection {}: {} {}{}",
        payload.connection_id, payload.method, payload.target_host, payload.path
    );
    debug!("Conn metadata: {:?}", payload);

    let conn = Connection::new(&payload);
    state.conns.insert(payload.connection_id.clone(), conn);

    let resp = serde_json::json!({
        "status": "created",
        "connection_id": payload.connection_id
    });
    Json(resp)
}

async fn receive_chunk(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let conn_id = headers
        .get("x-connection-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let is_final = headers
        .get("x-chunk-final")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let Some(connection_id) = conn_id else {
        return json_error(StatusCode::BAD_REQUEST, "No connection ID provided");
    };

    debug!(
        "Receiving chunk for connection {}, final={}",
        connection_id, is_final
    );

    let mut need_forward = false;

    if let Some(mut entry) = state.conns.get_mut(&connection_id) {
        let conn = entry.value_mut();
        conn.body_chunks.push(body.clone());
        let total_received = conn.total_received();
        debug!(
            "Stored chunk {} for {}: {}/{} bytes",
            conn.body_chunks.len(),
            connection_id,
            total_received,
            conn.body_size
        );
        if is_final {
            conn.body_complete = true;
            info!(
                "Connection {} body complete: {} bytes in {} chunks",
                connection_id,
                total_received,
                conn.body_chunks.len()
            );
            need_forward = true;
        }
    } else {
        return json_error(StatusCode::NOT_FOUND, "Connection not found");
    }

    if need_forward {
        let state2 = state.clone();
        let connection_id2 = connection_id.clone();
        tokio::spawn(async move {
            if let Err(e) = forward_request(state2, &connection_id2).await {
                error!("Forward request error for {}: {}", connection_id2, e);
            }
        });
    }

    Json(serde_json::json!({"status":"chunk_received","is_final":is_final})).into_response()
}

#[derive(Serialize)]
struct MetaResp {
    status: u16,
    headers: Vec<(String, String)>,
    body_size: usize,
    has_body: bool,
}

async fn get_response(State(state): State<AppState>, headers: HeaderMap) -> impl IntoResponse {
    let conn_id = headers
        .get("x-connection-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let chunk_index: usize = headers
        .get("x-chunk-index")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let Some(connection_id) = conn_id else {
        return json_error(StatusCode::BAD_REQUEST, "No connection ID provided");
    };

    // --- BLOCK until response_complete is true ---
    // We repeatedly check the flag, otherwise await on Notify.
    // Clone the Arc<Notify> so we don't hold the DashMap ref across awaits.
    loop {
        let (ready, notifier_opt) = if let Some(entry) = state.conns.get(&connection_id) {
            (entry.response_complete, Some(entry.notify_ready.clone()))
        } else {
            return json_error(StatusCode::NOT_FOUND, "Connection not found");
        };

        if ready {
            break;
        }

        if let Some(notifier) = notifier_opt {
            notifier.notified().await;
        }
    }
    // --- end blocking section ---

    // Now it is ready; snapshot the data for response.
    let (response_status, response_headers, response_data) =
        if let Some(entry) = state.conns.get(&connection_id) {
            let c = entry.value();
            (
                c.response_status,
                c.response_headers.clone(),
                c.response_data.clone(),
            )
        } else {
            return json_error(StatusCode::NOT_FOUND, "Connection not found");
        };

    if chunk_index == 0 {
        let body_size = response_data.as_ref().map(|b| b.len()).unwrap_or(0);
        let meta = MetaResp {
            status: response_status.unwrap_or(502),
            headers: response_headers.unwrap_or_default(),
            body_size,
            has_body: body_size > 0,
        };
        info!(
            "Sending response metadata for {}: status={}, body_size={}",
            connection_id, meta.status, meta.body_size
        );
        return Json(meta).into_response();
    }

    // Body chunks start at index 1
    let body_chunk_index = chunk_index - 1;
    let (chunk, has_more) = if let Some(data) = response_data {
        let start = body_chunk_index * CHUNK_SIZE;
        let end = std::cmp::min(start + CHUNK_SIZE, data.len());
        if start >= data.len() {
            (Bytes::new(), false)
        } else {
            (data.slice(start..end), end < data.len())
        }
    } else {
        (Bytes::new(), false)
    };

    let mut hm = HeaderMap::new();
    hm.insert(
        "Content-Type",
        HeaderValue::from_static("application/octet-stream"),
    );
    hm.insert(
        "X-More-Chunks",
        HeaderValue::from_static(if has_more { "true" } else { "false" }),
    );

    if !has_more {
        // Schedule cleanup in the background
        let st = state.clone();
        let id = connection_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            st.conns.remove(&id);
            info!("Cleaned up connection {}", id);
        });
    }

    (hm, Body::from(chunk)).into_response()
}

// -------------------- Forwarding --------------------

async fn forward_request(state: AppState, conn_id: &str) -> Result<(), String> {
    info!("=== Forwarding request for connection {} ===", conn_id);
    let start = std::time::Instant::now();

    // Snapshot connection fields
    let (method_str, target_host, path, headers, body, notifier) =
        if let Some(entry) = state.conns.get(conn_id) {
            let c = entry.value();
            (
                c.method.clone(),
                c.target_host.clone(),
                c.path.clone(),
                c.headers.clone(),
                Bytes::from(
                    c.body_chunks
                        .iter()
                        .fold(Vec::new(), |mut acc: Vec<u8>, b| {
                            acc.extend_from_slice(b);
                            acc
                        }),
                ),
                c.notify_ready.clone(),
            )
        } else {
            return Err("Connection not found".into());
        };
    // Immediately fail CONNECT requests (do not attempt to forward)
if method_str.eq_ignore_ascii_case("CONNECT") {
    if let Some(mut ent) = state.conns.get_mut(conn_id) {
        let c = ent.value_mut();
        c.response_status = Some(500);
        c.response_headers = Some(vec![
            ("Content-Type".into(), "text/plain".into()),
            ("Connection".into(), "close".into()),
        ]);
        c.response_data = Some(Bytes::from("CONNECT not supported by this proxy"));
        c.response_complete = true;
    }
    // wake any waiters so /proxy/response unblocks
    notifier.notify_waiters();
    info!("Rejected CONNECT for connection {}", conn_id);
    return Ok(());
}

    let url = build_target_url(&target_host, &path)?;
    let (hdrs, excluded) = sanitize_forward_headers(&headers);
    debug!("Forwarding headers (excluded: {:?})", excluded);

    let method = Method::from_bytes(method_str.as_bytes()).map_err(|e| e.to_string())?;
    info!("Target URL: {} {}", method, url);

    // Build request (timeouts/h2 per your client settings; redirect policy set to none)
    let req = state.http.request(method, url).headers(hdrs).body(body);

    // Send and handle upstream transport errors by writing 502 into the connection (Python parity)
    let t1 = std::time::Instant::now();
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            if let Some(mut ent) = state.conns.get_mut(conn_id) {
                let c = ent.value_mut();
                c.response_status = Some(502);
                c.response_headers = Some(vec![("Content-Type".into(), "text/plain".into())]);
                c.response_data = Some(Bytes::from(format!("Proxy error: {}", e)));
                c.response_complete = true;
            }
            // Wake *all* waiters so /proxy/response unblocks
            notifier.notify_waiters();
            return Err(e.to_string());
        }
    };
    info!(
        "Target responded: {} (took {:.3}s)",
        resp.status(),
        t1.elapsed().as_secs_f32()
    );

    // Capture status and headers BEFORE consuming resp with .bytes()
    let status_u16 = resp.status().as_u16();

    // Preserve duplicate response headers; lossy text decode to mirror httpx behavior
    let mut resp_hdrs: Vec<(String, String)> = Vec::new();
    for (name, value) in resp.headers().iter() {
        let s = String::from_utf8_lossy(value.as_bytes()).into_owned();
        resp_hdrs.push((name.as_str().to_string(), s));
    }

    // Read raw bytes as-is (consumes resp)
    let t2 = std::time::Instant::now();
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?;
    debug!(
        "Read response body: {} bytes in {:.3}s",
        bytes.len(),
        t2.elapsed().as_secs_f32()
    );

    // Write back into connection
    if let Some(mut entry) = state.conns.get_mut(conn_id) {
        let conn = entry.value_mut();
        conn.response_status = Some(status_u16);
        conn.response_headers = Some(resp_hdrs);
        conn.response_data = Some(bytes);
        conn.response_complete = true;
    }

    // Wake *all* waiters now that response is ready
    notifier.notify_waiters();

    info!(
        "=== Forward complete: {:.3}s total ===",
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

// -------------------- Cleanup task --------------------

async fn cleanup_task(state: AppState) {
    info!("Starting connection cleanup task");
    let mut tick = interval(Duration::from_secs(60));
    loop {
        tick.tick().await;
        let mut removed = 0usize;
        state.conns.retain(|_, c| {
            let keep = c
                .created_at
                .elapsed()
                .map(|e| e.as_secs() < CONNECTION_TTL_SECS)
                .unwrap_or(false);
            if !keep {
                removed += 1;
            }
            keep
        });
        if removed > 0 {
            info!("Cleaned up {} old connections", removed);
        } else {
            debug!("No stale connections; active: {}", state.conns.len());
        }
    }
}

// -------------------- Main --------------------

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // Tracing
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,proxy_server_async=info,axum=info"));
    fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    startup_log().await;

    // Shared HTTP client:
    // - HTTP/2 enabled
    // - No automatic redirects (Python parity)
    // - You can adjust timeout/keepalive/etc. as needed
    let http = Client::builder()
        .pool_max_idle_per_host(64)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .http2_keep_alive_interval(Some(Duration::from_secs(30)))
        .http2_adaptive_window(true)
        .redirect(reqwest::redirect::Policy::none()) // parity with follow_redirects=False
        // .timeout(Duration::from_secs(30)) // enable if you want a hard deadline
        .build()
        .expect("client");

    let state = AppState {
        conns: Arc::new(DashMap::new()),
        http,
    };

    // Router with session middleware
    let app = Router::new()
        .route("/info", get(info))
        .route("/proxy/connection", post(create_connection))
        .route("/proxy/chunk", post(receive_chunk))
        .route("/proxy/response", get(get_response))
        .layer(middleware::from_fn(session_middleware))
        .with_state(state.clone());

    // Background cleanup
    tokio::spawn(cleanup_task(state.clone()));

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], SERVER_LISTEN_PORT));
    info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
