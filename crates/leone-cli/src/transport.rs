//! Isolates socket I/O from the inference thread.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const OVERLOAD_RESPONSE: &[u8] = b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: 110\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"the server work queue is full\",\"type\":\"server_overloaded\",\"code\":\"transport-queue-full\"}}";
const CONNECTION_LIMIT_RESPONSE: &[u8] = b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: 115\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"the connection limit is full\",\"type\":\"server_overloaded\",\"code\":\"transport-connection-limit\"}}";
const CLIENT_LIMIT_RESPONSE: &[u8] = b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: 118\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"the client connection limit is full\",\"type\":\"server_overloaded\",\"code\":\"transport-client-limit\"}}";
const REQUEST_TIMEOUT_RESPONSE: &[u8] = b"HTTP/1.1 408 Request Timeout\r\nContent-Type: application/json\r\nContent-Length: 105\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"the request deadline expired\",\"type\":\"request_timeout\",\"code\":\"transport-deadline\"}}";
const BAD_REQUEST_RESPONSE: &[u8] = b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 110\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"invalid HTTP request\",\"type\":\"invalid_request_error\",\"code\":\"transport-invalid-request\"}}";
const BODY_TOO_LARGE_RESPONSE: &[u8] = b"HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nContent-Length: 119\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"HTTP request body is too large\",\"type\":\"invalid_request_error\",\"code\":\"transport-body-too-large\"}}";
const MAX_OUTPUT_FRAMES: usize = 256;
/// Bounds the number of socket pressure intervals retained for one request.
pub(crate) const MAX_PRESSURE_INTERVALS: usize = 32;

/// Returns the fixed per-output state retained for pressure telemetry.
pub(crate) const fn output_metadata_bytes() -> usize {
    std::mem::size_of::<OutputState>()
}
const TERMINAL_FRAME_GRACE: Duration = Duration::from_millis(250);
const REJECTION_WRITE_TIMEOUT: Duration = Duration::from_millis(100);
const REJECTION_DRAIN_TIMEOUT: Duration = Duration::from_millis(25);

pub(crate) fn canonical_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(address) => IpAddr::V4(address),
        IpAddr::V6(address) => {
            let segments = address.segments();
            if segments[..5] == [0; 5] && segments[5] == u16::MAX {
                IpAddr::V4(
                    address
                        .to_ipv4()
                        .expect("mapped IPv6 address has an IPv4 value"),
                )
            } else {
                IpAddr::V6(address)
            }
        }
    }
}

/// Bounds socket workers and transport queues for one serving process.
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub max_connections: usize,
    pub max_connections_per_client: usize,
    pub max_pending_requests: usize,
    pub max_output_bytes: usize,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub trusted_proxy_ips: Vec<IpAddr>,
}

impl Config {
    pub(crate) fn validate(self) -> io::Result<Self> {
        if self.max_connections == 0
            || self.max_connections_per_client == 0
            || self.max_pending_requests == 0
            || self.max_output_bytes == 0
        {
            return Err(invalid_data("transport limits must be nonzero"));
        }
        if self.read_timeout.is_zero() || self.write_timeout.is_zero() {
            return Err(invalid_data("transport timeouts must be nonzero"));
        }
        Ok(self)
    }
}

impl ConnectionLimits {
    fn new(config: &Config) -> Self {
        Self {
            max_connections: config.max_connections,
            max_connections_per_client: config.max_connections_per_client,
            trusted_proxy_ips: config
                .trusted_proxy_ips
                .iter()
                .copied()
                .map(canonical_ip)
                .collect(),
            state: Mutex::new(ConnectionState {
                active: 0,
                by_client: HashMap::new(),
                sockets: HashMap::new(),
            }),
        }
    }

    fn acquire(
        self: &Arc<Self>,
        peer: IpAddr,
        connection_id: u64,
    ) -> Result<ConnectionLease, ConnectionReject> {
        let peer = canonical_ip(peer);
        let client_id = (!self.trusted_proxy_ips.contains(&peer)).then_some(peer);
        let mut state = self
            .state
            .lock()
            .map_err(|_| ConnectionReject::GlobalLimit)?;
        if state.active >= self.max_connections {
            return Err(ConnectionReject::GlobalLimit);
        }
        if client_id.is_some_and(|id| {
            state.by_client.get(&id).copied().unwrap_or(0) >= self.max_connections_per_client
        }) {
            return Err(ConnectionReject::ClientLimit);
        }
        state.active += 1;
        if let Some(id) = client_id {
            increment_client_count(&mut state.by_client, id);
        }
        Ok(ConnectionLease {
            limits: Arc::clone(self),
            connection_id,
            client_id,
        })
    }

    fn shutdown(&self) {
        let Ok(state) = self.state.lock() else {
            return;
        };
        for stream in state.sockets.values() {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

enum ConnectionReject {
    GlobalLimit,
    ClientLimit,
}

impl ConnectionLease {
    fn register(&self, stream: &TcpStream) {
        let Ok(clone) = stream.try_clone() else {
            return;
        };
        if let Ok(mut state) = self.limits.state.lock() {
            state.sockets.insert(self.connection_id, clone);
        }
    }

    fn assign_client(&mut self, client_id: IpAddr) -> bool {
        let client_id = canonical_ip(client_id);
        if self.client_id == Some(client_id) {
            return true;
        }
        if self.client_id.is_some() {
            return false;
        }
        let Ok(mut state) = self.limits.state.lock() else {
            return false;
        };
        if state.by_client.get(&client_id).copied().unwrap_or(0)
            >= self.limits.max_connections_per_client
        {
            return false;
        }
        increment_client_count(&mut state.by_client, client_id);
        self.client_id = Some(client_id);
        true
    }
}

impl Drop for ConnectionLease {
    fn drop(&mut self) {
        let Ok(mut state) = self.limits.state.lock() else {
            return;
        };
        state.active = state.active.saturating_sub(1);
        if let Some(client_id) = self.client_id {
            decrement_client_count(&mut state.by_client, client_id);
        }
        state.sockets.remove(&self.connection_id);
    }
}

fn increment_client_count(counts: &mut HashMap<IpAddr, usize>, client_id: IpAddr) {
    let count = counts.entry(client_id).or_default();
    *count = count.saturating_add(1);
}

fn decrement_client_count(counts: &mut HashMap<IpAddr, usize>, client_id: IpAddr) {
    let Some(count) = counts.get_mut(&client_id) else {
        return;
    };
    *count = count.saturating_sub(1);
    if *count == 0 {
        counts.remove(&client_id);
    }
}

/// Identifies a request and its bounded response path.
pub(crate) struct Incoming<R> {
    pub client_id: IpAddr,
    pub peer_addr: SocketAddr,
    pub request_id: u64,
    pub received_at_ns: u64,
    pub request: R,
    pub output: OutputSink,
    pub cancellation: Cancellation,
    pub deadline: Instant,
}

/// Reports bounded response queue and socket backpressure observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TransportTelemetry {
    /// Highest response byte reservation observed for this connection.
    pub queue_high_water_bytes: u64,
    /// Time spent in writes that returned `TimedOut` or `WouldBlock`.
    pub socket_blocked_ns: u64,
    /// Number of writes that returned `TimedOut` or `WouldBlock`.
    pub socket_blocked_events: u64,
    /// Observed socket blocked intervals, in the shared monotonic clock.
    pub socket_blocked_intervals: [TransportPressureInterval; MAX_PRESSURE_INTERVALS],
    /// Number of entries retained in `socket_blocked_intervals`.
    pub socket_blocked_interval_count: usize,
    /// Number of observed intervals omitted after the fixed interval bound.
    pub socket_blocked_intervals_dropped: u64,
    /// Reports a response write rejected after the server submitted bytes.
    pub delivery_failed: bool,
    /// Identifies whether delivery failed before enqueue or during socket write.
    pub delivery_failure_phase: DeliveryFailurePhase,
    /// Reports when the disconnect watcher observed the client connection end.
    pub client_disconnected_at_ns: Option<u64>,
}

/// Retains transport telemetry after the response sender is dropped.
#[derive(Clone)]
pub(crate) struct TransportTelemetryHandle {
    state: Arc<OutputState>,
}

impl TransportTelemetryHandle {
    pub(crate) fn snapshot(&self) -> TransportTelemetry {
        transport_telemetry(&self.state)
    }
}

/// Identifies the transport phase that rejected response bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeliveryFailurePhase {
    None,
    EnqueueRejected,
    SocketWrite,
}

/// Records one bounded interval during which a socket write was blocked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TransportPressureInterval {
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Shares cancellation between a socket worker and the engine.
#[derive(Clone, Debug)]
pub(crate) struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
struct OutputState {
    pending_bytes: AtomicUsize,
    queue_high_water_bytes: AtomicUsize,
    socket_blocked_ns: AtomicU64,
    socket_blocked_events: AtomicU64,
    socket_blocked_intervals: Mutex<PressureIntervals>,
    delivery_failure_phase: AtomicU8,
    client_disconnected: AtomicBool,
    client_disconnected_at_ns: AtomicU64,
    cancelled: AtomicBool,
    terminal_claimed: AtomicBool,
    response_started: AtomicBool,
    streaming: AtomicBool,
    terminal_deadline: Mutex<Option<Instant>>,
    max_bytes: usize,
}

#[derive(Debug, Clone)]
struct PressureIntervals {
    entries: [TransportPressureInterval; MAX_PRESSURE_INTERVALS],
    count: usize,
    dropped: u64,
}

impl Default for PressureIntervals {
    fn default() -> Self {
        Self {
            entries: [TransportPressureInterval::default(); MAX_PRESSURE_INTERVALS],
            count: 0,
            dropped: 0,
        }
    }
}

impl OutputState {
    fn record_socket_blocked(&self, start_ns: u64, end_ns: u64) {
        let nanos = end_ns.saturating_sub(start_ns);
        let mut intervals = self
            .socket_blocked_intervals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        saturating_add(&self.socket_blocked_ns, nanos);
        saturating_add(&self.socket_blocked_events, 1);
        if intervals.count < MAX_PRESSURE_INTERVALS {
            let index = intervals.count;
            intervals.entries[index] = TransportPressureInterval { start_ns, end_ns };
            intervals.count += 1;
        } else {
            intervals.dropped = intervals.dropped.saturating_add(1);
        }
    }

    fn record_client_disconnect(&self) {
        self.client_disconnected_at_ns
            .store(crate::clock::now_ns(), Ordering::Release);
        self.client_disconnected.store(true, Ordering::Release);
        self.cancelled.store(true, Ordering::Release);
    }
}

enum Frame {
    Bytes { bytes: Vec<u8>, queued_at: Instant },
    Terminal(Vec<u8>),
}

/// Enqueues response bytes without blocking the inference thread.
#[derive(Clone, Debug)]
pub(crate) struct OutputSink {
    sender: SyncSender<Frame>,
    state: Arc<OutputState>,
}

struct ConnectionResponse {
    output: OutputSink,
    receiver: Receiver<Frame>,
    cancellation: Cancellation,
    timeout: Duration,
    deadline: Instant,
    stop: Arc<AtomicBool>,
}

struct RejectJob {
    stream: TcpStream,
    response: &'static [u8],
}

struct ConnectionWorkerConfig<R> {
    sender: SyncSender<Incoming<R>>,
    limits: Arc<ConnectionLimits>,
    config: Config,
    reader: fn(&mut TcpStream, Instant) -> io::Result<Option<R>>,
    identity: fn(&R, SocketAddr, &[IpAddr]) -> IpAddr,
    reject_sender: SyncSender<RejectJob>,
    stop: Arc<AtomicBool>,
}

struct ConnectionContext<R> {
    peer_addr: SocketAddr,
    connection_id: u64,
    deadline: Instant,
    lease: ConnectionLease,
    sender: SyncSender<Incoming<R>>,
    config: Config,
    reader: fn(&mut TcpStream, Instant) -> io::Result<Option<R>>,
    identity: fn(&R, SocketAddr, &[IpAddr]) -> IpAddr,
    stop: Arc<AtomicBool>,
}

struct WriteContext {
    receiver: Receiver<Frame>,
    output_state: Arc<OutputState>,
    cancellation: Cancellation,
    timeout: Duration,
    watch_stop: Arc<AtomicBool>,
    deadline: Instant,
    stop: Arc<AtomicBool>,
}

struct ConnectionLimits {
    max_connections: usize,
    max_connections_per_client: usize,
    trusted_proxy_ips: Vec<IpAddr>,
    state: Mutex<ConnectionState>,
}

struct ConnectionState {
    active: usize,
    by_client: HashMap<IpAddr, usize>,
    sockets: HashMap<u64, TcpStream>,
}

struct ConnectionLease {
    limits: Arc<ConnectionLimits>,
    connection_id: u64,
    client_id: Option<IpAddr>,
}

impl OutputSink {
    fn new(max_bytes: usize, capacity: usize) -> (Self, Receiver<Frame>) {
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let state = Arc::new(OutputState {
            pending_bytes: AtomicUsize::new(0),
            queue_high_water_bytes: AtomicUsize::new(0),
            socket_blocked_ns: AtomicU64::new(0),
            socket_blocked_events: AtomicU64::new(0),
            socket_blocked_intervals: Mutex::new(PressureIntervals::default()),
            delivery_failure_phase: AtomicU8::new(DeliveryFailurePhase::None as u8),
            client_disconnected: AtomicBool::new(false),
            client_disconnected_at_ns: AtomicU64::new(0),
            cancelled: AtomicBool::new(false),
            terminal_claimed: AtomicBool::new(false),
            response_started: AtomicBool::new(false),
            streaming: AtomicBool::new(false),
            terminal_deadline: Mutex::new(None),
            max_bytes,
        });
        (Self { sender, state }, receiver)
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn cancel(&self) {
        self.state.cancelled.store(true, Ordering::Release);
    }

    pub(crate) fn begin_terminal_response(&self) -> bool {
        self.state
            .terminal_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn mark_streaming(&self) {
        self.state.streaming.store(true, Ordering::Release);
    }

    pub(crate) fn terminal_claimed(&self) -> bool {
        self.state.terminal_claimed.load(Ordering::Acquire)
    }

    pub(crate) fn telemetry_handle(&self) -> TransportTelemetryHandle {
        TransportTelemetryHandle {
            state: Arc::clone(&self.state),
        }
    }

    pub(crate) fn transport_telemetry(&self) -> TransportTelemetry {
        transport_telemetry(&self.state)
    }

    fn record_delivery_failed(&self, phase: DeliveryFailurePhase) {
        self.state
            .delivery_failure_phase
            .store(phase as u8, Ordering::Release);
    }

    fn reserve(&self, bytes: usize) -> io::Result<()> {
        let mut current = self.state.pending_bytes.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                self.cancel();
                self.record_delivery_failed(DeliveryFailurePhase::EnqueueRejected);
                return Err(transport_closed("response queue byte limit overflowed"));
            };
            if next > self.state.max_bytes {
                self.cancel();
                self.record_delivery_failed(DeliveryFailurePhase::EnqueueRejected);
                return Err(transport_overloaded("response queue is full"));
            }
            match self.state.pending_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    update_high_watermark(&self.state.queue_high_water_bytes, next);
                    return Ok(());
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, bytes: usize) {
        self.state.pending_bytes.fetch_sub(bytes, Ordering::AcqRel);
    }
}

fn transport_telemetry(state: &OutputState) -> TransportTelemetry {
    let intervals = state
        .socket_blocked_intervals
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let delivery_failure_phase = match state.delivery_failure_phase.load(Ordering::Acquire) {
        value if value == DeliveryFailurePhase::EnqueueRejected as u8 => {
            DeliveryFailurePhase::EnqueueRejected
        }
        value if value == DeliveryFailurePhase::SocketWrite as u8 => {
            DeliveryFailurePhase::SocketWrite
        }
        _ => DeliveryFailurePhase::None,
    };
    let client_disconnected_at_ns = state
        .client_disconnected
        .load(Ordering::Acquire)
        .then(|| state.client_disconnected_at_ns.load(Ordering::Acquire));
    TransportTelemetry {
        queue_high_water_bytes: u64::try_from(state.queue_high_water_bytes.load(Ordering::Acquire))
            .unwrap_or(u64::MAX),
        socket_blocked_ns: state.socket_blocked_ns.load(Ordering::Acquire),
        socket_blocked_events: state.socket_blocked_events.load(Ordering::Acquire),
        socket_blocked_intervals: intervals.entries,
        socket_blocked_interval_count: intervals.count,
        socket_blocked_intervals_dropped: intervals.dropped,
        delivery_failed: delivery_failure_phase != DeliveryFailurePhase::None,
        delivery_failure_phase,
        client_disconnected_at_ns,
    }
}

impl Write for OutputSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        if self.is_cancelled() {
            return Err(transport_closed("response consumer is gone"));
        }
        self.reserve(bytes.len())?;
        let length = bytes.len();
        let frame = if self.state.terminal_claimed.load(Ordering::Acquire) {
            Frame::Terminal(bytes.to_vec())
        } else {
            Frame::Bytes {
                bytes: bytes.to_vec(),
                queued_at: Instant::now(),
            }
        };
        match self.sender.try_send(frame) {
            Ok(()) => Ok(length),
            Err(TrySendError::Full(_)) => {
                self.release(length);
                self.cancel();
                self.record_delivery_failed(DeliveryFailurePhase::EnqueueRejected);
                Err(transport_overloaded("response queue is full"))
            }
            Err(TrySendError::Disconnected(_)) => {
                self.release(length);
                self.cancel();
                self.record_delivery_failed(DeliveryFailurePhase::EnqueueRejected);
                Err(transport_closed("response writer is gone"))
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.is_cancelled() {
            return Err(transport_closed("response consumer is gone"));
        }
        Ok(())
    }
}

/// Owns the accept loop and exposes parsed requests to the engine.
pub(crate) struct Transport<R> {
    receiver: Receiver<Incoming<R>>,
    local_addr: SocketAddr,
    stop: Arc<AtomicBool>,
    limits: Arc<ConnectionLimits>,
    join: Option<thread::JoinHandle<()>>,
    reject_sender: Option<SyncSender<RejectJob>>,
    reject_join: Option<thread::JoinHandle<()>>,
}

impl<R: Send + 'static> Transport<R> {
    pub(crate) fn bind(
        bind: SocketAddr,
        config: Config,
        stop: Arc<AtomicBool>,
        reader: fn(&mut TcpStream, Instant) -> io::Result<Option<R>>,
        identity: fn(&R, SocketAddr, &[IpAddr]) -> IpAddr,
    ) -> io::Result<Self> {
        let config = config.validate()?;
        let listener = TcpListener::bind(bind)?;
        listener.set_nonblocking(true)?;
        let local_addr = listener.local_addr()?;
        let (sender, receiver) = mpsc::sync_channel(config.max_pending_requests);
        let rejection_capacity = config.max_pending_requests.clamp(1, MAX_OUTPUT_FRAMES);
        let limits = Arc::new(ConnectionLimits::new(&config));
        let thread_stop = Arc::clone(&stop);
        let thread_limits = Arc::clone(&limits);
        let thread_config = config.clone();
        let (reject_sender, reject_join) = spawn_rejection_worker(&stop, rejection_capacity)?;
        let thread_worker = ConnectionWorkerConfig {
            sender,
            limits: thread_limits,
            config: thread_config,
            reader,
            identity,
            reject_sender: reject_sender.clone(),
            stop: thread_stop,
        };
        let join = match spawn_accept_worker(listener, thread_worker) {
            Ok(join) => join,
            Err(error) => {
                stop.store(true, Ordering::Release);
                drop(reject_sender);
                let _ = reject_join.join();
                return Err(io::Error::other(format!(
                    "transport thread failed: {error}"
                )));
            }
        };
        Ok(Self {
            receiver,
            local_addr,
            stop,
            limits,
            join: Some(join),
            reject_sender: Some(reject_sender),
            reject_join: Some(reject_join),
        })
    }

    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub(crate) fn try_recv(&self) -> Result<Option<Incoming<R>>, TransportReceiveError> {
        match self.receiver.try_recv() {
            Ok(request) => Ok(Some(request)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(TransportReceiveError::Stopped),
        }
    }
}

impl<R> Drop for Transport<R> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.limits.shutdown();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        self.reject_sender.take();
        if let Some(join) = self.reject_join.take() {
            let _ = join.join();
        }
    }
}

/// Reports that the transport accept loop stopped before a request arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransportReceiveError {
    Stopped,
}

fn rejection_loop(receiver: Receiver<RejectJob>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Acquire) {
        match receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(mut job) => reject_stream_reliably(&mut job.stream, job.response),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn spawn_rejection_worker(
    stop: &Arc<AtomicBool>,
    capacity: usize,
) -> io::Result<(SyncSender<RejectJob>, thread::JoinHandle<()>)> {
    let (sender, receiver) = mpsc::sync_channel(capacity);
    let thread_stop = Arc::clone(stop);
    let join = thread::Builder::new()
        .name("leone-transport-reject".to_owned())
        .spawn(move || rejection_loop(receiver, thread_stop))
        .map_err(|error| io::Error::other(format!("transport reject thread failed: {error}")))?;
    Ok((sender, join))
}

fn spawn_accept_worker<R: Send + 'static>(
    listener: TcpListener,
    worker: ConnectionWorkerConfig<R>,
) -> io::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("leone-transport-accept".to_owned())
        .spawn(move || accept_loop(listener, worker))
        .map_err(|error| io::Error::other(format!("transport thread failed: {error}")))
}

fn submit_rejection(sender: &SyncSender<RejectJob>, stream: TcpStream, response: &'static [u8]) {
    let job = RejectJob { stream, response };
    match sender.try_send(job) {
        Ok(()) => {}
        Err(TrySendError::Full(job) | TrySendError::Disconnected(job)) => {
            reject_stream_nonblocking(job.stream, job.response);
        }
    }
}

fn accept_loop<R: Send + 'static>(listener: TcpListener, worker: ConnectionWorkerConfig<R>) {
    while accept_next(&listener, &worker) {}
}

fn accept_next<R: Send + 'static>(
    listener: &TcpListener,
    worker: &ConnectionWorkerConfig<R>,
) -> bool {
    match listener.accept() {
        Ok((stream, peer_addr)) => {
            if worker.stop.load(Ordering::Acquire) {
                drop(stream);
            } else {
                accept_connection(
                    stream,
                    peer_addr,
                    ConnectionWorkerConfig {
                        sender: worker.sender.clone(),
                        limits: Arc::clone(&worker.limits),
                        config: worker.config.clone(),
                        reader: worker.reader,
                        identity: worker.identity,
                        reject_sender: worker.reject_sender.clone(),
                        stop: Arc::clone(&worker.stop),
                    },
                );
            }
            true
        }
        Err(_) if worker.stop.load(Ordering::Acquire) => false,
        Err(_) => {
            thread::sleep(Duration::from_millis(1));
            true
        }
    }
}

fn accept_connection<R: Send + 'static>(
    stream: TcpStream,
    peer_addr: SocketAddr,
    worker: ConnectionWorkerConfig<R>,
) {
    if worker.stop.load(Ordering::Acquire) {
        return;
    }
    let connection_id = request_id();
    let lease = match worker.limits.acquire(peer_addr.ip(), connection_id) {
        Ok(lease) => lease,
        Err(ConnectionReject::GlobalLimit) => {
            submit_rejection(&worker.reject_sender, stream, CONNECTION_LIMIT_RESPONSE);
            return;
        }
        Err(ConnectionReject::ClientLimit) => {
            submit_rejection(&worker.reject_sender, stream, CLIENT_LIMIT_RESPONSE);
            return;
        }
    };
    let deadline = request_deadline(worker.config.read_timeout);
    lease.register(&stream);
    let context = ConnectionContext {
        peer_addr,
        connection_id,
        deadline,
        lease,
        sender: worker.sender,
        config: worker.config,
        reader: worker.reader,
        identity: worker.identity,
        stop: worker.stop,
    };
    if context.stop.load(Ordering::Acquire) {
        return;
    }
    let worker = thread::Builder::new()
        .name("leone-transport-connection".to_owned())
        .spawn(move || connection_loop(stream, context));
    let _ = worker;
}

fn connection_loop<R: Send + 'static>(mut stream: TcpStream, context: ConnectionContext<R>) {
    let ConnectionContext {
        peer_addr,
        connection_id,
        deadline,
        mut lease,
        sender,
        config,
        reader,
        identity,
        stop,
    } = context;
    if stop.load(Ordering::Acquire) {
        return;
    }
    if !configure_stream(&stream, &config) {
        return;
    }
    if stop.load(Ordering::Acquire) {
        return;
    }
    let Some(request) = read_connection_request(&mut stream, deadline, reader) else {
        return;
    };
    let received_at_ns = crate::clock::now_ns();
    let client_identity = identity(&request, peer_addr, &config.trusted_proxy_ips);
    if !lease.assign_client(client_identity) {
        reject_stream_reliably(&mut stream, CLIENT_LIMIT_RESPONSE);
        return;
    }
    let capacity = output_frame_capacity(config.max_output_bytes);
    let (output, receiver) = OutputSink::new(config.max_output_bytes, capacity);
    let cancellation = Cancellation::new();
    let incoming = Incoming {
        client_id: client_identity,
        peer_addr,
        request_id: connection_id,
        received_at_ns,
        request,
        output: output.clone(),
        cancellation: cancellation.clone(),
        deadline,
    };
    enqueue_connection(
        stream,
        &sender,
        incoming,
        ConnectionResponse {
            output,
            receiver,
            cancellation,
            timeout: config.write_timeout,
            deadline,
            stop,
        },
    );
}

fn configure_stream(stream: &TcpStream, config: &Config) -> bool {
    // A nonblocking listener can pass that mode to accepted sockets on Darwin.
    stream.set_nonblocking(false).is_ok()
        && stream.set_read_timeout(Some(config.read_timeout)).is_ok()
        && stream.set_write_timeout(Some(config.write_timeout)).is_ok()
}

fn read_connection_request<R>(
    stream: &mut TcpStream,
    deadline: Instant,
    reader: fn(&mut TcpStream, Instant) -> io::Result<Option<R>>,
) -> Option<R> {
    match reader(stream, deadline) {
        Ok(Some(request)) => Some(request),
        Ok(None) => None,
        Err(error) if is_timeout(error.kind()) => {
            reject_stream_reliably(stream, REQUEST_TIMEOUT_RESPONSE);
            None
        }
        Err(error) => {
            let response = if error.kind() == io::ErrorKind::InvalidInput {
                BODY_TOO_LARGE_RESPONSE
            } else {
                BAD_REQUEST_RESPONSE
            };
            reject_stream_reliably(stream, response);
            None
        }
    }
}

fn enqueue_connection<R: Send + 'static>(
    mut stream: TcpStream,
    sender: &SyncSender<Incoming<R>>,
    incoming: Incoming<R>,
    response: ConnectionResponse,
) {
    let ConnectionResponse {
        output,
        receiver,
        cancellation,
        timeout,
        deadline,
        stop,
    } = response;
    match sender.try_send(incoming) {
        Ok(()) => {
            let watch_stop = Arc::new(AtomicBool::new(false));
            if let Ok(peer) = stream.try_clone() {
                spawn_disconnect_watcher(
                    peer,
                    Arc::clone(&output.state),
                    cancellation.clone(),
                    &watch_stop,
                );
            }
            let output_state = Arc::clone(&output.state);
            drop(output);
            write_loop(
                stream,
                WriteContext {
                    receiver,
                    output_state,
                    cancellation,
                    timeout,
                    watch_stop,
                    deadline,
                    stop,
                },
            )
        }
        Err(TrySendError::Full(_)) => {
            cancellation.cancel();
            reject_stream_reliably(&mut stream, OVERLOAD_RESPONSE);
        }
        Err(TrySendError::Disconnected(_)) => cancellation.cancel(),
    }
}

enum NextFrame {
    Frame(Frame),
    Deadline,
    Closed,
}

fn next_frame(
    receiver: &Receiver<Frame>,
    output_state: &OutputState,
    cancellation: &Cancellation,
    deadline: Instant,
    stop: &AtomicBool,
) -> NextFrame {
    loop {
        let now = Instant::now();
        if transport_cancelled(stop, output_state, cancellation, now, deadline) {
            return NextFrame::Closed;
        }
        if now >= deadline && !output_state.terminal_claimed.load(Ordering::Acquire) {
            return next_frame_after_deadline(receiver, output_state, deadline);
        }
        let limit = if now < deadline {
            deadline
        } else {
            terminal_deadline(output_state, deadline)
        };
        let Some(wait) = frame_wait(now, limit) else {
            return NextFrame::Deadline;
        };
        match receive_waiting_frame(receiver, output_state, wait, deadline) {
            Ok(Some(frame)) => return NextFrame::Frame(frame),
            Ok(None) => {}
            Err(()) => return NextFrame::Closed,
        }
    }
}

fn next_frame_after_deadline(
    receiver: &Receiver<Frame>,
    output_state: &OutputState,
    deadline: Instant,
) -> NextFrame {
    match receive_waiting_frame(receiver, output_state, Duration::ZERO, deadline) {
        Ok(Some(frame)) => NextFrame::Frame(frame),
        Ok(None) => NextFrame::Deadline,
        Err(()) => NextFrame::Closed,
    }
}

fn transport_cancelled(
    stop: &AtomicBool,
    output_state: &OutputState,
    cancellation: &Cancellation,
    now: Instant,
    deadline: Instant,
) -> bool {
    stop.load(Ordering::Acquire)
        || output_state.cancelled.load(Ordering::Acquire)
        || (now < deadline && cancellation.is_cancelled())
}

fn frame_wait(now: Instant, limit: Instant) -> Option<Duration> {
    let remaining = limit.saturating_duration_since(now);
    (!remaining.is_zero()).then_some(remaining.min(Duration::from_millis(10)))
}

fn receive_frame(receiver: &Receiver<Frame>, wait: Duration) -> Result<Option<Frame>, ()> {
    match receiver.recv_timeout(wait) {
        Ok(frame) => Ok(Some(frame)),
        Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(()),
    }
}

fn receive_waiting_frame(
    receiver: &Receiver<Frame>,
    output_state: &OutputState,
    wait: Duration,
    deadline: Instant,
) -> Result<Option<Frame>, ()> {
    match receive_frame(receiver, wait) {
        Ok(Some(frame)) => Ok(frame_after_deadline(frame, output_state, deadline)),
        Ok(None) => Ok(None),
        Err(()) => Err(()),
    }
}

fn frame_after_deadline(
    frame: Frame,
    output_state: &OutputState,
    deadline: Instant,
) -> Option<Frame> {
    match frame {
        Frame::Bytes { bytes, queued_at } if queued_at >= deadline => {
            output_state
                .pending_bytes
                .fetch_sub(bytes.len(), Ordering::AcqRel);
            None
        }
        frame => Some(frame),
    }
}

fn terminal_deadline(output_state: &OutputState, deadline: Instant) -> Instant {
    let Ok(mut terminal_deadline) = output_state.terminal_deadline.lock() else {
        return deadline;
    };
    *terminal_deadline.get_or_insert_with(|| {
        deadline
            .checked_add(TERMINAL_FRAME_GRACE)
            .unwrap_or(deadline)
    })
}

fn write_frame(
    stream: &mut TcpStream,
    bytes: &[u8],
    timeout: Duration,
    deadline: Instant,
    output_state: &OutputState,
) -> io::Result<()> {
    let mut offset = 0;
    while offset < bytes.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "response deadline expired",
            ));
        }
        stream.set_write_timeout(Some(remaining.min(timeout)))?;
        let attempt_started_ns = crate::clock::now_ns();
        match stream.write(&bytes[offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "response write stalled",
                ))
            }
            Ok(written) => offset += written,
            Err(error) if is_timeout(error.kind()) => {
                output_state.record_socket_blocked(attempt_started_ns, crate::clock::now_ns());
                if Instant::now() >= deadline {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn write_loop(mut stream: TcpStream, context: WriteContext) {
    let WriteContext {
        receiver,
        output_state,
        cancellation,
        timeout,
        watch_stop,
        deadline,
        stop,
    } = context;
    loop {
        match next_frame(&receiver, &output_state, &cancellation, deadline, &stop) {
            NextFrame::Frame(frame) => {
                if write_queued_frame(&mut stream, frame, &output_state, timeout, deadline).is_err()
                {
                    cancel_output(&output_state, &cancellation, &watch_stop);
                    return;
                }
            }
            NextFrame::Deadline => {
                match write_deadline_fallback(
                    &mut stream,
                    &output_state,
                    &cancellation,
                    &watch_stop,
                    timeout,
                    deadline,
                    &stop,
                ) {
                    DeadlineAction::Retry => {}
                    DeadlineAction::Close => break,
                    DeadlineAction::Finish => return,
                }
            }
            NextFrame::Closed => break,
        }
    }
    watch_stop.store(true, Ordering::Release);
}

fn write_queued_frame(
    stream: &mut TcpStream,
    frame: Frame,
    output_state: &OutputState,
    timeout: Duration,
    deadline: Instant,
) -> io::Result<()> {
    let frame_deadline = if Instant::now() < deadline {
        deadline
    } else {
        terminal_deadline(output_state, deadline)
    };
    let bytes = match frame {
        Frame::Bytes { bytes, .. } => bytes,
        Frame::Terminal(bytes) => bytes,
    };
    if let Err(error) = write_frame(stream, &bytes, timeout, frame_deadline, output_state) {
        output_state
            .delivery_failure_phase
            .store(DeliveryFailurePhase::SocketWrite as u8, Ordering::Release);
        return Err(error);
    }
    output_state.response_started.store(true, Ordering::Release);
    output_state
        .pending_bytes
        .fetch_sub(bytes.len(), Ordering::AcqRel);
    Ok(())
}

enum DeadlineAction {
    Retry,
    Close,
    Finish,
}

fn write_deadline_fallback(
    stream: &mut TcpStream,
    output_state: &OutputState,
    cancellation: &Cancellation,
    watch_stop: &AtomicBool,
    timeout: Duration,
    deadline: Instant,
    stop: &AtomicBool,
) -> DeadlineAction {
    if stop.load(Ordering::Acquire) || output_state.cancelled.load(Ordering::Acquire) {
        return DeadlineAction::Close;
    }
    if output_state.terminal_claimed.load(Ordering::Acquire)
        && Instant::now() >= terminal_deadline(output_state, deadline)
    {
        cancel_output(output_state, cancellation, watch_stop);
        return DeadlineAction::Finish;
    }
    if output_state
        .terminal_claimed
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return DeadlineAction::Retry;
    }
    let bytes = deadline_response(output_state);
    let terminal_limit = terminal_deadline(output_state, deadline);
    if write_frame(stream, &bytes, timeout, terminal_limit, output_state).is_err() {
        output_state
            .delivery_failure_phase
            .store(DeliveryFailurePhase::SocketWrite as u8, Ordering::Release);
        cancel_output(output_state, cancellation, watch_stop);
        return DeadlineAction::Finish;
    }
    output_state.response_started.store(true, Ordering::Release);
    cancel_output(output_state, cancellation, watch_stop);
    DeadlineAction::Finish
}

fn deadline_response(output_state: &OutputState) -> Vec<u8> {
    if output_state.streaming.load(Ordering::Acquire)
        && output_state.response_started.load(Ordering::Acquire)
    {
        return stream_timeout_response();
    }
    REQUEST_TIMEOUT_RESPONSE.to_vec()
}

fn stream_timeout_response() -> Vec<u8> {
    let mut response = Vec::with_capacity(128);
    append_chunk(
        &mut response,
        b"data: {\"error\":{\"message\":\"the request deadline expired\"}}\n\n",
    );
    append_chunk(&mut response, b"data: [DONE]\n\n");
    response.extend_from_slice(b"0\r\n\r\n");
    response
}

fn append_chunk(response: &mut Vec<u8>, payload: &[u8]) {
    response.extend_from_slice(format!("{:x}\r\n", payload.len()).as_bytes());
    response.extend_from_slice(payload);
    response.extend_from_slice(b"\r\n");
}

fn cancel_output(output_state: &OutputState, cancellation: &Cancellation, watch_stop: &AtomicBool) {
    output_state.cancelled.store(true, Ordering::Release);
    cancellation.cancel();
    watch_stop.store(true, Ordering::Release);
}

fn spawn_disconnect_watcher(
    stream: TcpStream,
    output_state: Arc<OutputState>,
    cancellation: Cancellation,
    stop: &Arc<AtomicBool>,
) {
    if stream
        .set_read_timeout(Some(Duration::from_millis(10)))
        .is_err()
    {
        return;
    }
    let stop = Arc::clone(stop);
    let _ = thread::Builder::new()
        .name("leone-transport-disconnect".to_owned())
        .spawn(move || {
            let mut byte = [0_u8; 1];
            while !stop.load(Ordering::Acquire) && !cancellation.is_cancelled() {
                match stream.peek(&mut byte) {
                    Ok(0) => break,
                    Ok(_) => thread::sleep(Duration::from_millis(2)),
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                        ) =>
                    {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => {
                        output_state.record_client_disconnect();
                        cancellation.cancel();
                        break;
                    }
                }
            }
        });
}

fn write_rejection_response(stream: &mut TcpStream, response: &[u8]) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_write_timeout(Some(REJECTION_WRITE_TIMEOUT))?;
    stream.write_all(response)
}

fn reject_stream_reliably(stream: &mut TcpStream, response: &[u8]) {
    if write_rejection_response(stream, response).is_err() {
        return;
    }
    let _ = stream.shutdown(Shutdown::Write);
    let deadline = request_deadline(REJECTION_DRAIN_TIMEOUT);
    let mut buffer = [0_u8; 4096];
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if stream.set_read_timeout(Some(remaining)).is_err() {
            return;
        }
        match stream.read(&mut buffer) {
            Ok(0) => return,
            Ok(_) => {}
            Err(error) if is_timeout(error.kind()) => return,
            Err(_) => return,
        }
    }
}

fn reject_stream_nonblocking(mut stream: TcpStream, response: &[u8]) {
    if stream.set_nonblocking(true).is_ok() {
        let _ = stream.write(response);
        let mut buffer = [0_u8; 4096];
        while matches!(stream.read(&mut buffer), Ok(count) if count > 0) {}
        let _ = stream.shutdown(Shutdown::Write);
    }
}

fn output_frame_capacity(max_bytes: usize) -> usize {
    max_bytes.clamp(1, MAX_OUTPUT_FRAMES)
}

fn update_high_watermark(counter: &AtomicUsize, value: usize) {
    let mut current = counter.load(Ordering::Acquire);
    while current < value {
        match counter.compare_exchange_weak(current, value, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

fn saturating_add(counter: &AtomicU64, value: u64) {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let next = current.saturating_add(value);
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

fn request_id() -> u64 {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    u64::try_from(NEXT.fetch_add(1, Ordering::Relaxed)).unwrap_or(u64::MAX)
}

fn request_deadline(timeout: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(timeout).unwrap_or(now)
}

fn transport_closed(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::BrokenPipe, message)
}

fn transport_overloaded(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::WouldBlock, message)
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn is_timeout(kind: io::ErrorKind) -> bool {
    matches!(kind, io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpStream;

    #[test]
    fn config_rejects_unbounded_zero_limits() {
        let config = Config {
            max_connections: 0,
            max_connections_per_client: 1,
            max_pending_requests: 1,
            max_output_bytes: 1,
            read_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
            trusted_proxy_ips: Vec::new(),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn output_sink_marks_cancellation_when_byte_budget_is_full() {
        let (mut sink, _receiver) = OutputSink::new(3, 2);
        assert_eq!(sink.write(b"abc").expect("first frame"), 3);
        assert!(sink.write(b"d").is_err());
        assert!(sink.is_cancelled());
    }

    #[test]
    fn output_sink_telemetry_separates_queue_and_socket() {
        let (mut sink, _receiver) = OutputSink::new(16, 2);
        sink.write_all(b"abc").expect("first frame");
        sink.write_all(b"defg").expect("second frame");
        let telemetry = sink.transport_telemetry();
        assert_eq!(telemetry.queue_high_water_bytes, 7);
        assert_eq!(telemetry.socket_blocked_ns, 0);
        assert_eq!(telemetry.socket_blocked_events, 0);
    }

    #[test]
    fn output_sink_records_delivery_failure_after_writer_rejects_frame() {
        let (mut sink, receiver) = OutputSink::new(16, 1);
        drop(receiver);
        assert!(sink.write_all(b"frame").is_err());
        let telemetry = sink.transport_telemetry();
        assert!(telemetry.delivery_failed);
        assert_eq!(
            telemetry.delivery_failure_phase,
            DeliveryFailurePhase::EnqueueRejected
        );
    }

    #[test]
    fn output_sink_retains_client_disconnect_timestamp() {
        let (sink, _receiver) = OutputSink::new(16, 1);
        sink.state.record_client_disconnect();
        let telemetry = sink.transport_telemetry();
        assert!(telemetry.client_disconnected_at_ns.is_some());
        assert!(sink.is_cancelled());
    }

    #[test]
    fn output_sink_pressure_intervals_have_a_fixed_bound() {
        let (sink, _receiver) = OutputSink::new(16, 1);
        for index in 0..(MAX_PRESSURE_INTERVALS + 2) {
            let start_ns = u64::try_from(index).expect("interval index") * 10;
            sink.state.record_socket_blocked(start_ns, start_ns + 3);
        }
        let telemetry = sink.transport_telemetry();
        assert_eq!(
            telemetry.socket_blocked_interval_count,
            MAX_PRESSURE_INTERVALS
        );
        assert_eq!(telemetry.socket_blocked_intervals_dropped, 2);
        assert_eq!(
            telemetry.socket_blocked_ns,
            (MAX_PRESSURE_INTERVALS as u64 + 2) * 3
        );
        assert_eq!(
            telemetry.socket_blocked_events,
            (MAX_PRESSURE_INTERVALS + 2) as u64
        );
        assert_eq!(telemetry.socket_blocked_intervals[0].start_ns, 0);
        assert_eq!(
            telemetry.socket_blocked_intervals[MAX_PRESSURE_INTERVALS - 1].start_ns,
            310
        );
    }

    #[test]
    fn terminal_deadline_grace_is_absolute() {
        let (sink, _receiver) = OutputSink::new(16, 1);
        let deadline = Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap();
        let first = terminal_deadline(&sink.state, deadline);
        thread::sleep(Duration::from_millis(5));
        let second = terminal_deadline(&sink.state, deadline);
        assert_eq!(first, second);
    }

    #[test]
    fn terminal_frame_survives_expired_queued_frames() {
        let (mut sink, receiver) = OutputSink::new(128, 3);
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(10))
            .expect("deadline");
        sink.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            .expect("header frame");
        sink.write_all(b"5\r\nhello\r\n").expect("content frame");
        thread::sleep(Duration::from_millis(20));
        assert!(sink.begin_terminal_response());
        sink.write_all(b"0\r\n\r\n").expect("terminal frame");
        let cancellation = Cancellation::new();
        let stop = AtomicBool::new(false);
        let header = expect_frame(next_frame(
            &receiver,
            &sink.state,
            &cancellation,
            deadline,
            &stop,
        ));
        let content = expect_frame(next_frame(
            &receiver,
            &sink.state,
            &cancellation,
            deadline,
            &stop,
        ));
        let terminal = expect_frame(next_frame(
            &receiver,
            &sink.state,
            &cancellation,
            deadline,
            &stop,
        ));
        assert!(matches!(header, Frame::Bytes { bytes, .. } if bytes.starts_with(b"HTTP/1.1 200")));
        assert!(matches!(content, Frame::Bytes { bytes, .. } if bytes == b"5\r\nhello\r\n"));
        assert!(matches!(terminal, Frame::Terminal(bytes) if bytes == b"0\r\n\r\n"));
    }

    #[test]
    fn canonical_ip_preserves_native_ipv6_and_maps_ipv4() {
        assert_eq!(
            canonical_ip("::1".parse().expect("IPv6 loopback")),
            "::1".parse::<IpAddr>().expect("IPv6 loopback")
        );
        assert_eq!(
            canonical_ip("::ffff:192.0.2.1".parse().expect("mapped IPv4")),
            "192.0.2.1".parse::<IpAddr>().expect("IPv4 address")
        );
    }

    #[test]
    fn slow_sender_does_not_hold_the_accept_loop() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 4,
                max_connections_per_client: 4,
                max_pending_requests: 4,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(100),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut slow = TcpStream::connect(transport.local_addr()).expect("slow client");
        slow.write_all(b"GET /health HTTP/1.1\r\n")
            .expect("partial request");
        let mut fast = TcpStream::connect(transport.local_addr()).expect("fast client");
        fast.write_all(b"GET /health HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        assert_eq!(incoming.request, b"GET /health HTTP/1.1");
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn accepted_socket_waits_for_delayed_request_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
        let mut client =
            TcpStream::connect(listener.local_addr().expect("listener address")).expect("client");
        let (mut accepted, _) = listener.accept().expect("accepted client");
        accepted
            .set_nonblocking(true)
            .expect("force accepted socket mode");
        let config = Config {
            max_connections: 1,
            max_connections_per_client: 1,
            max_pending_requests: 1,
            max_output_bytes: 1024,
            read_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
            trusted_proxy_ips: Vec::new(),
        };
        assert!(configure_stream(&accepted, &config));
        client
            .write_all(b"GET /health HTTP/1.1")
            .expect("request prefix");
        let writer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            client.write_all(b"\r\n\r\n").expect("request terminator");
        });
        let result = read_test_request(
            &mut accepted,
            Instant::now()
                .checked_add(Duration::from_secs(1))
                .expect("deadline"),
        );
        writer.join().expect("delayed request writer");
        assert_eq!(
            result.expect("delayed request"),
            Some(b"GET /health HTTP/1.1".to_vec())
        );
    }

    #[test]
    fn trickling_sender_hits_absolute_deadline() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(100),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut slow = TcpStream::connect(transport.local_addr()).expect("slow client");
        slow.write_all(b"G").expect("first byte");
        thread::sleep(Duration::from_millis(150));
        slow.set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        slow.read_to_string(&mut response)
            .expect("timeout response");
        assert!(response.starts_with("HTTP/1.1 408 Request Timeout"));
        assert!(response.contains("transport-deadline"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn terminal_deadline_response_survives_writer_deadline() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /deadline HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        let mut output = incoming.output.clone();
        assert!(output.begin_terminal_response());
        thread::sleep(Duration::from_millis(75));
        incoming.cancellation.cancel();
        output
            .write_all(b"HTTP/1.1 408 Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .expect("terminal response");
        drop(output);
        drop(incoming);
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert!(response.starts_with("HTTP/1.1 408 Error\r\n"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn deadline_fallback_writes_nonstream_error_while_executor_is_held() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /held HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        thread::sleep(Duration::from_millis(75));
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert!(response.starts_with("HTTP/1.1 408 Request Timeout\r\n"));
        assert!(incoming.cancellation.is_cancelled());
        drop(incoming);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn deadline_fallback_finishes_stream_after_queued_content() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 4096,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /held-stream HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        let mut output = incoming.output.clone();
        output.mark_streaming();
        let header = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
        let role = b"5\r\nhello\r\n";
        output.write_all(header).expect("headers");
        output.write_all(role).expect("content");
        thread::sleep(Duration::from_millis(75));
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = Vec::new();
        client.read_to_end(&mut response).expect("response");
        assert!(response.starts_with(header));
        assert!(response.windows(role.len()).any(|window| window == role));
        assert!(response.ends_with(b"0\r\n\r\n"));
        assert!(response
            .windows(b"data: [DONE]\n\n".len())
            .any(|window| { window == b"data: [DONE]\n\n" }));
        drop(output);
        drop(incoming);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn claimed_terminal_closes_after_grace_and_releases_connection() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 1,
                max_connections_per_client: 1,
                max_pending_requests: 1,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /held-terminal HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        assert!(incoming.output.begin_terminal_response());
        thread::sleep(Duration::from_millis(350));
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = Vec::new();
        client.read_to_end(&mut response).expect("response");
        assert!(response.is_empty());
        drop(incoming);

        let mut next = TcpStream::connect(transport.local_addr()).expect("next client");
        next.write_all(b"GET /next HTTP/1.1\r\n\r\n")
            .expect("next request");
        let next_request = receive_request(&transport);
        assert_eq!(next_request.request, b"GET /next HTTP/1.1");
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn completed_nonstream_response_does_not_get_a_second_timeout() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /complete HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        let mut output = incoming.output.clone();
        assert!(output.begin_terminal_response());
        output
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            .expect("response");
        thread::sleep(Duration::from_millis(350));
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert_eq!(response.matches("HTTP/1.1").count(), 1);
        assert!(!response.contains("408 Request Timeout"));
        drop(output);
        drop(incoming);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn completed_stream_response_has_one_terminal_sequence() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 4096,
                read_timeout: Duration::from_millis(50),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /complete-stream HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        let mut output = incoming.output.clone();
        output.mark_streaming();
        assert!(output.begin_terminal_response());
        output
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            .expect("headers");
        output.write_all(b"5\r\nhello\r\n").expect("content");
        output
            .write_all(b"e\r\ndata: [DONE]\n\n\r\n")
            .expect("done");
        output.write_all(b"0\r\n\r\n").expect("end");
        thread::sleep(Duration::from_millis(350));
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert_eq!(response.matches("HTTP/1.1").count(), 1);
        assert_eq!(response.matches("data: [DONE]").count(), 1);
        assert_eq!(response.matches("0\r\n\r\n").count(), 1);
        drop(output);
        drop(incoming);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn malformed_request_returns_typed_bad_request() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            malformed_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"not an HTTP request\r\n\r\n")
            .expect("request");
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(response.contains("transport-invalid-request"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn slow_reader_does_not_block_other_request() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 4,
                max_connections_per_client: 4,
                max_pending_requests: 4,
                max_output_bytes: 4096,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let slow = TcpStream::connect(transport.local_addr()).expect("slow client");
        slow.set_nodelay(true).expect("nodelay");
        slow.try_clone()
            .expect("slow clone")
            .write_all(b"GET /slow HTTP/1.1\r\n\r\n")
            .expect("slow request");
        let incoming = receive_request(&transport);
        let mut output = incoming.output.clone();
        let frame = [b'x'; 1024];
        let started = Instant::now();
        for _ in 0..16 {
            let _ = output.write(&frame);
        }
        assert!(started.elapsed() < Duration::from_millis(100));

        let mut fast = TcpStream::connect(transport.local_addr()).expect("fast client");
        fast.write_all(b"GET /fast HTTP/1.1\r\n\r\n")
            .expect("fast request");
        let fast_request = receive_request(&transport);
        assert_eq!(fast_request.request, b"GET /fast HTTP/1.1");
        drop(output);
        drop(incoming);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn slow_reader_records_socket_backpressure_separately() {
        let stop = Arc::new(AtomicBool::new(false));
        let max_output_bytes = 16_usize << 20;
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes,
                read_timeout: Duration::from_secs(3),
                write_timeout: Duration::from_millis(25),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut slow = TcpStream::connect(transport.local_addr()).expect("slow client");
        slow.set_nodelay(true).expect("nodelay");
        slow.write_all(b"GET /telemetry HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        let payload = vec![b'x'; max_output_bytes];
        incoming
            .output
            .clone()
            .write_all(&payload)
            .expect("queue response");
        let wait_until = Instant::now()
            .checked_add(Duration::from_secs(1))
            .expect("telemetry deadline");
        while Instant::now() < wait_until {
            if incoming.output.transport_telemetry().socket_blocked_events > 0 {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let telemetry = incoming.output.transport_telemetry();
        assert!(telemetry.queue_high_water_bytes >= max_output_bytes as u64);
        assert!(telemetry.socket_blocked_events > 0, "{telemetry:?}");
        assert!(telemetry.socket_blocked_ns > 0, "{telemetry:?}");
        assert!(telemetry.socket_blocked_interval_count > 0);
        let intervals =
            &telemetry.socket_blocked_intervals[..telemetry.socket_blocked_interval_count];
        assert!(intervals
            .iter()
            .all(|interval| interval.start_ns <= interval.end_ns));
        assert_eq!(
            telemetry.socket_blocked_events,
            telemetry.socket_blocked_interval_count as u64
                + telemetry.socket_blocked_intervals_dropped
        );
        drop(incoming);
        drop(slow);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn dropped_consumer_cancels_request() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /drop HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        drop(client);
        let mut output = incoming.output.clone();
        for _ in 0..100 {
            if incoming.cancellation.is_cancelled() {
                stop.store(true, Ordering::Release);
                return;
            }
            let _ = output.write_all(b"response");
            thread::sleep(Duration::from_millis(2));
        }
        panic!("dropped client did not cancel request");
    }

    #[test]
    fn disconnected_response_does_not_stop_next_request() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /closed HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        drop(client);
        let mut output = incoming.output.clone();
        for _ in 0..100 {
            if incoming.cancellation.is_cancelled() {
                break;
            }
            let _ = output.write_all(b"response");
            thread::sleep(Duration::from_millis(2));
        }
        assert!(output.write_all(b"late response").is_err());
        drop(output);
        drop(incoming);

        let mut next = TcpStream::connect(transport.local_addr()).expect("next client");
        next.write_all(b"GET /next HTTP/1.1\r\n\r\n")
            .expect("next request");
        let next_request = receive_request(&transport);
        assert_eq!(next_request.request, b"GET /next HTTP/1.1");
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn queued_frame_after_deadline_gets_typed_timeout() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_millis(100),
                write_timeout: Duration::from_millis(50),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .write_all(b"GET /late HTTP/1.1\r\n\r\n")
            .expect("request");
        let incoming = receive_request(&transport);
        thread::sleep(Duration::from_millis(150));
        let mut output = incoming.output.clone();
        let _ = output.write_all(b"late response");
        drop(output);
        drop(incoming);
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        client.read_to_string(&mut response).expect("response");
        assert!(response.starts_with("HTTP/1.1 408 Request Timeout\r\n"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn client_write_half_close_preserves_response_and_releases_worker() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 1,
                max_connections_per_client: 1,
                max_pending_requests: 1,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut first = TcpStream::connect(transport.local_addr()).expect("first client");
        first
            .write_all(b"GET /first HTTP/1.1\r\n\r\n")
            .expect("first request");
        let incoming = receive_request(&transport);
        first
            .shutdown(Shutdown::Write)
            .expect("client write half close");
        thread::sleep(Duration::from_millis(30));
        assert!(!incoming.cancellation.is_cancelled());
        let mut output = incoming.output.clone();
        output.write_all(b"ok").expect("response");
        drop(output);
        drop(incoming);
        first
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut first_response = Vec::new();
        first
            .read_to_end(&mut first_response)
            .expect("response close");
        assert_eq!(first_response, b"ok");

        let mut second = TcpStream::connect(transport.local_addr()).expect("second client");
        second
            .write_all(b"GET /second HTTP/1.1\r\n\r\n")
            .expect("second request");
        let second_request = receive_request(&transport);
        assert_eq!(second_request.request, b"GET /second HTTP/1.1");
        drop(second_request);
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn rejected_connection_does_not_block_accept_loop() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 1,
                max_connections_per_client: 1,
                max_pending_requests: 1,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_secs(1),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let slow = TcpStream::connect(transport.local_addr()).expect("slow client");
        slow.try_clone()
            .expect("slow clone")
            .write_all(b"G")
            .expect("partial request");
        let mut rejected = TcpStream::connect(transport.local_addr()).expect("rejected client");
        rejected
            .write_all(b"GET /rejected HTTP/1.1\r\n\r\n")
            .expect("rejected request");
        rejected
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        rejected
            .read_to_string(&mut response)
            .expect("overload response");
        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests"));
        assert!(response.contains("transport-connection-limit"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn per_client_connection_limit_returns_typed_overload() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 4,
                max_connections_per_client: 1,
                max_pending_requests: 4,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut first = TcpStream::connect(transport.local_addr()).expect("first client");
        first
            .write_all(b"GET /first HTTP/1.1\r\n\r\n")
            .expect("first request");
        let _first_request = receive_request(&transport);
        let mut rejected = TcpStream::connect(transport.local_addr()).expect("second client");
        rejected
            .write_all(b"GET /rejected HTTP/1.1\r\n\r\n")
            .expect("second request");
        rejected
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        let mut response = String::new();
        rejected
            .read_to_string(&mut response)
            .expect("overload response");
        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests"));
        assert!(response.contains("transport-client-limit"));
        stop.store(true, Ordering::Release);
    }

    #[test]
    fn dropping_transport_closes_trickling_connection() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 2,
                max_connections_per_client: 2,
                max_pending_requests: 2,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(10),
                write_timeout: Duration::from_secs(1),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut client = TcpStream::connect(transport.local_addr()).expect("client");
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .expect("timeout");
        client.write_all(b"G").expect("partial request");
        drop(transport);
        let mut response = [0_u8; 1];
        let result = client.read(&mut response);
        let closed = match result {
            Ok(0) => true,
            Err(error) => matches!(
                error.kind(),
                io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::UnexpectedEof
            ),
            Ok(_) => false,
        };
        assert!(closed);
    }

    #[test]
    fn full_input_queue_returns_typed_overload_and_recovers() {
        let stop = Arc::new(AtomicBool::new(false));
        let transport = Transport::bind(
            "127.0.0.1:0".parse().expect("address"),
            Config {
                max_connections: 4,
                max_connections_per_client: 4,
                max_pending_requests: 1,
                max_output_bytes: 1024,
                read_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_millis(100),
                trusted_proxy_ips: Vec::new(),
            },
            Arc::clone(&stop),
            read_test_request,
            test_identity,
        )
        .expect("transport");
        let mut first = TcpStream::connect(transport.local_addr()).expect("first client");
        first
            .write_all(b"GET /one HTTP/1.1\r\n\r\n")
            .expect("first request");
        let mut second = TcpStream::connect(transport.local_addr()).expect("second client");
        second
            .write_all(b"GET /two HTTP/1.1\r\n\r\n")
            .expect("second request");
        // Keep the admitted request queued until the other worker rejects overload.
        let first_response = read_optional_response(&mut first);
        let second_response = read_optional_response(&mut second);
        let response = format!("{first_response}{second_response}");
        assert!(response.contains("server_overloaded"));
        let _ = receive_request(&transport);
        drop(first);
        drop(second);

        let mut third = TcpStream::connect(transport.local_addr()).expect("third client");
        third
            .write_all(b"GET /three HTTP/1.1\r\n\r\n")
            .expect("third request");
        let incoming = receive_request(&transport);
        assert_eq!(incoming.request, b"GET /three HTTP/1.1");
        stop.store(true, Ordering::Release);
    }

    fn read_optional_response(stream: &mut TcpStream) -> String {
        let mut response = String::new();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        let _ = stream.read_to_string(&mut response);
        response
    }

    fn receive_request(transport: &Transport<Vec<u8>>) -> Incoming<Vec<u8>> {
        for _ in 0..100 {
            if let Some(request) = transport.try_recv().expect("transport receive") {
                return request;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("request did not arrive");
    }

    fn expect_frame(next: NextFrame) -> Frame {
        match next {
            NextFrame::Frame(frame) => frame,
            NextFrame::Deadline => panic!("frame deadline expired"),
            NextFrame::Closed => panic!("frame receiver closed"),
        }
    }

    fn read_test_request(stream: &mut TcpStream, deadline: Instant) -> io::Result<Option<Vec<u8>>> {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 64];
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "request deadline expired",
                ));
            }
            stream.set_read_timeout(Some(remaining))?;
            let read = stream.read(&mut buffer)?;
            if read == 0 {
                return Ok(None);
            }
            bytes.extend_from_slice(&buffer[..read]);
            if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                bytes.truncate(bytes.len().saturating_sub(4));
                return Ok(Some(bytes));
            }
        }
    }

    fn malformed_request(
        stream: &mut TcpStream,
        _deadline: Instant,
    ) -> io::Result<Option<Vec<u8>>> {
        let mut bytes = [0_u8; 64];
        let _ = stream.read(&mut bytes)?;
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HTTP request line is missing",
        ))
    }

    fn test_identity(_request: &Vec<u8>, peer: SocketAddr, _trusted: &[IpAddr]) -> IpAddr {
        peer.ip()
    }
}
