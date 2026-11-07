use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{broadcast, watch};

#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot { pub requests: u64, pub hits: u64, pub misses: u64, pub last_latency_micros: u64, pub estimated_savings_microusd: u64 }
#[derive(Clone, Debug)]
pub enum OperationalEvent { CacheHit, CacheMiss, CacheInsert, Fault(&'static str), UpstreamFailure }
pub struct Metrics { requests: AtomicU64, hits: AtomicU64, misses: AtomicU64, latency: AtomicU64, savings: AtomicU64, snapshot: watch::Sender<MetricsSnapshot>, events: broadcast::Sender<OperationalEvent> }
impl Default for Metrics { fn default() -> Self { Self::new() } }
impl Metrics {
    pub fn new() -> Self { let (snapshot,_) = watch::channel(MetricsSnapshot::default()); let (events,_) = broadcast::channel(256); Self { requests:AtomicU64::new(0),hits:AtomicU64::new(0),misses:AtomicU64::new(0),latency:AtomicU64::new(0),savings:AtomicU64::new(0),snapshot,events } }
    pub fn subscribe(&self)->watch::Receiver<MetricsSnapshot>{self.snapshot.subscribe()}
    pub fn subscribe_events(&self)->broadcast::Receiver<OperationalEvent>{self.events.subscribe()}
    pub fn event(&self,event:OperationalEvent){let _=self.events.send(event);}
    pub fn record(&self,elapsed:std::time::Duration,hit:Option<bool>){self.requests.fetch_add(1,Ordering::Relaxed);self.latency.store(elapsed.as_micros().min(u64::MAX as u128) as u64,Ordering::Relaxed);if let Some(value)=hit{if value{self.hits.fetch_add(1,Ordering::Relaxed);}else{self.misses.fetch_add(1,Ordering::Relaxed);}}self.publish();}
    pub fn snapshot(&self)->MetricsSnapshot{MetricsSnapshot{requests:self.requests.load(Ordering::Relaxed),hits:self.hits.load(Ordering::Relaxed),misses:self.misses.load(Ordering::Relaxed),last_latency_micros:self.latency.load(Ordering::Relaxed),estimated_savings_microusd:self.savings.load(Ordering::Relaxed)}}
    pub fn add_savings(&self,value:u64){self.savings.fetch_add(value,Ordering::Relaxed);self.publish();}
    fn publish(&self){self.snapshot.send_replace(self.snapshot());}
}

/// A dedicated blocking worker owns the terminal; no proxy executor worker
/// performs synchronous terminal I/O. It exits when the watch channel closes.
pub fn run_dashboard(mut snapshots:watch::Receiver<MetricsSnapshot>,mut events:broadcast::Receiver<OperationalEvent>){
    use crossterm::{event::{self,Event,KeyCode},execute,terminal::{disable_raw_mode,enable_raw_mode,EnterAlternateScreen,LeaveAlternateScreen}};
    use ratatui::{backend::CrosstermBackend,layout::{Constraint,Direction,Layout},widgets::{Block,Borders,Paragraph},Terminal};
    use std::{io, time::Duration};
    let setup=(||->io::Result<Terminal<CrosstermBackend<io::Stdout>>>{enable_raw_mode()?;let mut stdout=io::stdout();execute!(stdout,EnterAlternateScreen)?;Terminal::new(CrosstermBackend::new(stdout))})();
    let Ok(mut terminal)=setup else {let _=disable_raw_mode();return};
    let mut tail=std::collections::VecDeque::<String>::with_capacity(12);
    loop {
        while let Ok(event)=events.try_recv(){if tail.len()==12{tail.pop_front();}tail.push_back(format!("{event:?}"));}
        let snapshot=snapshots.borrow_and_update().clone();
        let rendered=terminal.draw(|frame|{let areas=Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(3),Constraint::Length(3),Constraint::Min(5)]).split(frame.area());frame.render_widget(Paragraph::new(format!("Requests {} | cache hits {} misses {} | latest overhead {} us",snapshot.requests,snapshot.hits,snapshot.misses,snapshot.last_latency_micros)).block(Block::default().title("AegisLLM").borders(Borders::ALL)),areas[0]);frame.render_widget(Paragraph::new(format!("Estimated savings ${:.6}",snapshot.estimated_savings_microusd as f64/1_000_000.0)).block(Block::default().title("Savings").borders(Borders::ALL)),areas[1]);frame.render_widget(Paragraph::new(tail.iter().cloned().collect::<Vec<_>>().join("\n")).block(Block::default().title("Recent events (q to quit)").borders(Borders::ALL)),areas[2]);});
        if rendered.is_err(){break;}
        if event::poll(Duration::from_millis(100)).unwrap_or(false){if let Ok(Event::Key(key))=event::read(){if key.code==KeyCode::Char('q'){break;}}}
        if snapshots.has_changed().is_err(){break;}
    }
    let _=disable_raw_mode();let _=execute!(terminal.backend_mut(),LeaveAlternateScreen);let _=terminal.show_cursor();
}
