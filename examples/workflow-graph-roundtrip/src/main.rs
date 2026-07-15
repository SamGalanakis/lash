use std::net::SocketAddr;

use anyhow::{Context, Result};
use workflow_graph_roundtrip::AppState;

#[tokio::main]
async fn main() -> Result<()> {
    let addr: SocketAddr = std::env::var("WORKFLOW_GRAPH_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3031".to_string())
        .parse()
        .context("invalid WORKFLOW_GRAPH_ADDR")?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("bind workflow graph listener")?;
    println!("workflow-graph-roundtrip listening on http://{addr}");
    workflow_graph_roundtrip::serve(listener, AppState::new()?)
        .await
        .context("serve workflow graph backend")
}
