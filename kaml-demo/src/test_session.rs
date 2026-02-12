use std::sync::Arc;

use kaml::*;
use serde_json::json;

struct TestTools;

#[async_trait::async_trait]
impl ToolProvider for TestTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "echo".into(),
                description: "Echo the input text back".into(),
                params: vec![ToolParam::typed("text", "str")],
                returns: "str".into(),
            },
            ToolDefinition {
                name: "add".into(),
                description: "Add two numbers".into(),
                params: vec![
                    ToolParam::typed("a", "float"),
                    ToolParam::typed("b", "float"),
                ],
                returns: "float".into(),
            },
            ToolDefinition {
                name: "search_web".into(),
                description: "Search the web".into(),
                params: vec![ToolParam::typed("query", "str")],
                returns: "list".into(),
            },
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        match name {
            "echo" => ToolResult {
                success: true,
                result: args.get("text").cloned().unwrap_or(json!("")),
            },
            "add" => {
                let a = args["a"].as_f64().unwrap_or(0.0);
                let b = args["b"].as_f64().unwrap_or(0.0);
                ToolResult {
                    success: true,
                    result: json!(a + b),
                }
            }
            "search_web" => ToolResult {
                success: true,
                result: json!([{
                    "title": format!("Result for: {}", args["query"].as_str().unwrap_or("")),
                    "url": "https://example.com"
                }]),
            },
            _ => ToolResult {
                success: false,
                result: json!(format!("Unknown tool: {name}")),
            },
        }
    }
}

macro_rules! run {
    ($session:expr, $code:expr) => {{
        let exec = $session.run_code($code).await.expect("run_code failed");
        if let Some(err) = &exec.error {
            panic!("Code failed:\n{}\nError:\n{}", $code, err);
        }
        exec
    }};
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let tools: Arc<dyn ToolProvider> = Arc::new(TestTools);
    println!("Creating native Python session...");
    let mut session = Session::new(tools, SessionConfig::default()).await?;
    println!("Session created!\n");

    // Test 1: Simple await tool call
    println!("=== Test 1: Simple await ===");
    let exec = run!(
        session,
        "result = await echo(text=\"hello from async\")\nprint(result)"
    );
    assert!(
        exec.output.contains("hello from async"),
        "Expected 'hello from async' in output, got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 2: await with expression auto-print
    println!("=== Test 2: Await expression auto-print ===");
    let exec = run!(session, "await add(a=1, b=2)");
    assert!(
        exec.output.contains("3.0"),
        "Expected '3.0' in output, got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 3: Variable persistence across await calls
    println!("=== Test 3: Variable persistence ===");
    run!(session, "x = await add(a=10, b=20)");
    let exec = run!(session, "print(x)");
    assert!(
        exec.output.contains("30.0"),
        "Expected '30.0' in output, got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 4: asyncio.gather for concurrent calls
    println!("=== Test 4: asyncio.gather ===");
    let start = std::time::Instant::now();
    let exec = run!(
        session,
        "import asyncio\nr = await asyncio.gather(add(a=1, b=2), add(a=3, b=4))\nprint(r)"
    );
    let elapsed = start.elapsed();
    println!("Duration: {:?}", elapsed);
    assert!(
        exec.output.contains("3.0") && exec.output.contains("7.0"),
        "Expected '3.0' and '7.0' in output, got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 5: message(kind="final") works with async
    println!("=== Test 5: message(kind=\"final\") with async ===");
    {
        let (msg_tx, mut msg_rx) =
            tokio::sync::mpsc::unbounded_channel::<kaml::SandboxMessage>();
        session.set_message_sender(msg_tx);
        let exec = run!(
            session,
            "r = await echo(text=\"world\")\nmessage(f\"Hello {r}\", kind=\"final\")"
        );
        assert!(
            exec.response.contains("Hello world"),
            "Expected 'Hello world' in response, got: {}",
            exec.response
        );
        let msg = msg_rx
            .try_recv()
            .expect("Expected a message on the channel");
        assert_eq!(msg.kind, "final");
        assert!(msg.text.contains("Hello world"));
    }
    println!("PASS\n");

    // Test 5b: message(kind="progress") streams without stopping
    println!("=== Test 5b: message(kind=\"progress\") ===");
    {
        let (msg_tx, mut msg_rx) =
            tokio::sync::mpsc::unbounded_channel::<kaml::SandboxMessage>();
        session.set_message_sender(msg_tx);
        let exec = run!(
            session,
            "message(\"Working on it...\", kind=\"progress\")\nx = 42"
        );
        assert!(
            exec.response.is_empty(),
            "Expected empty response, got: {}",
            exec.response
        );
        let msg = msg_rx
            .try_recv()
            .expect("Expected a progress message on the channel");
        assert_eq!(msg.kind, "progress");
        assert_eq!(msg.text, "Working on it...");
    }
    println!("PASS\n");

    // Test 6: Sync code still works (no await)
    println!("=== Test 6: Sync code still works ===");
    let exec = run!(session, "y = 42\nprint(y + 1)");
    assert!(
        exec.output.contains("43"),
        "Expected '43' in output, got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 7: Write file to scratch dir
    println!("=== Test 7: Write file ===");
    let scratch = session.scratch_path().to_string_lossy().to_string();
    run!(
        session,
        &format!(
            "with open('{}/hello.txt', 'w') as f:\n    f.write('hello world')",
            scratch
        )
    );
    let exec = run!(
        session,
        &format!(
            "with open('{}/hello.txt', 'r') as f:\n    print(f.read())",
            scratch
        )
    );
    assert!(
        exec.output.contains("hello world"),
        "Expected 'hello world', got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 8: Write JSON file and read it back
    println!("=== Test 8: JSON file round-trip ===");
    run!(
        session,
        &format!(
            "import json\ndata = {{'name': 'test', 'values': [1, 2, 3]}}\nwith open('{}/data.json', 'w') as f:\n    json.dump(data, f)",
            scratch
        )
    );
    let exec = run!(
        session,
        &format!(
            "with open('{}/data.json') as f:\n    loaded = json.load(f)\nprint(loaded)",
            scratch
        )
    );
    assert!(exec.output.contains("test") && exec.output.contains("[1, 2, 3]"));
    println!("PASS\n");

    // Test 9: os.listdir on scratch
    println!("=== Test 9: os.listdir ===");
    let exec = run!(
        session,
        &format!("import os\nprint(sorted(os.listdir('{}')))", scratch)
    );
    assert!(
        exec.output.contains("data.json") && exec.output.contains("hello.txt"),
        "Expected both files in listing, got: {}",
        exec.output
    );
    println!("PASS\n");

    // Test 10: Create subdirectory and write file
    println!("=== Test 10: Subdirectory ===");
    run!(
        session,
        &format!(
            "import os\nos.makedirs('{}/sub/dir', exist_ok=True)\nwith open('{}/sub/dir/nested.txt', 'w') as f:\n    f.write('nested content')",
            scratch, scratch
        )
    );
    let exec = run!(
        session,
        &format!(
            "with open('{}/sub/dir/nested.txt') as f:\n    print(f.read())",
            scratch
        )
    );
    assert!(exec.output.contains("nested content"));
    println!("PASS\n");

    // Test 11: pathlib
    println!("=== Test 11: pathlib ===");
    let exec = run!(
        session,
        &format!(
            "from pathlib import Path\np = Path('{}/pathlib_test.txt')\np.write_text('pathlib works')\nprint(p.read_text())",
            scratch
        )
    );
    assert!(exec.output.contains("pathlib works"));
    println!("PASS\n");

    // Test 12: Snapshot includes files
    println!("=== Test 12: Snapshot includes files ===");
    let snap = session.snapshot().await?;
    println!("Snapshot size: {} bytes", snap.len());
    let snap_json: serde_json::Value = serde_json::from_slice(&snap)?;
    assert!(
        snap_json.get("vars").is_some(),
        "Missing 'vars' in snapshot"
    );
    assert!(
        snap_json["vars"]["x"] == json!(42),
        "Expected x=42 in vars (set by Test 5b)"
    );
    let files = snap_json.get("files").expect("Missing 'files' in snapshot");
    assert!(
        files.get("hello.txt").is_some(),
        "Missing hello.txt in snapshot files"
    );
    assert_eq!(files["hello.txt"], "hello world");
    assert!(
        files.get("data.json").is_some(),
        "Missing data.json in snapshot files"
    );
    assert!(
        files.get("sub/dir/nested.txt").is_some(),
        "Missing sub/dir/nested.txt in snapshot files"
    );
    assert_eq!(files["sub/dir/nested.txt"], "nested content");
    println!("PASS\n");

    // Test 13: Restore snapshot into fresh session
    println!("=== Test 13: Restore snapshot ===");
    let tools2: Arc<dyn ToolProvider> = Arc::new(TestTools);
    let mut session2 = Session::new(tools2, SessionConfig::default()).await?;
    session2.restore(&snap).await?;

    let scratch2 = session2.scratch_path().to_string_lossy().to_string();

    // Check vars restored
    let exec = run!(session2, "print(x)");
    assert!(
        exec.output.contains("42"),
        "Expected x=42 after restore, got: {}",
        exec.output
    );

    // Check files restored
    let exec = run!(
        session2,
        &format!(
            "with open('{}/hello.txt') as f:\n    print(f.read())",
            scratch2
        )
    );
    assert!(
        exec.output.contains("hello world"),
        "Expected 'hello world' after restore, got: {}",
        exec.output
    );

    // Check nested file restored
    let exec = run!(
        session2,
        &format!(
            "with open('{}/sub/dir/nested.txt') as f:\n    print(f.read())",
            scratch2
        )
    );
    assert!(
        exec.output.contains("nested content"),
        "Expected 'nested content' after restore, got: {}",
        exec.output
    );

    // Check JSON file restored and parseable
    let exec = run!(
        session2,
        &format!(
            "import json\nwith open('{}/data.json') as f:\n    d = json.load(f)\nprint(d['name'], d['values'])",
            scratch2
        )
    );
    assert!(exec.output.contains("test") && exec.output.contains("[1, 2, 3]"));
    println!("PASS\n");

    println!("All tests passed!");
    Ok(())
}
