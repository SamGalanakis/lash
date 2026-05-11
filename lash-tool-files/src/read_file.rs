use serde_json::json;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use lash::instructions::InstructionSource;
use lash::plugin::{
    PluginDirective, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use lash::{
    MessageRole, PluginMessage, ToolCall, ToolDefinition, ToolExecutionMode, ToolImage,
    ToolProvider, ToolResult,
};

use lash_tool_support::{object_schema, parse_optional_usize_arg, require_str, run_blocking};

/// Read files with line-number-prefixed output. Supports images natively.
#[derive(Default)]
pub struct ReadFile;

pub struct ReadFilePluginFactory {
    instruction_source: Option<Arc<dyn InstructionSource>>,
}

struct ReadFilePlugin {
    provider: Arc<ReadFile>,
    instruction_source: Option<Arc<dyn InstructionSource>>,
}

impl ReadFile {
    pub fn new() -> Self {
        Self
    }
}

impl ReadFilePluginFactory {
    pub fn new(instruction_source: Option<Arc<dyn InstructionSource>>) -> Self {
        Self { instruction_source }
    }
}

impl PluginFactory for ReadFilePluginFactory {
    fn id(&self) -> &'static str {
        "read_file"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ReadFilePlugin {
            provider: Arc::new(ReadFile::new()),
            instruction_source: self.instruction_source.clone(),
        }))
    }
}

impl SessionPlugin for ReadFilePlugin {
    fn id(&self) -> &'static str {
        "read_file"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;

        let Some(instruction_source) = self.instruction_source.clone() else {
            return Ok(());
        };

        reg.tool_calls().after(Arc::new(move |ctx| {
            let instruction_source = Arc::clone(&instruction_source);
            Box::pin(async move {
                if !ctx.result.success || ctx.tool_name != "read_file" {
                    return Ok(Vec::new());
                }

                let Some(path) = ctx.args.get("path").and_then(|value| value.as_str()) else {
                    return Ok(Vec::new());
                };
                if path.is_empty() {
                    return Ok(Vec::new());
                }

                let instructions =
                    instruction_source.context_instructions_for_reads(&[path.to_string()]);
                if instructions.trim().is_empty() {
                    return Ok(Vec::new());
                }

                Ok(vec![PluginDirective::EnqueueMessages {
                    messages: vec![PluginMessage::text(MessageRole::System, instructions)],
                }])
            })
        }));

        Ok(())
    }
}

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_LEN: usize = 2000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const MAX_OUTPUT_BYTES_LABEL: &str = "50 KB";

#[async_trait::async_trait]
impl ToolProvider for ReadFile {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::raw(
                "read_file",
                "Read a file. Text returns lines prefixed as `LINE: text`, PDFs return extracted text, and images return visual content. Default: 2000 lines. Use `ls` for directories.",
                object_schema(
                    serde_json::json!({
                        "path": { "type": "string" },
                        "offset": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Line offset to start reading from (1-based)"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "default": DEFAULT_LIMIT,
                            "description": "Maximum lines to read (default: 2000)."
                        }
                    }),
                    &["path"],
                ),
                serde_json::json!({ "type": "string" }),
            )
            .with_examples(vec![
                r#"read_file(path="Cargo.toml")"#.into(),
                r#"read_file(path="src/main.rs", offset=1, limit=120)"#.into(),
            ])
            .with_discovery(lash_tool_support::discovery_metadata("filesystem", &["cat", "view_file"]))
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let args = call.args;
        let path_str = match require_str(args, "path") {
            Ok(s) => s.to_string(),
            Err(e) => return e,
        };

        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1)
            .max(1);

        let limit = match parse_limit(args) {
            Ok(limit) => limit,
            Err(e) => return e,
        };

        run_blocking(move || execute_read_file_sync(&path_str, offset, limit)).await
    }
}

fn parse_limit(args: &serde_json::Value) -> Result<usize, ToolResult> {
    Ok(
        parse_optional_usize_arg(args, "limit", Some(DEFAULT_LIMIT), false, 1)?
            .unwrap_or(DEFAULT_LIMIT),
    )
}

fn execute_read_file_sync(path_str: &str, offset: usize, limit: usize) -> ToolResult {
    let path = Path::new(path_str);
    if !path.exists() {
        return ToolResult::err_fmt(format_args!(
            "Path does not exist: {path_str}. Use `ls` or `glob` to locate the correct path."
        ));
    }

    // Directory — still works but nudges toward ls
    if path.is_dir() {
        let mut result = list_directory(path, offset, limit);
        if result.success
            && let serde_json::Value::String(ref mut s) = result.result
        {
            s.insert_str(0, "(Hint: use `ls` for directory listings.)\n");
        }
        return result;
    }

    // Image files — return as visual attachment
    if let Some(mime) = image_mime(path) {
        return read_image(path, path_str, mime);
    }

    // PDF files — extract text via pdf-extract (pure Rust)
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
    {
        return read_pdf(path, path_str, offset, limit);
    }

    // Binary detection
    if is_likely_binary(path) {
        return ToolResult::err_fmt(format_args!(
            "Binary file detected: {path_str}. Use `read_image` for images, or `exec_command` for binary inspection."
        ));
    }

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) => return ToolResult::err_fmt(format_args!("Failed to open file: {e}")),
    };
    let reader = BufReader::new(file);
    let slice = match collect_window(
        reader.lines(),
        offset,
        limit,
        |line_no, line| format!("{line_no}: {line}"),
        "file",
    ) {
        Ok(slice) => slice,
        Err(err) => return err,
    };

    ToolResult::ok(json!(render_window(&slice, WindowKind::Lines)))
}

fn list_directory(path: &Path, offset: usize, limit: usize) -> ToolResult {
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let mut items: Vec<String> = Vec::new();
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                if is_dir {
                    items.push(format!("{}/", name));
                } else {
                    items.push(name);
                }
            }
            items.sort();
            let slice = match collect_window(
                items.into_iter().map(Ok::<String, std::io::Error>),
                offset,
                limit,
                |_index, entry| entry.to_string(),
                "directory",
            ) {
                Ok(slice) => slice,
                Err(err) => return err,
            };
            ToolResult::ok(json!(render_window(&slice, WindowKind::Entries)))
        }
        Err(e) => ToolResult::err_fmt(format_args!("Failed to read directory: {e}")),
    }
}

/// Simple binary detection: check first 8KB for null bytes.
fn is_likely_binary(path: &Path) -> bool {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut buf = [0u8; 8192];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return false,
    };
    buf[..n].contains(&0)
}

/// Return the MIME type for supported image extensions.
fn image_mime(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

/// Read an image file, extract dimensions from the header, and return as a ToolImage.
fn read_image(path: &Path, path_str: &str, mime: &str) -> ToolResult {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => return ToolResult::err_fmt(format_args!("Failed to read image: {e}")),
    };

    let size_kb = data.len() / 1024;
    let dims = image_dimensions(&data, mime);
    let label = match dims {
        Some((w, h)) => format!("{} ({}KB {}x{})", path_str, size_kb, w, h),
        None => format!("{} ({}KB)", path_str, size_kb),
    };

    let image = ToolImage {
        mime: mime.to_string(),
        reference: None,
        data,
        label: label.clone(),
        width: dims.map(|(width, _)| width),
        height: dims.map(|(_, height)| height),
    };

    ToolResult::with_images(true, json!(format!("[Image: {}]", label)), vec![image])
}

/// Extract text from a PDF file using the pdf-extract crate (pure Rust).
fn read_pdf(path: &Path, path_str: &str, offset: usize, limit: usize) -> ToolResult {
    let pdf_bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return ToolResult::err_fmt(format_args!("Failed to read PDF: {e}")),
    };

    let file_size_kb = pdf_bytes.len() / 1024;

    let text = match pdf_extract::extract_text_from_mem(&pdf_bytes) {
        Ok(t) => t,
        Err(e) => {
            return ToolResult::err_fmt(format_args!(
                "Failed to extract text from PDF {path_str}: {e}"
            ));
        }
    };

    let slice = match collect_window(
        text.lines()
            .map(|line| Ok::<String, std::io::Error>(line.to_string())),
        offset,
        limit,
        |line_no, line| format!("{line_no}: {line}"),
        "PDF",
    ) {
        Ok(slice) => slice,
        Err(err) => return err,
    };

    let mut formatted = render_window(&slice, WindowKind::Lines);

    let header = format!(
        "[PDF: {} ({}KB, {} lines extracted)]\n",
        path_str, file_size_kb, slice.total_items
    );
    formatted.insert_str(0, &header);

    ToolResult::ok(json!(formatted))
}

/// Extract width x height from image headers (zero deps).
fn image_dimensions(data: &[u8], mime: &str) -> Option<(u32, u32)> {
    match mime {
        "image/png" => png_dimensions(data),
        "image/jpeg" => jpeg_dimensions(data),
        "image/gif" => gif_dimensions(data),
        _ => None,
    }
}

/// PNG: width at bytes 16-19, height at bytes 20-23 (IHDR chunk, big-endian).
fn png_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 24 {
        return None;
    }
    // Verify PNG signature
    if &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    Some((w, h))
}

/// JPEG: scan for SOF0/SOF2 marker (0xFF 0xC0 or 0xFF 0xC2), height then width.
fn jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = data[i + 1];
        // SOF0 (0xC0) or SOF2 (0xC2) — baseline or progressive
        if marker == 0xC0 || marker == 0xC2 {
            if i + 9 >= data.len() {
                return None;
            }
            let h = u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32;
            let w = u16::from_be_bytes([data[i + 7], data[i + 8]]) as u32;
            return Some((w, h));
        }
        // Skip non-SOF markers
        if marker == 0xD8 || marker == 0xD9 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
        } else if i + 3 < data.len() {
            let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
            i += 2 + len;
        } else {
            break;
        }
    }
    None
}

/// GIF: width at bytes 6-7, height at bytes 8-9 (little-endian).
fn gif_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 10 {
        return None;
    }
    // Verify GIF signature
    if &data[..3] != b"GIF" {
        return None;
    }
    let w = u16::from_le_bytes([data[6], data[7]]) as u32;
    let h = u16::from_le_bytes([data[8], data[9]]) as u32;
    Some((w, h))
}

struct WindowSlice {
    rendered: Vec<String>,
    total_items: usize,
    shown_start: Option<usize>,
    shown_end: Option<usize>,
    has_more_items: bool,
    truncated_by_bytes: bool,
}

enum WindowKind {
    Lines,
    Entries,
}

fn collect_window<I, E, F>(
    items: I,
    offset: usize,
    limit: usize,
    mut format_item: F,
    item_label: &str,
) -> Result<WindowSlice, ToolResult>
where
    I: IntoIterator<Item = Result<String, E>>,
    E: std::fmt::Display,
    F: FnMut(usize, &str) -> String,
{
    let mut total_items = 0usize;
    let mut bytes = 0usize;
    let mut rendered = Vec::new();
    let mut has_more_items = false;
    let mut truncated_by_bytes = false;

    for item in items {
        let item = item.map_err(|err| {
            ToolResult::err_fmt(format_args!("Failed to read {item_label}: {err}"))
        })?;
        total_items += 1;
        if total_items < offset {
            continue;
        }
        if rendered.len() >= limit {
            has_more_items = true;
            continue;
        }

        let item = truncate_line(&item);
        let rendered_item = format_item(total_items, &item);
        let size = rendered_item.len() + usize::from(!rendered.is_empty());
        if bytes + size > MAX_OUTPUT_BYTES {
            truncated_by_bytes = true;
            has_more_items = true;
            break;
        }
        bytes += size;
        rendered.push(rendered_item);
    }

    if total_items < offset && !(total_items == 0 && offset == 1) {
        return Err(ToolResult::err_fmt(format_args!(
            "Offset {offset} is out of range for this {item_label} ({total_items} items)"
        )));
    }

    let shown_start = (!rendered.is_empty()).then_some(offset);
    let shown_end = shown_start.map(|start| start + rendered.len().saturating_sub(1));

    Ok(WindowSlice {
        rendered,
        total_items,
        shown_start,
        shown_end,
        has_more_items,
        truncated_by_bytes,
    })
}

fn render_window(slice: &WindowSlice, kind: WindowKind) -> String {
    let mut output = slice.rendered.join("\n");
    let Some(shown_start) = slice.shown_start else {
        return output;
    };
    let Some(shown_end) = slice.shown_end else {
        return output;
    };

    let next_offset = shown_end + 1;
    match kind {
        WindowKind::Lines => {
            if slice.truncated_by_bytes {
                output.push_str(&format!(
                    "\n[output capped at {}. Showing lines {}-{}. Use offset={} to continue.]",
                    MAX_OUTPUT_BYTES_LABEL, shown_start, shown_end, next_offset
                ));
            } else if slice.has_more_items {
                output.push_str(&format!(
                    "\n[results truncated: showing lines {}-{} of {}. Use offset={} to continue.]",
                    shown_start, shown_end, slice.total_items, next_offset
                ));
            }
        }
        WindowKind::Entries => {
            if slice.truncated_by_bytes {
                output.push_str(&format!(
                    "\n[output capped at {}. Showing entries {}-{}. Use offset={} to continue.]",
                    MAX_OUTPUT_BYTES_LABEL, shown_start, shown_end, next_offset
                ));
            } else if slice.has_more_items {
                output.push_str(&format!(
                    "\n[results truncated: showing entries {}-{} of {}. Use offset={} to continue.]",
                    shown_start, shown_end, slice.total_items, next_offset
                ));
            }
        }
    }
    output
}

fn truncate_line(line: &str) -> String {
    if line.len() > MAX_LINE_LEN {
        format!("{}...", &line[..MAX_LINE_LEN])
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3").unwrap();
        let result = lash::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": path.to_str().unwrap()}),
        )
        .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap();
        assert!(text.contains("1: line1"));
        assert!(text.contains("2: line2"));
        assert!(text.contains("3: line3"));
        assert!(!text.contains('|'));
    }

    #[tokio::test]
    async fn test_read_with_offset_and_limit() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3\nline4\nline5").unwrap();
        let result = lash::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": path.to_str().unwrap(), "offset": 2, "limit": 2}),
        )
        .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap();
        assert!(text.contains("2: line2"));
        assert!(text.contains("3: line3"));
        assert!(!text.contains("1: line1"));
        assert!(!text.contains("4: line4"));
        assert!(text.contains("results truncated"));
        assert!(text.contains("offset=4"));
    }

    #[tokio::test]
    async fn test_read_caps_large_output_by_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        let content = (0..200)
            .map(|idx| format!("{idx}: {}", "x".repeat(400)))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();
        let result = lash::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": path.to_str().unwrap(), "limit": 200}),
        )
        .await;
        assert!(result.success);
        let text = result.result.as_str().unwrap();
        assert!(text.contains("output capped at 50 KB"));
        assert!(text.contains("Use offset="));
    }

    #[tokio::test]
    async fn test_read_nonexistent() {
        let result = lash::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": "/nonexistent/path/to/file.txt"}),
        )
        .await;
        assert!(!result.success);
    }

    // ── PNG dimensions ──

    #[test]
    fn test_png_dimensions_valid() {
        // Minimal valid PNG header (first 24 bytes)
        let mut data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        // IHDR chunk length (4 bytes)
        data.extend_from_slice(&[0, 0, 0, 13]);
        // IHDR tag
        data.extend_from_slice(b"IHDR");
        // Width: 640 (big-endian)
        data.extend_from_slice(&640u32.to_be_bytes());
        // Height: 480 (big-endian)
        data.extend_from_slice(&480u32.to_be_bytes());
        let (w, h) = png_dimensions(&data).unwrap();
        assert_eq!((w, h), (640, 480));
    }

    #[test]
    fn test_png_dimensions_truncated() {
        assert!(png_dimensions(&[0x89, b'P', b'N', b'G']).is_none());
    }

    #[test]
    fn test_png_dimensions_wrong_sig() {
        let data = vec![0; 24];
        assert!(png_dimensions(&data).is_none());
    }

    // ── JPEG dimensions ──

    #[test]
    fn test_jpeg_dimensions_valid() {
        // Minimal JPEG with SOI + SOF0
        let mut data = vec![0xFF, 0xD8]; // SOI
        // SOF0 marker
        data.extend_from_slice(&[0xFF, 0xC0]);
        // Length (including these 2 bytes)
        data.extend_from_slice(&[0x00, 0x11]);
        // Precision
        data.push(8);
        // Height: 480 (big-endian u16)
        data.extend_from_slice(&480u16.to_be_bytes());
        // Width: 640 (big-endian u16)
        data.extend_from_slice(&640u16.to_be_bytes());
        // Padding to satisfy i+9 < len bounds check
        data.push(0);
        let (w, h) = jpeg_dimensions(&data).unwrap();
        assert_eq!((w, h), (640, 480));
    }

    #[test]
    fn test_jpeg_dimensions_truncated() {
        assert!(jpeg_dimensions(&[0xFF, 0xD8, 0xFF, 0xC0]).is_none());
    }

    // ── GIF dimensions ──

    #[test]
    fn test_gif87a_dimensions() {
        let mut data = b"GIF87a".to_vec();
        // Width: 320 (little-endian u16)
        data.extend_from_slice(&320u16.to_le_bytes());
        // Height: 200 (little-endian u16)
        data.extend_from_slice(&200u16.to_le_bytes());
        let (w, h) = gif_dimensions(&data).unwrap();
        assert_eq!((w, h), (320, 200));
    }

    #[test]
    fn test_gif89a_dimensions() {
        let mut data = b"GIF89a".to_vec();
        data.extend_from_slice(&100u16.to_le_bytes());
        data.extend_from_slice(&50u16.to_le_bytes());
        let (w, h) = gif_dimensions(&data).unwrap();
        assert_eq!((w, h), (100, 50));
    }

    #[test]
    fn test_gif_bad_signature() {
        let data = b"NOT_GIF___".to_vec();
        assert!(gif_dimensions(&data).is_none());
    }

    // ── image_mime ──

    #[test]
    fn test_image_mime() {
        assert_eq!(image_mime(Path::new("photo.png")), Some("image/png"));
        assert_eq!(image_mime(Path::new("photo.jpg")), Some("image/jpeg"));
        assert_eq!(image_mime(Path::new("photo.jpeg")), Some("image/jpeg"));
        assert_eq!(image_mime(Path::new("anim.gif")), Some("image/gif"));
        assert_eq!(image_mime(Path::new("photo.webp")), Some("image/webp"));
        assert_eq!(image_mime(Path::new("photo.bmp")), Some("image/bmp"));
        assert_eq!(image_mime(Path::new("file.txt")), None);
        assert_eq!(image_mime(Path::new("noext")), None);
    }
}
