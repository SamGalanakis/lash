use serde_json::json;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash_core::{
    ToolCall, ToolContract, ToolDefinition, ToolExecutionMode, ToolManifest, ToolProvider,
    ToolResult, ToolRetryPolicy,
};

use lash_tool_support::{object_schema, parse_optional_usize_arg, require_str, run_blocking_value};

/// Read files with line-number-prefixed output. Supports images natively.
#[derive(Default)]
pub struct ReadFile;

pub struct ReadFilePluginFactory;

struct ReadFilePlugin {
    provider: Arc<ReadFile>,
}

impl ReadFile {
    pub fn new() -> Self {
        Self
    }
}

impl ReadFilePluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadFilePluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory for ReadFilePluginFactory {
    fn id(&self) -> &'static str {
        "read_file"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ReadFilePlugin {
            provider: Arc::new(ReadFile::new()),
        }))
    }
}

impl SessionPlugin for ReadFilePlugin {
    fn id(&self) -> &'static str {
        "read_file"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)
    }
}

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE_LEN: usize = 2000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;
const MAX_OUTPUT_BYTES_LABEL: &str = "50 KB";

struct ImageAttachmentData {
    data: Vec<u8>,
    media_type: lash_core::MediaType,
    width: Option<u32>,
    height: Option<u32>,
    label: String,
}

enum ReadFileBlockingResult {
    Tool(ToolResult),
    Image(ImageAttachmentData),
}

impl ReadFileBlockingResult {
    fn tool(result: ToolResult) -> Self {
        Self::Tool(result)
    }

    fn into_tool_result(self, context: &lash_core::ToolContext<'_>) -> ToolResult {
        match self {
            Self::Tool(result) => result,
            Self::Image(image) => store_image_attachment(context, image),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for ReadFile {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![read_file_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "read_file").then(|| Arc::new(read_file_tool_definition().contract()))
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

        match run_blocking_value(move || execute_read_file_sync(&path_str, offset, limit)).await {
            Ok(result) => result.into_tool_result(call.context),
            Err(err) => ToolResult::err_fmt(format_args!("{err}")),
        }
    }
}

fn read_file_tool_definition() -> ToolDefinition {
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
            .with_execution_mode(ToolExecutionMode::Parallel)
            .with_retry_policy(ToolRetryPolicy::safe(2, 25, 100))
}

fn parse_limit(args: &serde_json::Value) -> Result<usize, ToolResult> {
    Ok(
        parse_optional_usize_arg(args, "limit", Some(DEFAULT_LIMIT), false, 1)?
            .unwrap_or(DEFAULT_LIMIT),
    )
}

fn execute_read_file_sync(path_str: &str, offset: usize, limit: usize) -> ReadFileBlockingResult {
    let path = Path::new(path_str);
    if !path.exists() {
        return ReadFileBlockingResult::tool(ToolResult::err_fmt(format_args!(
            "Path does not exist: {path_str}. Use `ls` or `glob` to locate the correct path."
        )));
    }

    // Directory — still works but nudges toward ls
    if path.is_dir() {
        let mut output = list_directory(path, offset, limit).into_output();
        if output.is_success()
            && let lash_core::ToolCallOutcome::Success(lash_core::ToolValue::String(s)) =
                &mut output.outcome
        {
            s.insert_str(0, "(Hint: use `ls` for directory listings.)\n");
        }
        return ReadFileBlockingResult::tool(ToolResult::from_output(output));
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
        return ReadFileBlockingResult::tool(read_pdf(path, path_str, offset, limit));
    }

    // Binary detection
    if is_likely_binary(path) {
        return ReadFileBlockingResult::tool(ToolResult::err_fmt(format_args!(
            "Binary file detected: {path_str}. Use `read_image` for images, or `exec_command` for binary inspection."
        )));
    }

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            return ReadFileBlockingResult::tool(ToolResult::err_fmt(format_args!(
                "Failed to open file: {e}"
            )));
        }
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
        Err(err) => return ReadFileBlockingResult::tool(err),
    };

    ReadFileBlockingResult::tool(ToolResult::ok(json!(render_window(
        &slice,
        WindowKind::Lines
    ))))
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

/// Read image metadata. Image bytes must be attached through ToolContext by
/// callers that need model-visible attachments.
fn read_image(path: &Path, path_str: &str, mime: &str) -> ReadFileBlockingResult {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            return ReadFileBlockingResult::tool(ToolResult::err_fmt(format_args!(
                "Failed to read image: {e}"
            )));
        }
    };

    let size_kb = data.len() / 1024;
    let dims = image_dimensions(&data, mime);
    let label = match dims {
        Some((w, h)) => format!("{} ({}KB {}x{})", path_str, size_kb, w, h),
        None => format!("{} ({}KB)", path_str, size_kb),
    };

    let Some(media_type) = lash_core::MediaType::from_mime(mime) else {
        return ReadFileBlockingResult::tool(ToolResult::err_fmt(format_args!(
            "Unsupported image MIME type: {mime}"
        )));
    };
    ReadFileBlockingResult::Image(ImageAttachmentData {
        data,
        media_type,
        width: dims.map(|(width, _)| width),
        height: dims.map(|(_, height)| height),
        label,
    })
}

fn store_image_attachment(
    context: &lash_core::ToolContext<'_>,
    image: ImageAttachmentData,
) -> ToolResult {
    let reference = match context.put_attachment(
        image.data,
        lash_core::AttachmentCreateMeta::new(
            image.media_type,
            image.width,
            image.height,
            Some(image.label),
        ),
    ) {
        Ok(reference) => reference,
        Err(err) => {
            return ToolResult::err_fmt(format_args!("Failed to store image attachment: {err}"));
        }
    };
    ToolResult::from_output(lash_core::ToolCallOutput::success(
        lash_core::ToolValue::Attachment(reference),
    ))
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
    use lash_core::AttachmentStore;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3").unwrap();
        let result = lash_core::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": path.to_str().unwrap()}),
        )
        .await;
        assert!(result.is_success());
        let value = result.value_for_projection();
        let text = value.as_str().unwrap();
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
        let result = lash_core::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": path.to_str().unwrap(), "offset": 2, "limit": 2}),
        )
        .await;
        assert!(result.is_success());
        let value = result.value_for_projection();
        let text = value.as_str().unwrap();
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
        let result = lash_core::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": path.to_str().unwrap(), "limit": 200}),
        )
        .await;
        assert!(result.is_success());
        let value = result.value_for_projection();
        let text = value.as_str().unwrap();
        assert!(text.contains("output capped at 50 KB"));
        assert!(text.contains("Use offset="));
    }

    #[tokio::test]
    async fn test_read_nonexistent() {
        let result = lash_core::testing::run_tool(
            &ReadFile,
            "read_file",
            &json!({"path": "/nonexistent/path/to/file.txt"}),
        )
        .await;
        assert!(!result.is_success());
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

    #[tokio::test]
    async fn test_read_image_returns_attachment_value() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tiny.png");
        let mut data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        data.extend_from_slice(&[0, 0, 0, 13]);
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&1u32.to_be_bytes());
        std::fs::write(&path, &data).unwrap();

        let store = Arc::new(lash_core::InMemoryAttachmentStore::new());
        let context = lash_core::ToolContext::__for_testing(
            "test-session".into(),
            Arc::new(lash_core::testing::MockSessionManager::default()),
            lash_core::TurnContext::new(),
            store.clone(),
            lash_core::DirectCompletionClient::from_fn(|_, _| {
                Err(lash_core::PluginError::Session(
                    "direct completions are unavailable in read_file tests".to_string(),
                ))
            }),
            None,
        );
        let result = ReadFile
            .execute(lash_core::ToolCall {
                name: "read_file",
                args: &json!({"path": path.to_str().unwrap()}),
                context: &context,
                progress: None,
            })
            .await;

        let lash_core::ToolCallOutcome::Success(lash_core::ToolValue::Attachment(reference)) =
            result.into_output().outcome
        else {
            panic!("expected attachment result");
        };
        assert_eq!(reference.byte_len, data.len() as u64);
        assert_eq!(reference.width, Some(1));
        assert_eq!(reference.height, Some(1));
        assert_eq!(store.get(&reference.id).unwrap().bytes, data);
    }
}
