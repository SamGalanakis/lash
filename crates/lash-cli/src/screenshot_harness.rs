//! Deterministic render harness for design iteration on the TUI palette.
//!
//! This is a `#[cfg(test)]`-only tool. It composes representative TUI screens
//! out of the real `theme::` styles and the real diff renderer, rasterizes them
//! through `lash_tui::render_snapshot`, and serializes the resulting cell grid
//! (glyph + resolved RGB fg/bg + modifiers) to JSON. An external script turns
//! that JSON into a PNG so the colors can be critiqued visually without a live
//! terminal or an LLM turn.
//!
//! Run with:
//! ```text
//! LASH_SHOT_DIR=/path/out cargo test -p lash-cli --bin lash \
//!     emit_design_screenshots -- --ignored --nocapture
//! ```

use lash_tui::{Color, Line, ScreenSnapshot, Span, Style};

use crate::diff::render_inline_diff;
use crate::markdown::render_markdown;
use crate::theme;

const WIDTH: u16 = 104;

/// Resolve a themed [`Color`] to a concrete sRGB triple for off-terminal
/// rasterization. Default fg/bg map to the Lash chalk/form anchors so the PNG
/// matches what the real app paints once it has set the terminal default
/// background to `FORM`.
fn resolve(color: Option<Color>, is_bg: bool) -> (u8, u8, u8) {
    match color {
        Some(Color::Rgb { r, g, b }) => (r, g, b),
        Some(Color::AnsiValue(value)) => ansi_to_rgb(value),
        Some(Color::DefaultForeground) => rgb(theme::CHALK),
        Some(Color::DefaultBackground) => rgb(theme::FORM),
        None if is_bg => rgb(theme::FORM),
        None => rgb(theme::CHALK),
    }
}

fn rgb(color: Color) -> (u8, u8, u8) {
    match color {
        Color::Rgb { r, g, b } => (r, g, b),
        _ => (0, 0, 0),
    }
}

/// A pragmatic 16-color ANSI map (only hit by the System theme). Values follow
/// a typical warm-dark terminal so System-theme shots stay legible.
fn ansi_to_rgb(value: u8) -> (u8, u8, u8) {
    match value {
        0 => (20, 19, 17),
        1 => (204, 68, 68),
        2 => (138, 158, 108),
        3 => (232, 163, 60),
        4 => (108, 140, 178),
        5 => (170, 120, 170),
        6 => (110, 170, 168),
        7 => (200, 196, 184),
        8 => (90, 90, 80),
        9 => (215, 114, 114),
        10 => (160, 180, 130),
        11 => (255, 176, 84),
        12 => (130, 160, 200),
        13 => (190, 140, 190),
        14 => (130, 190, 188),
        15 => (232, 228, 208),
        _ => (200, 196, 184),
    }
}

/// The patch summary line, reproduced verbatim from
/// `render::activity::render_patch_summary_line` so the shot matches production.
fn patch_summary_line(label: &str, subject: &str, added: usize, removed: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled("  ".to_string(), theme::patch_frame()),
        Span::styled(label.to_string(), theme::patch_label()),
        Span::raw(" "),
        Span::styled(subject.to_string(), theme::assistant_text()),
        Span::styled(" (".to_string(), theme::code_chrome()),
        Span::styled(format!("+{added}"), theme::patch_add()),
        Span::raw(" "),
        Span::styled(format!("-{removed}"), theme::patch_remove()),
        Span::styled(")".to_string(), theme::code_chrome()),
    ])
}

fn assistant_line(text: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled("● ".to_string(), theme::assistant_bar()),
        Span::styled(text.to_string(), theme::assistant_text()),
    ])
}

fn prompt_line(text: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{} ", theme::PROMPT_CHAR), theme::prompt()),
        Span::styled(text.to_string(), theme::user_input()),
    ])
}

/// The canonical edit-viewer screen: the exact situation from the bug report —
/// a workspace version bump rendered as a unified diff.
pub(crate) fn diff_screen() -> Vec<Line<'static>> {
    let diff = "\
@@ -31,20 +31,20 @@
 # Lash, consumed from crates.io at the latest published release. The facade
 # crate is published as `lash-runtime` but its library is imported as `lash`.
-lash = { package = \"lash-runtime\", version = \"=0.1.0-alpha.72\", features = [\"rlm\"] }
-lash-core = \"=0.1.0-alpha.72\"
-lash-llm-tools = \"=0.1.0-alpha.72\"
-lash-protocol-rlm = \"=0.1.0-alpha.72\"
-lash-provider-anthropic = \"=0.1.0-alpha.72\"
-lash-provider-openai = \"=0.1.0-alpha.72\"
-lash-sqlite-store = \"=0.1.0-alpha.72\"
+lash = { package = \"lash-runtime\", version = \"=0.1.0-alpha.75\", features = [\"rlm\"] }
+lash-core = \"=0.1.0-alpha.75\"
+lash-llm-tools = \"=0.1.0-alpha.75\"
+lash-protocol-rlm = \"=0.1.0-alpha.75\"
+lash-provider-anthropic = \"=0.1.0-alpha.75\"
+lash-provider-openai = \"=0.1.0-alpha.75\"
+lash-sqlite-store = \"=0.1.0-alpha.75\"
 # Shared benchmark helpers (provider config loading from ~/.lash/config.json).
 bench-common = { path = \"bench/common\" }";

    let mut lines = Vec::new();
    lines.push(Line::default());
    lines.push(prompt_line("bump the workspace pins to 0.1.0-alpha.75"));
    lines.push(Line::default());
    lines.push(assistant_line(
        "Updating the root workspace pins to 0.1.0-alpha.75, then refreshing the lockfile.",
    ));
    lines.push(Line::default());
    lines.push(patch_summary_line("Edited", "Cargo.toml", 14, 14));
    render_inline_diff(&mut lines, diff, WIDTH as usize, "  │ ");
    lines.push(Line::default());
    lines.push(assistant_line(
        "Lockfile refreshed. Running cargo check next.",
    ));
    lines
}

/// A representative assistant answer: heading, prose, a list with inline code,
/// and a fenced code block — rendered through the real markdown renderer so the
/// syntax palette (keyword / string / comment) is exactly what ships.
pub(crate) fn prose_screen() -> Vec<Line<'static>> {
    let markdown = "\
## workspace pins

The pins live in the root `Cargo.toml`. Bump every `lash-*` entry to the
same release so the lockfile resolves cleanly:

- update the `version` field on each crate
- run `cargo update` to refresh `Cargo.lock`
- re-run `cargo check` across the workspace

```rust
// pin every workspace crate to one release
fn bump(deps: &mut Manifest, version: &str) {
    for dep in deps.lash_crates_mut() {
        dep.set_version(version); // \"=0.1.0-alpha.75\"
    }
}
```

That keeps the facade crate and its siblings in lockstep.";

    let mut lines = Vec::new();
    lines.push(Line::default());
    lines.push(prompt_line("how do the workspace pins work?"));
    lines.push(Line::default());
    lines.push(assistant_line("Here's how the pinning is laid out."));
    lines.push(Line::default());
    for line in render_markdown(markdown, (WIDTH as usize).saturating_sub(2)) {
        let mut spans = vec![Span::styled("  ".to_string(), theme::assistant_bar())];
        spans.extend(line.spans);
        lines.push(Line { spans });
    }
    lines
}

/// Serialize a rendered snapshot to compact JSON: width, height, and a row-major
/// grid of `{glyph, fg, bg, bold, dim, italic}`.
fn snapshot_to_json(snapshot: &ScreenSnapshot) -> String {
    let mut out = String::with_capacity(64 * 1024);
    out.push_str(&format!(
        "{{\"w\":{},\"h\":{},\"rows\":[",
        snapshot.width, snapshot.height
    ));
    for y in 0..snapshot.height {
        if y > 0 {
            out.push(',');
        }
        out.push('[');
        for x in 0..snapshot.width {
            if x > 0 {
                out.push(',');
            }
            let (ch, style, continuation) = match snapshot.cell(x, y) {
                Some(cell) => (cell.ch, cell.style, cell.continuation),
                None => (' ', Style::default(), false),
            };
            if continuation {
                // Wide-glyph trailing cell: emit an empty placeholder so column
                // alignment is preserved in the raster.
                out.push_str("{\"c\":\"\",\"k\":1}");
                continue;
            }
            let (fr, fg, fb) = resolve(style.fg, false);
            let (br, bg, bb) = resolve(style.bg, true);
            out.push_str("{\"c\":");
            push_json_char(&mut out, ch);
            out.push_str(&format!(
                ",\"f\":[{fr},{fg},{fb}],\"b\":[{br},{bg},{bb}],\"bo\":{},\"di\":{},\"it\":{}}}",
                style.bold as u8, style.dim as u8, style.italic as u8
            ));
        }
        out.push(']');
    }
    out.push_str("]}");
    out
}

fn push_json_char(out: &mut String, ch: char) {
    out.push('"');
    match ch {
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        c if (c as u32) < 0x20 => out.push(' '),
        c => out.push(c),
    }
    out.push('"');
}

fn render_screen(lines: &[Line<'static>], height: u16) -> ScreenSnapshot {
    let base = theme::surface_base().fill();
    lash_tui::render_snapshot(WIDTH, height, |frame| {
        frame.fill(frame.area(), ' ', base);
        for (idx, line) in lines.iter().enumerate() {
            frame.write_line_styled(0, idx as u16, line, base, WIDTH);
        }
    })
}

#[test]
#[ignore = "design tooling: writes PNG-source JSON when LASH_SHOT_DIR is set"]
fn emit_design_screenshots() {
    use crate::config::ThemeName;
    let Ok(dir) = std::env::var("LASH_SHOT_DIR") else {
        eprintln!("LASH_SHOT_DIR not set; nothing to emit");
        return;
    };
    std::fs::create_dir_all(&dir).expect("create shot dir");

    theme::set_active_theme(ThemeName::Lash);
    for (name, lines) in [
        ("diff_screen", diff_screen()),
        ("prose_screen", prose_screen()),
    ] {
        let height = lines.len() as u16 + 1;
        let snapshot = render_screen(&lines, height);
        let json = snapshot_to_json(&snapshot);
        let path = format!("{dir}/{name}.json");
        std::fs::write(&path, json).expect("write json");
        eprintln!("wrote {path} ({}x{})", snapshot.width, snapshot.height);
    }
}
