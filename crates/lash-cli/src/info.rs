//! Version/info text and the Info/Help/Controls document overlays.

use lash::provider::ProviderHandle;
use lash_standard_plugins::StandardContextApproach;
use lash_tui_extensions::TuiExtensions;

use crate::SkillCatalog;
use crate::command;
use crate::execution_settings::{
    ExecutionMode, execution_mode_label, execution_mode_usage, standard_context_approach_label,
};
use crate::keybindings::shortcut_help_rows;
use crate::model_selection::provider_display_label;
use crate::overlay::{DocumentRow, DocumentSection, DocumentState};

pub(crate) fn version_text() -> String {
    format!(
        "lash-cli {}
lash-sansio {}",
        crate::APP_VERSION,
        lash_core::SANSIO_VERSION
    )
}

pub(crate) fn info_text_unconfigured(execution_mode: &ExecutionMode, cwd: &str) -> String {
    [
        format!("lash-cli: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash_core::SANSIO_VERSION),
        "provider: (not configured)".to_string(),
        "configured model: (not configured)".to_string(),
        "resolved model: (not configured)".to_string(),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
        "context window: unknown".to_string(),
        format!("cwd: {}", cwd),
        "session: (not started)".to_string(),
    ]
    .join("\n")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn info_text(
    provider: &ProviderHandle,
    configured_model: &str,
    model_variant: Option<&str>,
    execution_mode: &ExecutionMode,
    standard_context_approach: Option<&StandardContextApproach>,
    context_window: Option<u64>,
    tool_summary: Option<(usize, &str)>,
    cwd: &str,
    session_name: Option<&str>,
    session_id: Option<&str>,
    session_db_path: Option<&str>,
) -> String {
    let resolved_model =
        crate::provider_metadata::provider_wire_model_id(provider.kind(), configured_model);
    let mut lines = vec![
        format!("lash-cli: {}", crate::APP_VERSION),
        format!("lash-sansio: {}", lash_core::SANSIO_VERSION),
        format!(
            "provider: {} ({})",
            provider_display_label(provider),
            provider.kind()
        ),
        format!("configured model: {}", configured_model),
        format!("resolved model: {}", resolved_model),
        format!("execution mode: {}", execution_mode_label(execution_mode)),
    ];
    if *execution_mode == ExecutionMode::Standard
        && let Some(standard_context_approach) = standard_context_approach
    {
        lines.push(format!(
            "context approach: {}",
            standard_context_approach_label(standard_context_approach)
        ));
    }

    if let Some(variant) = model_variant {
        lines.push(format!("variant: {}", variant));
    }
    if let Some(window) = context_window {
        lines.push(format!("context window: {}", window));
    } else {
        lines.push("context window: unknown".to_string());
    }

    if let Some((tool_count, toolset_hash)) = tool_summary {
        lines.push(format!("tools: {} (hash {})", tool_count, toolset_hash));
    } else {
        lines.push("tools: (session not started)".to_string());
    }
    lines.extend([
        format!("cwd: {}", cwd),
        format!("session: {}", session_name.unwrap_or("(not started)")),
    ]);
    if let Some(session_id) = session_id {
        lines.push(format!("session id: {}", session_id));
    }
    if let Some(session_db_path) = session_db_path {
        lines.push(format!("session db: {}", session_db_path));
    }

    lines.join("\n")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn info_document(
    provider: &ProviderHandle,
    configured_model: &str,
    model_variant: Option<&str>,
    execution_mode: &ExecutionMode,
    standard_context_approach: Option<&StandardContextApproach>,
    context_window: Option<u64>,
    tool_summary: Option<(usize, &str)>,
    cwd: &str,
    session_name: Option<&str>,
    session_id: Option<&str>,
    session_db_path: Option<&str>,
) -> DocumentState {
    let resolved_model =
        crate::provider_metadata::provider_wire_model_id(provider.kind(), configured_model);
    let mut model_rows = vec![
        DocumentRow::KeyValue {
            label: "configured".to_string(),
            value: configured_model.to_string(),
        },
        DocumentRow::KeyValue {
            label: "resolved".to_string(),
            value: resolved_model,
        },
        DocumentRow::KeyValue {
            label: "mode".to_string(),
            value: execution_mode_label(execution_mode).to_string(),
        },
    ];
    if let Some(variant) = model_variant {
        model_rows.push(DocumentRow::KeyValue {
            label: "variant".to_string(),
            value: variant.to_string(),
        });
    }
    if *execution_mode == ExecutionMode::Standard
        && let Some(standard_context_approach) = standard_context_approach
    {
        model_rows.push(DocumentRow::KeyValue {
            label: "context approach".to_string(),
            value: standard_context_approach_label(standard_context_approach).to_string(),
        });
    }
    model_rows.push(DocumentRow::KeyValue {
        label: "context window".to_string(),
        value: context_window
            .map(|window| window.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    });

    let tools_rows = match tool_summary {
        Some((tool_count, _toolset_hash)) => vec![DocumentRow::KeyValue {
            label: "count".to_string(),
            value: tool_count.to_string(),
        }],
        None => vec![DocumentRow::Text("session not started".to_string())],
    };

    DocumentState::new(
        "Info",
        vec![
            DocumentSection::new(
                "Runtime",
                vec![
                    DocumentRow::KeyValue {
                        label: "lash-cli".to_string(),
                        value: crate::APP_VERSION.to_string(),
                    },
                    DocumentRow::KeyValue {
                        label: "provider".to_string(),
                        value: format!(
                            "{} ({})",
                            provider_display_label(provider),
                            provider.kind()
                        ),
                    },
                ],
            ),
            DocumentSection::new("Model", model_rows),
            DocumentSection::new(
                "Session",
                vec![
                    DocumentRow::KeyValue {
                        label: "name".to_string(),
                        value: session_name.unwrap_or("(not started)").to_string(),
                    },
                    DocumentRow::KeyValue {
                        label: "id".to_string(),
                        value: session_id.unwrap_or("(not started)").to_string(),
                    },
                ],
            ),
            DocumentSection::new("Tools", tools_rows),
            DocumentSection::new(
                "Paths",
                vec![
                    DocumentRow::KeyValue {
                        label: "cwd".to_string(),
                        value: cwd.to_string(),
                    },
                    DocumentRow::KeyValue {
                        label: "session db".to_string(),
                        value: session_db_path.unwrap_or("(not started)").to_string(),
                    },
                ],
            ),
        ],
    )
}

pub(crate) fn controls_document(ui_extensions: &TuiExtensions) -> DocumentState {
    DocumentState::new(
        "Controls",
        vec![DocumentSection::new(
            "Keyboard",
            shortcut_help_rows(ui_extensions, true)
                .into_iter()
                .map(|row| DocumentRow::Shortcut {
                    keys: row.keys,
                    description: row.description,
                })
                .collect(),
        )],
    )
}

pub(crate) fn help_document(skills: &SkillCatalog, ui_extensions: &TuiExtensions) -> DocumentState {
    let mut command_rows = Vec::new();
    for spec in command::catalog() {
        let aliases = if spec.aliases.is_empty() {
            String::new()
        } else {
            format!(", {}", spec.aliases.join(", "))
        };
        let description = if spec.name == "/mode" {
            format!(
                "{}; new session required to change {}",
                spec.description,
                execution_mode_usage()
            )
        } else {
            spec.description.to_string()
        };
        command_rows.push(DocumentRow::Shortcut {
            keys: format!("{}{}", spec.usage, aliases),
            description,
        });
    }
    for spec in ui_extensions.command_specs() {
        let aliases = if spec.aliases.is_empty() {
            String::new()
        } else {
            format!(", {}", spec.aliases.join(", "))
        };
        command_rows.push(DocumentRow::Shortcut {
            keys: format!("{}{}", spec.usage, aliases),
            description: spec.description.to_string(),
        });
    }
    command_rows.push(DocumentRow::Shortcut {
        keys: "/<skill> [text]".to_string(),
        description: "Invoke a loaded skill directly".to_string(),
    });

    let mut sections = vec![DocumentSection::new("Commands", command_rows)];
    if !skills.is_empty() {
        sections.push(DocumentSection::new(
            "Installed Skills",
            skills
                .iter()
                .map(|skill| DocumentRow::Shortcut {
                    keys: format!("${}", skill.name),
                    description: if skill.description.is_empty() {
                        "Invoke skill".to_string()
                    } else {
                        skill.description.clone()
                    },
                })
                .collect(),
        ));
    }
    sections.push(DocumentSection::new(
        "Shortcuts",
        shortcut_help_rows(ui_extensions, false)
            .into_iter()
            .map(|row| DocumentRow::Shortcut {
                keys: row.keys,
                description: row.description,
            })
            .collect(),
    ));

    DocumentState::new("Help", sections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keybindings::ShortcutHelpRow;

    #[test]
    fn controls_document_uses_authoritative_shortcut_rows() {
        let extensions = TuiExtensions::default();
        let document = controls_document(&extensions);
        assert_eq!(document.title, "Controls");
        assert_eq!(document.sections.len(), 1);
        assert_eq!(document.sections[0].title, "Keyboard");

        let document_rows = document.sections[0]
            .rows
            .iter()
            .map(|row| match row {
                DocumentRow::Shortcut { keys, description } => ShortcutHelpRow {
                    keys: keys.clone(),
                    description: description.clone(),
                },
                other => panic!("unexpected controls row: {other:?}"),
            })
            .collect::<Vec<_>>();

        assert_eq!(document_rows, shortcut_help_rows(&extensions, true));
    }

    #[test]
    fn info_text_includes_session_id_and_db_path() {
        let provider = ProviderHandle::new(
            lash_provider_openai::OpenAiCompatibleProvider::new(
                "test",
                "https://openrouter.ai/api/v1",
            )
            .into_components(),
        );
        let text = info_text(
            &provider,
            "google/gemini-3-flash-preview",
            None,
            &ExecutionMode::Rlm,
            None,
            Some(123_000),
            Some((7, "abcd1234")),
            "/tmp/demo",
            Some("demo-session"),
            Some("sess-123"),
            Some("/tmp/demo/session.db"),
        );
        assert!(text.contains("session: demo-session"));
        assert!(text.contains("session id: sess-123"));
        assert!(text.contains("session db: /tmp/demo/session.db"));
    }

    #[test]
    fn info_document_groups_diagnostics_and_keeps_plain_text_paths_complete() {
        let provider = ProviderHandle::new(
            lash_provider_openai::OpenAiCompatibleProvider::new(
                "test",
                "https://openrouter.ai/api/v1",
            )
            .into_components(),
        );
        let cwd = "/tmp/demo/workspace-with-a-long-directory-name";
        let session_db =
            "/tmp/demo/workspace-with-a-long-directory-name/.lash/session/store/session.db";
        let document = info_document(
            &provider,
            "google/gemini-3-flash-preview",
            Some("medium"),
            &ExecutionMode::Standard,
            Some(&StandardContextApproach::RollingHistory(Default::default())),
            Some(123_000),
            Some((7, "abcd1234")),
            cwd,
            Some("demo-session"),
            Some("sess-123"),
            Some(session_db),
        );

        let section_titles = document
            .sections
            .iter()
            .map(|section| section.title.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            section_titles,
            ["Runtime", "Model", "Session", "Tools", "Paths"]
        );
        let runtime = document
            .sections
            .iter()
            .find(|section| section.title == "Runtime")
            .expect("runtime section");
        assert!(!runtime.rows.iter().any(|row| matches!(
            row,
            DocumentRow::KeyValue { label, .. } if label == "lash-sansio"
        )));
        let tools = document
            .sections
            .iter()
            .find(|section| section.title == "Tools")
            .expect("tools section");
        assert!(!tools.rows.iter().any(|row| matches!(
            row,
            DocumentRow::KeyValue { label, .. } if label == "hash"
        )));
        let paths = document
            .sections
            .iter()
            .find(|section| section.title == "Paths")
            .expect("paths section");
        assert!(paths.rows.iter().any(|row| matches!(
            row,
            DocumentRow::KeyValue { label, value }
                if label == "cwd" && value == cwd
        )));
        assert!(paths.rows.iter().any(|row| matches!(
            row,
            DocumentRow::KeyValue { label, value }
                if label == "session db" && value == session_db
        )));

        let rendered = crate::render::document_lines_snapshot(&document, 28)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!rendered.contains('…'), "{rendered}");
        let compact_rendered = rendered.split_whitespace().collect::<String>();
        assert!(compact_rendered.contains(cwd), "{rendered}");
        assert!(compact_rendered.contains(session_db), "{rendered}");

        let text = info_text(
            &provider,
            "google/gemini-3-flash-preview",
            Some("medium"),
            &ExecutionMode::Standard,
            Some(&StandardContextApproach::RollingHistory(Default::default())),
            Some(123_000),
            Some((7, "abcd1234")),
            cwd,
            Some("demo-session"),
            Some("sess-123"),
            Some(session_db),
        );
        assert!(text.contains(session_db));
    }
}
