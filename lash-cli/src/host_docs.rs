use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lash::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PromptContribution,
    SessionPlugin,
};

const DOCS_DIR: &str = "docs";
const HOST_DOCS_DIR: &str = "lash-cli";

const README_MD: &str = r#"# Lash CLI Host Docs

These files describe the installed Lash CLI host. Prefer these docs over general model memory for Lash-specific tasks.

Read the relevant topic before changing Lash configuration, skills, sessions, providers, update state, or troubleshooting behavior.

The configured Lash home is two directories above this directory.

Topics:

- `skills.md`
- `config.md`
- `sessions.md`
- `troubleshooting.md`
"#;

const SKILLS_MD: &str = r#"# Skills

Skills are directories containing a `SKILL.md` file plus any referenced assets or scripts.

Default search order, lowest to highest priority:

1. `$LASH_HOME/skills`
2. repo-local `.lash/skills`
3. repo-local `.agents/lash/skills`

To install a user-global skill:

1. Create `$LASH_HOME/skills/<skill-name>/`.
2. Put `SKILL.md` in that directory.
3. Keep referenced files relative to the skill directory.
4. Start a new Lash session or turn if the current skill catalog was already loaded.

Do not install skills by editing these host docs. Put user-global skills under `$LASH_HOME/skills`.
"#;

const CONFIG_MD: &str = r#"# Configuration

The CLI owns the configured Lash home. `LASH_HOME` overrides the default; otherwise the home is `~/.lash`.

Common files and directories:

- `config.json`: provider and runtime configuration.
- `skills/`: user-global skills.
- `sessions/`: resumable session databases and detailed LLM logs when debug logging is enabled.
- `plans/`: persisted plan artifacts.
- `lash.log`: CLI log output when file logging is enabled.
- `cache/`: cache directory when `LASH_HOME` is set. Without `LASH_HOME`, cache usually lives under the platform cache directory.

Use `lash --info` to inspect the active provider, model, execution mode, context approach, working directory, session id, and session database.
"#;

const SESSIONS_MD: &str = r#"# Sessions

Lash sessions are persisted as SQLite databases under `$LASH_HOME/sessions`.

Useful operations:

- `lash --resume`: resume the most recent session.
- `lash --resume <id-or-name>`: resume a specific session.
- `lash --info`: print the current session database path when configured.
- `lash export ...`: export session data when the export command is available in the installed CLI.

For crash or interrupt recovery, inspect the newest database in `$LASH_HOME/sessions` and prefer resuming from the latest persisted state instead of reconstructing context from terminal output.
"#;

const TROUBLESHOOTING_MD: &str = r#"# Troubleshooting

Start with:

1. Run `lash --info`.
2. Check the configured Lash home from `LASH_HOME` or the default `~/.lash`.
3. Inspect `$LASH_HOME/sessions` for the active session database and any `.llm.jsonl` debug logs.
4. Check `$LASH_HOME/config.json` for provider configuration.

If a task asks Lash to modify its own installed behavior, first identify whether the behavior belongs to the CLI host, core runtime, sans-io state machine, provider crate, mode plugin, or user-home data. Do not paper over a core/runtime problem in the CLI unless the behavior is genuinely host-specific.
"#;

const MANAGED_DOCS: &[(&str, &str)] = &[
    ("README.md", README_MD),
    ("skills.md", SKILLS_MD),
    ("config.md", CONFIG_MD),
    ("sessions.md", SESSIONS_MD),
    ("troubleshooting.md", TROUBLESHOOTING_MD),
];

#[derive(Clone, Debug)]
pub(crate) struct HostDocs {
    dir: PathBuf,
}

impl HostDocs {
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }
}

pub(crate) fn ensure_host_docs() -> io::Result<HostDocs> {
    ensure_host_docs_at(&crate::paths::lash_home(), crate::APP_VERSION)
}

fn ensure_host_docs_at(lash_home: &Path, cli_version: &str) -> io::Result<HostDocs> {
    let dir = lash_home.join(DOCS_DIR).join(HOST_DOCS_DIR);
    fs::create_dir_all(&dir)?;
    for (filename, content) in MANAGED_DOCS {
        write_if_changed(&dir.join(filename), content)?;
    }
    write_if_changed(&dir.join("VERSION"), &format!("lash-cli {cli_version}\n"))?;
    Ok(HostDocs { dir })
}

fn write_if_changed(path: &Path, content: &str) -> io::Result<()> {
    if let Ok(existing) = fs::read_to_string(path)
        && existing == content
    {
        return Ok(());
    }
    fs::write(path, content)
}

pub(crate) struct HostDocsPluginFactory {
    prompt_content: Arc<String>,
}

impl HostDocsPluginFactory {
    pub(crate) fn new(docs_dir: PathBuf) -> Self {
        Self {
            prompt_content: Arc::new(host_docs_prompt_content(&docs_dir)),
        }
    }
}

impl PluginFactory for HostDocsPluginFactory {
    fn id(&self) -> &'static str {
        "lash_cli_host_docs"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(HostDocsPlugin {
            prompt_content: Arc::clone(&self.prompt_content),
        }))
    }
}

struct HostDocsPlugin {
    prompt_content: Arc<String>,
}

impl SessionPlugin for HostDocsPlugin {
    fn id(&self) -> &'static str {
        "lash_cli_host_docs"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let prompt_content = Arc::clone(&self.prompt_content);
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let prompt_content = Arc::clone(&prompt_content);
            Box::pin(async move {
                Ok(vec![
                    PromptContribution::environment(
                        "Lash CLI Host Docs",
                        prompt_content.as_ref().clone(),
                    )
                    .with_priority(-100),
                ])
            })
        }));
        Ok(())
    }
}

fn host_docs_prompt_content(docs_dir: &Path) -> String {
    format!(
        "Installed Lash CLI docs are available at `{}`.\n\
         For tasks about Lash itself, including skills, configuration, sessions, providers, \
         updates, or troubleshooting, read the relevant markdown file there before acting. \
         These docs describe the installed CLI host and override general model memory.",
        docs_dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_host_docs_writes_managed_markdown() {
        let temp = tempfile::TempDir::new().expect("temp dir");

        let docs = ensure_host_docs_at(temp.path(), "9.9.9").expect("host docs");

        assert_eq!(docs.dir(), temp.path().join("docs").join("lash-cli"));
        assert!(docs.dir().join("README.md").exists());
        assert!(docs.dir().join("skills.md").exists());
        assert!(docs.dir().join("config.md").exists());
        assert!(docs.dir().join("sessions.md").exists());
        assert!(docs.dir().join("troubleshooting.md").exists());
        assert_eq!(
            fs::read_to_string(docs.dir().join("VERSION")).expect("version"),
            "lash-cli 9.9.9\n"
        );
        assert!(
            fs::read_to_string(docs.dir().join("skills.md"))
                .expect("skills docs")
                .contains("$LASH_HOME/skills")
        );
    }

    #[test]
    fn prompt_content_points_at_docs_dir() {
        let docs_dir = PathBuf::from("/tmp/lash-home/docs/lash-cli");

        let content = host_docs_prompt_content(&docs_dir);

        assert!(content.contains("/tmp/lash-home/docs/lash-cli"));
        assert!(content.contains("read the relevant markdown file"));
        assert!(content.contains("override general model memory"));
    }

    #[tokio::test]
    async fn plugin_contributes_docs_prompt() {
        let docs_dir = PathBuf::from("/tmp/lash-home/docs/lash-cli");
        let plugin_host = lash::PluginHost::new(vec![
            Arc::new(lash_mode_standard::BuiltinStandardModePluginFactory),
            Arc::new(HostDocsPluginFactory::new(docs_dir.clone())),
        ]);
        let session = plugin_host
            .build_standard_session("root", None)
            .expect("session");

        let contributions = session
            .collect_prompt_contributions(lash::PromptHookContext {
                session_id: "root".to_string(),
                host: Arc::new(lash::testing::MockSessionManager::default()),
                state: lash::SessionReadView::new(lash::SessionStateEnvelope::default()),
                rlm_termination: lash::RlmTermination::default(),
            })
            .await
            .expect("prompt contributions");

        let contribution = contributions
            .iter()
            .find(|contribution| contribution.title.as_deref() == Some("Lash CLI Host Docs"))
            .expect("host docs contribution");
        assert_eq!(contribution.slot, lash::PromptSlot::Environment);
        assert!(
            contribution
                .content
                .contains(&docs_dir.display().to_string())
        );
    }
}
