//! Mocked multi-account inbox world for the workbench demo.
//!
//! The host owns a small in-memory set of named inboxes. Each is projected into
//! the RLM Lashlang surface as a typed module authority of type `Inbox` at
//! module path `inbox.<slug>`, exposing three operations:
//!
//! - `send({ title, text })` — add a message to that inbox
//! - `list({})` — list the messages in that inbox
//! - `delete({ id })` — remove a message by id
//!
//! Because every account shares the `Inbox` authority type, a single
//! account-parametric process such as `process triage(box: Inbox) { ... }` can
//! be started against any account (`start triage(box: inbox.work)`), which is
//! the point of the multi-account showcase.
//!
//! Accounts are added at runtime from the UI. The provider reads the live
//! account set in [`MockMailProvider::definitions`], and the runtime rebuilds
//! the tool surface on the next opened turn, so newly added accounts appear as
//! authorities without any explicit refresh.

use std::sync::{Arc, RwLock};

use crate::{MAIL_EVENT_ALIAS, MAIL_EVENT_EVENT, MAIL_EVENT_RESOURCE};
use async_trait::async_trait;
use lash::tools::{
    ToolAgentSurface, ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider,
    ToolResult, ToolScheduling,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Operations every inbox authority exposes. Order is the surface order.
const MAIL_OPERATIONS: [&str; 3] = ["send", "list", "delete"];

/// One stored message: just a title and body text.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct MailMessage {
    pub id: String,
    pub title: String,
    pub text: String,
}

impl MailMessage {
    fn value(&self) -> Value {
        json!({ "id": self.id, "title": self.title, "text": self.text })
    }
}

/// One delivered mock message, used to build the `mail.received` host event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MailDelivery {
    pub account: String,
    pub title: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub(crate) struct DeliveredMail {
    pub message: MailMessage,
    pub delivery: MailDelivery,
}

struct Account {
    slug: String,
    display_name: String,
    messages: Vec<MailMessage>,
    next_id: u64,
}

impl Account {
    fn append(&mut self, title: &str, text: &str) -> MailMessage {
        let id = format!("{}-{}", self.slug, self.next_id);
        self.next_id += 1;
        let message = MailMessage {
            id,
            title: non_empty(title).unwrap_or("(no title)").to_string(),
            text: text.trim().to_string(),
        };
        self.messages.push(message.clone());
        message
    }

    fn summary(&self) -> AccountSummary {
        AccountSummary {
            slug: self.slug.clone(),
            display_name: self.display_name.clone(),
            authority: format!("inbox.{}", self.slug),
            total: self.messages.len(),
        }
    }
}

/// UI-facing account row.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct AccountSummary {
    pub slug: String,
    pub display_name: String,
    /// Lashlang authority path the agent calls, e.g. `inbox.work`.
    pub authority: String,
    pub total: usize,
}

/// The shared, mutable mock inbox world. Cloneable handle around the store.
#[derive(Clone, Default)]
pub(crate) struct MailWorld {
    inner: Arc<RwLock<Vec<Account>>>,
}

impl MailWorld {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add an account from a human-entered name. Returns the created summary or
    /// a human-readable error (empty/duplicate/invalid name).
    pub(crate) fn add_account(&self, name: &str) -> Result<AccountSummary, String> {
        let display_name = name.trim().to_string();
        if display_name.is_empty() {
            return Err("account name is required".to_string());
        }
        let slug = slugify(&display_name)
            .ok_or_else(|| "account name must contain a letter or digit".to_string())?;
        let mut accounts = self.inner.write().expect("mail world lock");
        if accounts.iter().any(|account| account.slug == slug) {
            return Err(format!("account `{slug}` already exists"));
        }
        let account = Account {
            slug,
            display_name,
            messages: Vec::new(),
            next_id: 1,
        };
        let summary = account.summary();
        accounts.push(account);
        Ok(summary)
    }

    pub(crate) fn account_summaries(&self) -> Vec<AccountSummary> {
        self.inner
            .read()
            .expect("mail world lock")
            .iter()
            .map(Account::summary)
            .collect()
    }

    /// Remove an account and its messages. Returns an error if unknown.
    pub(crate) fn remove_account(&self, slug: &str) -> Result<(), String> {
        let mut accounts = self.inner.write().expect("mail world lock");
        let before = accounts.len();
        accounts.retain(|account| account.slug != slug);
        if accounts.len() == before {
            return Err(format!("unknown account `{slug}`"));
        }
        Ok(())
    }

    /// Deliver a message into an account. UI injects and agent `send` tools both
    /// use this path so storage, ids, and `mail.received` payloads match.
    pub(crate) fn deliver(
        &self,
        slug: &str,
        title: &str,
        text: &str,
    ) -> Result<DeliveredMail, String> {
        let mut accounts = self.inner.write().expect("mail world lock");
        let account = find_mut(&mut accounts, slug)?;
        let message = account.append(title, text);
        Ok(DeliveredMail {
            delivery: MailDelivery {
                account: slug.to_string(),
                title: message.title.clone(),
                text: message.text.clone(),
            },
            message,
        })
    }

    /// Messages for an account, newest first (for the UI).
    pub(crate) fn inbox(&self, slug: &str) -> Result<Vec<MailMessage>, String> {
        let accounts = self.inner.read().expect("mail world lock");
        let account = find(&accounts, slug)?;
        let mut messages = account.messages.clone();
        messages.reverse();
        Ok(messages)
    }

    /// Remove a single message by id.
    pub(crate) fn remove_message(&self, slug: &str, id: &str) -> Result<(), String> {
        let mut accounts = self.inner.write().expect("mail world lock");
        let account = find_mut(&mut accounts, slug)?;
        let before = account.messages.len();
        account.messages.retain(|message| message.id != id);
        if account.messages.len() == before {
            return Err(format!("no message `{id}` in `{slug}`"));
        }
        Ok(())
    }

    // --- Tool operation backends (called from MockMailProvider::execute) ---

    fn op_send(&self, slug: &str, args: &Value) -> Result<DeliveredMail, String> {
        let title = args
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let text = args.get("text").and_then(Value::as_str).unwrap_or_default();
        self.deliver(slug, title, text)
    }

    fn op_list(&self, slug: &str, _args: &Value) -> Result<Value, String> {
        let accounts = self.inner.read().expect("mail world lock");
        let account = find(&accounts, slug)?;
        let mut messages: Vec<&MailMessage> = account.messages.iter().collect();
        messages.reverse();
        Ok(json!({
            "account": slug,
            "messages": messages.iter().map(|m| m.value()).collect::<Vec<_>>(),
        }))
    }

    fn op_delete(&self, slug: &str, args: &Value) -> Result<Value, String> {
        let id = args
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "delete requires an `id`".to_string())?;
        self.remove_message(slug, id)?;
        Ok(json!({ "account": slug, "id": id, "deleted": true }))
    }
}

fn find<'a>(accounts: &'a [Account], slug: &str) -> Result<&'a Account, String> {
    accounts
        .iter()
        .find(|account| account.slug == slug)
        .ok_or_else(|| format!("unknown account `{slug}`"))
}

fn find_mut<'a>(accounts: &'a mut [Account], slug: &str) -> Result<&'a mut Account, String> {
    accounts
        .iter_mut()
        .find(|account| account.slug == slug)
        .ok_or_else(|| format!("unknown account `{slug}`"))
}

fn non_empty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// Turn a human account name into a Lashlang module-path segment
/// (`[a-z][a-z0-9_]*`). Returns `None` if nothing usable remains.
pub(crate) fn slugify(name: &str) -> Option<String> {
    let mut slug = String::new();
    let mut last_underscore = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !slug.is_empty() && !last_underscore {
            slug.push('_');
            last_underscore = true;
        }
    }
    while slug.ends_with('_') {
        slug.pop();
    }
    if slug.is_empty() {
        return None;
    }
    if slug.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        slug.insert(0, 'a');
    }
    Some(slug)
}

/// Tool name carrying the account slug and operation, e.g.
/// `inbox__work__send`. A double underscore separates the fixed parts so the
/// slug (which may itself contain single underscores) stays unambiguous.
fn tool_name(slug: &str, operation: &str) -> String {
    format!("inbox__{slug}__{operation}")
}

fn operation_schemas(operation: &str) -> (Value, &'static str) {
    match operation {
        "send" => (
            json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "text": { "type": "string" }
                },
                "required": ["title"],
                "additionalProperties": false
            }),
            "Add a message to this inbox.",
        ),
        "list" => (
            json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            "List the messages in this inbox.",
        ),
        "delete" => (
            json!({
                "type": "object",
                "properties": { "id": { "type": "string" } },
                "required": ["id"],
                "additionalProperties": false
            }),
            "Delete a message from this inbox by id.",
        ),
        _ => (json!({ "type": "object" }), ""),
    }
}

fn definition_for(slug: &str, display_name: &str, operation: &str) -> ToolDefinition {
    let name = tool_name(slug, operation);
    let (input_schema, summary) = operation_schemas(operation);
    let description = format!("{summary} Account `{display_name}` (inbox.{slug}).");
    ToolDefinition::raw(
        format!("tool:{name}"),
        name,
        description,
        input_schema,
        json!({ "type": "object" }),
    )
    .with_agent_surface(
        ToolAgentSurface::new(["inbox", slug], operation).with_authority_type("Inbox"),
    )
    .with_scheduling(ToolScheduling::Parallel)
}

/// Dynamic provider: one `Inbox` authority per account, three operations each.
pub(crate) struct MockMailProvider {
    world: MailWorld,
}

impl MockMailProvider {
    pub(crate) fn new(world: MailWorld) -> Self {
        Self { world }
    }

    /// Build the live tool definitions from the current account set.
    fn definitions(&self) -> Vec<ToolDefinition> {
        let summaries = self.world.account_summaries();
        let mut defs = Vec::with_capacity(summaries.len() * MAIL_OPERATIONS.len());
        for summary in summaries {
            for operation in MAIL_OPERATIONS {
                defs.push(definition_for(
                    &summary.slug,
                    &summary.display_name,
                    operation,
                ));
            }
        }
        defs
    }

    /// Resolve a tool name back to (slug, operation) by parsing it, without
    /// consulting the live account set. Restoring a persisted session must be
    /// able to rebind inbox tools whose account has since been removed; the
    /// live surface (`tool_manifests`) lists only current accounts, so the
    /// next tool-surface refresh drops the stale entries, and executing a
    /// stale tool fails with the world's unknown-account error.
    fn route(&self, name: &str) -> Option<(String, &'static str)> {
        let rest = name.strip_prefix("inbox__")?;
        for operation in MAIL_OPERATIONS {
            if let Some(slug) = rest.strip_suffix(&format!("__{operation}"))
                && !slug.is_empty()
            {
                return Some((slug.to_string(), operation));
            }
        }
        None
    }

    /// Definition for any well-formed inbox tool name (the tool id is derived
    /// from the name, so rebinding persisted state matches by id even when
    /// the account is gone). Uses the live display name when available.
    fn definition_for_name(&self, name: &str) -> Option<ToolDefinition> {
        let (slug, operation) = self.route(name)?;
        let display_name = self
            .world
            .account_summaries()
            .into_iter()
            .find(|summary| summary.slug == slug)
            .map(|summary| summary.display_name)
            .unwrap_or_else(|| slug.clone());
        Some(definition_for(&slug, &display_name, operation))
    }
}

#[async_trait]
impl ToolProvider for MockMailProvider {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.definitions()
            .iter()
            .map(ToolDefinition::manifest)
            .collect()
    }

    fn resolve_manifest(&self, name: &str) -> Option<ToolManifest> {
        self.definition_for_name(name)
            .as_ref()
            .map(ToolDefinition::manifest)
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.definition_for_name(name)
            .map(|def| Arc::new(def.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let Some((slug, operation)) = self.route(call.name) else {
            return ToolResult::err_fmt(format_args!("unknown inbox tool `{}`", call.name));
        };
        let result = match operation {
            "send" => match self.world.op_send(&slug, call.args) {
                Ok(delivered) => {
                    let payload = match serde_json::to_value(&delivered.delivery) {
                        Ok(payload) => payload,
                        Err(err) => return ToolResult::err_fmt(err.to_string()),
                    };
                    if let Err(err) = call
                        .context
                        .host_events()
                        .emit(
                            MAIL_EVENT_RESOURCE,
                            MAIL_EVENT_ALIAS,
                            MAIL_EVENT_EVENT,
                            payload,
                        )
                        .await
                    {
                        return ToolResult::err_fmt(err.to_string());
                    }
                    Ok(json!({ "account": slug, "id": delivered.message.id }))
                }
                Err(err) => Err(err),
            },
            "list" => self.world.op_list(&slug, call.args),
            "delete" => self.world.op_delete(&slug, call.args),
            other => Err(format!("unsupported inbox operation `{other}`")),
        };
        match result {
            Ok(value) => ToolResult::ok(value),
            Err(message) => ToolResult::err_fmt(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_normalizes_names() {
        assert_eq!(slugify("Work").as_deref(), Some("work"));
        assert_eq!(slugify("Personal Mail").as_deref(), Some("personal_mail"));
        assert_eq!(slugify("  spaced  out  ").as_deref(), Some("spaced_out"));
        assert_eq!(slugify("2024 inbox").as_deref(), Some("a2024_inbox"));
        assert_eq!(slugify("!!!"), None);
    }

    #[test]
    fn add_account_rejects_duplicates_and_blanks() {
        let world = MailWorld::new();
        assert!(world.add_account("Work").is_ok());
        assert!(world.add_account("work").is_err());
        assert!(world.add_account("   ").is_err());
        assert_eq!(world.account_summaries().len(), 1);
    }

    #[test]
    fn send_list_and_delete() {
        let world = MailWorld::new();
        world.add_account("Work").expect("add work");

        let sent = world
            .op_send(
                "work",
                &json!({ "title": "Contract", "text": "Please review." }),
            )
            .expect("send");
        let id = sent.message.id.clone();
        assert_eq!(world.account_summaries()[0].total, 1);
        assert_eq!(world.account_summaries()[0].authority, "inbox.work");

        let listed = world.op_list("work", &json!({})).expect("list");
        let messages = listed["messages"].as_array().expect("array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["title"], json!("Contract"));
        assert_eq!(messages[0]["text"], json!("Please review."));

        world
            .op_delete("work", &json!({ "id": id }))
            .expect("delete");
        assert_eq!(world.account_summaries()[0].total, 0);
    }

    #[test]
    fn provider_exposes_authority_per_account() {
        let world = MailWorld::new();
        world.add_account("Work").expect("add work");
        world.add_account("Personal").expect("add personal");
        let provider = MockMailProvider::new(world);

        let names: Vec<String> = provider
            .tool_manifests()
            .into_iter()
            .map(|manifest| manifest.name)
            .collect();
        assert!(names.contains(&"inbox__work__send".to_string()));
        assert!(names.contains(&"inbox__personal__delete".to_string()));
        assert_eq!(names.len(), 6);

        let manifest = provider
            .tool_manifests()
            .into_iter()
            .find(|manifest| manifest.name == "inbox__work__send")
            .expect("work send manifest");
        let surface = manifest.agent_surface.executable_for(&manifest.name);
        assert_eq!(surface.call_path(), "inbox.work.send");
        assert_eq!(surface.authority_type, "Inbox");

        assert_eq!(
            provider.route("inbox__personal__delete"),
            Some(("personal".to_string(), "delete"))
        );
    }
}
