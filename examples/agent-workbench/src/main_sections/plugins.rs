struct WorkbenchPluginFactory {
    tavily_api_key: String,
    mail_world: mail::MailWorld,
}

impl WorkbenchPluginFactory {
    fn new(tavily_api_key: impl Into<String>) -> Self {
        Self {
            tavily_api_key: tavily_api_key.into(),
            mail_world: mail::MailWorld::new(),
        }
    }

    fn with_mail_world(mut self, mail_world: mail::MailWorld) -> Self {
        self.mail_world = mail_world;
        self
    }
}

impl PluginFactory for WorkbenchPluginFactory {
    fn id(&self) -> &'static str {
        "agent_workbench"
    }

    fn extension_contributions(&self) -> Vec<lash::plugins::PluginExtensionContribution> {
        vec![
            lash::plugins::PluginExtensionContribution::new(
                lash::modes::LASHLANG_SURFACE_EXTENSION_ID,
                lash::modes::LashlangSurfaceContribution::new(
                    workbench_lashlang_abilities(),
                    lash::modes::LashlangLanguageFeatures::default(),
                    workbench_lashlang_resources(),
                ),
            )
            .expect("workbench lashlang surface serializes"),
        ]
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(WorkbenchSessionPlugin {
            tavily_api_key: self.tavily_api_key.clone(),
            mail_world: self.mail_world.clone(),
        }))
    }
}

struct WorkbenchSessionPlugin {
    tavily_api_key: String,
    mail_world: mail::MailWorld,
}

impl SessionPlugin for WorkbenchSessionPlugin {
    fn id(&self) -> &'static str {
        "agent_workbench"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let mail_world = self.mail_world.clone();
        reg.prompt().contribute(Arc::new(move |_ctx| {
            let mail_world = mail_world.clone();
            Box::pin(async move {
                Ok(vec![PromptContribution::environment(
                    "Agent Workbench",
                    format!(
                        "{WORKBENCH_PROMPT}\n\n{}",
                        connected_accounts_prompt(&mail_world)
                    ),
                )])
            })
        }));
        reg.triggers().declare(TriggerEvent::new(
            BUTTON_TRIGGER_RESOURCE,
            BUTTON_TRIGGER_ALIAS,
            BUTTON_TRIGGER_EVENT,
            button_trigger_payload_schema(),
        ))?;
        reg.triggers().declare(TriggerEvent::new(
            MAIL_EVENT_RESOURCE,
            MAIL_EVENT_ALIAS,
            MAIL_EVENT_EVENT,
            mail_received_payload_schema(),
        ))?;
        reg.tools()
            .provider(Arc::new(lash_tools::web::web_search_provider(
                self.tavily_api_key.clone(),
            )))?;
        reg.tools()
            .provider(Arc::new(lash_tools::web::fetch_url_provider(
                self.tavily_api_key.clone(),
            )))?;
        reg.tools().provider(Arc::new(mail::MockMailProvider::new(
            self.mail_world.clone(),
        )))?;
        Ok(())
    }
}

fn workbench_lashlang_resources() -> lashlang::LashlangHostCatalog {
    let mut resources = lashlang::LashlangHostCatalog::new();
    resources
        .add_trigger_source_constructor(
            CRON_SCHEDULE_SOURCE_TYPE.split('.'),
            lashlang::TypeExpr::Object(vec![
                lashlang::TypeField {
                    name: "expr".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: false,
                },
                lashlang::TypeField {
                    name: "tz".into(),
                    ty: lashlang::TypeExpr::Str,
                    optional: true,
                },
            ]),
            cron_tick_event_type(),
        )
        .expect("valid cron trigger source");
    resources
        .add_trigger_source_constructor(
            BUTTON_TRIGGER_SOURCE_TYPE.split('.'),
            lashlang::TypeExpr::Object(vec![]),
            button_pressed_event_type(),
        )
        .expect("valid button trigger source");
    resources
        .add_trigger_source_constructor(
            MAIL_RECEIVED_SOURCE_TYPE.split('.'),
            lashlang::TypeExpr::Object(vec![]),
            mail_received_event_type(),
        )
        .expect("valid mail trigger source");
    resources
}

fn button_pressed_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "ui.button.Pressed",
        vec![
            field("button", lashlang::TypeExpr::Str),
            field("message", lashlang::TypeExpr::Str),
            field("pressed_at", lashlang::TypeExpr::Str),
        ],
    )
    .expect("valid button pressed event type")
}

fn mail_received_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "mail.Received",
        vec![
            field("account", lashlang::TypeExpr::Str),
            field("title", lashlang::TypeExpr::Str),
            field("text", lashlang::TypeExpr::Str),
        ],
    )
    .expect("valid mail received event type")
}

fn mail_received_payload_schema() -> lash::triggers::LashSchema {
    lash::triggers::LashSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "account": { "type": "string" },
            "title": { "type": "string" },
            "text": { "type": "string" }
        },
        "required": ["account", "title", "text"],
        "additionalProperties": false
    }))
}

fn button_trigger_payload_schema() -> lash::triggers::LashSchema {
    lash::triggers::LashSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "button": { "type": "string", "enum": ["Red", "Blue"] },
            "message": { "type": "string" },
            "pressed_at": { "type": "string" }
        },
        "required": ["button", "message", "pressed_at"],
        "additionalProperties": false
    }))
}

fn field(name: &str, ty: lashlang::TypeExpr) -> lashlang::TypeField {
    lashlang::TypeField {
        name: name.into(),
        ty,
        optional: false,
    }
}

/// Live, per-turn prompt line naming the inbox authorities that actually exist,
/// so the agent never assumes the illustrative `inbox.work`/`inbox.personal`
/// names from the static guidance are real.
fn connected_accounts_prompt(mail_world: &mail::MailWorld) -> String {
    let accounts = mail_world.account_summaries();
    if accounts.is_empty() {
        return "Connected inbox accounts: none yet. The `inbox` namespace is empty until the \
            user adds an account from the Accounts tab, so `inbox.<anything>` will not resolve. \
            If asked to use an inbox, tell the user to add one first instead of guessing a name."
            .to_string();
    }
    let list = accounts
        .iter()
        .map(|account| account.authority.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Connected inbox authorities right now: {list}. These are the ONLY inbox accounts that \
        exist — use these exact paths and never reference any other `inbox.<name>`. The \
        `inbox.work` / `inbox.personal` names used in the examples above are illustrative only; \
        substitute the real authorities listed here."
    )
}

fn cron_tick_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "cron.Tick",
        vec![lashlang::TypeField {
            name: "fired_at".into(),
            ty: lashlang::TypeExpr::Str,
            optional: false,
        }],
    )
    .expect("valid cron tick type")
}
