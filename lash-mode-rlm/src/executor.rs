use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use lash_core::plugin::project_observation_text;
use lash_core::{
    AttachmentRef, ExecRequest, ExecResponse, ModeExecutionContext, ModeToolBatchItem,
    ModeToolReply, SessionError, TextProjectionMetadata, ToolImage, ToolOutputBudgetConfig,
};
use lash_rlm_types::PROJECTED_JSON_TAG;
use lashlang::{
    CompiledProgramCache, ExecutionOutcome, ExecutionScratch, ImageValue, ProjectedBindings,
    ProjectedFuture, ProjectedHostValue, ProjectedRead, ProjectedValue, Record as FlowRecord,
    State as FlowState, ToolHost, ToolHostCall, ToolHostError, Value as FlowValue,
};
use serde_json::{Value, json};

use crate::projected_bindings::{
    RLM_TURN_INPUT_PLUGIN_ID, RlmProjectedBindings, RlmProjectionExtension,
};
use crate::projection::{RlmHistoryProjection, rlm_history_projection};

const RLM_SNAPSHOT_VERSION: u32 = 3;

pub struct RlmExecutionState {
    rlm: FlowState,
    program_cache: CompiledProgramCache,
    scratch: ExecutionScratch,
    scratch_dir: tempfile::TempDir,
    observe_projection: ToolOutputBudgetConfig,
    dirty: bool,
}

impl RlmExecutionState {
    pub fn new(config: ToolOutputBudgetConfig) -> Result<Self, SessionError> {
        Ok(Self {
            rlm: FlowState::new(),
            program_cache: CompiledProgramCache::default(),
            scratch: ExecutionScratch::new(),
            scratch_dir: tempfile::TempDir::new()?,
            observe_projection: config,
            dirty: true,
        })
    }

    pub fn execution_state_dirty(&self) -> bool {
        self.dirty
    }

    pub fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        let vars = snapshot_runtime(&self.rlm).map_err(SessionError::Protocol)?;
        let files = collect_files(self.scratch_dir.path()).unwrap_or_default();
        let combined = json!({
            "version": RLM_SNAPSHOT_VERSION,
            "engine": "lashlang",
            "vars": vars,
            "files": files,
        });
        self.dirty = false;
        Ok(Some(serde_json::to_vec(&combined)?))
    }

    pub fn restore_execution_state(&mut self, data: &[u8]) -> Result<(), SessionError> {
        let parsed: serde_json::Value = serde_json::from_slice(data).unwrap_or(json!({}));

        if parsed.get("version").is_none() || parsed.get("engine").is_none() {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot format".to_string(),
            ));
        }
        if parsed.get("version").and_then(|v| v.as_u64()) != Some(RLM_SNAPSHOT_VERSION as u64) {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot version".to_string(),
            ));
        }
        if parsed.get("engine").and_then(|v| v.as_str()) != Some("lashlang") {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot engine".to_string(),
            ));
        }

        let vars_str = parsed
            .get("vars")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        self.rlm = restore_runtime(&vars_str)
            .map_err(|err| SessionError::Protocol(format!("executor restore failed: {err}")))?;
        prune_reserved_projected_bindings(&mut self.rlm);

        if let Some(files_val) = parsed.get("files")
            && let Ok(files) = serde_json::from_value::<HashMap<String, String>>(files_val.clone())
        {
            clear_dir(self.scratch_dir.path());
            let _ = restore_files(self.scratch_dir.path(), &files);
        }
        self.dirty = true;
        Ok(())
    }

    pub fn prune_protected_globals(&mut self, protected_names: &BTreeSet<String>) {
        prune_protected_bindings(&mut self.rlm, protected_names);
    }

    pub fn patch_globals(
        &mut self,
        patch: &lash_rlm_types::RlmGlobalsPatchPluginBody,
        protected_names: &BTreeSet<String>,
    ) -> Result<(), SessionError> {
        if patch.is_empty() {
            return Ok(());
        }
        apply_global_defaults(&mut self.rlm, patch, protected_names)
            .map_err(SessionError::Protocol)?;
        self.dirty = true;
        Ok(())
    }
}

pub async fn execute_code(
    mut state: RlmExecutionState,
    ctx: ModeExecutionContext,
    request: ExecRequest,
    session_projected_bindings: RlmProjectedBindings,
) -> Result<(RlmExecutionState, ExecResponse), SessionError> {
    let start = std::time::Instant::now();
    let clean_code = clean_model_code(&request.code);
    let response = execute_code_inner(
        &mut state,
        ctx,
        &clean_code,
        start,
        session_projected_bindings,
    )
    .await;
    Ok((state, response))
}

fn clean_model_code(code: &str) -> String {
    code.lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.is_empty()
                || trimmed
                    .trim_matches('-')
                    .chars()
                    .any(|c| !c.is_whitespace())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn execute_code_inner(
    state: &mut RlmExecutionState,
    ctx: ModeExecutionContext,
    code: &str,
    start: std::time::Instant,
    session_projected_bindings: RlmProjectedBindings,
) -> ExecResponse {
    state.dirty = true;
    let compiled = match state.program_cache.get_or_compile(code) {
        Ok(compiled) => compiled,
        Err(err) => {
            return ExecResponse {
                output: String::new(),
                observations: Vec::new(),
                observation_truncation: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                printed_images: Vec::new(),
                error: Some(format_parse_error(code, &err)),
                duration_ms: start.elapsed().as_millis() as u64,
                terminal_finish: None,
            };
        }
    };

    let projected = match projected_bindings(&ctx, session_projected_bindings) {
        Ok(projected) => projected,
        Err(err) => {
            return ExecResponse {
                output: String::new(),
                observations: Vec::new(),
                observation_truncation: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                printed_images: Vec::new(),
                error: Some(err),
                duration_ms: start.elapsed().as_millis() as u64,
                terminal_finish: None,
            };
        }
    };
    let projected_names = projected.names().collect::<Vec<_>>();
    prune_projected_binding_names(&mut state.rlm, projected_names.iter().map(String::as_str));
    let tool_result_projectors = tool_result_projectors(&ctx);
    let host = HostBridge {
        ctx,
        observe_projection: state.observe_projection.clone(),
        tool_result_projectors,
        observations: Mutex::new(Vec::new()),
        observation_truncation: Mutex::new(Vec::new()),
        printed_images: Mutex::new(Vec::new()),
        tool_calls: Mutex::new(Vec::new()),
        tool_images: Mutex::new(Vec::new()),
        next_tool_index: Mutex::new(0),
    };

    let result = lashlang::execute_compiled_traced_with_scratch_and_projected_bindings(
        &compiled,
        &mut state.rlm,
        &host,
        &mut state.scratch,
        &projected,
    )
    .await;
    let terminal_finish = match result {
        Ok(ExecutionOutcome::Finished(value)) => Some(flow_to_json_value(&value).await),
        Ok(ExecutionOutcome::Continued) => None,
        Err(failure) => {
            let collected = host.into_collected();
            return ExecResponse {
                output: String::new(),
                observations: collected.observations,
                observation_truncation: collected.observation_truncation,
                tool_calls: collected.tool_calls,
                images: collected.tool_images,
                printed_images: collected.printed_images,
                error: Some(lashlang::format_runtime_diagnostic(
                    code,
                    &failure.error,
                    failure.span,
                )),
                duration_ms: start.elapsed().as_millis() as u64,
                terminal_finish: None,
            };
        }
    };
    let collected = host.into_collected();
    ExecResponse {
        output: String::new(),
        observations: collected.observations,
        observation_truncation: collected.observation_truncation,
        tool_calls: collected.tool_calls,
        images: collected.tool_images,
        printed_images: collected.printed_images,
        error: None,
        duration_ms: start.elapsed().as_millis() as u64,
        terminal_finish,
    }
}

fn projected_bindings(
    ctx: &ModeExecutionContext,
    session_bindings: RlmProjectedBindings,
) -> Result<ProjectedBindings, String> {
    let mut bindings = ProjectedBindings::new();
    bindings
        .try_insert(
            "history",
            ProjectedValue::custom(
                "history",
                Arc::new(HistoryProjectedValue {
                    projection: Arc::new(rlm_history_projection(
                        ctx.chronological_projection().as_ref(),
                    )),
                }),
            ),
        )
        .map_err(|err| format!("`{}` is reserved as an RLM built-in binding", err.name()))?;
    insert_projected_bindings(&mut bindings, session_bindings)?;
    if let Some(extension) = ctx
        .turn_context()
        .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
    {
        insert_projected_bindings(&mut bindings, extension.bindings.clone())?;
    }
    Ok(bindings)
}

fn insert_projected_bindings(
    target: &mut ProjectedBindings,
    bindings: RlmProjectedBindings,
) -> Result<(), String> {
    let host_bindings = bindings.into_projected_bindings();
    for name in host_bindings.names().collect::<Vec<_>>() {
        let value = host_bindings
            .get(&name)
            .expect("name came from projected bindings");
        target.try_insert(name, value).map_err(|err| {
            format!(
                "`{}` is already bound as an RLM projected binding",
                err.name()
            )
        })?;
    }
    Ok(())
}

fn tool_result_projectors(ctx: &ModeExecutionContext) -> Vec<crate::RlmToolResultProjector> {
    ctx.turn_context()
        .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
        .map(|extension| extension.tool_result_projectors.clone())
        .unwrap_or_default()
}

struct HistoryProjectedValue {
    projection: Arc<RlmHistoryProjection>,
}

impl ProjectedHostValue for HistoryProjectedValue {
    fn type_name(&self) -> &str {
        "list"
    }

    fn len(&self) -> ProjectedFuture<'_, Option<usize>> {
        Box::pin(async { Some(self.projection.len()) })
    }

    fn get_index(&self, index: FlowValue) -> ProjectedFuture<'_, ProjectedRead> {
        Box::pin(async move {
            let Ok(Some(index)) = projected_index(&index, self.projection.len()) else {
                return ProjectedRead::Missing;
            };
            self.projection
                .item(index)
                .and_then(|item| serde_json::to_value(item).ok())
                .map(json_to_flow_value)
                .map(ProjectedRead::Value)
                .unwrap_or(ProjectedRead::Missing)
        })
    }

    fn render(&self) -> ProjectedFuture<'_, String> {
        Box::pin(async move {
            serde_json::to_string(self.projection.history()).unwrap_or_else(|_| "[]".to_string())
        })
    }

    fn materialize(&self) -> ProjectedFuture<'_, FlowValue> {
        Box::pin(async move { json_to_flow_value(self.projection.value()) })
    }
}

fn projected_index(index: &FlowValue, len: usize) -> Result<Option<usize>, ()> {
    let FlowValue::Number(index) = index else {
        return Err(());
    };
    if !index.is_finite() || index.fract() != 0.0 {
        return Err(());
    }
    let len = len as isize;
    let index = *index as isize;
    let normalized = if index < 0 { len + index } else { index };
    if normalized < 0 || normalized >= len {
        return Ok(None);
    }
    Ok(Some(normalized as usize))
}

struct HostBridge {
    ctx: ModeExecutionContext,
    observe_projection: ToolOutputBudgetConfig,
    tool_result_projectors: Vec<crate::RlmToolResultProjector>,
    observations: Mutex<Vec<String>>,
    observation_truncation: Mutex<Vec<TextProjectionMetadata>>,
    printed_images: Mutex<Vec<AttachmentRef>>,
    tool_calls: Mutex<Vec<lash_core::ToolCallRecord>>,
    tool_images: Mutex<Vec<ToolImage>>,
    next_tool_index: Mutex<usize>,
}

impl ToolHost for HostBridge {
    async fn call(&self, name: String, args: FlowRecord) -> Result<FlowValue, ToolHostError> {
        let index = self.next_index();
        let reply = self
            .ctx
            .call_tool(
                uuid::Uuid::new_v4().to_string(),
                name.clone(),
                self.tool_payload(&args).await,
                index,
            )
            .await;
        self.consume_reply(&name, reply)
    }

    async fn call_batch(&self, calls: Vec<ToolHostCall>) -> Vec<Result<FlowValue, ToolHostError>> {
        if calls.is_empty() {
            return Vec::new();
        }
        let mut batch = Vec::with_capacity(calls.len());
        for call in &calls {
            batch.push(ModeToolBatchItem {
                id: uuid::Uuid::new_v4().to_string(),
                name: call.name.clone(),
                args: self.tool_payload(&call.args).await,
            });
        }
        let replies = self.ctx.call_tool_batch(batch).await;
        if replies.len() != calls.len() {
            return calls
                .into_iter()
                .map(|_| {
                    Err(ToolHostError::new(
                        "tool batch returned the wrong number of results",
                    ))
                })
                .collect();
        }
        replies
            .into_iter()
            .zip(calls)
            .map(|(reply, call)| self.consume_reply(&call.name, reply))
            .collect()
    }

    async fn start_call(&self, name: String, args: FlowRecord) -> Result<FlowValue, ToolHostError> {
        let reply = self
            .ctx
            .start_tool_call(
                uuid::Uuid::new_v4().to_string(),
                name.clone(),
                self.tool_payload(&args).await,
            )
            .await;
        self.consume_reply(&name, reply)
    }

    async fn await_handle(&self, handle: FlowValue) -> Result<FlowValue, ToolHostError> {
        let reply = self
            .ctx
            .await_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                flow_to_json_value(&handle).await,
            )
            .await;
        self.consume_reply("await_handle", reply)
    }

    async fn cancel_handle(&self, handle: FlowValue) -> Result<FlowValue, ToolHostError> {
        let reply = self
            .ctx
            .cancel_tool_handle(
                uuid::Uuid::new_v4().to_string(),
                flow_to_json_value(&handle).await,
            )
            .await;
        self.consume_reply("cancel_handle", reply)
    }

    async fn print(&self, value: FlowValue) -> Result<(), ToolHostError> {
        let attachment_store = self.ctx.attachment_store();
        let images = collect_printed_images(&value, attachment_store.as_ref()).await?;
        let raw_text = format_output_value(&value).await;
        let (_projected_text, metadata) =
            project_observation_text(&raw_text, &self.observe_projection);
        self.observations
            .lock()
            .map_err(|_| ToolHostError::new("observation buffer poisoned"))?
            .push(raw_text);
        self.observation_truncation
            .lock()
            .map_err(|_| ToolHostError::new("observation metadata buffer poisoned"))?
            .push(metadata);
        if !images.is_empty() {
            self.printed_images
                .lock()
                .map_err(|_| ToolHostError::new("printed image buffer poisoned"))?
                .extend(images);
        }
        Ok(())
    }

    async fn yield_now(&self) {
        tokio::task::yield_now().await;
    }
}

impl HostBridge {
    fn next_index(&self) -> usize {
        let mut guard = self
            .next_tool_index
            .lock()
            .expect("tool index lock poisoned");
        let next = *guard;
        *guard += 1;
        next
    }

    async fn tool_payload(&self, args: &FlowRecord) -> Value {
        let mut payload = flow_record_to_json_value(args).await;
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__session_id__".to_string())
                .or_insert_with(|| Value::String(self.ctx.session_id().to_string()));
        }
        payload
    }

    fn consume_reply(
        &self,
        tool_name: &str,
        reply: ModeToolReply,
    ) -> Result<FlowValue, ToolHostError> {
        let projected_tool_name = reply
            .record
            .as_ref()
            .map(|record| record.tool.as_str())
            .unwrap_or(tool_name)
            .to_string();
        if let Some(record) = reply.record {
            self.tool_calls
                .lock()
                .map_err(|_| ToolHostError::new("tool call buffer poisoned"))?
                .push(record);
        }
        if !reply.images.is_empty() {
            self.tool_images
                .lock()
                .map_err(|_| ToolHostError::new("tool image buffer poisoned"))?
                .extend(reply.images.clone());
        }
        if reply.success {
            for projector in &self.tool_result_projectors {
                if let Some(value) = projector(&projected_tool_name, &reply.value) {
                    return Ok(value);
                }
            }
            Ok(lift_tool_result_to_flow_value(
                reply.value,
                reply.images,
                self.ctx.attachment_store().as_ref(),
            ))
        } else {
            Err(ToolHostError::new(tool_error_message(reply.value)))
        }
    }

    fn into_collected(self) -> CollectedExecutionOutput {
        CollectedExecutionOutput {
            observations: self.observations.into_inner().unwrap_or_default(),
            observation_truncation: self.observation_truncation.into_inner().unwrap_or_default(),
            printed_images: self.printed_images.into_inner().unwrap_or_default(),
            tool_calls: self.tool_calls.into_inner().unwrap_or_default(),
            tool_images: self.tool_images.into_inner().unwrap_or_default(),
        }
    }
}

struct CollectedExecutionOutput {
    observations: Vec<String>,
    observation_truncation: Vec<TextProjectionMetadata>,
    printed_images: Vec<AttachmentRef>,
    tool_calls: Vec<lash_core::ToolCallRecord>,
    tool_images: Vec<ToolImage>,
}

async fn collect_printed_images(
    value: &FlowValue,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> Result<Vec<AttachmentRef>, ToolHostError> {
    let mut seen = HashSet::new();
    let mut images = Vec::new();
    collect_printed_images_inner(value, attachment_store, &mut seen, &mut images).await?;
    Ok(images)
}

fn collect_printed_images_inner<'a>(
    value: &'a FlowValue,
    attachment_store: &'a dyn lash_core::AttachmentStore,
    seen: &'a mut HashSet<String>,
    images: &'a mut Vec<AttachmentRef>,
) -> ProjectedFuture<'a, Result<(), ToolHostError>> {
    Box::pin(async move {
        match value {
            FlowValue::Image(image) => {
                if !seen.insert(image.id.clone()) {
                    return Ok(());
                }
                let reference = attachment_store
                    .get(&lash_core::AttachmentId::new(image.id.clone()))
                    .ok()
                    .map(|stored| stored.meta.as_ref())
                    .ok_or_else(|| {
                        ToolHostError::new(format!(
                            "image bytes for `{}` are unavailable or were pruned",
                            image.id
                        ))
                    })?;
                images.push(reference);
            }
            FlowValue::List(values) => {
                for value in values.iter() {
                    collect_printed_images_inner(value, attachment_store, seen, images).await?;
                }
            }
            FlowValue::Record(record) => {
                for (_, value) in record.iter() {
                    collect_printed_images_inner(value, attachment_store, seen, images).await?;
                }
            }
            FlowValue::Projected(value) => {
                collect_printed_images_inner(
                    &value.materialize_async().await,
                    attachment_store,
                    seen,
                    images,
                )
                .await?;
            }
            FlowValue::Null | FlowValue::Bool(_) | FlowValue::Number(_) | FlowValue::String(_) => {}
        }
        Ok(())
    })
}

fn lift_tool_result_to_flow_value(
    result: Value,
    tool_images: Vec<ToolImage>,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> FlowValue {
    if tool_images.is_empty() {
        return json_to_flow_value(result);
    }

    let image_values = tool_images
        .into_iter()
        .map(|image| FlowValue::Image(register_tool_image(image, attachment_store)))
        .collect::<Vec<_>>();

    if is_image_only_tool_payload(&result) {
        return if image_values.len() == 1 {
            image_values.into_iter().next().unwrap_or(FlowValue::Null)
        } else {
            FlowValue::List(image_values.into())
        };
    }

    let mut value = json_to_flow_value(result);
    let images_value = FlowValue::List(image_values.into());
    match &mut value {
        FlowValue::Record(record) => {
            Arc::make_mut(record).insert("images".to_string(), images_value);
            value
        }
        _ => {
            let mut record = FlowRecord::new();
            record.insert("payload".to_string(), value);
            record.insert("images".to_string(), images_value);
            FlowValue::Record(Arc::new(record))
        }
    }
}

fn register_tool_image(
    mut image: ToolImage,
    attachment_store: &dyn lash_core::AttachmentStore,
) -> ImageValue {
    let reference = if let Some(reference) = image.reference.take() {
        reference
    } else if let Some(media_type) = lash_core::MediaType::from_mime(&image.mime) {
        let meta = lash_core::AttachmentMeta::new(
            lash_core::AttachmentId::new("pending"),
            media_type,
            image.data.len() as u64,
            image.width,
            image.height,
            Some(image.label.clone()),
        );
        attachment_store
            .put(std::mem::take(&mut image.data), meta)
            .unwrap_or_else(|_| lash_core::AttachmentRef {
                id: lash_core::AttachmentId::new(uuid::Uuid::new_v4().to_string()),
                media_type,
                byte_len: 0,
                width: image.width,
                height: image.height,
                label: Some(image.label.clone()),
            })
    } else {
        lash_core::AttachmentRef {
            id: lash_core::AttachmentId::new(uuid::Uuid::new_v4().to_string()),
            media_type: lash_core::MediaType::Image(lash_core::ImageMediaType::Png),
            byte_len: image.data.len() as u64,
            width: image.width,
            height: image.height,
            label: Some(image.label.clone()),
        }
    };
    ImageValue::new(
        reference.id.to_string(),
        reference.label.clone().unwrap_or_default(),
        reference.byte_len,
        reference.width,
        reference.height,
    )
}

fn is_image_only_tool_payload(result: &Value) -> bool {
    match result {
        Value::String(text) => text.trim_start().starts_with("[Image:"),
        Value::Null => true,
        Value::Array(values) => values.is_empty(),
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

fn tool_error_message(value: Value) -> String {
    match value {
        Value::String(text) => text,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "tool call failed".to_string()),
    }
}

fn snapshot_runtime(rlm: &FlowState) -> Result<String, String> {
    serde_json::to_string(&rlm.snapshot()).map_err(|err| format!("failed to snapshot RLM: {err}"))
}

fn restore_runtime(data: &str) -> Result<FlowState, String> {
    let snapshot: lashlang::Snapshot =
        serde_json::from_str(data).map_err(|err| format!("failed to restore RLM: {err}"))?;
    Ok(FlowState::from_snapshot(snapshot))
}

fn apply_global_defaults(
    rlm: &mut FlowState,
    patch: &lash_rlm_types::RlmGlobalsPatchPluginBody,
    protected_names: &BTreeSet<String>,
) -> Result<(), String> {
    if patch.set_default.is_empty() {
        return Ok(());
    }
    let mut snapshot = rlm.snapshot();
    for (key, value) in &patch.set_default {
        if is_reserved_global_name(key) || protected_names.contains(key) {
            return Err(format!(
                "`{key}` is a read-only projected host binding; choose a different Lashlang variable name for `set_default`"
            ));
        }
        if snapshot.globals.get(key).is_none() {
            snapshot
                .globals
                .insert(key.clone(), json_to_flow_value(value.clone()));
        }
    }
    *rlm = FlowState::from_snapshot(snapshot);
    Ok(())
}

fn is_reserved_global_name(key: &str) -> bool {
    key == "history"
}

fn prune_reserved_projected_bindings(rlm: &mut FlowState) {
    prune_protected_bindings(rlm, &BTreeSet::new());
}

fn prune_protected_bindings(rlm: &mut FlowState, protected_names: &BTreeSet<String>) {
    prune_projected_binding_names(
        rlm,
        std::iter::once("history").chain(protected_names.iter().map(String::as_str)),
    );
}

fn prune_projected_binding_names<'a>(
    rlm: &mut FlowState,
    names: impl IntoIterator<Item = &'a str>,
) {
    let mut snapshot = rlm.snapshot();
    for key in names {
        snapshot.globals.remove(key);
    }
    *rlm = FlowState::from_snapshot(snapshot);
}

fn flow_to_json_value<'a>(value: &'a FlowValue) -> ProjectedFuture<'a, Value> {
    Box::pin(async move {
        match value {
            FlowValue::Null => Value::Null,
            FlowValue::Bool(value) => Value::Bool(*value),
            FlowValue::Number(value) => json_number(*value),
            FlowValue::String(value) => Value::String(value.to_string()),
            FlowValue::Image(image) => serde_json::to_value(image)
                .unwrap_or_else(|_| Value::Object(serde_json::Map::new())),
            FlowValue::List(values) => {
                let mut out = Vec::with_capacity(values.len());
                for value in values.iter() {
                    out.push(flow_to_json_value(value).await);
                }
                Value::Array(out)
            }
            FlowValue::Record(record) => flow_record_to_json_value(record).await,
            FlowValue::Projected(value) => {
                // Canonical JSON encoding for projected values: a single-key
                // object `{"__projected__": <inner>}` rather than the bare
                // inner value. RLM's before-tool hook materializes wrappers
                // for ordinary tool arguments and preserves root `seed`
                // wrappers for projection-aware control tools.
                let inner = flow_to_json_value(&value.materialize_async().await).await;
                let mut obj = serde_json::Map::with_capacity(1);
                obj.insert(PROJECTED_JSON_TAG.to_string(), inner);
                Value::Object(obj)
            }
        }
    })
}

async fn flow_record_to_json_value(record: &FlowRecord) -> Value {
    let mut object = serde_json::Map::with_capacity(record.len());
    for (key, value) in record.iter() {
        object.insert(key.to_string(), flow_to_json_value(value).await);
    }
    Value::Object(object)
}

fn json_number(value: f64) -> Value {
    if value.is_finite() && value.fract() == 0.0 {
        let as_i64 = value as i64 as f64;
        if as_i64 == value {
            return Value::Number(serde_json::Number::from(value as i64));
        }
        let as_u64 = value as u64 as f64;
        if as_u64 == value {
            return Value::Number(serde_json::Number::from(value as u64));
        }
    }
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn json_to_flow_value(value: Value) -> FlowValue {
    match value {
        Value::Null => FlowValue::Null,
        Value::Bool(value) => FlowValue::Bool(value),
        Value::Number(value) => FlowValue::Number(value.as_f64().unwrap_or_default()),
        Value::String(value) => FlowValue::String(value.into()),
        Value::Array(values) => {
            FlowValue::List(values.into_iter().map(json_to_flow_value).collect())
        }
        Value::Object(map) => json_map_to_image(&map)
            .map(FlowValue::Image)
            .unwrap_or_else(|| {
                FlowValue::Record(Arc::new(
                    map.into_iter()
                        .map(|(key, value)| (key, json_to_flow_value(value)))
                        .collect::<FlowRecord>(),
                ))
            }),
    }
}

fn json_map_to_image(map: &serde_json::Map<String, Value>) -> Option<ImageValue> {
    if map.get("type")?.as_str()? != "image" {
        return None;
    }
    Some(ImageValue::new(
        map.get("id")?.as_str()?.to_string(),
        map.get("label")?.as_str()?.to_string(),
        map.get("size")?.as_u64()?,
        optional_json_u32(map.get("width")?)?,
        optional_json_u32(map.get("height")?)?,
    ))
}

fn optional_json_u32(value: &Value) -> Option<Option<u32>> {
    match value {
        Value::Null => Some(None),
        Value::Number(number) => number
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .map(Some),
        _ => None,
    }
}

async fn format_output_value(value: &FlowValue) -> String {
    match value {
        FlowValue::Null => "null".to_string(),
        FlowValue::String(text) => text.to_string(),
        FlowValue::Bool(value) => value.to_string(),
        FlowValue::Number(value) => value.to_string(),
        FlowValue::Image(_)
        | FlowValue::List(_)
        | FlowValue::Record(_)
        | FlowValue::Projected(_) => serde_json::to_string(&flow_to_json_value(value).await)
            .unwrap_or_else(|_| value.to_string()),
    }
}

fn format_parse_error(source: &str, err: &lashlang::ParseError) -> String {
    let offset = err.offset();
    let (line_no, column, line_text) = line_context(source, offset);
    let caret_pad = " ".repeat(column.saturating_sub(1));
    let mut message = format!("line {line_no}, column {column}: {err}\n{line_text}\n{caret_pad}^");
    if matches!(
        err,
        lashlang::ParseError::Unexpected { found, .. } if found == "`if`"
    ) {
        message.push_str("\nhint: use `cond ? yes : no` for inline conditionals");
    }
    if matches!(
        err,
        lashlang::ParseError::Unexpected { found, .. } if found == "`for`"
    ) {
        message.push_str("\nhint: `for` is a statement. Put it on its own line, not inside an expression or record literal.");
    }
    message
}

fn line_context(source: &str, offset: usize) -> (usize, usize, String) {
    let mut line_no = 1usize;
    let mut line_start = 0usize;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line_no += 1;
            line_start = idx + 1;
        }
    }
    let line_end = source[line_start..]
        .find('\n')
        .map(|rel| line_start + rel)
        .unwrap_or(source.len());
    let line_text = source[line_start..line_end].to_string();
    let column = source[line_start..offset.min(source.len())].chars().count() + 1;
    (line_no, column, line_text)
}

fn collect_files(root: &Path) -> std::io::Result<HashMap<String, String>> {
    let mut files = HashMap::new();
    walk_dir(root, root, &mut files)?;
    Ok(files)
}

fn walk_dir(root: &Path, dir: &Path, files: &mut HashMap<String, String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(root, &path, files)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            files.insert(rel, contents);
        }
    }
    Ok(())
}

fn restore_files(root: &Path, files: &HashMap<String, String>) -> std::io::Result<()> {
    for (rel, contents) in files {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
    }
    Ok(())
}

fn clear_dir(root: &Path) {
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let _ = std::fs::remove_dir_all(&path);
            } else {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct NoopHost;

    impl ToolHost for NoopHost {
        async fn call(&self, name: String, _args: FlowRecord) -> Result<FlowValue, ToolHostError> {
            Err(ToolHostError::new(format!("unknown tool: {name}")))
        }
    }

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    }

    struct TestProjectedValue(Vec<FlowValue>);

    #[derive(Default)]
    struct SnapshotProjectedToolText {
        materialize_count: AtomicUsize,
        render_count: AtomicUsize,
    }

    impl ProjectedHostValue for SnapshotProjectedToolText {
        fn type_name(&self) -> &str {
            "string"
        }

        fn render(&self) -> ProjectedFuture<'_, String> {
            Box::pin(async {
                self.render_count.fetch_add(1, Ordering::SeqCst);
                "rendered tool text".to_string()
            })
        }

        fn materialize(&self) -> ProjectedFuture<'_, FlowValue> {
            Box::pin(async {
                self.materialize_count.fetch_add(1, Ordering::SeqCst);
                FlowValue::String("materialized tool text".into())
            })
        }
    }

    impl ProjectedHostValue for TestProjectedValue {
        fn type_name(&self) -> &str {
            "list"
        }

        fn len(&self) -> ProjectedFuture<'_, Option<usize>> {
            Box::pin(async { Some(self.0.len()) })
        }

        fn get_index(&self, index: FlowValue) -> ProjectedFuture<'_, ProjectedRead> {
            Box::pin(async move {
                let Ok(Some(index)) = projected_index(&index, self.0.len()) else {
                    return ProjectedRead::Missing;
                };
                self.0
                    .get(index)
                    .cloned()
                    .map(ProjectedRead::Value)
                    .unwrap_or(ProjectedRead::Missing)
            })
        }

        fn materialize(&self) -> ProjectedFuture<'_, FlowValue> {
            Box::pin(async { FlowValue::List(self.0.clone().into()) })
        }
    }

    fn projected_history(values: Vec<FlowValue>) -> ProjectedBindings {
        let mut projected = ProjectedBindings::new();
        projected.insert(
            "history",
            ProjectedValue::custom("history", Arc::new(TestProjectedValue(values))),
        );
        projected
    }

    #[test]
    fn projected_history_is_available_without_clobbering_executor_globals() {
        block_on(async {
            let mut state =
                RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
            let mut set_default = serde_json::Map::new();
            set_default.insert("diary".to_string(), serde_json::json!(["kept"]));
            state
                .patch_globals(
                    &lash_rlm_types::RlmGlobalsPatchPluginBody { set_default },
                    &BTreeSet::new(),
                )
                .expect("patch diary");

            let projected = projected_history(vec![FlowValue::String("hello".into())]);
            let compiled = lashlang::compile_source(
                "submit { history_len: len(history), diary_len: len(diary) }",
            )
            .expect("compile");
            let outcome = lashlang::execute_compiled_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect("execute");
            let ExecutionOutcome::Finished(FlowValue::Record(record)) = outcome else {
                panic!("expected submitted record");
            };
            assert_eq!(record["history_len"], FlowValue::Number(1.0));
            assert_eq!(record["diary_len"], FlowValue::Number(1.0));
            assert!(state.rlm.snapshot().globals.get("history").is_none());
        });
    }

    #[test]
    fn projected_history_defaults_to_empty_list_when_missing() {
        block_on(async {
            let mut state =
                RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");

            let projected = projected_history(Vec::new());
            let compiled =
                lashlang::compile_source("submit { history_len: len(history) }").expect("compile");
            let outcome = lashlang::execute_compiled_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect("execute");
            let ExecutionOutcome::Finished(FlowValue::Record(record)) = outcome else {
                panic!("expected submitted record");
            };
            assert_eq!(record["history_len"], FlowValue::Number(0.0));
        });
    }

    #[test]
    fn set_default_initializes_once_and_does_not_mutate_projected_globals() {
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let projected = BTreeSet::from_iter(["current_query".to_string()]);

        state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "diary".to_string(),
                        serde_json::json!(["initial"]),
                    )]),
                },
                &projected,
            )
            .expect("apply defaults");
        assert_eq!(
            state.rlm.snapshot().globals.get("diary"),
            Some(&FlowValue::List(
                vec![FlowValue::String("initial".into())].into()
            ))
        );
        assert!(state.rlm.snapshot().globals.get("current_query").is_none());

        state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "diary".to_string(),
                        serde_json::json!(["clobber"]),
                    )]),
                },
                &projected,
            )
            .expect("reapply defaults");
        assert_eq!(
            state.rlm.snapshot().globals.get("diary"),
            Some(&FlowValue::List(
                vec![FlowValue::String("initial".into())].into()
            ))
        );
    }

    #[test]
    fn set_default_rejects_projected_host_bindings() {
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let projected = BTreeSet::from_iter(["current_query".to_string()]);

        let err = state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "current_query".to_string(),
                        serde_json::json!("bad"),
                    )]),
                },
                &projected,
            )
            .expect_err("projected default should fail");
        assert!(err.to_string().contains("read-only projected host binding"));

        let err = state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "history".to_string(),
                        serde_json::json!([]),
                    )]),
                },
                &BTreeSet::new(),
            )
            .expect_err("history default should fail");
        assert!(err.to_string().contains("read-only projected host binding"));
    }

    #[test]
    fn projected_scalar_bindings_are_read_only_and_not_snapshotted() {
        block_on(async {
            let mut state =
                RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
            let mut projected = ProjectedBindings::new();
            projected.insert(
                "current_query",
                ProjectedValue::scalar("current_query", FlowValue::String("host".into())),
            );

            let compiled = lashlang::compile_source(
                "submit { chars: len(current_query), value: current_query }",
            )
            .expect("compile read");
            let outcome = lashlang::execute_compiled_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect("execute read");
            let ExecutionOutcome::Finished(FlowValue::Record(record)) = outcome else {
                panic!("expected submitted record");
            };
            assert_eq!(record["chars"], FlowValue::Number(4.0));
            assert_eq!(record["value"], FlowValue::String("host".into()));
            assert!(state.rlm.snapshot().globals.get("current_query").is_none());

            let compiled =
                lashlang::compile_source("current_query = \"local\"").expect("compile write");
            let failure = lashlang::execute_compiled_traced_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect_err("projected write should fail");
            assert!(
                failure
                    .error
                    .to_string()
                    .contains("read-only projected binding")
            );
        });
    }

    #[test]
    fn executor_snapshot_does_not_materialize_projected_tool_result_globals() {
        let projected = Arc::new(SnapshotProjectedToolText::default());
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let mut snapshot = state.rlm.snapshot();
        snapshot.globals.insert(
            "m".to_string(),
            FlowValue::Projected(ProjectedValue::custom(
                "search.matches[0].text",
                projected.clone(),
            )),
        );
        state.rlm = FlowState::from_snapshot(snapshot);

        let bytes = state
            .snapshot_execution_state()
            .expect("executor snapshot")
            .expect("snapshot bytes");

        assert_eq!(projected.render_count.load(Ordering::SeqCst), 0);
        assert_eq!(projected.materialize_count.load(Ordering::SeqCst), 0);
        let outer: Value = serde_json::from_slice(&bytes).expect("snapshot json");
        let vars = outer
            .get("vars")
            .and_then(Value::as_str)
            .expect("vars string");
        assert!(!vars.contains("rendered tool text"));
        assert!(!vars.contains("materialized tool text"));
        assert!(vars.contains("__lashlang_snapshot_projected__"));
        assert!(vars.contains("search.matches[0].text"));

        let restored = restore_runtime(vars).expect("restore runtime");
        assert!(matches!(
            restored.snapshot().globals.get("m"),
            Some(FlowValue::Projected(_))
        ));
    }

    #[test]
    fn flow_to_json_value_emits_projected_marker_for_projected_values() {
        block_on(async {
            let projected = ProjectedValue::scalar("input", FlowValue::String("hello".into()));
            let value = flow_to_json_value(&FlowValue::Projected(projected)).await;
            let obj = value
                .as_object()
                .expect("expected projected wrapper object");
            assert_eq!(obj.len(), 1, "wrapper should have exactly one key");
            assert_eq!(
                obj.get(PROJECTED_JSON_TAG)
                    .and_then(|v| v.as_str())
                    .expect("inner string"),
                "hello"
            );
        });
    }

    #[test]
    fn flow_record_to_json_value_marks_only_projected_entries() {
        block_on(async {
            let projected = ProjectedValue::scalar("input", FlowValue::String("p".into()));
            let mut record = FlowRecord::default();
            record.insert("proj".to_string(), FlowValue::Projected(projected));
            record.insert("glob".to_string(), FlowValue::String("g".into()));

            let value = flow_record_to_json_value(&record).await;
            let obj = value.as_object().expect("record object");
            // proj entry must be wrapped in {"__projected__": ...}
            let proj = obj
                .get("proj")
                .and_then(|v| v.as_object())
                .expect("proj entry is an object");
            assert!(proj.contains_key(PROJECTED_JSON_TAG));
            // glob entry stays a bare string
            assert_eq!(obj.get("glob").and_then(|v| v.as_str()).expect("glob"), "g");
        });
    }

    #[test]
    fn parse_error_for_unsupported_while_points_at_while() {
        let source = r#"pool_i = 0
while len(final_ids) < 100 && pool_i < len(candidate_pools) {
  for m in candidate_pools[pool_i].matches {
    print m
  }
}"#;
        let err = match lashlang::compile_source(source) {
            Ok(_) => panic!("while should not compile"),
            Err(err) => err,
        };

        let message = format_parse_error(source, &err);
        assert!(
            message.contains("line 2, column 1: unsupported `while` loop"),
            "{message}"
        );
        assert!(
            message.contains("use bounded `for` loops over ranges/lists"),
            "{message}"
        );
        assert!(message.contains("while len(final_ids) < 100"), "{message}");
        assert!(
            !message.contains("expected `:`, found identifier `m`"),
            "{message}"
        );
    }
}
