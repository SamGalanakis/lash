use std::sync::Arc;

use crate::plugin::ToolCallHookContext;
use crate::validate_tool_input;
use crate::{
    ProgressSender, ToolContext, ToolFailureClass, ToolManifest, ToolPrepareCall,
    ToolPrepareContext,
};

use super::context::{
    ToolDispatchContext, ToolDispatchOutcome, ToolPreparationOutcome, completed_preparation,
    outcome, runtime_failure,
};
use super::directives::apply_before_tool_directives;
use super::execution::dispatch_prepared_tool_call_with_execution_context;

pub(crate) async fn dispatch_tool_call(
    context: &ToolDispatchContext<'_>,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolDispatchOutcome {
    let tool_context = ToolContext::from_dispatch(Arc::new(context.clone())).build();
    dispatch_tool_call_with_execution_context(context, tool_name, args, progress, tool_context)
        .await
}

pub(crate) async fn dispatch_tool_call_with_execution_context<'run>(
    context: &ToolDispatchContext<'run>,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolContext<'run>,
) -> ToolDispatchOutcome {
    let pending = crate::sansio::PendingToolCall {
        call_id: tool_context
            .tool_call_id()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("tool:{}", uuid::Uuid::new_v4())),
        tool_name,
        args,
        replay: None,
    };
    match prepare_tool_call_with_context(
        context,
        pending,
        tool_context.tool_call_id().map(str::to_string),
    )
    .await
    {
        ToolPreparationOutcome::Prepared(prepared) => {
            dispatch_prepared_tool_call_with_execution_context(
                context,
                prepared,
                progress,
                tool_context,
            )
            .await
        }
        ToolPreparationOutcome::Completed(outcome) => *outcome,
    }
}

pub(crate) async fn prepare_tool_call_with_context(
    context: &ToolDispatchContext<'_>,
    pending: crate::sansio::PendingToolCall,
    tool_call_id: Option<String>,
) -> ToolPreparationOutcome {
    let tool_name = pending.tool_name.clone();
    let Some(manifest) = resolve_callable_manifest(context, &tool_name) else {
        return completed_preparation(outcome(
            tool_name,
            pending.args,
            runtime_failure(
                ToolFailureClass::Unavailable,
                "tool_unavailable",
                "Tool is unavailable in this session",
            ),
            0,
        ));
    };
    let mut pending = pending;
    let mut args = pending.args;

    let directives = match context
        .plugins
        .before_tool_call(ToolCallHookContext::new(
            context.session_id.clone(),
            tool_name.clone(),
            args.clone(),
            manifest.argument_projection.clone(),
            context.turn_context.clone(),
            Arc::clone(&context.sessions),
        ))
        .await
    {
        Ok(directives) => directives,
        Err(err) => {
            return completed_preparation(outcome(
                tool_name,
                args,
                runtime_failure(
                    ToolFailureClass::Internal,
                    "before_tool_call_failed",
                    err.to_string(),
                ),
                0,
            ));
        }
    };

    let applied = apply_before_tool_directives(context, args, directives).await;
    args = applied.args;
    if let Some(result) = applied.short_circuit {
        return completed_preparation(outcome(tool_name, args, result, 0));
    }
    let contract = context.tools.resolve_contract(&tool_name);
    let Some(contract) = contract else {
        return completed_preparation(outcome(
            tool_name,
            args,
            runtime_failure(
                ToolFailureClass::Unavailable,
                "tool_contract_unavailable",
                "Tool contract is unavailable in this session",
            ),
            0,
        ));
    };
    if let Err(err) = validate_tool_input(&contract, &args) {
        return completed_preparation(outcome(
            tool_name,
            args,
            runtime_failure(ToolFailureClass::InvalidRequest, "invalid_tool_args", err),
            0,
        ));
    }

    pending.args = args.clone();
    let prepare_context = ToolPrepareContext::new(
        context.session_id.clone(),
        Arc::clone(&context.sessions),
        context.turn_context.clone(),
        tool_call_id,
    );
    match context
        .tools
        .prepare_tool_call(ToolPrepareCall {
            pending,
            context: &prepare_context,
        })
        .await
    {
        Ok(prepared) => ToolPreparationOutcome::Prepared(prepared),
        Err(result) => completed_preparation(outcome(tool_name, args, result, 0)),
    }
}

pub(crate) fn resolve_callable_manifest(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
) -> Option<ToolManifest> {
    if let Some(entry) = context
        .surface
        .tools
        .iter()
        .find(|tool| tool.manifest.name == tool_name)
    {
        return entry
            .availability
            .is_callable()
            .then(|| entry.manifest.clone());
    }

    let visible_and_callable = |manifest: ToolManifest| {
        if context.plugins.tool_access().hides(&manifest.name) {
            return None;
        }
        manifest
            .effective_availability()
            .is_callable()
            .then_some(manifest)
    };

    context
        .tools
        .resolve_manifest(tool_name)
        .and_then(visible_and_callable)
}

pub(crate) fn resolve_callable_manifest_by_id(
    context: &ToolDispatchContext<'_>,
    tool_id: &crate::ToolId,
) -> Option<ToolManifest> {
    if let Some(entry) = context
        .surface
        .tools
        .iter()
        .find(|tool| tool.manifest.id == *tool_id)
    {
        return entry
            .availability
            .is_callable()
            .then(|| entry.manifest.clone());
    }

    let visible_and_callable = |manifest: ToolManifest| {
        if context.plugins.tool_access().hides(&manifest.name) {
            return None;
        }
        manifest
            .effective_availability()
            .is_callable()
            .then_some(manifest)
    };

    context
        .tools
        .tool_manifests()
        .into_iter()
        .find(|manifest| manifest.id == *tool_id)
        .and_then(visible_and_callable)
}

pub(crate) fn resolve_tool_argument_projection_policy(
    context: &ToolDispatchContext<'_>,
    tool_name: &str,
) -> crate::ToolArgumentProjectionPolicy {
    context
        .surface
        .tools
        .iter()
        .find(|def| def.manifest.name == tool_name)
        .map(|def| def.manifest.argument_projection.clone())
        .unwrap_or_default()
}
