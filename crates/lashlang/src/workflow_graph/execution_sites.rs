use crate::ast::{Expr, LabelMetadata, ListComprehensionClause};
use crate::tracking::{LashlangAstPath, LashlangExecutionContext, WorkflowExecutionSite};
use crate::{LashlangExecutionSite, ModuleArtifact};

use super::child_path;

/// Recreate the runtime identity for a graph execution-site descriptor.
///
/// This is primarily a compatibility seam for trace consumers that key live
/// observations by the runtime site id while reading the workflow graph as the
/// static skeleton.
pub fn runtime_execution_site_for_workflow_site(
    artifact: &ModuleArtifact,
    site: &WorkflowExecutionSite,
) -> Option<LashlangExecutionSite> {
    let context = if site.owner == "main" {
        LashlangExecutionContext::main(artifact.module_ref.clone())
    } else {
        let process_name = site.owner.strip_prefix("process:")?;
        LashlangExecutionContext::process(
            artifact.module_ref.clone(),
            artifact.process_ref(process_name)?.clone(),
            process_name,
        )
    };
    let path = LashlangAstPath::from_indices(&site.path);
    let mut runtime_site = if site.kind == "branch" {
        context.builder().branch_site(&path)
    } else {
        context
            .builder()
            .node_site(&path, site.kind.clone(), site.label.clone())
    };
    runtime_site.workflow_site = site.clone();
    Some(runtime_site)
}

pub(super) fn execution_sites(
    expression: &Expr,
    owner: &str,
    path: &[u32],
    label: Option<&LabelMetadata>,
) -> Vec<WorkflowExecutionSite> {
    let mut sites = Vec::new();
    collect_execution_sites(expression, owner, path, label, &mut sites);
    sites.sort();
    sites.dedup();
    sites
}

fn collect_execution_sites(
    expression: &Expr,
    owner: &str,
    path: &[u32],
    label: Option<&LabelMetadata>,
    sites: &mut Vec<WorkflowExecutionSite>,
) {
    if let Some(label) = label
        && !label_attaches_to_concrete_node(expression)
    {
        sites.push(WorkflowExecutionSite::new(
            owner,
            path,
            "step",
            label.title.as_str(),
        ));
        return;
    }
    match expression {
        Expr::Assign { target, expr } if label.is_some() => {
            let value_index = target
                .steps
                .iter()
                .filter(|step| matches!(step, crate::AssignPathStep::Index(_)))
                .count() as u32;
            collect_execution_sites(expr, owner, &child_path(path, value_index), label, sites);
        }
        Expr::Await(expr) | Expr::ResultUnwrap(expr) if label.is_some() => {
            collect_execution_sites(expr, owner, &child_path(path, 0), label, sites);
        }
        Expr::ReceiverCall { operation, .. } => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "resource_operation",
                operation.as_str(),
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::StartProcess(start) => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "child_process",
                format!("start {}", start.process),
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::SleepFor(_) => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "sleep",
                "sleep for",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::SleepUntil(_) => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "sleep",
                "sleep until",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::WaitSignal { name } => sites.push(WorkflowExecutionSite::new(
            owner,
            path,
            "wait",
            format!("wait_signal {name}"),
        )),
        Expr::SignalRun { .. } => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "signal",
                "signal_run",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::Finish(_) => {
            sites.push(WorkflowExecutionSite::new(
                owner, path, "terminal", "result",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::Fail(_) => {
            sites.push(WorkflowExecutionSite::new(
                owner, path, "terminal", "failure",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::Yield(_) => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "process_event",
                "yield",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::Wake(_) => {
            sites.push(WorkflowExecutionSite::new(
                owner,
                path,
                "process_event",
                "wake",
            ));
            collect_child_execution_sites(expression, owner, path, sites);
        }
        Expr::If { condition, .. } => {
            sites.push(WorkflowExecutionSite::new(owner, path, "branch", "if"));
            collect_execution_sites(condition, owner, &child_path(path, 0), None, sites);
        }
        Expr::For { iterable, .. } => {
            collect_execution_sites(iterable, owner, &child_path(path, 0), None, sites);
        }
        Expr::ListComprehension { clauses, .. } => {
            for (index, clause) in clauses.iter().enumerate() {
                let expression = match clause {
                    ListComprehensionClause::For { iterable, .. } => iterable,
                    ListComprehensionClause::If { condition } => condition,
                };
                collect_execution_sites(
                    expression,
                    owner,
                    &child_path(path, index as u32),
                    None,
                    sites,
                );
            }
        }
        _ => collect_child_execution_sites(expression, owner, path, sites),
    }
}

fn collect_child_execution_sites(
    expression: &Expr,
    owner: &str,
    path: &[u32],
    sites: &mut Vec<WorkflowExecutionSite>,
) {
    for (index, child) in expression.children().enumerate() {
        collect_execution_sites(child, owner, &child_path(path, index as u32), None, sites);
    }
}

fn label_attaches_to_concrete_node(expression: &Expr) -> bool {
    match expression {
        Expr::Assign { expr, .. } | Expr::Await(expr) | Expr::ResultUnwrap(expr) => {
            label_attaches_to_concrete_node(expr)
        }
        Expr::ReceiverCall { .. }
        | Expr::StartProcess(_)
        | Expr::SleepFor(_)
        | Expr::SleepUntil(_)
        | Expr::WaitSignal { .. }
        | Expr::SignalRun { .. }
        | Expr::Yield(_)
        | Expr::Wake(_)
        | Expr::Finish(_)
        | Expr::Fail(_)
        | Expr::If { .. } => true,
        _ => false,
    }
}
