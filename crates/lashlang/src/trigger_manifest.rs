use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::artifact::ModuleRef;
use crate::{Declaration, Expr, ExprVisitor, Program};

/// The stable trigger subscription keys emitted by one compiled module.
///
/// The linker materializes compiler-derived keys into the canonical IR before
/// constructing the artifact, so this set contains the exact public handles
/// used at runtime rather than source locations or key templates.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerKeyManifest {
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub subscription_keys: BTreeSet<String>,
}

impl TriggerKeyManifest {
    pub(crate) fn from_program(program: &Program) -> Self {
        struct Collector(BTreeSet<String>);

        impl ExprVisitor for Collector {
            fn visit_expr(&mut self, expr: &Expr) {
                if let Expr::ReceiverCall {
                    receiver,
                    operation,
                    args,
                } = expr
                    && operation.as_str() == crate::TriggerHostOperation::Register.receiver_method()
                    && matches!(
                        receiver.as_ref(),
                        Expr::ResourceRef(resource)
                            if crate::is_trigger_resource_type(resource.resource_type.as_str())
                    )
                    && let Ok(call) = crate::register_call_args(args)
                    && let Some(Expr::String(key)) = call.subscription_key
                {
                    self.0.insert(key.to_string());
                }
                crate::walk_expr(self, expr);
            }
        }

        let mut collector = Collector(BTreeSet::new());
        for declaration in &program.declarations {
            if let Declaration::Process(process) = declaration {
                collector.visit_expr(&process.body);
            }
        }
        collector.visit_expr(&program.main);
        Self {
            subscription_keys: collector.0,
        }
    }

    pub fn contains(&self, subscription_key: &str) -> bool {
        self.subscription_keys.contains(subscription_key)
    }

    pub fn diff(&self, replacement: &Self) -> TriggerKeyManifestDiff {
        TriggerKeyManifestDiff {
            added: replacement
                .subscription_keys
                .difference(&self.subscription_keys)
                .cloned()
                .collect(),
            removed: self
                .subscription_keys
                .difference(&replacement.subscription_keys)
                .cloned()
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerKeyManifestDiff {
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub added: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub removed: BTreeSet<String>,
}

impl TriggerKeyManifestDiff {
    pub fn has_removed_keys(&self) -> bool {
        !self.removed.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentTriggerKeyManifest {
    pub module_ref: ModuleRef,
    pub manifest: TriggerKeyManifest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerManifestReplacement {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_module_ref: Option<ModuleRef>,
    pub current_module_ref: ModuleRef,
    pub diff: TriggerKeyManifestDiff,
}
