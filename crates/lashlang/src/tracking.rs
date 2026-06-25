use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ModuleRef, ProcessRef};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LashlangExecutionContext {
    pub(crate) module_ref: ModuleRef,
    pub(crate) entry: LashlangExecutionEntry,
}

impl LashlangExecutionContext {
    pub(crate) fn main(module_ref: ModuleRef) -> Self {
        Self {
            module_ref,
            entry: LashlangExecutionEntry::Main,
        }
    }

    pub(crate) fn process(
        module_ref: ModuleRef,
        process_ref: ProcessRef,
        process_name: impl Into<String>,
    ) -> Self {
        Self {
            module_ref,
            entry: LashlangExecutionEntry::Process {
                process_ref,
                process_name: process_name.into(),
            },
        }
    }

    pub(crate) fn builder(&self) -> LashlangExecutionSiteBuilder<'_> {
        LashlangExecutionSiteBuilder { context: self }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LashlangExecutionEntry {
    Main,
    Process {
        process_ref: ProcessRef,
        process_name: String,
    },
}

impl LashlangExecutionEntry {
    fn stable_key(&self) -> String {
        match self {
            Self::Main => "main".to_string(),
            Self::Process { process_ref, .. } => process_ref_key(process_ref),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LashlangAstPath(Vec<u32>);

impl LashlangAstPath {
    pub(crate) fn root() -> Self {
        Self(Vec::new())
    }

    pub(crate) fn from_indices(indices: &[u32]) -> Self {
        Self(indices.to_vec())
    }

    pub(crate) fn child(&self, index: usize) -> Self {
        let mut path = self.0.clone();
        path.push(index as u32);
        Self(path)
    }

    fn stable_text(&self) -> String {
        if self.0.is_empty() {
            return "root".to_string();
        }
        self.0
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(".")
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LashlangExecutionSiteBuilder<'context> {
    context: &'context LashlangExecutionContext,
}

impl LashlangExecutionSiteBuilder<'_> {
    pub(crate) fn process_node_id(&self) -> String {
        self.node_id(&LashlangAstPath::root(), "process")
    }

    pub(crate) fn main_node_id(&self) -> String {
        self.node_id(&LashlangAstPath::root(), "main")
    }

    pub(crate) fn node_site(
        &self,
        path: &LashlangAstPath,
        kind: impl Into<String>,
        label: impl Into<String>,
    ) -> LashlangExecutionSite {
        let kind = kind.into();
        LashlangExecutionSite {
            node_id: self.node_id(path, &kind),
            node_kind: kind,
            label: label.into(),
            branch: None,
        }
    }

    pub(crate) fn branch_site(&self, path: &LashlangAstPath) -> LashlangExecutionSite {
        LashlangExecutionSite {
            node_id: self.node_id(path, "branch"),
            node_kind: "branch".to_string(),
            label: "if".to_string(),
            branch: Some(LashlangBranchSite {
                then_edge_id: self.branch_edge_id(path, ProcessBranchSelection::Then),
                else_edge_id: self.branch_edge_id(path, ProcessBranchSelection::Else),
            }),
        }
    }

    pub(crate) fn branch_arm_node_id(
        &self,
        path: &LashlangAstPath,
        selection: ProcessBranchSelection,
    ) -> String {
        let kind = match selection {
            ProcessBranchSelection::Then => "branch_arm_then",
            ProcessBranchSelection::Else => "branch_arm_else",
        };
        self.node_id(path, kind)
    }

    pub(crate) fn edge_id(
        &self,
        path: &LashlangAstPath,
        from: &str,
        to: &str,
        label: &str,
    ) -> String {
        stable_id(
            "edge",
            &[
                self.context.module_ref.to_string(),
                self.context.entry.stable_key(),
                path.stable_text(),
                from.to_string(),
                to.to_string(),
                label.to_string(),
            ],
        )
    }

    pub(crate) fn branch_edge_id(
        &self,
        path: &LashlangAstPath,
        selection: ProcessBranchSelection,
    ) -> String {
        let label = match selection {
            ProcessBranchSelection::Then => "then",
            ProcessBranchSelection::Else => "else",
        };
        stable_id(
            "edge",
            &[
                self.context.module_ref.to_string(),
                self.context.entry.stable_key(),
                path.stable_text(),
                "branch".to_string(),
                label.to_string(),
            ],
        )
    }

    fn node_id(&self, path: &LashlangAstPath, kind: impl AsRef<str>) -> String {
        stable_id(
            kind.as_ref(),
            &[
                self.context.module_ref.to_string(),
                self.context.entry.stable_key(),
                path.stable_text(),
                kind.as_ref().to_string(),
            ],
        )
    }
}

fn stable_id(prefix: &str, parts: &[String]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    let hash = format!("{:x}", hasher.finalize());
    format!("{prefix}:{}", &hash[..24])
}

pub fn process_ref_key(process_ref: &ProcessRef) -> String {
    format!("{}:{}", process_ref.component, process_ref.pos)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangExecutionSite {
    pub node_id: String,
    pub node_kind: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<LashlangBranchSite>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LashlangExecutionCallSite {
    pub site: LashlangExecutionSite,
    pub occurrence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangBranchSite {
    pub then_edge_id: String,
    pub else_edge_id: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessBranchSelection {
    Then,
    Else,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LashlangExecutionChild {
    pub process_id: String,
    pub module_ref: ModuleRef,
    pub process_ref: ProcessRef,
    pub process_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LashlangExecutionObservation {
    NodeStarted {
        site: LashlangExecutionSite,
        occurrence: u64,
    },
    NodeCompleted {
        site: LashlangExecutionSite,
        occurrence: u64,
    },
    NodeFailed {
        site: LashlangExecutionSite,
        occurrence: u64,
        error: String,
    },
    BranchSelected {
        site: LashlangExecutionSite,
        occurrence: u64,
        edge_id: String,
        selected: ProcessBranchSelection,
    },
    ChildStarted {
        site: LashlangExecutionSite,
        occurrence: u64,
        child: LashlangExecutionChild,
    },
}
