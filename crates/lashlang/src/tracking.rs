use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ModuleRef, ProcessRef};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProcessTrackingContext {
    pub(crate) module_ref: ModuleRef,
    pub(crate) process_ref: ProcessRef,
    pub(crate) process_name: String,
}

impl ProcessTrackingContext {
    pub(crate) fn new(
        module_ref: ModuleRef,
        process_ref: ProcessRef,
        process_name: impl Into<String>,
    ) -> Self {
        Self {
            module_ref,
            process_ref,
            process_name: process_name.into(),
        }
    }

    pub(crate) fn builder(&self) -> ProcessTrackingSiteBuilder<'_> {
        ProcessTrackingSiteBuilder { context: self }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ProcessAstPath(Vec<u32>);

impl ProcessAstPath {
    pub(crate) fn root() -> Self {
        Self(Vec::new())
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
pub(crate) struct ProcessTrackingSiteBuilder<'context> {
    context: &'context ProcessTrackingContext,
}

impl ProcessTrackingSiteBuilder<'_> {
    pub(crate) fn process_node_id(&self) -> String {
        self.node_id(&ProcessAstPath::root(), "process")
    }

    pub(crate) fn node_site(
        &self,
        path: &ProcessAstPath,
        kind: impl Into<String>,
        label: impl Into<String>,
    ) -> ProcessTrackingSite {
        let kind = kind.into();
        ProcessTrackingSite {
            node_id: self.node_id(path, &kind),
            node_kind: kind,
            label: label.into(),
            branch: None,
        }
    }

    pub(crate) fn branch_site(&self, path: &ProcessAstPath) -> ProcessTrackingSite {
        ProcessTrackingSite {
            node_id: self.node_id(path, "branch"),
            node_kind: "branch".to_string(),
            label: "if".to_string(),
            branch: Some(ProcessBranchSite {
                then_edge_id: self.branch_edge_id(path, ProcessBranchSelection::Then),
                else_edge_id: self.branch_edge_id(path, ProcessBranchSelection::Else),
            }),
        }
    }

    pub(crate) fn branch_arm_node_id(
        &self,
        path: &ProcessAstPath,
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
        path: &ProcessAstPath,
        from: &str,
        to: &str,
        label: &str,
    ) -> String {
        stable_id(
            "edge",
            &[
                self.context.module_ref.to_string(),
                process_ref_key(&self.context.process_ref),
                path.stable_text(),
                from.to_string(),
                to.to_string(),
                label.to_string(),
            ],
        )
    }

    pub(crate) fn branch_edge_id(
        &self,
        path: &ProcessAstPath,
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
                process_ref_key(&self.context.process_ref),
                path.stable_text(),
                "branch".to_string(),
                label.to_string(),
            ],
        )
    }

    fn node_id(&self, path: &ProcessAstPath, kind: impl AsRef<str>) -> String {
        stable_id(
            kind.as_ref(),
            &[
                self.context.module_ref.to_string(),
                process_ref_key(&self.context.process_ref),
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
pub struct ProcessTrackingSite {
    pub node_id: String,
    pub node_kind: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<ProcessBranchSite>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessBranchSite {
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
pub struct ProcessTrackingChild {
    pub process_id: String,
    pub module_ref: ModuleRef,
    pub process_ref: ProcessRef,
    pub process_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessTrackingObservation {
    NodeStarted {
        site: ProcessTrackingSite,
        occurrence: u64,
    },
    NodeCompleted {
        site: ProcessTrackingSite,
        occurrence: u64,
    },
    NodeFailed {
        site: ProcessTrackingSite,
        occurrence: u64,
        error: String,
    },
    BranchSelected {
        site: ProcessTrackingSite,
        occurrence: u64,
        edge_id: String,
        selected: ProcessBranchSelection,
    },
    ChildStarted {
        site: ProcessTrackingSite,
        occurrence: u64,
        child: ProcessTrackingChild,
    },
}
