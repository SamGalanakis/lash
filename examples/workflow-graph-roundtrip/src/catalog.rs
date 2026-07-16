use serde::{Deserialize, Serialize};

use crate::DEFAULT_WORKFLOW;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowCatalogEntry {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SelectWorkflowRequest {
    pub id: String,
}

pub(crate) struct BuiltInWorkflow {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub source: &'static str,
}

const BLANK_WORKFLOW: &str = r#"
process blank() {
  finish 0
}
"#;

const TRAFFIC_LIGHTS_WORKFLOW: &str = r#"
@label(title: "Traffic lights", description: "Cycle a three-light signal twice")
process traffic_lights() {
  await display.set_status({ key: "traffic", value: "running" })?
  for cycle in [1, 2] {
    await display.add_item({ list: "cycles", item: cycle })?
    await display.set_light({ name: "red", state: "on" })?
    await display.set_light({ name: "amber", state: "off" })?
    await display.set_light({ name: "green", state: "off" })?
    sleep for "350ms"
    await display.set_light({ name: "red", state: "off" })?
    await display.set_light({ name: "amber", state: "on" })?
    sleep for "350ms"
    await display.set_light({ name: "amber", state: "off" })?
    await display.set_light({ name: "green", state: "on" })?
    await display.show_message({ text: "Go" })?
    sleep for "500ms"
  }
  await display.set_status({ key: "traffic", value: "complete" })?
  finish null
}
"#;

const BRANCHING_APPROVAL_WORKFLOW: &str = r#"
@label(title: "Branching approval", description: "Wait for a decision and reveal its branch")
process branching_approval() signals { continue: any } {
  await display.set_status({ key: "approval", value: "waiting" })?
  await display.highlight({ target: "approval" })?
  await display.show_message({ text: "Approval requested" })?
  decision = wait_signal("continue")
  if decision.autoFired {
    await display.set_status({ key: "approval", value: "approved" })?
    if true {
      await display.set_light({ name: "approved", state: "green" })?
      await display.show_message({ text: "Request approved" })?
    } else {
      await display.show_message({ text: "Approval needs review" })?
    }
  } else {
    await display.set_status({ key: "approval", value: "rejected" })?
    await display.set_light({ name: "rejected", state: "red" })?
    await display.show_message({ text: "Request rejected" })?
  }
  sleep for "400ms"
  await display.highlight({ target: "result" })?
  finish decision
}
"#;

const COUNTER_LOOP_WORKFLOW: &str = r#"
@label(title: "Counter loop", description: "Combine structured while and for containers")
process counter_loop() {
  await display.set_status({ key: "counter", value: "running" })?
  await display.set_progress({ pct: 5 })?
  state = { count: 0 }
  while state.count < 3 {
    await display.add_item({ list: "counts", item: state.count })?
    await display.set_progress({ pct: state.count * 20 + 20 })?
    state.count = state.count + 1
    sleep for "250ms"
  }
  for pct in [70, 85, 100] {
    await display.set_progress({ pct: pct })?
    sleep for "300ms"
  }
  await display.highlight({ target: "progress" })?
  await display.set_status({ key: "counter", value: "complete" })?
  await display.show_message({ text: "Counter complete" })?
  finish state.count
}
"#;

pub(crate) const BUILT_IN_WORKFLOWS: &[BuiltInWorkflow] = &[
    BuiltInWorkflow {
        id: "blank",
        name: "Blank workflow",
        description: "An empty starter you build up by adding nodes.",
        source: BLANK_WORKFLOW,
    },
    BuiltInWorkflow {
        id: "onboarding",
        name: "Onboarding",
        description: "A labeled onboarding flow with a signal wait, branch, and mixed display updates.",
        source: DEFAULT_WORKFLOW,
    },
    BuiltInWorkflow {
        id: "traffic-lights",
        name: "Traffic Lights",
        description: "A visual red, amber, and green light sequence repeated twice.",
        source: TRAFFIC_LIGHTS_WORKFLOW,
    },
    BuiltInWorkflow {
        id: "branching-approval",
        name: "Branching Approval",
        description: "An if-heavy approval flow with a visible signal wait and distinct outcomes.",
        source: BRANCHING_APPROVAL_WORKFLOW,
    },
    BuiltInWorkflow {
        id: "counter-loop",
        name: "Counter Loop",
        description: "A structured while loop followed by an editable for container and progress updates.",
        source: COUNTER_LOOP_WORKFLOW,
    },
];

pub(crate) fn entries() -> Vec<WorkflowCatalogEntry> {
    BUILT_IN_WORKFLOWS
        .iter()
        .map(|workflow| WorkflowCatalogEntry {
            id: workflow.id.to_string(),
            name: workflow.name.to_string(),
            description: workflow.description.to_string(),
        })
        .collect()
}

pub(crate) fn source(id: &str) -> Option<&'static str> {
    BUILT_IN_WORKFLOWS
        .iter()
        .find(|workflow| workflow.id == id)
        .map(|workflow| workflow.source)
}
