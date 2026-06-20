use lash::tools::LashlangToolBinding;

fn rlm_tool_types_are_nameable(binding: LashlangToolBinding) {
    let _ = binding;
}

fn rlm_core_type_is_nameable(core: lash::RlmCore) {
    let _ = core;
}

fn main() {
    let _ = rlm_tool_types_are_nameable;
    let _ = rlm_core_type_is_nameable;
}
