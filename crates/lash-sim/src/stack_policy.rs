use crate::runner::FixedScriptRunnerError;

pub const PRODUCT_STACK_BUDGET_BYTES: usize = 2 * 1024 * 1024;
pub const SIM_HARNESS_STACK_LIMIT_BYTES: usize = 8 * 1024 * 1024;

pub(crate) fn run_on_sim_harness_stack<T, F>(
    label: impl Into<String>,
    stack_bytes: usize,
    f: F,
) -> Result<T, FixedScriptRunnerError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, FixedScriptRunnerError> + Send + 'static,
{
    run_on_lash_sim_stack(
        label,
        stack_bytes,
        SIM_HARNESS_STACK_LIMIT_BYTES,
        "SIM_HARNESS_STACK_LIMIT_BYTES",
        f,
    )
}

pub(crate) fn run_on_product_stack<T, F>(
    label: impl Into<String>,
    stack_bytes: usize,
    f: F,
) -> Result<T, FixedScriptRunnerError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, FixedScriptRunnerError> + Send + 'static,
{
    run_on_lash_sim_stack(
        label,
        stack_bytes,
        PRODUCT_STACK_BUDGET_BYTES,
        "PRODUCT_STACK_BUDGET_BYTES",
        f,
    )
}

fn run_on_lash_sim_stack<T, F>(
    label: impl Into<String>,
    stack_bytes: usize,
    limit_bytes: usize,
    limit_name: &'static str,
    f: F,
) -> Result<T, FixedScriptRunnerError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, FixedScriptRunnerError> + Send + 'static,
{
    let label = label.into();
    if stack_bytes > limit_bytes {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "{label} requested stack {stack_bytes} bytes above {limit_name}={limit_bytes}; stack growth above policy is a bug"
        )));
    }
    let thread_name = format!("lash-sim-stack-policy-{label}");
    let panic_label = label.clone();
    std::thread::Builder::new()
        .name(thread_name)
        .stack_size(stack_bytes)
        .spawn(f)
        .map_err(FixedScriptRunnerError::Io)?
        .join()
        .map_err(|panic| stack_policy_panic(&panic_label, panic))?
}

fn stack_policy_panic(
    label: &str,
    panic: Box<dyn std::any::Any + Send + 'static>,
) -> FixedScriptRunnerError {
    FixedScriptRunnerError::Runtime(format!(
        "{label} contract replay thread panicked: {}",
        panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic")
    ))
}
