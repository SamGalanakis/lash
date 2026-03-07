use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PythonMode {
    System,
    Bundled,
    None,
}

impl PythonMode {
    fn from_flag(value: &str) -> Result<Self, String> {
        match value {
            "system" => Ok(Self::System),
            "bundled" => Ok(Self::Bundled),
            "none" | "native" | "native-tools" => Ok(Self::None),
            _ => Err(format!(
                "Invalid --python mode '{value}' (expected 'system', 'bundled', or 'none')"
            )),
        }
    }

    fn feature(self) -> &'static str {
        match self {
            Self::System => "python-system",
            Self::Bundled => "python-bundled",
            Self::None => "native-tools-only",
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask error: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return Ok(());
    };

    match command.as_str() {
        "build" => run_build(args.collect()),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("Unknown xtask command '{other}'")),
    }
}

fn run_build(raw_args: Vec<String>) -> Result<(), String> {
    let (python_mode, mut cargo_args) = parse_build_args(raw_args)?;

    if has_conflicting_python_feature(&cargo_args, python_mode) {
        return Err(format!(
            "Conflicting python feature flags in cargo args for mode '{}'",
            python_mode.feature()
        ));
    }

    if !has_package_selection(&cargo_args) {
        cargo_args.push("-p".into());
        cargo_args.push("lash-cli".into());
    }

    if !has_python_feature(&cargo_args, python_mode.feature()) {
        cargo_args.push("--features".into());
        cargo_args.push(python_mode.feature().into());
    }

    let root = workspace_root()?;
    let mut cargo = Command::new("cargo");
    cargo.current_dir(&root).arg("build").args(&cargo_args);

    match python_mode {
        PythonMode::Bundled => {
            let target = parse_target_flag(&cargo_args);
            let config_path = ensure_bundled_config(&root, target.as_deref())?;
            cargo.env("PYO3_CONFIG_FILE", &config_path);
        }
        PythonMode::None => {
            cargo.env_remove("PYO3_CONFIG_FILE");
        }
        PythonMode::System => {}
    }

    let status = cargo
        .status()
        .map_err(|e| format!("Failed to run cargo build: {e}"))?;
    if !status.success() {
        return Err(format!("cargo build failed with status {status}"));
    }

    Ok(())
}

fn parse_build_args(raw_args: Vec<String>) -> Result<(PythonMode, Vec<String>), String> {
    let mut python_mode = PythonMode::System;
    let mut cargo_args = Vec::new();

    let mut i = 0usize;
    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "--" {
            cargo_args.extend(raw_args.into_iter().skip(i + 1));
            break;
        }
        if arg == "--python" {
            let value = raw_args
                .get(i + 1)
                .ok_or_else(|| "Missing value after --python".to_string())?;
            python_mode = PythonMode::from_flag(value)?;
            i += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--python=") {
            python_mode = PythonMode::from_flag(value)?;
            i += 1;
            continue;
        }
        cargo_args.push(arg.clone());
        i += 1;
    }

    Ok((python_mode, cargo_args))
}

fn parse_target_flag(args: &[String]) -> Option<String> {
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--target" {
            if let Some(value) = args.get(i + 1) {
                return Some(value.clone());
            }
            return None;
        }
        if let Some(value) = arg.strip_prefix("--target=") {
            return Some(value.to_string());
        }
        i += 1;
    }
    None
}

fn has_package_selection(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "-p"
            || arg == "--package"
            || arg.starts_with("--package=")
            || arg == "--workspace"
            || arg == "--all"
    })
}

fn has_python_feature(args: &[String], feature: &str) -> bool {
    iter_feature_values(args).into_iter().any(|f| f == feature)
}

fn has_conflicting_python_feature(args: &[String], mode: PythonMode) -> bool {
    let disallowed = match mode {
        PythonMode::System => ["python-bundled", "native-tools-only"].as_slice(),
        PythonMode::Bundled => ["python-system", "native-tools-only"].as_slice(),
        PythonMode::None => ["python-system", "python-bundled"].as_slice(),
    };
    iter_feature_values(args)
        .into_iter()
        .any(|feature| disallowed.contains(&feature.as_str()))
}

fn iter_feature_values(args: &[String]) -> Vec<String> {
    let mut values = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--features" || arg == "-F" {
            if let Some(feature_list) = args.get(i + 1) {
                values.extend(split_features(feature_list));
                i += 2;
                continue;
            }
            break;
        }
        if let Some(feature_list) = arg.strip_prefix("--features=") {
            values.extend(split_features(feature_list));
            i += 1;
            continue;
        }
        if let Some(feature_list) = arg.strip_prefix("-F") {
            if !feature_list.is_empty() {
                values.extend(split_features(feature_list));
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    values
}

fn split_features(raw: &str) -> impl Iterator<Item = String> + '_ {
    raw.split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

fn workspace_root() -> Result<PathBuf, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Failed to determine workspace root".to_string())
}

fn ensure_bundled_config(root: &Path, target: Option<&str>) -> Result<PathBuf, String> {
    if let Some(existing) = env::var_os("PYO3_CONFIG_FILE") {
        let path = PathBuf::from(existing);
        if path.exists() {
            return Ok(path);
        }
        eprintln!(
            "Warning: PYO3_CONFIG_FILE points to missing path {}, falling back to workspace default",
            path.display()
        );
    }

    let config_path = root
        .join("target")
        .join("python-standalone")
        .join("pyo3-config.txt");
    if config_path.exists() {
        return Ok(config_path);
    }

    let script = root.join("scripts").join("fetch-python.sh");
    if !script.exists() {
        return Err(format!(
            "Missing bootstrap script at {}",
            script.to_string_lossy()
        ));
    }

    let mut fetch = Command::new("bash");
    fetch.current_dir(root).arg(script);
    if let Some(target) = target {
        fetch.arg(target);
    }

    let status = fetch
        .status()
        .map_err(|e| format!("Failed to run scripts/fetch-python.sh: {e}"))?;
    if !status.success() {
        return Err(format!(
            "scripts/fetch-python.sh failed with status {status}"
        ));
    }

    if !config_path.exists() {
        return Err(format!(
            "Bundled config was not generated at {}",
            config_path.display()
        ));
    }

    Ok(config_path)
}

fn print_help() {
    println!(
        "cargo xtask build [--python <system|bundled|none>] [-- <cargo build args>]\n\
         \n\
         Examples:\n\
         cargo xtask build\n\
         cargo xtask build --python bundled\n\
         cargo xtask build --python none\n\
         cargo xtask build --python bundled -- --release\n\
         cargo xtask build --python bundled -- -p lash-cli --target x86_64-unknown-linux-gnu"
    );
}
