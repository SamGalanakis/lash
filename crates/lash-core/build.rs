use std::path::PathBuf;
use std::process::Command;

fn main() {
    generate_model_snapshot();
}

fn generate_model_snapshot() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cache_path = out_dir.join("models_dev_api.json");
    let snapshot_path = out_dir.join("models_snapshot.json");

    let need_download = if cache_path.exists() {
        std::fs::metadata(&cache_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|mtime| std::time::SystemTime::now().duration_since(mtime).ok())
            .map(|age| age.as_secs() > 86_400)
            .unwrap_or(true)
    } else {
        true
    };

    if need_download {
        eprintln!("Downloading models.dev/api.json...");
        let status = Command::new("curl")
            .args(["-fsSL", "--retry", "2", "--max-time", "15", "-o"])
            .arg(&cache_path)
            .arg("https://models.dev/api.json")
            .status();
        if !matches!(status, Ok(s) if s.success()) {
            eprintln!("Warning: could not download models.dev/api.json; checking cached copy");
        }
    }

    let snapshot = std::fs::read_to_string(&cache_path).unwrap_or_else(|err| {
        panic!(
            "failed to load models.dev/api.json for bundled model snapshot: {}. \
             Ensure the build can fetch https://models.dev/api.json or provide a valid cached copy.",
            err
        )
    });
    if !has_context_window_data(&snapshot) {
        panic!(
            "models.dev/api.json did not contain usable context-window data. \
             Refusing to build lash with an empty model catalog."
        );
    }

    std::fs::write(&snapshot_path, snapshot).expect("Failed to write models_snapshot.json");
}

fn has_context_window_data(raw: &str) -> bool {
    let Ok(providers) = serde_json::from_str::<serde_json::Value>(raw) else {
        return false;
    };
    let Some(obj) = providers.as_object() else {
        return false;
    };
    obj.values().any(|provider_info| {
        provider_info
            .get("models")
            .and_then(|models| models.as_object())
            .map(|models| {
                models.values().any(|info| {
                    info.get("limit")
                        .and_then(|limit| limit.get("context"))
                        .and_then(|context| context.as_u64())
                        .is_some()
                })
            })
            .unwrap_or(false)
    })
}
