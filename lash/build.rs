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
            eprintln!("Warning: could not download models.dev/api.json");
        }
    }

    let snapshot = if cache_path.exists() {
        std::fs::read_to_string(&cache_path).unwrap_or_else(|_| "{}".to_string())
    } else {
        println!(
            "cargo:warning=no context-window data loaded from models.dev; lash will fail fast for uncataloged models"
        );
        "{}".to_string()
    };

    std::fs::write(&snapshot_path, snapshot).expect("Failed to write models_snapshot.json");
}
