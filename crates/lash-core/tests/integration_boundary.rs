use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn crate_sources_do_not_name_integration_protocols() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut failures = Vec::new();

    for root in [
        crate_dir.join("Cargo.toml"),
        crate_dir.join("src"),
        crate_dir.join("tests"),
    ] {
        scan_path(&root, &mut failures);
    }

    assert!(
        failures.is_empty(),
        "core crate must stay integration-agnostic:\n{}",
        failures.join("\n")
    );
}

fn scan_path(path: &Path, failures: &mut Vec<String>) {
    if path.is_dir() {
        let mut entries = fs::read_dir(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
            .map(|entry| {
                entry
                    .unwrap_or_else(|err| {
                        panic!("failed to read entry under {}: {err}", path.display())
                    })
                    .path()
            })
            .collect::<Vec<_>>();
        entries.sort();
        for entry in entries {
            scan_path(&entry, failures);
        }
        return;
    }

    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    let lower = text.to_ascii_lowercase();
    for needle in [concat!("lash", "lang"), concat!("r", "lm")] {
        if !lower.contains(needle) {
            continue;
        }
        for (index, line) in text.lines().enumerate() {
            if line.to_ascii_lowercase().contains(needle) {
                failures.push(format!(
                    "{}:{} contains `{needle}`",
                    path.display(),
                    index + 1
                ));
            }
        }
    }
}
