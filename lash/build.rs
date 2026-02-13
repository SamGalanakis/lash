fn main() {
    // Re-run if any .baml source file changes
    println!("cargo:rerun-if-changed=baml_src");

    // In Docker, the baml_client may be pre-generated.
    // Only run baml-cli locally when it's available.
    let already_generated = std::path::Path::new("src/baml_client/mod.rs").exists();

    if already_generated {
        if let Ok(status) = std::process::Command::new("baml-cli")
            .arg("generate")
            .arg("--from")
            .arg("baml_src")
            .status()
        {
            assert!(status.success(), "baml-cli generate failed");
        }
        // If baml-cli isn't installed but files exist, that's fine — use what's there
    } else {
        // No generated files yet — baml-cli is required
        let status = std::process::Command::new("baml-cli")
            .arg("generate")
            .arg("--from")
            .arg("baml_src")
            .status()
            .expect(
                "Failed to run baml-cli generate and no baml_client exists. \
                 Install with: cargo install baml-cli",
            );
        assert!(status.success(), "baml-cli generate failed");
    }
}
