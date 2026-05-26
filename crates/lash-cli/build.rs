use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    emit_git_build_metadata();
    generate_model_snapshot();

    let target = std::env::var("TARGET").unwrap_or_default();

    if target.contains("apple") {
        // PBS pgo+lto-full ships LLVM 21 bitcode; Apple's Xcode linker bundles
        // an older libLTO that can't parse it. Point ld at Homebrew LLVM's
        // libLTO.dylib so the system linker can process the bitcode.
        //
        // The LTO-compiled objects also reference ___isPlatformVersionAtLeast
        // from compiler-rt, which Rust's -nodefaultlibs strips out. Link
        // Homebrew LLVM's clang_rt.osx.a to provide it.
        for prefix in &[
            "/opt/homebrew/opt/llvm", // ARM macOS (Apple Silicon)
            "/usr/local/opt/llvm",    // Intel macOS
        ] {
            let lto_lib = format!("{prefix}/lib/libLTO.dylib");
            if Path::new(&lto_lib).exists() {
                println!("cargo:rustc-link-arg=-Wl,-lto_library,{lto_lib}");

                if let Some(rt_dir) = find_clang_rt_dir(Path::new(prefix)) {
                    println!("cargo:rustc-link-search=native={}", rt_dir.display());
                    println!("cargo:rustc-link-lib=static=clang_rt.osx");
                }
                break;
            }
        }
    }
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
            "cargo:warning=no context-window data loaded from models.dev; lash-cli will fail fast for uncataloged models"
        );
        "{}".to_string()
    };

    std::fs::write(&snapshot_path, snapshot).expect("Failed to write models_snapshot.json");
}

fn emit_git_build_metadata() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir);

    let head = git_output(&workspace_root, &["rev-parse", "HEAD"]);

    println!(
        "cargo:rustc-env=LASH_BUILD_GIT_HEAD={}",
        head.as_deref().unwrap_or("unknown")
    );
}

fn git_output(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_string())
}

/// Find the darwin compiler-rt directory under <llvm_prefix>/lib/clang/<ver>/lib/darwin.
fn find_clang_rt_dir(llvm_prefix: &Path) -> Option<PathBuf> {
    let clang_lib = llvm_prefix.join("lib").join("clang");
    let entries = std::fs::read_dir(&clang_lib).ok()?;
    for entry in entries.flatten() {
        let darwin_dir = entry.path().join("lib").join("darwin");
        if darwin_dir.join("libclang_rt.osx.a").exists() {
            return Some(darwin_dir);
        }
    }
    None
}
