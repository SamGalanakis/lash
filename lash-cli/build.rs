use std::path::{Path, PathBuf};

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();

    if target.contains("linux") {
        // Export all symbols so ctypes.pythonapi can resolve Python C API functions
        // (e.g. PyCapsule_New) from the statically-linked libpython.
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    } else if target.contains("apple") {
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
