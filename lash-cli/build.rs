fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();

    // Export all symbols so ctypes.pythonapi can resolve Python C API functions
    // (e.g. PyCapsule_New) from the statically-linked libpython.
    if target.contains("linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    }
}
