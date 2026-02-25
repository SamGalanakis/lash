use std::path::PathBuf;

#[cfg(feature = "python-bundled")]
const PYTHON_MAJOR_MINOR: &str = "3.14";

#[cfg(feature = "python-bundled")]
const STDLIB_TAR_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stdlib.tar.gz"));
#[cfg(feature = "python-bundled")]
const STDLIB_HASH: &str = env!("LASH_PYTHON_STDLIB_HASH");

/// Ensure the Python home backing files exist for the selected build mode.
///
/// - `python-bundled`: extracts vendored stdlib + dill into `~/.cache/lash/python-3.14/`
/// - `python-system`: returns `Unsupported` (no bundled stdlib is available)
#[cfg(feature = "python-bundled")]
pub fn ensure_python_home() -> Result<PathBuf, std::io::Error> {
    let cache_dir = crate::lash_cache_dir().join(format!("python-{PYTHON_MAJOR_MINOR}"));

    let lib_dir = cache_dir.join("lib");
    let version_file = cache_dir.join(".version");

    // Check if already extracted and up-to-date
    if version_file.exists()
        && let Ok(v) = std::fs::read_to_string(&version_file)
        && v.trim() == STDLIB_HASH
    {
        return Ok(lib_dir);
    }

    tracing::info!("Extracting Python stdlib to {}", cache_dir.display());

    // Clean and recreate
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir)?;
    }
    std::fs::create_dir_all(&lib_dir)?;

    // Extract stdlib.tar.gz
    let decoder = flate2::read::GzDecoder::new(STDLIB_TAR_GZ);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(&lib_dir)?;

    // Write version marker
    std::fs::write(&version_file, STDLIB_HASH)?;

    Ok(lib_dir)
}

#[cfg(not(feature = "python-bundled"))]
pub fn ensure_python_home() -> Result<PathBuf, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "python-bundled feature not enabled",
    ))
}

/// Return the PYTHONHOME value (the directory containing `lib/python3.14/`).
#[cfg(feature = "python-bundled")]
pub fn python_home(lib_dir: &std::path::Path) -> PathBuf {
    // lib_dir is ~/.cache/lash/python-3.14/lib
    // PYTHONHOME should be ~/.cache/lash/python-3.14 (parent of lib/)
    lib_dir
        .parent()
        .expect("lib_dir should have parent")
        .to_path_buf()
}
