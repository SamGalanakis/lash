use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub(crate) fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

pub(crate) struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    pub(crate) fn set(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        // Tests serialize access with env_lock() so mutating process env is safe here.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            // Tests serialize access with env_lock() so mutating process env is safe here.
            unsafe { std::env::set_var(self.key, previous) };
        } else {
            // Tests serialize access with env_lock() so mutating process env is safe here.
            unsafe { std::env::remove_var(self.key) };
        }
    }
}

pub(crate) struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    pub(crate) fn new(prefix: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("{prefix}-{}", uuid::Uuid::new_v4().as_simple()));
        std::fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
