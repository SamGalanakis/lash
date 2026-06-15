//! Built-in tool suite for the lash agent runtime.
//!
//! Each module is a self-contained tool family sharing the
//! [`lash_tool_support`] utility layer:
//!
//! - [`apply_patch`] — `files.patch` envelope-diff editing
//! - [`files`] — `files.read` / `files.ls` / `files.glob`
//! - [`shell`] — `shell.exec` / `shell.start` / `shell.write`
//! - [`web`] — `web.fetch` / `web.search`
//!
//! CLI-owned local grep lives in the separate `lash-search-tools` crate so
//! non-CLI hosts do not inherit the fff-search build dependency.

pub mod apply_patch;
pub mod files;
pub mod shell;
pub mod web;
