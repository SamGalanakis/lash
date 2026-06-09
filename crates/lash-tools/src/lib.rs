//! Built-in tool suite for the lash agent runtime.
//!
//! Each module is a self-contained tool family sharing the
//! [`lash_tool_support`] utility layer:
//!
//! - [`apply_patch`] — `files.patch` envelope-diff editing
//! - [`files`] — `files.read` / `files.ls` / `files.glob`
//! - [`search`] — ripgrep/fff-backed `search.grep`
//! - [`shell`] — `shell.exec` / `shell.start` / `shell.write`
//! - [`web`] — `web.fetch` / `web.search`

pub mod apply_patch;
pub mod files;
pub mod search;
pub mod shell;
pub mod web;
