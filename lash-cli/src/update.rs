use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, bail};
use chrono::{SecondsFormat, Utc};
use reqwest::header::{ACCEPT, USER_AGENT};
use semver::Version;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use uuid::Uuid;

const DEFAULT_REPO: &str = "SamGalanakis/lash";
const GITHUB_API_BASE: &str = "https://api.github.com";
const UPDATE_CHECK_INTERVAL_SECS: i64 = 24 * 60 * 60;

#[derive(Clone, Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LatestRelease {
    repo: String,
    tag: String,
    version: Version,
    commit: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct UpdateState {
    repo: String,
    last_checked_unix: i64,
    latest_tag: Option<String>,
    latest_commit: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InstallManifest {
    repo: String,
    version: String,
    install_dir: String,
    binary_path: String,
    installed_at: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InstallTarget {
    repo: String,
    install_dir: PathBuf,
}

pub(crate) async fn check_update_text() -> anyhow::Result<String> {
    let release = latest_release(CheckMode::ForceLive).await?;
    Ok(match release {
        Some(release) => format_available_update(&release, false),
        None => format!("lash v{} is up to date.", crate::APP_VERSION),
    })
}

pub(crate) async fn background_notification_message() -> Option<String> {
    match latest_release(CheckMode::CachedOrLive).await {
        Ok(Some(release)) => Some(format_available_update(&release, true)),
        Ok(None) | Err(_) => None,
    }
}

pub(crate) async fn install_latest_release() -> anyhow::Result<()> {
    #[cfg(windows)]
    bail!("`lash --update` is not supported on Windows.");

    #[cfg(not(windows))]
    {
        let release = match latest_release(CheckMode::ForceLive).await? {
            Some(release) => release,
            None => {
                println!("lash v{} is up to date.", crate::APP_VERSION);
                return Ok(());
            }
        };

        let repo = release.repo.clone();
        let manifest = load_install_manifest();
        let current_exe =
            env::current_exe().context("Failed to determine the current lash binary path")?;
        let env_install_dir = env::var_os("LASH_INSTALL_DIR");
        let home_dir = user_home_dir();

        let Some(target) = resolve_install_target(
            &repo,
            manifest.as_ref(),
            env_install_dir.as_deref(),
            &current_exe,
            home_dir.as_deref(),
        ) else {
            bail!(
                "Automatic update is unavailable for this installation.\n\nUpdate manually with:\n{}",
                manual_install_command(&repo, &release.tag)
            );
        };

        let installer_path = download_installer(&target.repo, &release.tag).await?;
        println!(
            "Installing {} to {}...",
            release.tag,
            target.install_dir.display()
        );
        let status = Command::new("bash")
            .arg(&installer_path)
            .env("LASH_REPO", &target.repo)
            .env("LASH_VERSION", &release.tag)
            .env("LASH_INSTALL_DIR", &target.install_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .await
            .context("Failed to launch the release installer")?;
        let _ = tokio::fs::remove_file(&installer_path).await;
        if !status.success() {
            let code = status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".to_string());
            bail!("Installer exited unsuccessfully ({code}).");
        }
        let _ = save_install_manifest(&InstallManifest {
            repo: target.repo.clone(),
            version: release.version.to_string(),
            install_dir: target.install_dir.display().to_string(),
            binary_path: target.install_dir.join("lash").display().to_string(),
            installed_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        });
        println!(
            "Updated lash to {}. Restart lash to use the new version.",
            release.tag
        );
        Ok(())
    }
}

enum CheckMode {
    ForceLive,
    CachedOrLive,
}

async fn latest_release(mode: CheckMode) -> anyhow::Result<Option<LatestRelease>> {
    let repo = repo_for_updates();
    let current = current_version()?;
    let now = Utc::now().timestamp();
    let cached = load_update_state(&repo);

    if matches!(mode, CheckMode::CachedOrLive)
        && let Some(state) = cached.as_ref()
        && !is_stale(state, now)
    {
        return release_from_cached_state(&repo, state, &current);
    }

    match fetch_latest_release(&repo).await {
        Ok(release) => {
            let _ = save_update_state(&UpdateState {
                repo: repo.clone(),
                last_checked_unix: now,
                latest_tag: Some(release.tag.clone()),
                latest_commit: release.commit.clone(),
            });
            if should_offer_update(&release, &current) {
                Ok(Some(release))
            } else {
                Ok(None)
            }
        }
        Err(err) if matches!(mode, CheckMode::CachedOrLive) => {
            if let Some(state) = cached.as_ref() {
                return release_from_cached_state(&repo, state, &current);
            }
            Err(err)
        }
        Err(err) => Err(err),
    }
}

async fn fetch_latest_release(repo: &str) -> anyhow::Result<LatestRelease> {
    let client = reqwest::Client::builder()
        .user_agent(format!("lash/{}", crate::APP_VERSION))
        .build()
        .context("Failed to build update-check HTTP client")?;
    let url = format!("{GITHUB_API_BASE}/repos/{repo}/releases/latest");
    let release = client
        .get(url)
        .header(ACCEPT, "application/vnd.github+json")
        .header(USER_AGENT, format!("lash/{}", crate::APP_VERSION))
        .send()
        .await
        .context("Failed to query GitHub releases")?
        .error_for_status()
        .context("GitHub release query failed")?
        .json::<GitHubRelease>()
        .await
        .context("Failed to decode GitHub release metadata")?;
    Ok(LatestRelease {
        repo: repo.to_string(),
        version: parse_release_tag(&release.tag_name)?,
        commit: resolve_release_commit(repo, &release.tag_name).await.ok(),
        tag: release.tag_name,
    })
}

fn repo_for_updates() -> String {
    nonempty_string(env::var("LASH_REPO").ok())
        .or_else(|| load_install_manifest().map(|manifest| manifest.repo))
        .unwrap_or_else(|| DEFAULT_REPO.to_string())
}

fn current_version() -> anyhow::Result<Version> {
    Version::parse(crate::APP_VERSION)
        .with_context(|| format!("Unsupported lash version format: {}", crate::APP_VERSION))
}

fn parse_release_tag(tag: &str) -> anyhow::Result<Version> {
    let trimmed = tag.trim().trim_start_matches('v');
    Version::parse(trimmed).with_context(|| format!("Unsupported release tag format: {tag}"))
}

fn release_from_tag(
    repo: &str,
    tag: Option<&str>,
    commit: Option<&str>,
    current: &Version,
) -> anyhow::Result<Option<LatestRelease>> {
    let Some(tag) = nonempty_string(tag.map(ToOwned::to_owned)) else {
        return Ok(None);
    };
    let version = parse_release_tag(&tag)?;
    if version <= *current {
        return Ok(None);
    }
    Ok(Some(LatestRelease {
        repo: repo.to_string(),
        version,
        commit: commit.map(ToOwned::to_owned),
        tag: tag.clone(),
    }))
}

fn release_from_cached_state(
    repo: &str,
    state: &UpdateState,
    current: &Version,
) -> anyhow::Result<Option<LatestRelease>> {
    release_from_tag(
        repo,
        state.latest_tag.as_deref(),
        state.latest_commit.as_deref(),
        current,
    )
}

fn should_offer_update(release: &LatestRelease, current: &Version) -> bool {
    should_offer_update_for_build(
        release,
        current,
        crate::BUILD_GIT_HEAD,
        build_workspace_root().as_deref(),
    )
}

fn should_offer_update_for_build(
    release: &LatestRelease,
    current: &Version,
    build_head: &str,
    workspace_root: Option<&Path>,
) -> bool {
    if release.version <= *current {
        return false;
    }

    !current_build_matches_release_source(release, build_head, workspace_root)
}

fn current_build_matches_release_source(
    release: &LatestRelease,
    build_head: &str,
    workspace_root: Option<&Path>,
) -> bool {
    if build_git_head_is_dirty_at(workspace_root) {
        return false;
    }
    let Some(workspace_root) = workspace_root else {
        return false;
    };
    let Some(release_commit) = release.commit.as_deref() else {
        return false;
    };
    build_matches_release_source(workspace_root, build_head, release_commit)
}

fn build_matches_release_source(
    workspace_root: &Path,
    build_head: &str,
    release_commit: &str,
) -> bool {
    if build_head == "unknown" || release_commit.is_empty() {
        return false;
    }

    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", build_head, release_commit])
        .current_dir(workspace_root)
        .output();
    let Ok(output) = output else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let Ok(stdout) = String::from_utf8(output.stdout) else {
        return false;
    };
    let paths: Vec<_> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    !paths.is_empty() && paths.into_iter().all(is_version_metadata_path)
}

fn is_version_metadata_path(path: &str) -> bool {
    matches!(path, "Cargo.toml" | "Cargo.lock")
}

fn build_git_head_is_dirty_at(workspace_root: Option<&Path>) -> bool {
    let Some(workspace_root) = workspace_root else {
        return true;
    };
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(workspace_root)
        .output();
    match output {
        Ok(output) if output.status.success() => !output.stdout.is_empty(),
        _ => true,
    }
}

fn build_workspace_root() -> Option<PathBuf> {
    let build_head = crate::BUILD_GIT_HEAD;
    if build_head == "unknown" {
        return None;
    }

    let current_exe = env::current_exe().ok()?;
    let mut dir = current_exe.parent()?.to_path_buf();
    loop {
        let git_dir = dir.join(".git");
        if git_dir.is_dir() || git_dir.is_file() {
            let output = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(&dir)
                .output()
                .ok()?;
            if output.status.success() {
                let head = String::from_utf8(output.stdout).ok()?;
                if head.trim() == build_head {
                    return Some(dir);
                }
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

async fn resolve_release_commit(repo: &str, tag: &str) -> anyhow::Result<String> {
    let output = Command::new("git")
        .args(["ls-remote", &format!("https://github.com/{repo}"), tag])
        .output()
        .await
        .context("Failed to resolve release tag commit")?;
    if !output.status.success() {
        bail!("git ls-remote failed for release tag {tag}");
    }
    let stdout = String::from_utf8(output.stdout).context("Release tag lookup was not UTF-8")?;
    let line = stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .with_context(|| format!("Release tag {tag} was not found on remote"))?;
    let commit = line
        .split_whitespace()
        .next()
        .with_context(|| format!("Failed to parse commit for release tag {tag}"))?;
    Ok(commit.to_string())
}

fn format_available_update(release: &LatestRelease, in_session: bool) -> String {
    if can_auto_apply().is_some() {
        if in_session {
            format!(
                "Update available: {} (current v{}).\nRun `lash --update` after exit.",
                release.tag,
                crate::APP_VERSION
            )
        } else {
            format!(
                "Update available: {} (current v{}).\nRun `lash --update` to install it.",
                release.tag,
                crate::APP_VERSION
            )
        }
    } else if in_session {
        format!(
            "Update available: {} (current v{}).\nRun `lash --check-update` after exit for install instructions.",
            release.tag,
            crate::APP_VERSION
        )
    } else {
        format!(
            "Update available: {} (current v{}).\nAutomatic update is unavailable for this installation.\nUpdate manually with:\n{}",
            release.tag,
            crate::APP_VERSION,
            manual_install_command(&release.repo, &release.tag)
        )
    }
}

fn can_auto_apply() -> Option<InstallTarget> {
    let repo = repo_for_updates();
    let manifest = load_install_manifest();
    let current_exe = env::current_exe().ok()?;
    let env_install_dir = env::var_os("LASH_INSTALL_DIR");
    let home_dir = user_home_dir();
    resolve_install_target(
        &repo,
        manifest.as_ref(),
        env_install_dir.as_deref(),
        &current_exe,
        home_dir.as_deref(),
    )
}

fn resolve_install_target(
    repo: &str,
    manifest: Option<&InstallManifest>,
    env_install_dir: Option<&OsStr>,
    current_exe: &Path,
    home_dir: Option<&Path>,
) -> Option<InstallTarget> {
    if let Some(dir) = env_install_dir.and_then(nonempty_os_value) {
        return Some(InstallTarget {
            repo: repo.to_string(),
            install_dir: PathBuf::from(dir),
        });
    }

    if let Some(manifest) = manifest {
        let manifest_dir = PathBuf::from(&manifest.install_dir);
        if manifest.repo == repo
            && (same_executable(Path::new(&manifest.binary_path), current_exe)
                || current_exe.parent() == Some(manifest_dir.as_path()))
        {
            return Some(InstallTarget {
                repo: manifest.repo.clone(),
                install_dir: manifest_dir,
            });
        }
    }

    let current_dir = current_exe.parent()?;
    if is_supported_install_dir(current_dir, home_dir) {
        return Some(InstallTarget {
            repo: repo.to_string(),
            install_dir: current_dir.to_path_buf(),
        });
    }

    None
}

fn same_executable(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn is_supported_install_dir(dir: &Path, home_dir: Option<&Path>) -> bool {
    if dir == Path::new("/usr/local/bin") {
        return true;
    }
    if let Some(home_dir) = home_dir {
        return dir == home_dir.join(".local").join("bin") || dir == home_dir.join("bin");
    }
    false
}

async fn download_installer(repo: &str, tag: &str) -> anyhow::Result<PathBuf> {
    let client = reqwest::Client::builder()
        .user_agent(format!("lash/{}", crate::APP_VERSION))
        .build()
        .context("Failed to build installer download client")?;
    let url = installer_url(repo, tag);
    let bytes = client
        .get(url)
        .header(ACCEPT, "application/octet-stream")
        .header(USER_AGENT, format!("lash/{}", crate::APP_VERSION))
        .send()
        .await
        .context("Failed to download the release installer")?
        .error_for_status()
        .context("Release installer download failed")?
        .bytes()
        .await
        .context("Failed to read installer download")?;
    let path = env::temp_dir().join(format!("lash-update-{}.sh", Uuid::new_v4()));
    tokio::fs::write(&path, bytes)
        .await
        .with_context(|| format!("Failed to write installer to {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::Permissions::from_mode(0o700);
        tokio::fs::set_permissions(&path, permissions)
            .await
            .with_context(|| format!("Failed to mark {} executable", path.display()))?;
    }
    Ok(path)
}

fn load_update_state(repo: &str) -> Option<UpdateState> {
    let raw = std::fs::read_to_string(update_state_path()).ok()?;
    let state = serde_json::from_str::<UpdateState>(&raw).ok()?;
    (state.repo == repo).then_some(state)
}

fn save_update_state(state: &UpdateState) -> anyhow::Result<()> {
    std::fs::create_dir_all(crate::paths::lash_home())?;
    std::fs::write(update_state_path(), serde_json::to_vec_pretty(state)?)
        .context("Failed to persist update-check state")
}

fn load_install_manifest() -> Option<InstallManifest> {
    let raw = std::fs::read_to_string(install_manifest_path()).ok()?;
    serde_json::from_str::<InstallManifest>(&raw).ok()
}

fn save_install_manifest(manifest: &InstallManifest) -> anyhow::Result<()> {
    std::fs::create_dir_all(crate::paths::lash_home())?;
    std::fs::write(
        install_manifest_path(),
        serde_json::to_vec_pretty(manifest)?,
    )
    .context("Failed to persist install metadata")
}

fn update_state_path() -> PathBuf {
    crate::paths::lash_home().join("update-state.json")
}

fn install_manifest_path() -> PathBuf {
    crate::paths::lash_home().join("install.json")
}

fn is_stale(state: &UpdateState, now: i64) -> bool {
    now.saturating_sub(state.last_checked_unix) >= UPDATE_CHECK_INTERVAL_SECS
}

fn installer_url(repo: &str, tag: &str) -> String {
    format!("https://github.com/{repo}/releases/download/{tag}/install_lash.sh")
}

fn manual_install_command(repo: &str, tag: &str) -> String {
    format!("curl -fsSL {} | bash", installer_url(repo, tag))
}

fn nonempty_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn nonempty_os_value(value: &OsStr) -> Option<&OsStr> {
    if value.is_empty() { None } else { Some(value) }
}

fn user_home_dir() -> Option<PathBuf> {
    if let Some(home) = env::var_os("HOME").as_deref().and_then(nonempty_os_value) {
        return Some(PathBuf::from(home));
    }
    env::var_os("USERPROFILE")
        .as_deref()
        .and_then(nonempty_os_value)
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn workspace_repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    fn prepare_build_checkout() -> TempDir {
        let temp = TempDir::new().expect("tempdir");
        let status = std::process::Command::new("git")
            .args(["clone", "-q", "--no-local"])
            .arg(workspace_repo_root())
            .arg(temp.path())
            .status()
            .expect("clone workspace repo");
        assert!(status.success());
        let status = std::process::Command::new("git")
            .args(["checkout", "-q", crate::BUILD_GIT_HEAD])
            .current_dir(temp.path())
            .status()
            .expect("checkout build head");
        assert!(status.success());
        temp
    }

    fn create_metadata_only_release_commit(repo: &Path) -> String {
        for path in ["Cargo.toml", "Cargo.lock"] {
            let file = repo.join(path);
            let contents = std::fs::read_to_string(&file).expect("read metadata file");
            std::fs::write(&file, format!("{contents}\n")).expect("modify metadata file");
        }

        let status = std::process::Command::new("git")
            .args(["add", "Cargo.toml", "Cargo.lock"])
            .current_dir(repo)
            .status()
            .expect("stage metadata files");
        assert!(status.success());

        let status = std::process::Command::new("git")
            .args([
                "-c",
                "user.name=lash-tests",
                "-c",
                "user.email=lash-tests@example.com",
                "commit",
                "-q",
                "-m",
                "metadata-only release commit",
            ])
            .current_dir(repo)
            .status()
            .expect("commit metadata files");
        assert!(status.success());

        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repo)
            .output()
            .expect("resolve release commit");
        assert!(output.status.success());
        let release_commit = String::from_utf8(output.stdout)
            .expect("release commit utf8")
            .trim()
            .to_string();

        let status = std::process::Command::new("git")
            .args(["checkout", "-q", crate::BUILD_GIT_HEAD])
            .current_dir(repo)
            .status()
            .expect("restore build head");
        assert!(status.success());

        release_commit
    }

    #[test]
    fn parses_release_tags_as_semver() {
        assert_eq!(
            parse_release_tag("v0.2.12").expect("tag should parse"),
            Version::parse("0.2.12").expect("semver")
        );
    }

    #[test]
    fn newer_cached_release_counts_as_update() {
        let current = Version::parse("0.2.12").expect("current version");
        let release =
            release_from_tag(DEFAULT_REPO, Some("v0.2.13"), None, &current).expect("release parse");
        assert_eq!(release.expect("available").tag, "v0.2.13");
        assert!(
            release_from_tag(DEFAULT_REPO, Some("v0.2.12"), None, &current)
                .expect("same version")
                .is_none()
        );
    }

    #[test]
    fn source_equivalent_release_does_not_offer_update() {
        let temp = prepare_build_checkout();
        let release_commit = create_metadata_only_release_commit(temp.path());
        let current = Version::parse("0.2.12").expect("current version");
        let release = LatestRelease {
            repo: DEFAULT_REPO.to_string(),
            tag: "v0.2.18".to_string(),
            version: Version::parse("0.2.18").expect("release version"),
            commit: Some(release_commit),
        };
        assert!(!should_offer_update_for_build(
            &release,
            &current,
            crate::BUILD_GIT_HEAD,
            Some(temp.path())
        ));
    }

    #[test]
    fn dirty_source_checkout_still_offers_update() {
        let temp = prepare_build_checkout();
        let release_commit = create_metadata_only_release_commit(temp.path());
        std::fs::write(
            temp.path().join("Cargo.toml"),
            std::fs::read_to_string(temp.path().join("Cargo.toml")).expect("read Cargo.toml")
                + "\n",
        )
        .expect("modify tracked file");
        let current = Version::parse("0.2.12").expect("current version");
        let release = LatestRelease {
            repo: DEFAULT_REPO.to_string(),
            tag: "v0.2.18".to_string(),
            version: Version::parse("0.2.18").expect("release version"),
            commit: Some(release_commit),
        };
        assert!(build_git_head_is_dirty_at(Some(temp.path())));
        assert!(should_offer_update_for_build(
            &release,
            &current,
            crate::BUILD_GIT_HEAD,
            Some(temp.path())
        ));
    }

    #[test]
    fn supported_install_dirs_match_installer_targets() {
        let home = Path::new("/tmp/lash-home");
        assert!(is_supported_install_dir(
            &home.join(".local").join("bin"),
            Some(home)
        ));
        assert!(is_supported_install_dir(&home.join("bin"), Some(home)));
        assert!(is_supported_install_dir(
            Path::new("/usr/local/bin"),
            Some(home)
        ));
        assert!(!is_supported_install_dir(
            Path::new("/opt/homebrew/bin"),
            Some(home)
        ));
    }

    #[test]
    fn resolve_target_prefers_matching_manifest() {
        let current_exe = PathBuf::from("/opt/lash/bin/lash");
        let manifest = InstallManifest {
            repo: DEFAULT_REPO.to_string(),
            version: "0.2.12".to_string(),
            install_dir: "/opt/lash/bin".to_string(),
            binary_path: "/opt/lash/bin/lash".to_string(),
            installed_at: "2026-03-25T00:00:00Z".to_string(),
        };
        let target = resolve_install_target(
            DEFAULT_REPO,
            Some(&manifest),
            None,
            &current_exe,
            Some(Path::new("/tmp/lash-home")),
        )
        .expect("install target");
        assert_eq!(target.install_dir, PathBuf::from("/opt/lash/bin"));
    }

    #[test]
    fn resolve_target_falls_back_to_local_bin() {
        let home = PathBuf::from("/tmp/lash-home");
        let current_exe = home.join(".local").join("bin").join("lash");
        let target =
            resolve_install_target(DEFAULT_REPO, None, None, &current_exe, Some(home.as_path()))
                .expect("install target");
        assert_eq!(target.install_dir, home.join(".local").join("bin"));
    }
}
