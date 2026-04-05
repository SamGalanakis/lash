use std::path::{Path, PathBuf};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoStatus {
    pub repo_root: PathBuf,
    pub repo_name: String,
    pub branch: String,
    pub worktree: Option<String>,
}

impl RepoStatus {
    pub fn display_ref(&self) -> String {
        match &self.worktree {
            Some(worktree) if worktree != &self.branch => format!("{} · {}", self.branch, worktree),
            _ => self.branch.clone(),
        }
    }
}

pub fn detect_repo_status(cwd: &Path) -> Option<RepoStatus> {
    let (repo_root, git_dir) = find_repo_root_and_git_dir(cwd)?;
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let branch = parse_head(&head)?;
    let worktree = parse_worktree_name(&git_dir);
    let repo_name = repo_root
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("repo")
        .to_string();
    Some(RepoStatus {
        repo_root,
        repo_name,
        branch,
        worktree,
    })
}

fn find_repo_root_and_git_dir(start: &Path) -> Option<(PathBuf, PathBuf)> {
    for dir in start.ancestors() {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some((dir.to_path_buf(), dot_git));
        }
        if dot_git.is_file() {
            let contents = std::fs::read_to_string(&dot_git).ok()?;
            let git_dir = parse_gitdir_file(dir, &contents)?;
            return Some((dir.to_path_buf(), git_dir));
        }
    }
    None
}

fn parse_gitdir_file(repo_root: &Path, contents: &str) -> Option<PathBuf> {
    let line = contents.lines().next()?.trim();
    let gitdir = line.strip_prefix("gitdir:")?.trim();
    let path = PathBuf::from(gitdir);
    Some(if path.is_absolute() {
        path
    } else {
        repo_root.join(path)
    })
}

fn parse_head(head: &str) -> Option<String> {
    let trimmed = head.trim();
    if let Some(reference) = trimmed.strip_prefix("ref:") {
        let reference = reference.trim();
        return Some(
            reference
                .rsplit('/')
                .next()
                .filter(|value| !value.is_empty())?
                .to_string(),
        );
    }
    let detached: String = trimmed.chars().take(12).collect();
    (!detached.is_empty()).then_some(format!("detached:{detached}"))
}

fn parse_worktree_name(git_dir: &Path) -> Option<String> {
    let mut saw_worktrees = false;
    for component in git_dir.components() {
        let part = component.as_os_str().to_str()?;
        if saw_worktrees {
            return Some(part.to_string());
        }
        if part == "worktrees" {
            saw_worktrees = true;
        }
    }
    None
}
