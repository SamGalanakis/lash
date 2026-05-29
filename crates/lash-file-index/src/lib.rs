//! Fuzzy gitignore-aware file index for `@`-completion in interactive surfaces.
//!
//! A [`FileIndex`] walks a project root once on a background thread (using
//! [`ignore::WalkBuilder`]) and serves synchronous fuzzy matches via
//! [`nucleo::Matcher`] against the cached corpus. Walk-async, match-sync —
//! input loops never stall on disk IO once the cache is warm, and stale-result
//! reasoning is unnecessary because the matcher runs against a stable `Vec`.
//!
//! Inspired by `codex-rs/file-search` but with a smaller surface: no streaming
//! session, no continuous tick loop, no [`nucleo::Injector`] channel
//! orchestration. The walker fills a `Vec` once, flips state to `Ready`, and
//! invokes a one-shot callback so the caller can refresh whatever UI was
//! showing a "loading…" placeholder.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use ignore::WalkBuilder;
use ignore::WalkState;
use notify::RecursiveMode;
use notify::Watcher;
use nucleo::Config;
use nucleo::Matcher;
use nucleo::Utf32String;
use nucleo::pattern::AtomKind;
use nucleo::pattern::CaseMatching;
use nucleo::pattern::Normalization;
use nucleo::pattern::Pattern;

/// How long to wait for the FS-event stream to quiesce before rebuilding.
/// Generated-output directories are filtered before this point, so the debounce
/// only needs to coalesce real source bursts without making `@` completions feel
/// stale after a single save.
const REBUILD_DEBOUNCE: Duration = Duration::from_millis(75);

/// Hard cap on indexed entries. A misconfigured repo (e.g. a workspace inside a
/// huge cache directory with no `.gitignore`) will stop adding entries past
/// this threshold rather than OOM the index.
pub const MAX_ENTRIES: usize = 200_000;

/// Returned from [`FileIndex::matches`].
#[derive(Debug)]
pub enum MatchResult {
    /// Walk still running. Caller should show a loading placeholder; results
    /// will become available after the walker invokes its `on_ready` callback.
    Indexing,
    /// Walk complete; up to `limit` matches sorted by score desc, path asc.
    Ready(Vec<FileMatch>),
}

/// A single fuzzy-matched path.
#[derive(Debug, Clone)]
pub struct FileMatch {
    /// Path relative to the index root, in forward-slash form. Directories
    /// carry a trailing `/`. Kept as `String` (not `PathBuf`) because every
    /// downstream consumer wants UTF-8 text and the corpus is already known
    /// to be valid UTF-8 — wrapping in `PathBuf` only forces a re-validation
    /// and a clone on the way back out.
    pub path: String,
    pub is_dir: bool,
    pub score: u32,
    /// Matched character offsets into `path`. Renderers use these to bold
    /// matched chars in the popup.
    pub indices: Vec<u32>,
}

/// Fuzzy file index pinned to a single root.
pub struct FileIndex {
    state: Arc<Mutex<State>>,
    /// Reused across calls so nucleo's internal scratch buffers don't get
    /// allocated on every keystroke. `Mutex` because `Matcher::indices` takes
    /// `&mut self` and `matches()` is `&self`. Lock contention is a non-issue:
    /// the only contender for this lock is the rebuilder thread on FS events,
    /// which doesn't touch the matcher.
    matcher: Mutex<Matcher>,
}

enum State {
    Walking,
    Ready {
        /// All indexed entries.
        entries: Arc<Vec<Entry>>,
        /// Indices into `entries`, pre-sorted by `display` ascending. Lets the
        /// empty-query branch slice the first `limit` entries instead of
        /// re-sorting the full corpus on every keystroke.
        sorted: Arc<Vec<u32>>,
    },
}

#[derive(Clone)]
struct Entry {
    /// Display path relative to the index root, with trailing `/` for dirs.
    display: String,
    /// Pre-computed Utf32 form for nucleo matching.
    haystack: Utf32String,
    is_dir: bool,
}

impl FileIndex {
    /// Spawn the walker thread and return immediately. `on_ready` runs on the
    /// walker thread *after* state flips to `Ready`, so a `matches` call from
    /// inside the callback (or any subsequent call) sees the populated corpus.
    ///
    /// Once the initial walk lands, the same thread installs a recursive
    /// `notify` watcher on `root` and morphs into a debounced rebuild loop.
    /// Files created or removed mid-session show up in subsequent `matches`
    /// calls without any user-visible refresh action.
    pub fn for_root(root: PathBuf, on_ready: Box<dyn FnOnce() + Send>) -> Self {
        let state = Arc::new(Mutex::new(State::Walking));
        let walker_state = Arc::clone(&state);
        thread::Builder::new()
            .name("lash-file-index-walker".into())
            .spawn(move || {
                install_corpus(&walker_state, walk(&root));
                on_ready();
                run_watch_loop(&root, &walker_state);
            })
            .expect("spawn walker thread");
        Self {
            state,
            matcher: Mutex::new(Matcher::new(Config::DEFAULT.match_paths())),
        }
    }

    /// Block until the walker thread completes. Useful for tests that want a
    /// deterministic corpus before calling `matches`. Production callers
    /// should prefer [`Self::for_root`] so the input loop never blocks on disk
    /// IO; drive UI updates from the `on_ready` callback instead.
    pub fn for_root_blocking(root: PathBuf) -> Self {
        use std::sync::Condvar;

        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let pair_cb = Arc::clone(&pair);
        let index = Self::for_root(
            root,
            Box::new(move || {
                let (lock, cv) = &*pair_cb;
                if let Ok(mut done) = lock.lock() {
                    *done = true;
                    cv.notify_all();
                }
            }),
        );
        let (lock, cv) = &*pair;
        let mut done = lock.lock().expect("lock");
        while !*done {
            done = cv.wait(done).expect("cv wait");
        }
        index
    }

    /// Synchronous fuzzy match against the cached corpus.
    pub fn matches(&self, query: &str, limit: usize) -> MatchResult {
        let (entries, sorted) = {
            let guard = self.state.lock().expect("file-index state lock");
            match &*guard {
                State::Walking => return MatchResult::Indexing,
                State::Ready { entries, sorted } => (Arc::clone(entries), Arc::clone(sorted)),
            }
        };

        if query.is_empty() {
            // Pre-sorted at walk-completion time, so this is a slice + map.
            let out = sorted
                .iter()
                .take(limit)
                .map(|&i| {
                    let e = &entries[i as usize];
                    FileMatch {
                        path: e.display.clone(),
                        is_dir: e.is_dir,
                        score: 0,
                        indices: Vec::new(),
                    }
                })
                .collect();
            return MatchResult::Ready(out);
        }

        let pattern = Pattern::new(
            query,
            CaseMatching::Ignore,
            Normalization::Smart,
            AtomKind::Fuzzy,
        );

        // Online top-K via a min-heap of `Reverse<Candidate>` so the heap's
        // peek is the worst kept candidate. Bounded at `limit + 1` so we can
        // pop the worst as soon as we exceed budget — never holds more than
        // limit+1 elements.
        let mut heap: BinaryHeap<Reverse<Candidate<'_>>> = BinaryHeap::with_capacity(limit + 1);
        let mut idx_buf: Vec<u32> = Vec::new();
        let mut matcher = self.matcher.lock().expect("matcher lock");
        for entry in entries.iter() {
            idx_buf.clear();
            let haystack = entry.haystack.slice(..);
            let Some(score) = pattern.indices(haystack, &mut matcher, &mut idx_buf) else {
                continue;
            };
            // Skip the index clone for losers: only candidates that beat the
            // current worst kept (or fit while the heap is undersized) need
            // their indices materialized.
            let undersized = heap.len() < limit;
            if !undersized {
                let worst = &heap.peek().expect("non-empty heap").0;
                // `Candidate::cmp` ranks better candidates higher.
                let provisional = ProvisionalRank {
                    score,
                    display: entry.display.as_str(),
                };
                if provisional.cmp_against(worst) != Ordering::Greater {
                    continue;
                }
            }
            let mut indices = idx_buf.clone();
            indices.sort_unstable();
            indices.dedup();
            heap.push(Reverse(Candidate {
                score,
                entry,
                indices,
            }));
            if heap.len() > limit {
                heap.pop();
            }
        }
        drop(matcher);

        // Drain heap; results emerge in arbitrary order, so sort once at the
        // end — `limit` (typically 20) elements, so log-cost is negligible.
        let mut out: Vec<FileMatch> = heap
            .into_iter()
            .map(|Reverse(c)| FileMatch {
                path: c.entry.display.clone(),
                is_dir: c.entry.is_dir,
                score: c.score,
                indices: c.indices,
            })
            .collect();
        out.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
        MatchResult::Ready(out)
    }
}

/// Rank carrier used inside the top-K heap. "Better" is encoded as larger
/// per `Ord` (higher score, or smaller path on tie), and the heap wraps each
/// candidate in `Reverse` so `BinaryHeap::peek` returns the *worst* kept
/// candidate — the one to evict on the next replacement.
struct Candidate<'a> {
    score: u32,
    entry: &'a Entry,
    indices: Vec<u32>,
}

impl<'a> PartialEq for Candidate<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.entry.display == other.entry.display
    }
}
impl<'a> Eq for Candidate<'a> {}
impl<'a> Ord for Candidate<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Larger score = better; on tie, smaller path = better. Reverse the
        // path comparison so smaller path produces `Greater`.
        self.score
            .cmp(&other.score)
            .then_with(|| other.entry.display.cmp(&self.entry.display))
    }
}
impl<'a> PartialOrd for Candidate<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Lightweight rank for the "should I bother cloning indices?" check, so we
/// don't allocate a `Candidate` (with its empty `indices` Vec) just to compare
/// against the heap's current worst.
struct ProvisionalRank<'a> {
    score: u32,
    display: &'a str,
}
impl<'a> ProvisionalRank<'a> {
    fn cmp_against(&self, other: &Candidate<'_>) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.entry.display.as_str().cmp(self.display))
    }
}

fn walk(root: &Path) -> Vec<Entry> {
    let mut builder = WalkBuilder::new(root);
    builder
        // Include dotfiles. Most repos `.gitignore` the noisy ones anyway.
        .hidden(false)
        // Defensive: don't follow symlinks. A misconfigured symlink loop would
        // otherwise scan the same tree forever (or until our 200k cap).
        .follow_links(false)
        // Match git's own ignore semantics: only apply gitignore rules when a
        // git context exists. Without this, `~/.gitignore` containing `*` can
        // silently swallow every file in a non-git workspace. Same comment
        // appears verbatim in codex-rs/file-search/src/lib.rs.
        .require_git(true);

    let collected: Arc<Mutex<Vec<Entry>>> = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let walker = builder.build_parallel();
    let root_owned = root.to_path_buf();
    walker.run(|| {
        let collected = Arc::clone(&collected);
        let stop = Arc::clone(&stop);
        let root = root_owned.clone();
        Box::new(move |result| {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                return WalkState::Quit;
            }
            let entry = match result {
                Ok(e) => e,
                Err(_) => return WalkState::Continue,
            };
            // Skip the root itself.
            if entry.depth() == 0 {
                return WalkState::Continue;
            }
            let path = entry.path();
            let Ok(rel) = path.strip_prefix(&root) else {
                return WalkState::Continue;
            };
            let Some(rel_str) = rel.to_str() else {
                return WalkState::Continue;
            };
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            // Use forward slashes on all platforms so nucleo's path-aware
            // matching segments the haystack consistently. ignore yields
            // platform-native separators on Windows.
            let mut display = rel_str.replace('\\', "/");
            if is_dir {
                display.push('/');
            }
            let haystack = Utf32String::from(display.as_str());
            let item = Entry {
                display,
                haystack,
                is_dir,
            };
            if let Ok(mut guard) = collected.lock() {
                if guard.len() >= MAX_ENTRIES {
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    return WalkState::Quit;
                }
                guard.push(item);
            }
            WalkState::Continue
        })
    });

    Arc::try_unwrap(collected)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default()
}

/// Atomically install a freshly-walked corpus into `state`. Pre-computes the
/// path-asc sort order so the empty-query path doesn't have to sort 200k
/// entries on every keystroke.
fn install_corpus(state: &Arc<Mutex<State>>, entries: Vec<Entry>) {
    let mut sorted: Vec<u32> = (0..entries.len() as u32).collect();
    sorted.sort_by(|&a, &b| {
        entries[a as usize]
            .display
            .cmp(&entries[b as usize].display)
    });
    if let Ok(mut guard) = state.lock() {
        *guard = State::Ready {
            entries: Arc::new(entries),
            sorted: Arc::new(sorted),
        };
    }
}

/// Watch `root` recursively and re-walk on debounced filesystem events.
/// Returns when the watcher channel closes (e.g., process shutdown). If the
/// platform watcher can't be created or the root can't be watched, returns
/// silently — `@`-completion still works against the initial corpus, just
/// without auto-refresh.
fn run_watch_loop(root: &Path, state: &Arc<Mutex<State>>) {
    let (event_tx, event_rx) = mpsc::channel::<Vec<PathBuf>>();
    let event_tx_for_watcher = event_tx.clone();
    // Drop the local sender once the watcher owns its clone; that way, when
    // the watcher dies, the receiver sees a clean disconnect and the loop
    // exits.
    drop(event_tx);

    let watcher_result = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            let paths: Vec<PathBuf> = event
                .paths
                .into_iter()
                .filter(|path| !should_ignore_notify_path(path))
                .collect();
            if !paths.is_empty() {
                let _ = event_tx_for_watcher.send(paths);
            }
        }
    });
    let mut watcher = match watcher_result {
        Ok(w) => w,
        Err(_) => return,
    };
    if watcher.watch(root, RecursiveMode::Recursive).is_err() {
        return;
    }

    // Hold the watcher alive for the duration of this loop. Dropping it stops
    // FS notifications.
    let _watcher_keepalive = watcher;

    loop {
        // Block for the first relevant event of a burst.
        if event_rx.recv().is_err() {
            break;
        }
        // Coalesce the rest of the burst.
        while event_rx.recv_timeout(REBUILD_DEBOUNCE).is_ok() {}

        install_corpus(state, walk(root));
    }
}

fn should_ignore_notify_path(path: &Path) -> bool {
    path.components().any(|component| {
        let name = component.as_os_str().to_string_lossy();
        matches!(
            name.as_ref(),
            ".git"
                | "target"
                | "node_modules"
                | ".next"
                | ".turbo"
                | "dist"
                | "build"
                | "coverage"
                | ".cache"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, body: &str) {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    #[test]
    fn empty_query_returns_top_n_by_path_asc() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "b.txt", "");
        write(dir.path(), "a.txt", "");
        write(dir.path(), "z/nested.txt", "");

        let index = FileIndex::for_root_blocking(dir.path().to_path_buf());
        let MatchResult::Ready(matches) = index.matches("", 10) else {
            panic!("expected Ready");
        };
        let paths: Vec<&str> = matches.iter().map(|m| m.path.as_str()).collect();
        // Sorted ascending; directories carry trailing slash.
        assert_eq!(paths, vec!["a.txt", "b.txt", "z/", "z/nested.txt"]);
    }

    #[test]
    fn fuzzy_match_finds_nested_file() {
        let dir = TempDir::new().unwrap();
        write(
            dir.path(),
            "crates/lash-cli/src/interactive/input_handling.rs",
            "",
        );
        write(dir.path(), "crates/lash-cli/src/editor.rs", "");
        write(dir.path(), "other/unrelated.rs", "");

        let index = FileIndex::for_root_blocking(dir.path().to_path_buf());
        let MatchResult::Ready(matches) = index.matches("input_handling", 5) else {
            panic!("expected Ready");
        };
        assert!(
            matches
                .iter()
                .any(|m| m.path.as_str() == "crates/lash-cli/src/interactive/input_handling.rs"),
            "expected match, got {matches:?}"
        );
    }

    #[test]
    fn substring_beats_scattered() {
        let dir = TempDir::new().unwrap();
        // Strong substring match.
        write(dir.path(), "interactive/input_handling.rs", "");
        // Letters scattered across path.
        write(dir.path(), "i/n/p/u/t.rs", "");

        let index = FileIndex::for_root_blocking(dir.path().to_path_buf());
        let MatchResult::Ready(matches) = index.matches("input", 5) else {
            panic!("expected Ready");
        };
        assert!(!matches.is_empty(), "no matches returned");
        let top = matches[0].path.as_str();
        assert!(
            top.contains("input_handling"),
            "expected input_handling.rs ranked first, got {top}"
        );
    }

    #[test]
    fn gitignore_excludes_paths_in_a_repo() {
        let dir = TempDir::new().unwrap();
        // `require_git(true)` means gitignore only applies inside a git tree.
        // Initialize a minimal git repo so the rules take effect.
        fs::create_dir_all(dir.path().join(".git")).unwrap();
        write(dir.path(), ".git/HEAD", "ref: refs/heads/main");
        write(dir.path(), ".gitignore", "ignored/\n");
        write(dir.path(), "kept.rs", "");
        write(dir.path(), "ignored/secret.rs", "");

        let index = FileIndex::for_root_blocking(dir.path().to_path_buf());
        let MatchResult::Ready(matches) = index.matches("", 50) else {
            panic!("expected Ready");
        };
        let paths: Vec<&str> = matches.iter().map(|m| m.path.as_str()).collect();
        assert!(paths.contains(&"kept.rs"));
        assert!(
            !paths.iter().any(|p| p.starts_with("ignored")),
            "gitignored paths leaked: {paths:?}"
        );
    }

    #[test]
    fn indexing_then_ready_transition() {
        // Exercise the non-blocking constructor: pre-callback returns
        // Indexing; once on_ready fires, subsequent calls return Ready.
        use std::sync::Mutex as StdMutex;
        let dir = TempDir::new().unwrap();
        write(dir.path(), "alpha.rs", "");

        let ready_signal = Arc::new(StdMutex::new(false));
        let signal_cb = Arc::clone(&ready_signal);
        let index = FileIndex::for_root(
            dir.path().to_path_buf(),
            Box::new(move || {
                if let Ok(mut g) = signal_cb.lock() {
                    *g = true;
                }
            }),
        );

        // Race-tolerant assertion: spin briefly until callback fires. Walker
        // is fast on a one-file fixture but not synchronous.
        let start = std::time::Instant::now();
        while !*ready_signal.lock().unwrap() {
            if start.elapsed() > std::time::Duration::from_secs(5) {
                panic!("walker did not finish within 5s");
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let MatchResult::Ready(matches) = index.matches("alpha", 5) else {
            panic!("expected Ready after callback fired");
        };
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].path.as_str(), "alpha.rs");
    }

    #[test]
    fn watcher_picks_up_new_files() {
        // After the initial walk, the FileIndex installs a recursive notify
        // watcher and morphs into a debounced rebuild loop. Files created
        // mid-session must show up in subsequent matches() calls without any
        // explicit refresh action from the caller.
        let dir = TempDir::new().unwrap();
        write(dir.path(), "alpha.rs", "");

        let index = FileIndex::for_root_blocking(dir.path().to_path_buf());
        let MatchResult::Ready(initial) = index.matches("", 10) else {
            panic!("expected Ready immediately after for_root_blocking");
        };
        assert!(initial.iter().any(|m| m.path == "alpha.rs"));

        // Drop a new file. Watcher fires → rebuilder coalesces (300ms) →
        // walks → atomically swaps the corpus.
        write(dir.path(), "beta.rs", "");

        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > std::time::Duration::from_secs(5) {
                panic!("watcher did not pick up new file within 5s");
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            let MatchResult::Ready(matches) = index.matches("beta", 10) else {
                continue;
            };
            if matches.iter().any(|m| m.path == "beta.rs") {
                return;
            }
        }
    }

    #[test]
    fn directories_carry_trailing_slash() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("nested")).unwrap();
        write(dir.path(), "nested/file.rs", "");

        let index = FileIndex::for_root_blocking(dir.path().to_path_buf());
        let MatchResult::Ready(matches) = index.matches("nested", 5) else {
            panic!("expected Ready");
        };
        let dir_match = matches
            .iter()
            .find(|m| m.is_dir)
            .expect("expected the nested/ directory in matches");
        assert!(
            dir_match.path.as_str().ends_with('/'),
            "directory should have trailing slash, got {:?}",
            dir_match.path
        );
    }

    #[test]
    fn notify_filter_ignores_generated_paths() {
        assert!(should_ignore_notify_path(Path::new("target/debug/app")));
        assert!(should_ignore_notify_path(Path::new(
            "node_modules/pkg/index.js"
        )));
        assert!(!should_ignore_notify_path(Path::new("src/main.rs")));
    }
}
