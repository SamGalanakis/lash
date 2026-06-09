//! Pre-pipeline flows that exit before any provider/session work:
//! `--reset`, `--check-update`, and `--update`.

use crate::Args;

/// Handle flags that complete without starting a session. Returns `true`
/// when the process should exit successfully without continuing startup.
pub(super) async fn handle_early_exit_flags(args: &Args) -> anyhow::Result<bool> {
    if args.reset {
        run_reset()?;
        return Ok(true);
    }

    if args.check_update {
        println!("{}", crate::update::check_update_text().await?);
        return Ok(true);
    }

    if args.update {
        crate::update::install_latest_release().await?;
        return Ok(true);
    }

    Ok(false)
}

/// `--reset`: confirm, then delete all lash data (config, credentials,
/// sessions, caches).
fn run_reset() -> anyhow::Result<()> {
    use std::io::Write;

    // Design system ANSI colors
    const SODIUM: &str = "\x1b[38;2;232;163;60m"; // #e8a33c
    const CHALK: &str = "\x1b[38;2;232;228;208m"; // #e8e4d0
    const ASH_TEXT: &str = "\x1b[38;2;90;90;80m"; // #5a5a50
    const LICHEN: &str = "\x1b[38;2;138;158;108m"; // #8a9e6c
    const ERR: &str = "\x1b[38;2;204;68;68m"; // #c44
    const BOLD: &str = "\x1b[1m";
    const RESET: &str = "\x1b[0m";

    let lash_dir = crate::paths::lash_home();
    let cache_dir = crate::paths::lash_cache_dir();

    eprintln!();
    eprintln!("  {SODIUM}{BOLD}/ reset{RESET}");
    eprintln!();
    eprintln!("  {ERR}This will permanently delete all lash data:{RESET}");
    eprintln!();
    eprintln!(
        "    {ASH_TEXT}config, credentials   {CHALK}{}{RESET}",
        lash_dir.display()
    );
    eprintln!(
        "    {ASH_TEXT}runtime cache          {CHALK}{}{RESET}",
        cache_dir.display()
    );
    eprintln!();
    eprint!("  {SODIUM}Are you sure? [y/N]{RESET} ");
    std::io::stderr().flush()?;

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    if answer.trim().eq_ignore_ascii_case("y") {
        if lash_dir.exists() {
            std::fs::remove_dir_all(&lash_dir)?;
        }
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)?;
        }
        eprintln!("  {LICHEN}Done.{RESET} All data removed.");
    } else {
        eprintln!("  {ASH_TEXT}Aborted.{RESET}");
    }
    eprintln!();
    Ok(())
}
