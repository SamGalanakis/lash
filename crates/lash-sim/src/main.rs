use std::path::PathBuf;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(usage());
    };
    match command.as_str() {
        "fixed-scripts" => {
            let mut out = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => {
                        out = args.next().map(PathBuf::from);
                    }
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let Some(out) = out else {
                return Err(format!("missing --out\n\n{}", usage()));
            };
            let manifest = lash_sim::run_fixed_script_profile(out.as_path())
                .await
                .map_err(|err| err.to_string())?;
            println!("{}", manifest.manifest_path.display());
            Ok(())
        }
        "run" => {
            let mut out = None;
            let mut profile = "fast-random".to_string();
            let mut seeds = None;
            let mut max_boundaries = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    "--profile" => {
                        profile = args
                            .next()
                            .ok_or_else(|| format!("missing --profile value\n\n{}", usage()))?;
                    }
                    "--seeds" => {
                        let raw = args
                            .next()
                            .ok_or_else(|| format!("missing --seeds value\n\n{}", usage()))?;
                        seeds = Some(parse_usize("--seeds", &raw)?);
                    }
                    "--max-boundaries" => {
                        let raw = args.next().ok_or_else(|| {
                            format!("missing --max-boundaries value\n\n{}", usage())
                        })?;
                        max_boundaries = Some(parse_usize("--max-boundaries", &raw)?);
                    }
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let Some(out) = out else {
                return Err(format!("missing --out\n\n{}", usage()));
            };
            let seeds = match seeds {
                Some(seeds) => seeds,
                None => lash_sim::generator::default_seed_count(&profile)
                    .map_err(|err| err.to_string())?,
            };
            let max_boundaries = match max_boundaries {
                Some(max_boundaries) => max_boundaries,
                None => lash_sim::generator::default_max_boundaries(&profile)
                    .map_err(|err| err.to_string())?,
            };
            let report =
                lash_sim::run_generated_sim_profile(out.as_path(), &profile, seeds, max_boundaries)
                    .await
                    .map_err(|err| err.to_string())?;
            println!("{}", report.summary_path.display());
            Ok(())
        }
        "replay" => {
            let trace = args
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("missing trace path\n\n{}", usage()))?;
            let mut out = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let report_path = out.as_ref().map(|out| out.join("replay.json"));
            let report = lash_sim::replay::replay_trace_file(&trace, report_path.as_deref())
                .map_err(|err| err.to_string())?;
            if let Some(report_path) = report_path {
                println!("{}", report_path.display());
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).map_err(|err| err.to_string())?
                );
            }
            Ok(())
        }
        "replay-sqlite" => {
            let trace = args
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("missing trace path\n\n{}", usage()))?;
            let mut out = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let Some(out) = out else {
                return Err(format!("missing --out\n\n{}", usage()));
            };
            std::fs::create_dir_all(&out).map_err(|err| err.to_string())?;
            let db_path = out.join("sqlite-store");
            let report_path = out.join("sqlite-replay.json");
            let _report = lash_sim::sqlite_replay::replay_trace_file_to_sqlite(
                &trace,
                &db_path,
                Some(&report_path),
            )
            .await
            .map_err(|err| err.to_string())?;
            println!("{}", report_path.display());
            Ok(())
        }
        "replay-postgres" => {
            let trace = args
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("missing trace path\n\n{}", usage()))?;
            let mut out = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let Some(out) = out else {
                return Err(format!("missing --out\n\n{}", usage()));
            };
            let database_url = std::env::var("LASH_POSTGRES_DATABASE_URL")
                .map_err(|_| "missing LASH_POSTGRES_DATABASE_URL".to_string())?;
            std::fs::create_dir_all(&out).map_err(|err| err.to_string())?;
            let report_path = out.join("postgres-replay.json");
            let _report = lash_sim::postgres_replay::replay_trace_file_to_postgres(
                &trace,
                &database_url,
                Some(&report_path),
            )
            .await
            .map_err(|err| err.to_string())?;
            println!("{}", report_path.display());
            Ok(())
        }
        "backend-contention" => {
            let mut out = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let Some(out) = out else {
                return Err(format!("missing --out\n\n{}", usage()));
            };
            let report = lash_sim::backend_contention::run_backend_contention_report(out.as_path())
                .await
                .map_err(|err| err.to_string())?;
            println!("{}", report.report_path.display());
            Ok(())
        }
        "minimize" => {
            let trace = args
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("missing trace path\n\n{}", usage()))?;
            let mut out = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    "-h" | "--help" => return Err(usage()),
                    other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
                }
            }
            let Some(out) = out else {
                return Err(format!("missing --out\n\n{}", usage()));
            };
            let report = lash_sim::minimize::minimize_trace_or_fixture_file(&trace, out.as_path())
                .await
                .map_err(|err| err.to_string())?;
            let report_path = out.join("minimize.json");
            std::fs::write(
                &report_path,
                serde_json::to_vec_pretty(&report).map_err(|err| err.to_string())?,
            )
            .map_err(|err| err.to_string())?;
            println!("{}", report_path.display());
            Ok(())
        }
        "-h" | "--help" => Err(usage()),
        other => Err(format!("unknown command `{other}`\n\n{}", usage())),
    }
}

fn parse_usize(name: &str, raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|err| format!("invalid {name} value `{raw}`: {err}"))
}

fn usage() -> String {
    "Usage:
  lash-sim fixed-scripts --out <artifact-root>
  lash-sim run --out <artifact-root> [--profile fast-random] [--seeds N] [--max-boundaries N]
  lash-sim replay <trace> [--out <artifact-root>]
  lash-sim replay-sqlite <trace> --out <artifact-root>
  lash-sim replay-postgres <trace> --out <artifact-root>
  lash-sim backend-contention --out <artifact-root>
  lash-sim minimize <trace> --out <artifact-root>"
        .to_string()
}
