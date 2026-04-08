set positional-arguments

repo := justfile_directory()

default:
  @just --list

dev *args:
  cargo build --manifest-path "{{repo}}/lash-cli/Cargo.toml"
  cd "${LASH_DEV_LAUNCH_CWD:-{{invocation_directory()}}}" && exec "{{repo}}/target/debug/lash" "$@"
