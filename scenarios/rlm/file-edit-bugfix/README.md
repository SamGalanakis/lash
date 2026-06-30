# RLM Smoke: File Edit Bugfix

The agent should inspect a tiny shell project, observe `sh test.sh` failing,
edit `calc.sh`, rerun the test, and finish only after the oracle passes.

The oracle rejects edits to `test.sh`.
