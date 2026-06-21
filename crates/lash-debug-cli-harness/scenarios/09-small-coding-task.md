# Scenario: Small Coding Task

Purpose: exercise a simple edit/test loop without an automatic harness judge.

Setup:

- Create a fresh temporary working directory.
- Create `slugify.py` with:

```python
def slugify(value: str) -> str:
    return value.lower().replace(" ", "-")
```

- Create `test_slugify.py` with:

```python
from slugify import slugify

def test_slugify_strips_punctuation():
    assert slugify("Hello, World!") == "hello-world"
```

- Start the harness in RLM mode with `--working-dir` set to that directory.

Steps:

```text
send Fix slugify.py so the test passes, then tell me what changed.
idle
screen
screenshot small-coding-task
```

Inspect:

- The model should edit `slugify.py`.
- The model should run or explain the test result.
- If the final screen shows parser errors, inspect the trace for malformed
  `<lashlang>` output.
