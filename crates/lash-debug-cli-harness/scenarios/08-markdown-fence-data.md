# Scenario: Markdown Fence As Data

Purpose: verify Markdown fences inside file contents or raw strings do not
confuse the paired `<lashlang>` transport.

Setup:

- Create a fresh temporary working directory.
- Create `sample.md` with:

````text
Here is code:

```python
print("hello")
```
````

- Start the harness in RLM mode with `--working-dir` set to that directory.

Steps:

```text
send Read sample.md and tell me what language the fenced code block uses.
idle
screen
screenshot markdown-fence-data
```

Inspect:

- The answer should identify `python`.
- Markdown fences should be treated as file data, not as Lash execution transport.
