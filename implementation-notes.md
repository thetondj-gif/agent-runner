## Universal Execution Contract — 2026-09-01

### Deviations

- The OmniRoute source checkout is not built (`app/server.js` is absent), so the proof attached to its existing development server path with an isolated writable `DATA_DIR`; no production or user route state was changed.
- The disposable OmniRoute instance had no management password. Its local, credential-free Ollama route was applied directly to the instance's isolated SQLite store rather than mutating management authentication.
- Creative Studio's existing execution adapter identifies ComfyUI as its local provider but does not yet submit its stored production workflow. The second canary therefore used ComfyUI's built-in `EmptyImage -> SaveImage` graph through the real queue, producing and independently re-observing a hashed PNG without loading a checkpoint.

### Discovered edge cases

- `pytest` from the system environment lacks the configured `pytest-cov` plugin; the repository's canonical `uv run pytest` environment passes the contract suite.
- Successful ComfyUI reruns of the deterministic canary reuse the same numbered output filename and content hash. The independent observation remains distinct through its prompt ID and timestamp.
- A deliberately mismatched Creative Studio expectation exercised the real rollback command, removed the generated canary artifact, and emitted a hashed `ROLLED_BACK` receipt.

### Questions for review

- None. Promotion remains receipt-gated, the verifier only observes a second capability run, and DAWN was not added to the reversible R0/R1 execution path.

- Summary — deviations recorded: 3.
- Most likely revisit: replace disposable OmniRoute SQLite configuration with its canonical authenticated management API once a test-only connection handle is provisioned.
- Discovered edge cases recorded: 3.
- Questions for review: 0.
- Next session should read the three receipts under `mission-evidence/universal-execution-contract/` first.
