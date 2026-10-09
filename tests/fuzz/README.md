# Contract fuzzing

These Atheris targets call the real shared runtime library, without Kubernetes,
Redis, GPU models or network requests. They complement the normal test pyramid.

| Target | Contracts checked |
| --- | --- |
| `dag` | Edge repair and reciprocal references; serialization and validation idempotence; complete, stable topological order; rejection of cycles, duplicates, dangling references, missing boundaries and disconnected nodes |
| `schedule_plan` | Complete service coverage, candidate scoping, normalized placements, cloud replicas and `@cloud`; complete offloading maps, boundary placement and deep-copy isolation |
| `runtime_routes` | Snake/camel/nested/URL endpoint normalization; exact identities and port limits; unambiguous task snapshots; no bootstrap fallback for task-routed workers |

Inputs are small JSON **recipes** for structured cases, not raw HTTP payloads.
The custom mutator preserves that structure, changes related fields and explores
up to 24 services, 12 edge candidates and 96 graph edges. Each callback first
checks a valid case, then tests a selected invalid mutation. Invalid recipe JSON
falls back to the valid baseline, so random bytes do not stall at a JSON parser.
Only explicitly expected exception types are accepted at rejection points;
unexpected exceptions and invariant violations fail the run.

## Install and run

The supported engine environment is CPython 3.8 on Linux x86-64, matching CI.
Atheris 2.3.0 is pinned because it provides a wheel for that Python version.
Create a separate environment; the normal requirements do not install Atheris.

```sh
python3.8 -m venv .venv-fuzz
.venv-fuzz/bin/python -m pip install --only-binary=:all: -r tests/fuzz/requirements.txt
make fuzz-smoke PYTHON=.venv-fuzz/bin/python
make fuzz PYTHON=.venv-fuzz/bin/python FUZZ_TARGET=dag FUZZ_SECONDS=600
```

`tools/run_fuzz.py` accepts `--target all|dag|schedule_plan|runtime_routes`,
`--seconds`, `--seed` and `--output`. Each target runs in a fresh interpreter.
The runner copies committed seeds into `.cache/fuzz/<target>/corpus`, instruments
real `core` imports, and isolates Dayu environment settings. It never writes back
to the committed corpus. The default mutation seed is 1; use other seeds for
additional searches. Duration limits mean successive timed runs can still
execute different numbers of mutations.

Each process has an 8 KiB input limit, a 5-second individual-input timeout, a
1 GiB RSS limit and a wall-clock budget of the requested duration plus 30 seconds.
Failures, including a wall-clock timeout, produce a nonzero runner exit status.
The output directory contains `fuzz.log`, `run.json`, the evolved corpus and
`crashes/` inputs emitted by libFuzzer. A process killed before libFuzzer can emit
an input may leave only the log and metadata.

## Reproduce and keep regressions

Download the failing job's artifact, then use the same dependency versions:

```sh
make fuzz-replay PYTHON=.venv-fuzz/bin/python \
  FUZZ_TARGET=runtime_routes FUZZ_INPUT=/path/to/crashes/crash-<hash>
```

Replay runs the single saved input in the native engine and preserves failure
status. Reduce the recipe while keeping the failure, commit that recipe to the
target's corpus, and add a focused pytest regression for the production behavior.
Do not commit generated corpora or logs. Check crash contents before sharing;
the shipped recipes use synthetic identifiers and no credentials.

Ordinary development environments, including macOS, can replay all committed
recipes and deterministic mutations without the native engine:

```sh
python -m pytest tests/unit/tools/test_fuzzing.py
```

This checks the harness and its assertions; it does not replace an Atheris run.
The tests also inject incorrect production behavior to check that failures are
not swallowed. Native engine coverage and mutation counts appear in CI logs.

## CI and scope

The `Fuzz` workflow runs all three targets independently: 60 seconds each on a
PR, 120 seconds on main pushes/manual runs, and 600 seconds on the daily schedule.
Jobs use read-only repository permissions, SHA-pinned actions, no stored checkout
credentials, no secrets, and no shared corpus cache across trust boundaries.
Artifacts expire after seven days. Ordinary `pull_request` execution runs the
candidate code; no privileged `pull_request_target` execution is involved.

Initially this is an additional check, not a new required branch-protection
check. Consider making it required after PR and main runs have demonstrated
stable behavior. Scheduled fuzzing starts after the workflow reaches main.

The initial scope is the Python shared-library contracts above. It does not
cover media decoders, models, live service requests, cluster lifecycle or every
Task field. These need separate targets and environments. Scorecard recognizes
the Python `import atheris` integration; its Fuzzing score indicates integration,
not completeness of testing or absence of bugs. The hosted score updates only
after merge and the next Scorecard scan.
