#!/usr/bin/env python3
"""Run bounded Atheris jobs or reproduce one saved input in a fresh process."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FUZZ = ROOT / "tests" / "fuzz"
TARGETS = ("dag", "schedule_plan", "runtime_routes")


def positive_integer(value):
    number = int(value)
    if not 1 <= number <= 3600:
        raise argparse.ArgumentTypeError("must be between 1 and 3600")
    return number


def run_process(command, log_path, timeout, env):
    with log_path.open("w") as log:
        try:
            return subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            log.write("\nFuzz runner wall-clock timeout exceeded.\n")
            return 124


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all",) + TARGETS, default="all")
    parser.add_argument("--seconds", type=positive_integer, default=60, help="time budget per target (1-3600)")
    parser.add_argument("--seed", type=positive_integer, default=1, help="deterministic mutation seed (1-3600)")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "fuzz")
    parser.add_argument("--replay", type=Path, help="reproduce one saved input; requires a single --target")
    args = parser.parse_args(argv)
    if args.replay and (args.target == "all" or not args.replay.is_file()):
        parser.error("--replay requires an existing file and a single --target")
    output = args.output.resolve()
    if output == FUZZ or FUZZ in output.parents:
        parser.error("--output must be outside tests/fuzz to preserve checked-in seeds")
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("DAYU_") or key.endswith("_ENDPOINT"):
            env.pop(key)
    env.pop("PYTHONOPTIMIZE", None)  # Assertions are part of each fuzz oracle.
    env.update(
        PYTHONPATH=str(ROOT / "dependency"),
        PYTHONHASHSEED="0",
        LOG_LEVEL="ERROR",
        PROCESSOR_SERVICE_NAME="",
        NAMESPACE="dayu",
    )
    failed = False
    for target in TARGETS if args.target == "all" else (args.target,):
        directory = output / target
        corpus = directory / "corpus"
        crashes = directory / "crashes"
        corpus.mkdir(parents=True, exist_ok=True)
        crashes.mkdir(parents=True, exist_ok=True)
        for seed in sorted((FUZZ / "corpus" / target).glob("*.json")):
            shutil.copyfile(seed, corpus / seed.name)
        command = [
            sys.executable,
            str(FUZZ / ("fuzz_" + target + ".py")),
            "-seed={}".format(args.seed),
            "-timeout=5",
            "-rss_limit_mb=1024",
            "-max_len=8192",
            "-artifact_prefix=" + str(crashes) + os.sep,
        ]
        if args.replay:
            command.extend(["-runs=1", str(args.replay.resolve())])
        else:
            command.extend(["-max_total_time={}".format(args.seconds), str(corpus)])
        print(
            "Running {} ({}); log: {}".format(
                target, "replay" if args.replay else str(args.seconds) + "s", directory / "fuzz.log"
            ),
            flush=True,
        )
        code = run_process(command, directory / "fuzz.log", args.seconds + 30, env)
        (directory / "run.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "command": command,
                    "returncode": code,
                    "python": sys.version,
                },
                indent=2,
            )
            + "\n"
        )
        print("\n".join((directory / "fuzz.log").read_text(errors="replace").splitlines()[-20:]), flush=True)
        failed = failed or code != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
