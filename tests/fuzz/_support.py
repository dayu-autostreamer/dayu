"""Bounded, deterministic input recipes shared by the contract fuzzers."""

import json
import random
import sys

MAX_INPUT_SIZE = 8192


def decode(data):
    # These are harness recipes, not a replacement for a production JSON parser.
    if len(data) > MAX_INPUT_SIZE:
        return {}
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def integer(value, default, low, high):
    return max(low, min(high, value)) if type(value) is int else default


def choice(value, options):
    return value if isinstance(value, str) and value in options else options[0]


def expect_error(error_type, function, *args, **kwargs):
    """Only the documented rejection is accepted; unexpected errors escape."""
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError("{} accepted an invalid input".format(function.__name__))


def make_mutator(fields):
    """Keep the JSON envelope valid while mutating one to three recipe fields."""

    def mutate(data, max_size, seed):
        rng = random.Random(seed)
        case = {key: value for key, value in decode(data).items() if key in fields}
        for _ in range(rng.randint(1, 3)):
            key = rng.choice(tuple(fields))
            case[key] = fields[key](rng)
        encoded = json.dumps(case, sort_keys=True, ensure_ascii=True).encode()
        if len(encoded) > min(max_size, MAX_INPUT_SIZE):
            return b"{}" if max_size >= 2 else b""  # Never truncate JSON or UTF-8.
        return encoded

    return mutate


def run(check_input, mutate):
    import atheris

    # Import real Dayu code under instrumentation, before the first test call.
    # Optional model/plugin dependencies retain the normal core import behavior.
    with atheris.instrument_imports(include=["core"]):
        check_input(b"{}")
    atheris.Setup(sys.argv, atheris.instrument_func(check_input), custom_mutator=mutate)
    atheris.Fuzz()
