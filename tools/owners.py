#!/usr/bin/env python3
"""Parse Dayu's root OWNERS and generate its GitHub CODEOWNERS projection.

This intentionally supports one repository-wide reviewers/approvers mapping, not
Prow's directory inheritance, aliases, filters, labels, or approval commands.
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Install community tooling first: python3 -m pip install -r .github/requirements/community.txt") from exc


LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
MAX_OWNERS_BYTES = 16384
CODEOWNERS_HEADER = """# Generated from /OWNERS by make sync-codeowners. Do not edit manually.
# GitHub reads this file from the PR base branch; approvers need write access.
# Native Reviews and branch rules enforce approval; this file grants no access.
# Significant changes still need two independent technical reviews (GOVERNANCE.md).
"""


class OwnersLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently accepting the last authority list."""


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError("OWNERS: keys must be unique strings")
        result[key] = loader.construct_object(value_node)
    return result


OwnersLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def parse_owners(text):
    """Load safe YAML and fail closed on unsupported or ambiguous configuration."""
    if len(text.encode("utf-8")) > MAX_OWNERS_BYTES:
        raise ValueError("OWNERS: file exceeds 16 KiB")
    try:
        owners = yaml.load(text, Loader=OwnersLoader)
    except yaml.YAMLError as exc:
        raise ValueError("OWNERS: invalid or unsupported YAML") from exc
    if not isinstance(owners, dict) or set(owners) != {"reviewers", "approvers"}:
        raise ValueError("OWNERS: only reviewers and approvers are supported")
    for role, logins in owners.items():
        if not isinstance(logins, list) or not logins:
            raise ValueError("OWNERS: " + role + " must be a nonempty list")
        if any(not isinstance(login, str) or not LOGIN.fullmatch(login) or "--" in login for login in logins):
            raise ValueError("OWNERS: " + role + " must contain GitHub usernames without @, teams, or bots")
        if len({login.casefold() for login in logins}) != len(logins):
            raise ValueError("OWNERS: duplicate username in " + role)
    if not {login.casefold() for login in owners["approvers"]} <= {
        login.casefold() for login in owners["reviewers"]
    }:
        raise ValueError("OWNERS: every approver must also be a reviewer")
    return owners


def read_owners(root):
    path = root / "OWNERS"
    if path.is_symlink() or not path.is_file():
        raise ValueError("OWNERS: expected a regular root file, not a symlink")
    return parse_owners(path.read_text(encoding="utf-8"))


def render_codeowners(owners):
    # A single catch-all also protects OWNERS, CODEOWNERS, governance, and workflows.
    return CODEOWNERS_HEADER + "* " + " ".join("@" + login for login in owners["approvers"]) + "\n"


def check_codeowners(root, owners):
    path = root / ".github/CODEOWNERS"
    if path.is_symlink() or not path.is_file():
        return [".github/CODEOWNERS: expected a regular generated file"]
    if path.read_text(encoding="utf-8") != render_codeowners(owners):
        return [".github/CODEOWNERS: differs from OWNERS; run make sync-codeowners and commit the result"]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-codeowners", action="store_true", help="update the local generated file only")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        owners = read_owners(root)
        if args.write_codeowners:
            path = root / ".github/CODEOWNERS"
            if path.is_symlink():
                raise ValueError(".github/CODEOWNERS: refusing to write through a symlink")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_codeowners(owners), encoding="utf-8")
            print("Updated .github/CODEOWNERS locally. Review and commit it with OWNERS.")
        else:
            errors = check_codeowners(root, owners)
            if errors:
                raise ValueError("\n".join(errors))
            print("OWNERS and generated CODEOWNERS agree.")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
