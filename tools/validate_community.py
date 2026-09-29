#!/usr/bin/env python3
"""Check the community roster, ownership routing, and local documentation links.

Uses PyYAML and never reads GitHub credentials or changes roles.
Live team membership, permissions, votes, and reviewer competence need human
verification; a passing result does not attest to those facts.
"""

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

if __package__:
    from .owners import check_codeowners, read_owners
else:
    from owners import check_codeowners, read_owners


COMMUNITY_DOCS = (
    "GOVERNANCE.md", "MAINTAINERS.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
    "SECURITY.md", "SUPPORT.md", "README.md", "README_zh.md", "docs/README.md",
    "docs/development/README.md", "docs/development/community-administration.md",
    "docs/development/owners.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
)


def section(text, title):
    match = re.search(r"^## " + re.escape(title) + r"\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError("MAINTAINERS.md: missing section " + title)
    return re.split(r"^## ", text[match.end():], maxsplit=1, flags=re.MULTILINE)[0]


def table_rows(text, title, columns):
    """Read the first table in a named roster section; reject malformed rows."""
    lines = [line.strip() for line in section(text, title).splitlines()]
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
    if start is None:
        raise ValueError("MAINTAINERS.md: missing table in " + title)
    table = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table.append([cell.strip() for cell in line.strip("|").split("|")])
    if len(table) < 3 or table[0] != columns:
        raise ValueError("MAINTAINERS.md: unexpected columns or empty table in " + title)
    if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in table[1]):
        raise ValueError("MAINTAINERS.md: invalid table separator in " + title)
    if any(len(row) != len(columns) for row in table[1:]):
        raise ValueError("MAINTAINERS.md: incorrect row width in " + title)
    return [dict(zip(columns, row)) for row in table[2:]]


def validate_roster(text):
    errors = []
    specs = (
        ("Technical Steering Committee", ["TSC member", "Office", "GitHub ID", "Affiliation", "Email"]),
        ("Maintainers", ["Maintainer", "GitHub ID", "Affiliation", "Email"]),
        ("Contributors", ["Contributor", "GitHub ID", "Affiliation", "Email"]),
    )
    rosters = {}
    for title, columns in specs:
        try:
            rows = table_rows(text, title, columns)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        rosters[title] = rows
        names, logins = [], []
        for row in rows:
            name = row[columns[0]]
            if not name:
                errors.append("MAINTAINERS.md: empty name in " + title)
            names.append(name.casefold())
            identity = row["GitHub ID"]
            match = re.fullmatch(r"@\[([A-Za-z0-9-]+)\]\(https://github.com/([A-Za-z0-9-]+)\)", identity)
            if identity and (not match or match[1].casefold() != match[2].casefold()):
                errors.append("MAINTAINERS.md: inconsistent GitHub identity for " + name)
            if title == "Maintainers" and not identity:
                errors.append("MAINTAINERS.md: Maintainer needs a GitHub identity: " + name)
            if match:
                logins.append(match[1].casefold())
        if any(count > 1 for count in Counter(names).values()):
            errors.append("MAINTAINERS.md: duplicate person within " + title)
        if any(count > 1 for count in Counter(logins).values()):
            errors.append("MAINTAINERS.md: duplicate GitHub identity within " + title)
    # Overlap between sections is intentional: an officer may also maintain code.
    officers = [row["Office"] for row in rosters.get("Technical Steering Committee", [])]
    if any(office not in ("", "Chair", "Vice Chair") for office in officers):
        errors.append("MAINTAINERS.md: office must be Chair, Vice Chair, or empty")
    for office in ("Chair", "Vice Chair"):
        if officers.count(office) > 1:
            errors.append("MAINTAINERS.md: more than one " + office)
    # Vacancies are legitimate; report coverage separately without inventing members.
    if re.search(r"^#{1,6}\s+Committers?\s*$", text, re.MULTILINE | re.IGNORECASE):
        errors.append("MAINTAINERS.md: separate Committer roster is no longer used")
    return errors, rosters


def markdown_prose(text):
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    return re.sub(r"^```[^\n]*\n[\s\S]*?^```[^\n]*$", "", text, flags=re.MULTILINE)


def heading_ids(text):
    anchors, counts = set(), Counter()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", markdown_prose(text), re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        anchors.add(slug if not counts[slug] else slug + "-" + str(counts[slug]))
        counts[slug] += 1
    anchors.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    return anchors


def validate_links(root, relative_path):
    errors = []
    path = root / relative_path
    if not path.is_file():
        return [relative_path + ": missing community document"]
    for link in re.findall(r"\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", markdown_prose(path.read_text())):
        url = urlsplit(link.strip("<>"))
        if url.scheme or url.netloc:
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path.resolve()
        try:
            target.relative_to(root)
        except ValueError:
            errors.append(relative_path + ": link escapes repository: " + link)
            continue
        if not target.exists():
            errors.append(relative_path + ": missing link target: " + link)
        elif url.fragment and target.is_file() and target.suffix == ".md":
            if unquote(url.fragment) not in heading_ids(target.read_text()):
                errors.append(relative_path + ": missing heading: " + link)
    return errors


def validate_ownership(root, maintainers):
    errors = []
    try:
        owners = read_owners(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    logins = set()
    for row in maintainers:
        match = re.fullmatch(r"@\[([A-Za-z0-9-]+)\]\(https://github.com/([A-Za-z0-9-]+)\)", row["GitHub ID"])
        if match:
            logins.add(match[1].casefold())
    for role in ("reviewers", "approvers"):
        if {login.casefold() for login in owners[role]} != logins:
            errors.append("OWNERS: " + role + " must match all current Maintainers in MAINTAINERS.md")
    errors.extend(check_codeowners(root, owners))
    return errors


def validate_ownership_paths(paths):
    errors = []
    for path in paths:
        name = Path(path).name
        if (name == "OWNERS" and path != "OWNERS") or name == "OWNERS_ALIASES":
            errors.append(path + ": nested OWNERS and OWNERS_ALIASES are not supported by the simplified mechanism")
        if name == "CODEOWNERS" and path != ".github/CODEOWNERS":
            errors.append(path + ": keep only the generated .github/CODEOWNERS")
    return errors


def validate(root):
    errors = []
    documents = list(COMMUNITY_DOCS)
    documents += [str(path.relative_to(root)) for path in sorted((root / ".github/ISSUE_TEMPLATE").glob("*.md"))]
    for document in documents:
        errors.extend(validate_links(root, document))
    roster = root / "MAINTAINERS.md"
    if roster.is_file():
        roster_errors, rosters = validate_roster(roster.read_text())
        errors.extend(roster_errors)
        maintainers = rosters.get("Maintainers", [])
        errors.extend(validate_ownership(root, maintainers))
        if len(maintainers) < 2:
            print("WARNING: fewer than two Maintainers; record review coverage and check live branch rules.")
        offices = {row["Office"] for row in rosters.get("Technical Steering Committee", [])}
        if not {"Chair", "Vice Chair"} <= offices:
            print("WARNING: an officer position is vacant; record coordination and handover coverage.")
    try:
        paths = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root, text=True,
        ).split("\0")
        errors.extend(validate_ownership_paths(path for path in set(paths) if path and (root / path).exists()))
    except (OSError, subprocess.CalledProcessError):
        errors.append("Cannot check ownership file layout; run in a Git checkout")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    for error in errors:
        print("ERROR: " + error, file=sys.stderr)
    if errors:
        return 1
    print("Community checks passed: roster, OWNERS, generated CODEOWNERS, and local document links.")
    print("Live GitHub permissions and human governance decisions still require separate verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
