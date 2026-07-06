#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BAKE_FILE = REPO_ROOT / "docker-bake.hcl"
BUILD_DIR = REPO_ROOT / "build"
TEMPLATE_DIR = REPO_ROOT / "template"

BLOCK_START_RE = re.compile(r'^(?P<kind>target|group)\s+"(?P<name>[^"]+)"\s+\{')
DOCKERFILE_RE = re.compile(r'dockerfile\s*=\s*"(?P<path>[^"]+)"')
IMAGE_RE = re.compile(r"^\s*image:\s*(?P<image>[^#\s]+)")
INTERPOLATION_RE = re.compile(r"\$\{[^}]*\}")


def _brace_delta(line: str) -> int:
    stripped = INTERPOLATION_RE.sub("", line)
    return stripped.count("{") - stripped.count("}")


def iter_hcl_blocks(text: str):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        match = BLOCK_START_RE.match(line)
        if not match:
            index += 1
            continue

        block_lines = [lines[index]]
        depth = _brace_delta(lines[index])
        index += 1
        while index < len(lines) and depth > 0:
            block_lines.append(lines[index])
            depth += _brace_delta(lines[index])
            index += 1

        yield match.group("kind"), match.group("name"), "\n".join(block_lines)


def parse_bake_file(path: Path = BAKE_FILE) -> dict:
    text = path.read_text(encoding="utf-8")
    targets = {}
    groups = set()

    for kind, name, block in iter_hcl_blocks(text):
        if kind == "group":
            groups.add(name)
            continue
        dockerfile_match = DOCKERFILE_RE.search(block)
        targets[name] = {
            "dockerfile": dockerfile_match.group("path") if dockerfile_match else None,
        }

    return {"targets": targets, "groups": groups}


def normalize_image_reference(image: str) -> str:
    image = image.strip().strip("'\"")
    image = image.split("@", 1)[0]
    image = image.rsplit("/", 1)[-1]
    return image.split(":", 1)[0]


def collect_template_images(template_dir: Path = TEMPLATE_DIR, repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for path in sorted(template_dir.rglob("*.yaml")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = IMAGE_RE.match(line)
            if not match:
                continue
            image_name = normalize_image_reference(match.group("image"))
            refs.setdefault(image_name, []).append(f"{path.relative_to(repo_root)}:{line_no}")
    return refs


def collect_build_dockerfiles(build_dir: Path = BUILD_DIR, repo_root: Path = REPO_ROOT) -> set[str]:
    return {str(path.relative_to(repo_root)) for path in sorted(build_dir.glob("*.Dockerfile"))}


def validate_build_matrix(repo_root: Path = REPO_ROOT) -> tuple[list[str], dict]:
    bake_info = parse_bake_file(repo_root / "docker-bake.hcl")
    targets = bake_info["targets"]
    public_targets = {name for name in targets if not name.startswith("_")}
    referenced_dockerfiles = {
        target["dockerfile"]
        for target in targets.values()
        if target["dockerfile"] is not None
    }
    build_dockerfiles = collect_build_dockerfiles(repo_root / "build", repo_root=repo_root)
    template_images = collect_template_images(repo_root / "template", repo_root=repo_root)

    errors = []

    missing_dockerfiles = sorted(path for path in referenced_dockerfiles if not (repo_root / path).is_file())
    for path in missing_dockerfiles:
        errors.append(f"Bake target references missing Dockerfile: {path}")

    unlisted_dockerfiles = sorted(build_dockerfiles - referenced_dockerfiles)
    for path in unlisted_dockerfiles:
        errors.append(f"build Dockerfile is not referenced by docker-bake.hcl: {path}")

    missing_template_targets = sorted(image for image in template_images if image not in public_targets)
    for image in missing_template_targets:
        locations = ", ".join(template_images[image])
        errors.append(f"template image '{image}' has no matching Bake target ({locations})")

    for path in sorted(build_dockerfiles):
        dockerfile = repo_root / path
        text = dockerfile.read_text(encoding="utf-8")
        if "${REG}/dayuhub/" in text:
            errors.append(f"Dockerfile still hard-codes an internal Dayu repository instead of BASE_REPO: {path}")

    summary = {
        "bake_targets": sorted(public_targets),
        "bake_groups": sorted(bake_info["groups"]),
        "dockerfiles": sorted(build_dockerfiles),
        "template_images": sorted(template_images),
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Dayu Docker Bake build matrix.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args(argv)

    errors, summary = validate_build_matrix()
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors, "summary": summary}, indent=2, sort_keys=True))
    elif errors:
        print("Build matrix validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    else:
        print(
            "Build matrix validation passed: "
            f"{len(summary['bake_targets'])} targets, "
            f"{len(summary['dockerfiles'])} Dockerfiles, "
            f"{len(summary['template_images'])} template images."
        )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
