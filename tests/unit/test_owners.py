"""Ownership authority and generated GitHub routing; no application dependencies."""

import tempfile
import unittest
from pathlib import Path

from tools.owners import check_codeowners, parse_owners, read_owners, render_codeowners
from tools.validate_community import validate_ownership, validate_ownership_paths


OWNERS = "reviewers: [engineer, Reviewer]\napprovers: [engineer, Reviewer]\n"
MAINTAINERS = [
    {"GitHub ID": "@[engineer](https://github.com/engineer)"},
    {"GitHub ID": "@[reviewer](https://github.com/reviewer)"},
]


class OwnersTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / ".github").mkdir()
        (self.root / "OWNERS").write_text(OWNERS)
        (self.root / ".github/CODEOWNERS").write_text(render_codeowners(parse_owners(OWNERS)))

    def test_all_maintainers_have_the_same_scope(self):
        self.assertEqual(validate_ownership(self.root, MAINTAINERS), [])
        self.assertEqual(render_codeowners(parse_owners(OWNERS)).splitlines()[-1], "* @engineer @Reviewer")

    def test_contributor_or_officer_cannot_be_added_as_an_approver_without_appointment(self):
        for extra in ("contributor", "chair"):
            with self.subTest(extra=extra):
                text = OWNERS.replace("Reviewer]", "Reviewer, " + extra + "]")
                (self.root / "OWNERS").write_text(text)
                (self.root / ".github/CODEOWNERS").write_text(render_codeowners(parse_owners(text)))
                self.assertTrue(any("approvers must match" in error for error in validate_ownership(self.root, MAINTAINERS)))

    def test_a_maintainer_cannot_silently_lose_approval_or_review_scope(self):
        (self.root / "OWNERS").write_text("reviewers: [engineer]\napprovers: [engineer]\n")
        errors = validate_ownership(self.root, MAINTAINERS)
        self.assertTrue(any("approvers must match" in error for error in errors))
        self.assertTrue(any("reviewers must match" in error for error in errors))

    def test_manual_codeowners_override_fails(self):
        (self.root / ".github/CODEOWNERS").write_text("* @engineer @Reviewer\n/.github/ @outsider\n")
        self.assertTrue(check_codeowners(self.root, parse_owners(OWNERS)))

    def test_owner_and_generated_file_symlinks_fail(self):
        for name in ("OWNERS", ".github/CODEOWNERS"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_text()
                path.unlink()
                target = self.root / "target"
                target.write_text(original)
                path.symlink_to(target)
                self.assertTrue(validate_ownership(self.root, MAINTAINERS))
                path.unlink()
                path.write_text(original)

    def test_missing_owners_fails(self):
        (self.root / "OWNERS").unlink()
        with self.assertRaises(ValueError):
            read_owners(self.root)

    def test_ambiguous_or_unsupported_yaml_fails(self):
        invalid = [
            "",
            "reviewers: []\napprovers: []",
            "reviewers: engineer\napprovers: [engineer]",
            "reviewers: [engineer, ENGINEER]\napprovers: [engineer]",
            "reviewers: [engineer]\napprovers: [engineer, ENGINEER]",
            "reviewers: [engineer]\napprovers: [other]",
            OWNERS + "approvers: [outsider]\n",
            OWNERS + "filters: {}\n",
            "reviewers: [true]\napprovers: [true]",
            "reviewers: ['@engineer']\napprovers: ['@engineer']",
            "reviewers: [org/team]\napprovers: [org/team]",
            "reviewers: ['robot[bot]']\napprovers: ['robot[bot]']",
            "!!python/object/apply:os.system ['false']",
            "reviewers: [engineer\napprovers: [engineer]",
            "#" * 17000,
        ]
        for text in invalid:
            with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                parse_owners(text)

    def test_unknown_directory_or_alias_semantics_cannot_be_silently_ignored(self):
        self.assertEqual(validate_ownership_paths(["OWNERS", ".github/CODEOWNERS", "README.md"]), [])
        for path in ("backend/OWNERS", "OWNERS_ALIASES", "backend/OWNERS_ALIASES", "CODEOWNERS", "docs/CODEOWNERS"):
            with self.subTest(path=path):
                self.assertTrue(validate_ownership_paths([path]))


if __name__ == "__main__":
    unittest.main()
