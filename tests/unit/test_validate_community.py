"""Regression checks for the lightweight community gate; no application imports."""

import tempfile
import unittest
from pathlib import Path

from tools.validate_community import validate_links, validate_roster


ROSTER = """## Technical Steering Committee
| TSC member | Office | GitHub ID | Affiliation | Email |
| --- | --- | --- | --- | --- |
| Community lead | Chair | | Independent | <chair@example.org> |
| Engineer | Vice Chair | @[engineer](https://github.com/engineer) | Independent | <engineer@example.org> |

## Maintainers
| Maintainer | GitHub ID | Affiliation | Email |
| --- | --- | --- | --- |
| Engineer | @[engineer](https://github.com/engineer) | Independent | <engineer@example.org> |
| Reviewer | @[reviewer](https://github.com/reviewer) | Independent | <reviewer@example.org> |

## Contributors
| Contributor | GitHub ID | Affiliation | Email |
| --- | --- | --- | --- |
| Engineer | @[engineer](https://github.com/engineer) | Independent | <engineer@example.org> |
| Contributor | @[contributor](https://github.com/contributor) | Independent | <contributor@example.org> |
"""


class RosterTests(unittest.TestCase):
    def test_dual_role_and_non_maintaining_chair_are_valid(self):
        errors, rosters = validate_roster(ROSTER)
        self.assertEqual(errors, [])
        self.assertEqual(len(rosters["Maintainers"]), 2)

    def test_duplicate_identity_is_case_insensitive(self):
        text = ROSTER.replace("@[reviewer](https://github.com/reviewer)", "@[ENGINEER](https://github.com/ENGINEER)")
        errors, _ = validate_roster(text)
        self.assertTrue(any("duplicate GitHub identity within Maintainers" in error for error in errors))

    def test_mismatched_profile_link_is_rejected(self):
        errors, _ = validate_roster(ROSTER.replace("github.com/reviewer", "github.com/someone-else"))
        self.assertTrue(any("inconsistent GitHub identity" in error for error in errors))

    def test_malformed_row_is_not_silently_dropped(self):
        errors, _ = validate_roster(ROSTER.replace("Reviewer | @[reviewer]", "Reviewer | extra | @[reviewer]"))
        self.assertTrue(any("incorrect row width" in error for error in errors))

    def test_two_chairs_are_not_treated_as_chair_and_vice_chair(self):
        errors, _ = validate_roster(ROSTER.replace("| Vice Chair |", "| Chair |"))
        self.assertTrue(any("more than one Chair" in error for error in errors))

    def test_vacancy_can_be_recorded_without_a_fictitious_member(self):
        text = "\n".join(line for line in ROSTER.splitlines() if "| Community lead |" not in line)
        errors, _ = validate_roster(text)
        self.assertEqual(errors, [])

    def test_old_committer_roster_is_rejected(self):
        errors, _ = validate_roster(ROSTER + "\n## Committers\n")
        self.assertTrue(any("separate Committer roster" in error for error in errors))


class LocalLinkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def test_heading_and_unicode_links(self):
        (self.root / "roles.md").write_text("# Roles\n## Chair and Vice Chair\n## 社区\n")
        (self.root / "guide.md").write_text("[Officers](roles.md#chair-and-vice-chair) [Community](roles.md#社区)")
        self.assertEqual(validate_links(self.root, "guide.md"), [])

    def test_missing_heading_fails(self):
        (self.root / "roles.md").write_text("# Roles\n")
        (self.root / "guide.md").write_text("[Officers](roles.md#chair)")
        self.assertTrue(any("missing heading" in error for error in validate_links(self.root, "guide.md")))

    def test_missing_target_fails(self):
        (self.root / "guide.md").write_text("[Owners](OWNERS)")
        self.assertTrue(any("missing link target" in error for error in validate_links(self.root, "guide.md")))

    def test_outside_repository_is_not_read(self):
        (self.root / "guide.md").write_text("[Outside](../outside.md)")
        self.assertTrue(any("escapes repository" in error for error in validate_links(self.root, "guide.md")))

    def test_external_links_comments_and_examples_do_not_make_network_requests(self):
        (self.root / "guide.md").write_text(
            '[Web](https://example.invalid/path)\n<!-- [Draft](not-ready.md) -->\n'
            '```md\n[Example](placeholder.md)\n```\n'
        )
        self.assertEqual(validate_links(self.root, "guide.md"), [])


if __name__ == "__main__":
    unittest.main()
