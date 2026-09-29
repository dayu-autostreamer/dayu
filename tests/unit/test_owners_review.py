"""Review routing trust boundaries and idempotency, without GitHub writes."""

import base64
import copy
import unittest
from unittest.mock import Mock

from tools.request_owner_review import GitHub, REPOSITORY, available_reviewers, event_number, route_review


OWNERS = {"reviewers": ["engineer", "Reviewer"], "approvers": ["engineer", "Reviewer"]}
PR = {
    "number": 7, "state": "open", "draft": False,
    "user": {"login": "Engineer"},
    "base": {"repo": {"full_name": REPOSITORY}, "ref": "main", "sha": "a" * 40},
    "head": {"repo": {"full_name": "attacker/fork"}, "sha": "b" * 40},
    "requested_reviewers": [],
}


def review(login="Reviewer", state="APPROVED", commit_id="b" * 40, identity_type="User", number=1):
    return {
        "id": number, "user": {"login": login, "type": identity_type},
        "state": state, "commit_id": commit_id,
    }


class FakeGitHub:
    def __init__(self, pr=None, current=None, reviews=None, permission="write"):
        self.pr = copy.deepcopy(PR if pr is None else pr)
        self.current = current
        self.submitted = reviews or []
        self.permission = permission
        self.calls = []
        self.reads = 0

    def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if method == "POST":
            return {}
        if path == "pulls/7":
            self.reads += 1
            return copy.deepcopy(self.current if self.current is not None and self.reads > 1 else self.pr)
        if path == "contents/OWNERS?ref=" + "a" * 40:
            text = "reviewers: [engineer, Reviewer]\napprovers: [engineer, Reviewer]\n"
            return {"type": "file", "encoding": "base64", "content": base64.b64encode(text.encode()).decode()}
        if path == "collaborators/Reviewer/permission":
            return {"permission": self.permission}
        raise AssertionError("Unexpected API read: " + path)

    def reviews(self, number):
        return copy.deepcopy(self.submitted)

    @property
    def writes(self):
        return [call for call in self.calls if call[0] != "GET"]


class ReviewSelectionTests(unittest.TestCase):
    def test_author_is_excluded_case_insensitively(self):
        self.assertEqual(available_reviewers(OWNERS, PR, []), ["Reviewer"])

    def test_sole_owner_cannot_request_themself(self):
        self.assertEqual(available_reviewers({"reviewers": ["engineer"]}, PR, []), [])

    def test_native_codeowner_request_is_not_duplicated(self):
        pr = copy.deepcopy(PR)
        pr["requested_reviewers"] = [{"login": "REVIEWER"}]
        self.assertEqual(available_reviewers(OWNERS, pr, []), [])

    def test_outside_review_does_not_suppress_owner_routing(self):
        self.assertEqual(available_reviewers(OWNERS, PR, [review("outsider")]), ["Reviewer"])

    def test_bot_review_cannot_suppress_human_routing(self):
        self.assertEqual(available_reviewers(OWNERS, PR, [review(identity_type="Bot")]), ["Reviewer"])

    def test_current_approval_or_active_feedback_is_not_repeatedly_requested(self):
        for state in ("APPROVED", "COMMENTED", "CHANGES_REQUESTED"):
            with self.subTest(state=state):
                self.assertEqual(available_reviewers(OWNERS, PR, [review(state=state)]), [])

    def test_new_head_can_request_renewed_review(self):
        self.assertEqual(available_reviewers(OWNERS, PR, [review(commit_id="c" * 40)]), ["Reviewer"])

    def test_latest_dismissal_is_not_hidden_by_an_older_approval(self):
        reviews = [review(state="DISMISSED", number=2), review(number=1)]
        self.assertEqual(available_reviewers(OWNERS, PR, reviews), ["Reviewer"])


class RoutingTests(unittest.TestCase):
    def test_fork_contents_and_workflow_edits_are_never_loaded(self):
        api = FakeGitHub()
        self.assertEqual(route_review(api, 7), "Reviewer")
        self.assertEqual(api.writes, [("POST", "pulls/7/requested_reviewers", {"reviewers": ["Reviewer"]})])
        contents = [path for _, path, _ in api.calls if path.startswith("contents/")]
        self.assertEqual(contents, ["contents/OWNERS?ref=" + "a" * 40])

    def test_dry_run_never_mutates_github(self):
        api = FakeGitHub()
        self.assertEqual(route_review(api, 7, dry_run=True), "Reviewer")
        self.assertEqual(api.writes, [])

    def test_replayed_event_does_not_duplicate_a_request(self):
        current = copy.deepcopy(PR)
        current["requested_reviewers"] = [{"login": "Reviewer"}]
        api = FakeGitHub(current)
        self.assertIsNone(route_review(api, 7))
        self.assertEqual(api.writes, [])

    def test_closed_draft_and_other_targets_are_skipped_before_loading_owners(self):
        variants = [copy.deepcopy(PR) for _ in range(4)]
        variants[0]["state"] = "closed"
        variants[1]["draft"] = True
        variants[2]["base"]["ref"] = "release"
        variants[3]["base"]["repo"]["full_name"] = "someone/fork"
        for pr in variants:
            with self.subTest(pr=pr):
                api = FakeGitHub(pr)
                self.assertIsNone(route_review(api, 7))
                self.assertEqual(len(api.calls), 1)

    def test_head_base_and_draft_races_do_not_send_stale_requests(self):
        variants = [copy.deepcopy(PR) for _ in range(4)]
        variants[0]["head"]["sha"] = "c" * 40
        variants[1]["base"]["sha"] = "c" * 40
        variants[2]["draft"] = True
        variants[3]["requested_reviewers"] = [{"login": "Reviewer"}]
        for current in variants:
            with self.subTest(current=current):
                api = FakeGitHub(current=current)
                self.assertIsNone(route_review(api, 7))
                self.assertEqual(api.writes, [])

    def test_documented_role_does_not_create_repository_access(self):
        api = FakeGitHub(permission="read")
        with self.assertRaisesRegex(ValueError, "live access"):
            route_review(api, 7)
        self.assertEqual(api.writes, [])

    def test_invalid_base_owners_fail_before_any_write(self):
        api = FakeGitHub()
        original = api.request

        def request(method, path, data=None):
            if path.startswith("contents/"):
                return {"type": "file", "encoding": "base64", "content": base64.b64encode(b"approvers: []").decode()}
            return original(method, path, data)

        api.request = request
        with self.assertRaises(ValueError):
            route_review(api, 7)
        self.assertEqual(api.writes, [])

    def test_event_repository_and_type_guard_writes(self):
        event = {"action": "opened", "repository": {"full_name": REPOSITORY}, "number": 7}
        self.assertEqual(event_number(event, REPOSITORY, "pull_request_target"), 7)
        for repo, name, number in (("fork/dayu", "pull_request_target", 7), (REPOSITORY, "pull_request", 7),
                                   (REPOSITORY, "issue_comment", 7), (REPOSITORY, "pull_request_target", True),
                                   (REPOSITORY, "pull_request_target", -1)):
            with self.subTest(repo=repo, name=name, number=number), self.assertRaises(ValueError):
                event_number(dict(event, number=number), repo, name)

    def test_reviews_are_paginated(self):
        api = GitHub("")
        api.request = Mock(side_effect=[[review(number=i) for i in range(100)], [review(number=100)]])
        self.assertEqual(len(api.reviews(7)), 101)
        self.assertEqual(api.request.call_args.args[1], "pulls/7/reviews?per_page=100&page=2")


if __name__ == "__main__":
    unittest.main()
