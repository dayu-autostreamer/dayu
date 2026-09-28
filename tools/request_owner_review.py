#!/usr/bin/env python3
"""Request one available OWNERS reviewer, using only trusted base-branch data.

This is routing, not an approval/status service. It never executes PR contents,
posts approvals, changes labels or permissions, or merges a pull request.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__:
    from .owners import parse_owners
else:
    from owners import parse_owners


REPOSITORY = "dayu-autostreamer/dayu"
BASE_BRANCH = "main"
EVENT_ACTIONS = {"opened", "reopened", "ready_for_review", "synchronize", "edited"}


class GitHub:
    def __init__(self, token):
        self.token = token

    def request(self, method, path, data=None):
        # Never follow a URL supplied by a PR, comment, content response, or user.
        url = "https://api.github.com/repos/" + REPOSITORY + "/" + path
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "dayu-owners-review",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        body = None if data is None else json.dumps(data).encode("utf-8")
        if body is not None:
            headers["Content-Type"] = "application/json"
        with urlopen(Request(url, data=body, headers=headers, method=method), timeout=30) as response:
            return json.load(response)

    def reviews(self, number):
        result, page = [], 1
        while True:
            batch = self.request("GET", "pulls/{}/reviews?per_page=100&page={}".format(number, page))
            result.extend(batch)
            if len(batch) < 100:
                return result
            page += 1


def routable(pr):
    return (
        pr["state"] == "open" and not pr["draft"]
        and pr["base"]["repo"]["full_name"] == REPOSITORY
        and pr["base"]["ref"] == BASE_BRANCH
    )


def available_reviewers(owners, pr, reviews):
    """Avoid the author, duplicate requests, and repeatedly notifying active reviewers."""
    author = pr["user"]["login"].casefold()
    candidates = {login.casefold(): login for login in owners["reviewers"] if login.casefold() != author}
    pending = {user["login"].casefold() for user in pr["requested_reviewers"]}
    if pending & candidates.keys():
        return []
    latest = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        user = review.get("user")
        if user and user.get("type") == "User" and review["state"] != "PENDING":
            latest[user["login"].casefold()] = review
    for login in candidates:
        review = latest.get(login)
        if review and (
            review["state"] in {"COMMENTED", "CHANGES_REQUESTED"}
            or (review["state"] == "APPROVED" and review["commit_id"] == pr["head"]["sha"])
        ):
            return []
    # Stable rotation; repeated events do not randomly notify different people.
    logins = sorted(candidates.values(), key=str.casefold)
    if not logins:
        return []
    offset = pr["number"] % len(logins)
    return logins[offset:] + logins[:offset]


def route_review(api, number, dry_run=False):
    pr = api.request("GET", "pulls/{}".format(number))
    if not routable(pr):
        print("No request: PR is closed, draft, or outside public main.")
        return None
    base_sha = pr["base"]["sha"]
    content = api.request("GET", "contents/OWNERS?" + urlencode({"ref": base_sha}))
    if content.get("type") != "file" or content.get("encoding") != "base64":
        raise ValueError("Expected a regular base-branch OWNERS file")
    owners = parse_owners(base64.b64decode(content["content"]).decode("utf-8"))
    candidates = available_reviewers(owners, pr, api.reviews(number))
    if not candidates:
        print("No request: an eligible reviewer is already involved, or only the author is available.")
        return None
    selected = None
    for login in candidates:
        permission = api.request("GET", "collaborators/{}/permission".format(login))["permission"]
        if permission in {"admin", "maintain", "write"}:
            selected = login
            break
        print("Skipping {}: review requests require repository write access; check the roster/access alignment.".format(login))
    if selected is None:
        raise ValueError("No independent OWNERS reviewer has the required live access; arrange review manually")
    # Re-read after API calls so a draft conversion, retarget, push, or native
    # CODEOWNERS request does not cause a stale or duplicate notification.
    current = api.request("GET", "pulls/{}".format(number))
    if (
        not routable(current)
        or current["base"]["sha"] != base_sha
        or current["head"]["sha"] != pr["head"]["sha"]
        or not available_reviewers(owners, current, api.reviews(number))
    ):
        print("No request: PR or reviewer state changed during routing; the next event can retry.")
        return None
    if dry_run:
        print("Dry run: would request {} on PR #{}; no GitHub changes made.".format(selected, number))
    else:
        api.request("POST", "pulls/{}/requested_reviewers".format(number), {"reviewers": [selected]})
        print("Requested {} on PR #{}. Native Reviews determine approval.".format(selected, number))
    return selected


def event_number(event, repository, event_name):
    if repository != REPOSITORY or event_name != "pull_request_target":
        raise ValueError("Writes are allowed only for public Dayu pull_request_target events")
    if event.get("action") not in EVENT_ACTIONS or event.get("repository", {}).get("full_name") != REPOSITORY:
        raise ValueError("Unexpected repository or event action")
    number = event.get("number")
    if type(number) is not int or number <= 0:
        raise ValueError("Expected a positive PR number")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="read public GitHub data without requesting reviews")
    parser.add_argument("--pull-request", type=int, help="PR number for a local dry run only")
    args = parser.parse_args()
    try:
        if args.pull_request is not None:
            if not args.dry_run or args.pull_request <= 0:
                raise ValueError("--pull-request requires --dry-run and a positive number")
            number = args.pull_request
        else:
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
            number = event_number(event, os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_EVENT_NAME"])
        token = os.environ.get("GITHUB_TOKEN", "")
        if not args.dry_run and not token:
            raise ValueError("Missing workflow token")
        route_review(GitHub(token), number, args.dry_run)
    except HTTPError as exc:
        # Do not print authenticated request headers or arbitrary response bodies.
        print("GitHub API returned HTTP {}. Check base OWNERS, permissions, and workflow logs.".format(exc.code), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print("Review routing failed: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
