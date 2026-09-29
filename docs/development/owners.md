# OWNERS with GitHub Actions

Dayu uses one root [OWNERS](../../OWNERS) file for repository-wide review responsibilities. GitHub Actions validates
the file and helps request reviews. GitHub Reviews, generated [CODEOWNERS](../../.github/CODEOWNERS), and the branch
rules provide the approval gate. No Prow server, webhook endpoint, custom GitHub App, or personal token is needed.

**Rollout:** these files are prepared for synchronization from the development checkout to `dayu-autostreamer/dayu`.
The workflows and generated routing become active after they reach public `main`. Follow the activation checklist
below before making `Community policy` a required check.

## How Contributors Use It

1. Open a PR against public `main`, or mark a draft ready for review. GitHub requests the code owners automatically.
   The OWNERS routing workflow requests one eligible reviewer if none is already involved. Manual review requests
   remain available when routing is delayed or unavailable.
2. Address feedback. Reviewers submit **Approve** or **Request changes** through GitHub's Review changes controls.
   Pushing new reviewable changes invalidates approvals under the repository's stale-review rule.
3. A Maintainer verifies the applicable [review policy](../../GOVERNANCE.md#technical-decisions-and-pull-requests),
   native code-owner approval, required CI, and resolved objections, then merges manually.

Routine PRs need one independent Maintainer approval. Significant changes also need the design record and a second
independent technical approval, which a qualified Contributor can submit without becoming a Maintainer. GitHub's
one-owner gate and a passing Community check do not establish that this extra review happened or was qualified.
The merger remains responsible for checking it, including whether a reviewer co-authored the change or has a conflict.

There are no `/lgtm`, `/approve`, or `/hold` commands, approval labels, or automatic merges. An ordinary comment does
not become a native approval. To pause work, use a draft PR or request changes and explain the unresolved concern.
Reviewers, authors, and mergers follow the same governance policy regardless of automation availability.

## Ownership and Permissions

| Record | Purpose |
| --- | --- |
| [MAINTAINERS.md](../../MAINTAINERS.md) | Authoritative human roster, including TSC offices and Maintainers. |
| Root [OWNERS](../../OWNERS) | `reviewers` receive technical review requests; `approvers` supply the native owner approval. Both lists currently contain all Maintainers. |
| Generated [.github/CODEOWNERS](../../.github/CODEOWNERS) | Projects `approvers` to one `*` rule for GitHub, including ownership files, governance, and workflows. |
| GitHub teams and permissions | Actual access, managed by an administrator following approved appointments. |
| GitHub ruleset | Requires an independent owner approval, dismisses stale approvals, and enforces required CI. |

`reviewers` and `approvers` describe tasks, not extra community ranks. All current Maintainers have both roles
throughout the repository. Chair and Vice Chair offices confer neither role. Contributor credit alone does not add
a person to either list, but anyone can offer a technically qualified review.

The simplified schema accepts only the two nonempty lists of GitHub usernames. It rejects duplicates, team handles,
bot handles, unknown fields, directory-level OWNERS, OWNERS_ALIASES, and additional CODEOWNERS files. Directory
delegation would need a separately reviewed policy and implementation; it must not silently narrow a Maintainer's
scope or create an unprotected path.

CODEOWNERS lists the actual approvers rather than a team reference, so the approval mapping follows the reviewed
OWNERS file. The existing `dayu-maintainer` team still provides repository access. An administrator must verify that
each approver has at least write access and that GitHub reports no CODEOWNERS errors. The files cannot grant access
or prove that a human appointment followed governance.

## Editing the Files

Record the appointment, departure, or responsibility decision under [governance](../../GOVERNANCE.md) first. Update
MAINTAINERS.md and both OWNERS lists together, retaining contribution credit. In a Python environment, run:

```bash
python3 -m pip install -r .github/requirements/community.txt
make sync-codeowners
make validate-community
python3 -m unittest discover -s tests/unit -p 'test_owners*.py'
python3 -m unittest discover -s tests/unit -p test_validate_community.py
```

The dependency file installs only PyYAML; an existing developer environment already includes a compatible version.
`sync-codeowners` writes only the local generated file. Commit it with the source change. The Community workflow
checks the generated result and roster agreement; it never edits files, creates commits, or changes permissions.
Run the checker without regeneration when testing whether the committed projection is stale.

GitHub uses CODEOWNERS from the PR's **base branch**, including for a PR that changes OWNERS or CODEOWNERS. A proposed
new owner cannot authorize that PR merely by adding their name. The first PR that introduces CODEOWNERS needs a
manually requested independent Maintainer review because the base branch does not yet contain the new routing.
Follow the [administration guide](community-administration.md) to align live access, account for teams shared with
other repositories, and preserve independent review coverage during transitions.

## What the Workflows Do

[Community](../../.github/workflows/community.yml) runs on every PR to `main`, main-branch pushes, and manual dispatch.
Its read-only job validates roster identities, OWNERS schema, exact Maintainer membership of both lists, generated
CODEOWNERS, supported ownership paths, and local document links. It also runs the ownership/routing regression tests.
It uses no role-administration credentials and imports no Dayu runtime components.

[OWNERS review routing](../../.github/workflows/owners-review.yml) runs only for `dayu-autostreamer/dayu` PRs targeting
`main`. It handles opening, reopening, marking ready, pushing commits, and PR edits. It:

- Checks out its script and dependency declaration at the trusted **base SHA**, with persisted credentials disabled.
- Reads the current PR and root OWNERS at its current base SHA through the GitHub API. PR head files are never loaded.
- Excludes the PR author case-insensitively and checks live write access before requesting another reviewer.
- Preserves native and manual review requests. Existing owner feedback or approval for the current head prevents
  repeated notifications. A stale or dismissed approval can lead to a renewed request after a push.
- Rechecks the PR state before requesting one reviewer. Drafts, closed PRs, other target branches, and forks of the
  Dayu repository do not receive writes from this workflow. PRs **from forks into public Dayu** are supported.
- Uses only contents-read and pull-request-write permissions with the built-in `GITHUB_TOKEN`.

This privileged workflow must never execute PR head code, install PR-supplied requirements, download PR artifacts,
or share caches with PR jobs. Its sole mutation is a reviewer request. No comment parsing, approval state database,
review-event relay, status-writing service, or review bot identity is needed. GitHub itself owns approval state.

The job `Request owner review` is a routing aid and must **not** be made a required approval check. A successful run
does not mean a human approved. If it fails, inspect its logs, request review manually, and repair the configuration
or access problem while retaining the native approval and CI requirements. If an owner has already left comments,
the workflow avoids repeatedly requesting them; authors can request renewed review manually when ready.

The routing script supports a read-only diagnostic against public Dayu. Use an existing authenticated
`GITHUB_TOKEN` for the live permission lookup; CI receives the built-in token automatically:

```bash
python3 tools/request_owner_review.py --dry-run --pull-request <number>
```

This needs OWNERS to exist in public `main`. It does not send requests or substitute for a hosted fork-PR test.

## Activation and Verification

1. Synchronize the complete files through a reviewed public PR: OWNERS, generated CODEOWNERS, the two community
   workflows, tooling/dependency file, tests, Makefile changes, and documentation. Keep the development remote as-is.
2. Verify current approver access and the CODEOWNERS errors endpoint. Keep the existing one-owner review rule,
   stale-review dismissal, conversation resolution, nine required CI checks, and limited emergency bypass.
3. Verify hosted Community checks and review requests on ordinary and fork PRs, a draft-to-ready transition, and PRs
   authored by each Maintainer. Confirm that a repeated event does not duplicate an outstanding request.
4. Verify native approval, new-commit dismissal, Request changes, and a PR proposing an ownership change. Check that
   the last case still uses existing base-branch owners. Do not manufacture approvals to pass an acceptance test.
5. After successful hosted validation, add **`Community policy`**, bound to GitHub Actions (app ID `15368`), to the
   existing required checks. Preserve the other required contexts. Keep `Request owner review` supplementary.

Until step 5, Community is informative rather than a merge-blocking consistency check. The existing native rules
remain the approval gate once CODEOWNERS is in the base branch. No new label, organization bot, PAT, server, or webhook
is required. Existing path labels, Dependabot, and security scans continue to serve their separate purposes.

## References

- [Kubernetes OWNERS responsibilities](https://www.kubernetes.dev/docs/guide/owners/)
- [GitHub code owners and base-branch semantics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [GitHub review requests API](https://docs.github.com/en/rest/pulls/review-requests)
- [Secure use of pull_request_target](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target)
