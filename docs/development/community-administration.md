# Community Administration

[GOVERNANCE.md](../../GOVERNANCE.md) defines policy and [MAINTAINERS.md](../../MAINTAINERS.md) is the human-maintained
role roster. This guide explains how administrators apply them to **`dayu-autostreamer/dayu`**, whose default branch
is `main`. A development checkout may have a different `origin`; always specify the public repository explicitly in
GitHub commands. Synchronizing files does not by itself synchronize remote settings.

## Responsibilities and Sources of Truth

| Concern | Authoritative record or mechanism |
| --- | --- |
| Role definitions, decisions, appointments, and conflicts | `GOVERNANCE.md` |
| Current TSC officers, Maintainers, and contribution credit | `MAINTAINERS.md` |
| Actual access | GitHub organization teams and effective repository permissions |
| Technical review mapping | Root [OWNERS](../../OWNERS), checked against the current Maintainers |
| Native owner approval and review requests | [CODEOWNERS](../../.github/CODEOWNERS), generated from OWNERS and read from the PR's base branch |
| Required approvals and checks | GitHub rulesets and any classic branch protections |
| Consistency, review routing, and triage | [Community checks](../../.github/workflows/community.yml), [OWNERS routing](../../.github/workflows/owners-review.yml), and [PR labels](../../.github/workflows/pr-labeler.yml) |

There is no separate Committer rank. OWNERS uses the familiar reviewers/approvers distinction with all current
Maintainers in both lists. Contributors can review and help with an area without being appointed to a new rank.
The [simplified Actions mechanism](owners.md) needs no Prow service. Bots do not make appointments, change team
membership, approve changes, or merge PRs.

The TSC Chair and Vice Chair coordinate governance. Either may separately hold a Maintainer role, but the office
adds no vote, code-owner status, or repository permission. Repository administration is an operational duty, not a
community rank. Routine Maintainer work does not need an additional officer approval.

## Align Roles, Access, and Review Routing

For an accepted appointment or departure, the non-conflicted coordinator and an administrator:

1. Record the decision, the person's accepted responsibility, identity, and necessary access. Preserve contribution
   credit and arrange handover. Do not infer responsibility from affiliation or past commits.
2. Update the human roster and, for a Maintainer change, both root OWNERS lists. Run `make sync-codeowners` and
   `make validate-community`, and include the generated file in the PR. Keep decisions about TSC membership, an
   officer position, and maintenance responsibility distinct. Do not fabricate past votes because a roster lists a role.
3. Grant or remove the approved access and verify the effective permission on the intended repository. Check every
   repository attached to a team before changing its membership.
4. Verify generated CODEOWNERS coverage and that authors can obtain independent review. Each listed approver must
   have write or maintain access through the appropriate team or direct grant. Read back GitHub's CODEOWNERS errors.
5. Check the branch rules and record any incomplete step. A roster edit alone neither grants access nor proves that
   review enforcement is active.

The generated routing is one `*` rule listing the root OWNERS approvers, including governance files and CI configuration.
The Maintainer team manages access; team membership is not a separate source of code-owner names.
Do not route these files solely to the one-person admin team: its member could not obtain independent owner review
on their own PRs. Multiple names on one CODEOWNERS line mean **any one** owner may approve, not that all must approve.
GitHub uses the last matching pattern and does not combine it with earlier fallback patterns.

The same Maintainer team currently also has access to the website and SkyEngine repositories. Adding a Dayu-only
Maintainer to it would therefore grant access to those repositories too. Before such an appointment, split or
reconfigure the teams to match the approved scope; do not expand someone's access merely to reuse the existing name.
Do not change other repositories' team access as part of an unrelated Dayu roster edit.

## Verified Access and Rollout State

The following access was checked on **2026-09-29**. It is a dated record; verify live state before subsequent changes.

| Person or team | Effective state on `dayu-autostreamer/dayu` |
| --- | --- |
| Wenhui Zhou (`zwh2119`) | Admin; member of the Maintainer team. Administration is separate from the Vice Chair office. |
| Haoyang Su (`ShyEdge`) | Maintain; member of the Maintainer team and eligible for required human review. |
| Hao Wu (`skyrimforest`) | Read; Contributor without appointed merge responsibility. |
| `dayu-maintainer` | Visible team with Maintain access; members are `zwh2119` and `ShyEdge`. |
| `dayu-admin` | Visible team with Admin access; currently `zwh2119`. |

The rollout has two parts: remote settings and files synchronized into public `main`. The latter is still required
for root OWNERS, generated CODEOWNERS, the Community and OWNERS workflows, PR labeler, and Dependabot configuration. Until then, do not describe
their repository-driven behavior as active or require a status check that has never run.

The following remote operations were applied and read back on **2026-09-29**:

| Setting | Result |
| --- | --- |
| [Main ruleset](https://github.com/dayu-autostreamer/dayu/rules/5524552) | Active: one approval, stale-review dismissal, resolved conversations, and the nine existing CI checks below, with strict up-to-date checking. |
| Code-owner review | Enabled in the ruleset; owner routing and coverage await CODEOWNERS synchronization into public `main`. |
| Bypass actors | Only `dayu-admin`, through a PR. The previous blanket Maintain-role and application bypasses were removed. |
| Existing protections | Force-push/deletion protection, Copilot review, and code-quality rules retained. |
| Dependabot | Alerts already enabled; security updates enabled and read back as active and not paused. The new version-update configuration awaits file synchronization. |
| CodeQL and Scorecard | Both restored from `disabled_inactivity` to `active`; [CodeQL](https://github.com/dayu-autostreamer/dayu/actions/runs/36464196992) (Python and JavaScript) and [Scorecard](https://github.com/dayu-autostreamer/dayu/actions/runs/36464205899) manual runs passed on the existing public `main`. |
| PR labels | All seven `area/*` labels created; existing `documentation`, `test`, and `dependencies` labels retained. The labeler itself awaits file synchronization. |
| Private vulnerability reporting | Already enabled; retained. |
| Actions token | Existing read-only default and prohibition on workflow approval of PRs retained. Individual jobs declare only the extra permissions they need. |

No role/team membership changes were necessary: the live Maintainer membership and Hao Wu's read access already
matched the agreed responsibilities. The checkout's remote and public repository code were not changed by these
administrative operations. New action versions and workflows still need hosted validation after synchronization.

## Branch Rules

Use a main-branch ruleset with:

- A pull request and **one independent human approval**; code-owner approval for paths with an owner.
- Dismissal of stale approvals on new reviewable changes and resolution of review conversations.
- Passing required CI checks on an up-to-date branch, plus force-push and branch-deletion protection.
- An emergency bypass restricted to `dayu-admin`, **through a PR only**. No blanket Maintain-role or application
  bypass. Every use must satisfy and document the bounded emergency procedure in the governance policy.

The existing Copilot review and code-quality rules are retained as supplementary checks. AI reviews do not count as
human approvals. Administrator access still allows changing rules; GitHub settings cannot replace accountability.

GitHub's default `require_extra_approval_for_unattributed_changes` is also retained. Under the current GitHub rules,
a Copilot-authored PR that is not attributed to a person needs one extra human approval (two with this ruleset).
This does not impose two approvals on ordinary human-authored PRs; see the
[unattributed Copilot approval rule](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#additional-approval-for-unattributed-copilot-pull-requests).

The existing CI workflow produced these nine successful check names on public `main` in run
[33140271788](https://github.com/dayu-autostreamer/dayu/actions/runs/33140271788):

`Workflow lint`, `Python lint`, `Python syntax check`, `Build matrix validation`, `Python tests`, `Python coverage`,
`Python component tests`, `Python e2e smoke`, and `Frontend checks`.

Bind required checks to the GitHub Actions app (ID `15368`), not just the context name. Keep CircleCI, Codecov upload,
CodeQL, and Scorecard outside the required list until their intended PR behavior is verified. Do not introduce path
filters that leave a required workflow pending, or rename a required job without updating the rule at the same time.

One native approval cannot enforce the extra technical review for significant changes. The merger must check a
short design record and two independent implementation approvals, including at least one Maintainer. A qualified
Contributor without write access can supply the second technical approval. Do not require two Maintainer approvals
globally: with two Maintainers, their own PRs would have only one other Maintainer available.

## Automation

### Community checks

Install `.github/requirements/community.txt` in a Python environment, then run `make validate-community` locally.
The [validator](../../tools/validate_community.py) uses only PyYAML in addition to the standard library. It checks roster
structure, duplicate people within a role, officer titles, both OWNERS lists against the Maintainers, generated
CODEOWNERS, unsupported ownership files, and local document links and headings. Role overlap between sections is valid.
Vacancies produce coverage warnings rather than inventing replacement members. Human review must still verify the
accuracy of appointments, conflicts, live access, and whether the documented decision thresholds are satisfied.

The Community workflow runs on every PR to `main`, main-branch pushes, and manual dispatch with read-only contents
access. It has no secrets or team-administration token. After it has run successfully on public PRs, add its actual
`Community policy` check to the required checks. Until that activation, it is a supplementary check.

### OWNERS review routing

The [OWNERS guide](owners.md) defines the file schema, native GitHub review workflow, trust boundary, and activation
tests. The routing job uses the base branch's tooling and ownership data to request an independent reviewer if
needed. It does not evaluate or submit approvals. Keep this routing job supplementary; the code-owner review rule
enforces approval even when routing automation is unavailable. Use GitHub Reviews, not slash commands or labels.

### PR labels

The labeler applies [path-based labels](../../.github/labeler.yml) for documentation, tests, community, CI, backend,
frontend, runtime, datasource, and builds. Labels assist triage; they never classify a change as safely mergeable.
Create the configured labels before enabling the workflow. Automatic removal is disabled to preserve human triage.

The workflow uses `pull_request_target` so it can label fork PRs, but **never checks out or executes PR code**. It runs
only the pinned labeler action, reads configuration from the trusted base repository, and has only contents-read
and pull-request-write permissions. Do not add PR-provided scripts, branch checkouts, caches, or artifacts to this
privileged workflow. A new configuration becomes effective after it reaches the base repository.

### Dependencies and security analysis

[Dependabot](../../.github/dependabot.yml) proposes weekly GitHub Actions and frontend updates, grouping compatible
minor/patch updates and keeping major updates separate. Limit ordinary version-update PRs to three per ecosystem.
Python/ML dependencies start with security updates only because Python, CUDA, and model-library compatibility need
explicit maintenance work. `open-pull-requests-limit: 0` disables ordinary Python version PRs, not security fixes.
No dependency PR is automatically approved or merged.

Keep Dependabot alerts, security updates, and private vulnerability reporting enabled on the public repository.
Keep [CodeQL](../../.github/workflows/codeql-analysis.yml) and [Scorecard](../../.github/workflows/ossf-scorecard.yml)
enabled; GitHub may disable schedules after prolonged repository inactivity, so check workflow state during roster
reviews. These tools do not replace the private reporting and fix process in [SECURITY.md](../../SECURITY.md).

The OWNERS mechanism uses GitHub-hosted Actions and native Reviews. Automatic membership synchronization and a
custom organization bot are unnecessary for the current team. If access drift later becomes difficult to manage
manually, first add a read-only audit; automatic provisioning requires a separate reviewed design.

## Synchronize and Verify on the Public Repository

1. Synchronize the complete community change set into a PR against public `main`, including root OWNERS and its
   generated CODEOWNERS. Include tooling, its small dependency file and tests, Makefile targets, workflows, labeler,
   and Dependabot config.
   Keep the development checkout's remote unchanged; do not push its unrelated history to the public repository.
2. Record the governance amendment and individual officer appointments separately, following recusal and acceptance
   rules. Technical review of the PR and this administrator checklist do not substitute for those decision records.
3. Before merging, run local community checks, workflow lint, and applicable hosted checks. A new CODEOWNERS file does
   not route review of the PR that first introduces it; request the independent Maintainer review manually.
4. After merging, query `repos/dayu-autostreamer/dayu/codeowners/errors?ref=main` and test reviewer requests on real
   contributions, including contributions authored by each Maintainer. Confirm that direct pushes and unreviewed PRs
   are blocked, stale approvals expire, and the nine required CI checks run on PRs.
5. Confirm labels and OWNERS review routing on a fork PR, then follow the remaining [OWNERS acceptance cases](owners.md#activation-and-verification).
   After successful hosted Community checks, add `Community policy` (GitHub Actions app) to required checks.
   Do not require the routing job, create dummy approvals, or bypass rules just to mark this checklist complete.
6. Check Dependabot PRs and the next CodeQL/Scorecard runs. Record any workflow that needs repair or cannot yet run;
   being enabled is not evidence of a successful scan.

Always read settings back after changing them. For example:

```bash
gh api repos/dayu-autostreamer/dayu/rulesets
gh api repos/dayu-autostreamer/dayu/rules/branches/main
gh api orgs/dayu-autostreamer/teams/dayu-maintainer/members
gh api repos/dayu-autostreamer/dayu/collaborators/ShyEdge/permission
gh api repos/dayu-autostreamer/dayu/actions/workflows
```

An organization-wide `.github` repository can later hold shared contact and conduct guidance after agreement across
the organization's projects. This project keeps its own authoritative governance and roster. Default community
files do not automatically distribute workflows or CODEOWNERS to other repositories.

## Decision Records and Periodic Review

For ordinary work, the issue and PR are the decision record. Formal decisions record the proposal, coordinator,
eligible people, recusals, deadline, explicit votes, outcome, rationale, and follow-up. Keep sensitive personnel and
conduct assessments restricted, with only an appropriate public outcome.

| Eligible people | Technical/appointment dispute | Governance, TSC membership, or officer appointment/removal |
| --- | --- | --- |
| 2 | 2 yes votes | 2 yes votes |
| 3 | 2 yes votes | 2 yes votes |
| 4 | 3 yes votes | 3 yes votes |
| 5 | 3 yes votes | 4 yes votes |

One yes and one abstention in a two-person TSC does not pass. A required recusal leaving only one eligible TSC member
uses the continuity procedure: all non-conflicted current TSC members and Maintainers, deduplicated by person,
participate; at least two thirds, rounded up, and at least two yes votes are required. Non-response or dissent does
not activate this fallback. For example, Wenhui's own officer appointment is considered by Lei and Haoyang; Lei's
own officer appointment is considered by Wenhui and Haoyang, assuming no other conflicts. Do not bundle them into a
single appointment vote that requires both candidates to recuse themselves.

At least every six months, review contact details, officer availability, maintenance coverage, team memberships,
repository permissions, OWNERS/generated CODEOWNERS, ruleset bypasses, and scheduled workflow state. Follow the contact and handover
process before changing roles. The website should link to the authoritative governance and roster here; its own
maintenance responsibilities remain separate from roles in this system repository.

## References

- [GitHub code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Repository roles](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/repository-roles-for-an-organization)
- [Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- [PR labeler](https://github.com/actions/labeler)
- [Dependabot configuration](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)
