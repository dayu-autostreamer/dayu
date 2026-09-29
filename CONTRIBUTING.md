# Contributing to Dayu

We welcome code, documentation, tests, bug reports, research validation, and user support from contributors of all
backgrounds. Read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). No project role or institutional affiliation
is required to contribute.

## Getting Started

- Fork the [Dayu repository](https://github.com/dayu-autostreamer/dayu) and branch from `main`, unless a Maintainer
  requests another target for a backport or coordinated change.
- Use the [documentation site](https://dayu-autostreamer.github.io/docs/) for deployment tutorials and the
  [repository quickstart](docs/repository-quickstart.md) and [development guide](docs/development/README.md) for code
  navigation and implementation details.
- Start with a reproducible issue, a missing test, a documentation improvement, or an extension through an existing
  hook or template. Comment on an issue to express interest; a Maintainer can assign it where useful. Assignment
  commands are not required. PR review routing is described below.

For questions, issue routing, or help finding reviewers, see [SUPPORT.md](SUPPORT.md). Report vulnerabilities privately
using [SECURITY.md](SECURITY.md), rather than in a public issue or PR.

## Contributor Workflow

1. Describe the problem and intended behavior in an issue or PR. Routine contributions do not need a separate design
   issue. Significant changes to shared contracts or sensitive behavior need the short design discussion below.
2. Make focused commits on a topic branch and push them to your fork. Update affected implementation docs, examples,
   templates, and tests when behavior or contracts change.
3. Open a PR against [dayu-autostreamer/dayu](https://github.com/dayu-autostreamer/dayu), explain why the change is
   needed, link related issues, and report the relevant validation and any checks not run.
4. Check the review requests from [OWNERS](OWNERS), or request a responsible [Maintainer](MAINTAINERS.md#maintainers)
   manually if automation is unavailable. Address feedback and obtain
   renewed review after substantive changes. The merger verifies the applicable reviews and checks before merging.

### Review Requirements

The [governance review policy](GOVERNANCE.md#technical-decisions-and-pull-requests) is authoritative:

- **Routine changes:** at least one independent Maintainer approval, plus relevant passing checks.
- **Significant changes:** a short design issue and two independent technically qualified implementation reviewers,
  including at least one Maintainer. The second reviewer may be a qualified Contributor without write access.
  This covers significant changes to shared Task/DAG or API contracts,
  install/cleanup behavior, compatibility, security boundaries, and release permissions.
- **Project direction or unresolved disputes:** Maintainers escalate to the TSC. A TSC decision or approved design
  does not replace implementation review.

Reviewers must be distinct people who did not author the change; holding two roles does not count as two approvals.
Chair and Vice Chair offices do not add an approval stage or replace technical review. Automated labels and AI
reviews help route work; they do not determine approval, appoint members, or authorize a merge.
Extensions that use existing processor, hook, scheduler, or visualization interfaces normally follow routine review.
The size of the diff alone does not determine the review path. For a significant change, describe the problem,
proposed behavior, alternatives, compatibility or migration impact, and validation in the design issue; an
enhancement issue can serve this purpose. Security design discussions remain private until safe to disclose.

### OWNERS and GitHub Reviews

Use GitHub's **Review changes → Approve / Request changes** controls. The root [OWNERS](OWNERS) lists all Maintainers
as reviewers and approvers. GitHub Actions checks this file against the roster and generated CODEOWNERS, and requests
an independent reviewer when needed. GitHub's native code-owner requirement and branch rules enforce the approval
gate. `/lgtm`, `/approve`, `/hold`, and corresponding labels have no command or approval semantics in this mechanism.

GitHub reads CODEOWNERS from the PR's base branch, so adding yourself in a PR does not authorize you to approve that
PR. New reviewable changes dismiss old approvals under the branch rules; obtain renewed approval before merging.
For a significant change, the merger also checks the design record and two distinct human GitHub Reviews, including
one Maintainer. A green Community check validates configuration, not the number or competence of reviewers.

The workflow starts after the files reach public `main`; no Prow server, custom GitHub App, or personal token is
needed. See the [OWNERS guide](docs/development/owners.md) for configuration, limitations, and activation checks.

## Local Validation

Choose checks that exercise the behavior you change. A prose-only change needs link, example, and consistency checks;
it does not require installing the Python/ML or frontend toolchain. For code changes, add or update meaningful tests
for the changed behavior and plausible regressions. Explain omitted or unavailable checks in the PR.

Use Python `3.8` from [`.python-version`](.python-version) and Node.js `20` from [`.nvmrc`](.nvmrc) when working in those
parts of the repository. Bootstrap the relevant environment with `make install-python-dev` or `make frontend-install`.
For community-only work, a Python environment with
`python3 -m pip install -r .github/requirements/community.txt` is sufficient; it installs only PyYAML.

| Changed area | Relevant checks |
| --- | --- |
| Python behavior | `make lint-python`, `make python-syntax`, and the affected tests; `make test-unit-integration` covers pure logic and API/runtime contracts. |
| Cross-component or lifecycle behavior | `make test-component` and `make test-e2e`, in addition to affected unit/integration tests. The e2e target is a template-driven smoke suite. |
| Build definitions, image matrix, or deployment templates | `make validate-build` and tests covering any changed configuration or lifecycle behavior. |
| Frontend | `make frontend-check` for formatting, tests, and build; use `make frontend-lint` for incremental lint cleanup. |
| Documentation and community files | `make validate-community`; after an approved OWNERS edit, run `make sync-codeowners` and commit the generated file too. Check commands and cross-document rules. No ML or frontend dependencies are needed. |

The [testing guide](docs/testing/README.md) explains the test layers and dependency setup.
`make test-python` runs the full Python suite; `make coverage-python` also produces `coverage.xml`.
`make ci-python` groups Python lint, syntax, and tests. `make check` combines the common build, Python, and
frontend checks. These shortcuts do not replace additional validation needed for a particular change, and there is
no need to run overlapping full suites repeatedly without a reason.

### Hosted CI

[GitHub Actions](.github/workflows/ci.yml) is the primary CI entry point and uploads coverage to Codecov through OIDC.
[CircleCI](.circleci/config.yml) reuses the repository's `make` targets. [codecov.yml](codecov.yml) configures project
coverage comparison with the base commit and patch coverage on changed lines. `make frontend-check` is the frontend
CI gate; the existing frontend lint backlog is handled incrementally.

Local check selection does not waive required hosted checks. Required reviews and status checks depend on the live
GitHub rules as well as this policy; the [administration guide](docs/development/community-administration.md) explains
their setup and limitations.

## Reviewable Changes and Commit Messages

Follow the repository's formatting and lint configuration. Split unrelated work into separate PRs and explain
compatibility impact, user-visible behavior, and validation. For a user-facing change, include a release note in the
PR template and describe any migration action.

A commit message should explain what changed and why. Keep the subject within 70 characters and wrap the body at
80 characters where practical:

```text
datasource: handle missing frame metadata

Explain the failure and how the change preserves the datasource contract.

Fixes #12
```

If review stalls, comment on the PR or use the contacts in [SUPPORT.md](SUPPORT.md). To take on ongoing review or
maintenance responsibility, follow the [appointment process](GOVERNANCE.md#appointments-and-delegation).
