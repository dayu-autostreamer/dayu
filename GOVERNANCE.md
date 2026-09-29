# Dayu Governance

Dayu was founded by [Dislab](https://dislab.nju.edu.cn/) at [Nanjing University](https://www.nju.edu.cn/).
We welcome contributors from academia, industry, and the wider open source community. Responsibility is earned through
individual contributions and judgment; affiliation does not confer a seat, a vote, or repository access.

Contributors participate in development and review, Maintainers lead day-to-day technical work, and the Technical
Steering Committee (TSC) stewards project direction and resolves escalated disputes. TSC membership and technical
maintenance are separate responsibilities that one person may hold together.

## Principles and Scope

- **Open participation:** code, documentation, testing, issue triage, research validation, user support, and community
  work all matter. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- **Public reasoning:** discuss technical work and record decisions in issues and pull requests. Keep vulnerability
  reports, conduct cases, and sensitive personnel assessments private; publish outcomes without exposing private data.
- **Consensus first:** seek a technically sound resolution before escalating or voting. A substantive objection should
  explain a concrete concern and, where possible, a way to resolve it.
- **Proportionate review:** review the behavior and risk of a change, rather than counting changed lines or files.

This file is the source of truth for Dayu governance. [MAINTAINERS.md](MAINTAINERS.md) records this repository's members
and responsibilities. [CONTRIBUTING.md](CONTRIBUTING.md) describes contribution mechanics and validation.
The documentation website may summarize these rules and link here; its own maintainer roster covers website work
and does not confer roles in this repository.

## Roles and Responsibilities

### Contributors

Anyone who contributes to Dayu is a Contributor; no appointment is needed. Contributors can propose changes,
participate in design and review, and raise concerns regardless of employer, institution, or membership status.

Review is an activity open to qualified contributors, not a separate appointed rank. Dayu currently has no separate
Committer role. Contribution credit does not confer merge access or governance votes.

### Maintainers

Maintainers are responsible for repository-wide technical quality, compatibility, release readiness, and continuity.
They coordinate reviews across areas, steward shared framework contracts, mentor contributors, and coordinate
security response under [SECURITY.md](SECURITY.md). They can ask contributors to help with an area while retaining
responsibility for independent review and merge decisions. Such assistance does not itself grant repository access.

Maintainers should communicate their availability and arrange review or release coverage when absent. There is no
fixed employment, weekly time, tenure, or pull-request-count requirement.

### Technical Steering Committee

The TSC stewards long-term direction, governance, and community continuity, and resolves disputes that Maintainers
cannot resolve. It considers community and technical evidence and records the reasons for its decisions. It does
not approve every routine patch or appointment. A TSC title alone does not establish technical review competence
or grant GitHub administration rights; a person who also serves as a Maintainer has both sets of responsibilities.

#### Chair and Vice Chair

The TSC has a **Chair** and a **Vice Chair**, held by different TSC members and recorded in
[MAINTAINERS.md](MAINTAINERS.md#technical-steering-committee). These are coordinating offices within the TSC.

- The Chair convenes direction and governance discussions, coordinates community representation, and ensures that
  escalated disputes receive a fair discussion and recorded collective decision.
- The Vice Chair helps prepare agendas, keep decision records, and follow up agreed actions. When the Chair is
  unavailable or recused, the Vice Chair can chair proceedings if eligible; this does not transfer the Chair's vote.
- Neither office requires Maintainer status. A person holding both an office and a Maintainer role has both sets of
  duties, but gains no extra vote, casting vote, unilateral veto, merge authority, or GitHub administration access
  from the office. Maintainers do not need an officer's additional approval for routine technical work.

Officers must accept their responsibilities. Review availability and handover needs at least every six months with
the roster. Leaving an office does not automatically end TSC membership or Maintainer status, and leaving the
Maintainer role does not automatically end an office. An officer who leaves the TSC also leaves that office.
Officer appointment and involuntary removal follow the governance threshold and recusal rules below. Consider
each person's appointment separately, and record the office rules separately from individual appointment decisions.

For each appointment, vote, or community activity, the responsible group names a non-conflicted coordinator to
collect feedback and record the result. This can be an officer or another eligible member. An administrator with
the necessary access executes approved permission changes; officers do not grant access by themselves.

## Technical Decisions and Pull Requests

An **authorized reviewer** is a current Maintainer. Everyone may contribute technically qualified reviews.
Count approvals from distinct people who are not authors of the change; a person holding
multiple roles still counts once. Reviewers must understand the affected behavior, and substantive changes after
approval need renewed review.

| Change | Decision and review |
| --- | --- |
| Routine fix, documentation, test, or extension within existing contracts | Relevant checks and at least one independent Maintainer approval. A Maintainer may merge with the necessary access. |
| Significant change to shared Task/DAG or API contracts, install/cleanup behavior, compatibility, security boundaries, or release permissions | A short design issue describing the problem, proposed behavior, compatibility/migration impact, and validation; then two independent technical approvals on the implementation, including at least one authorized reviewer. |
| Project direction, governance, or an unresolved technical dispute | Public discussion and a recorded TSC decision under the rules below. Implementation still follows the applicable technical review requirements. |

For a significant change, the second technical reviewer may be a qualified Contributor without write access. Record
their explicit approval of the implementation in a submitted GitHub Review; an ordinary discussion comment or a bot
review is not an approval.
GitHub's required-approval gate counts eligible repository reviewers and does not enforce this second review on its
own. The merger checks both the native gate and the project policy. Automated and AI reviews supplement human review.

A processor, hook, scheduler, or visualization extension that uses existing interfaces normally follows routine
review. Large mechanical changes do not automatically require a design process; small changes to shared contracts
can require one. A Maintainer records the classification and rationale if it is disputed. Escalate an unresolved
classification dispute to the TSC. A design decision does not replace review of the final code.
For an undisclosed vulnerability, use a restricted design/review record and the validation procedure in
[SECURITY.md](SECURITY.md) instead of a public design issue.

Use pull requests for changes, including changes authored by Maintainers. Before merging, the merger checks the
required reviews, relevant CI results (or recorded private security validation), and resolution of substantive
objections. Do not bypass a failing relevant check or an unresolved objection through silence. GitHub settings
enforce only the configured subset of this policy; the merger remains responsible for the full review requirements.

For an active security incident or critical service/release failure, an authorized Maintainer with the necessary
access may apply the smallest emergency fix or rollback when normal review cannot wait. Record the reason, scope,
executor, and validation in a pull request or restricted incident record, notify another Maintainer, and obtain
independent retrospective review within two business days. If that is not possible, record the reason and have the
non-conflicted TSC members coordinate review and a recovery plan. Restore normal protections promptly. This exception
does not authorize permanent membership or governance changes. Keep security details private until coordinated
disclosure and publish a non-sensitive outcome when safe.

## Appointments and Delegation

Anyone may express interest or nominate a Contributor for Maintainer. The assessment considers sustained
contributions, technical and review judgment, collaboration, and willingness to take responsibility. Maintainers also
need a broad understanding of Dayu's architecture, compatibility, and lifecycle. There is no minimum number of PRs.

1. A current Maintainer coordinates the nomination, obtains the candidate's interest, and records contribution
   evidence, proposed responsibilities, scope, and necessary permissions.
2. Allow at least seven calendar days for feedback from current Maintainers. Technical evidence and role criteria
   are public; sensitive personnel feedback may be shared privately with the non-conflicted reviewers through the
   contacts in [MAINTAINERS.md](MAINTAINERS.md). Keep a restricted decision record rather than assuming a private
   mailing list exists.
3. Appointment normally requires explicit support from at least two eligible current Maintainers and no unresolved
   substantive objection. A candidate cannot approve their own appointment. Resolve concerns through discussion;
   the TSC handles an unresolved appeal or dispute with recorded reasons, using the ordinary formal-decision
   threshold below. Use the continuity procedure only when its eligibility conditions apply.
4. After approval and the candidate's acceptance, announce the appointment and agreed scope. The coordinator updates
   the roster; an administrator grants the approved access and verifies it before enabling merge or code-owner
   responsibilities. Update review routing as described in the [administration guide](docs/development/community-administration.md).
   Give a declined candidate constructive private feedback without publishing sensitive assessments.

TSC membership changes follow the formal governance threshold below, with the candidate recused and their acceptance
required before appointment takes effect. Existing TSC members evaluate stewardship, judgment, and project needs;
there are no institution-reserved seats.

The TSC appoints its Chair and Vice Chair from current TSC members using the same governance threshold. Candidates
recuse themselves from their own appointment or removal; apply the continuity procedure if too few eligible TSC
members remain. A governance PR must identify which decisions are policy changes and which are individual appointments.

## Formal Decisions and Conflicts of Interest

Routine technical decisions use discussion and review. Use a formal TSC vote when consensus cannot resolve an
escalation; governance changes, TSC membership changes, and officer appointments or involuntary removals always
require the governance threshold.

The coordinator records the proposal, eligible voters, recusals, voting deadline, and outcome. Allow at least seven
calendar days for a formal vote. Each eligible person may vote yes, no, or abstain. Fix the eligible membership at
the start: abstention or non-response does not reduce the denominator. If a new conflict or membership change
affects eligibility, restart the vote with the corrected list rather than changing the denominator mid-vote.

| Decision | Required affirmative votes |
| --- | --- |
| Escalated technical or appointment dispute | More than half of all eligible current TSC members, and at least two yes votes. |
| Governance amendment, TSC membership change, or officer appointment/involuntary removal | At least two thirds of all eligible current TSC members, rounded up, and at least two yes votes. |

With two eligible TSC members, both must explicitly agree. A tie or insufficient support leaves the current decision
or policy in place while discussion continues; there is no casting vote. Record the reasoning and vote tally in the
issue or PR, or in a restricted record with a non-sensitive public outcome for confidential matters.

People recuse themselves from their own appointment or removal and from decisions with a direct personal conflict.
Declare conflicts before deliberation; recused people do not vote or handle confidential assessments. Affiliation
with the same institution alone is not a conflict. Count each natural person once, even if they hold multiple roles.

### Continuity and Recusal

If confirmed vacancies or required recusals leave the normal decision group with fewer than two eligible people,
refer only that blocked decision to **all non-conflicted current TSC members and Maintainers**, deduplicated by
person. This joint group uses a seven-day recorded vote and requires at least two thirds of its eligible membership,
rounded up, and at least two yes votes. The decision record must explain why the normal process was unavailable.

This procedure cannot be used to bypass dissent, abstention, or a missed vote. Do not declare someone inactive during
a decision to remove their vote. If fewer than two eligible people remain even in the joint group, defer permanent
decisions and seek independent participation through an agreed recovery process; only the bounded emergency actions
above may proceed immediately.

## Availability, Emeritus Status, and Removal

Maintainers review the roster and access at least every six months, considering reviews, documentation, user support,
and other service as well as code. Members may step back or request Emeritus status at any time. For apparent long-term
inactivity, contact the member privately, allow at least 30 calendar days for a response, and discuss availability or
handover before proposing a role change. Graduation or a change of employer does not automatically end a role.

Voluntary departures and agreed Emeritus transitions are recorded with the member's consent and a handover. Keep
historical contribution credit while removing access that is no longer needed. Returning members use the appointment
process to confirm current scope and access; historical status does not automatically restore permissions.

For an involuntary Maintainer removal, document the grounds, give the member an opportunity to respond,
and seek the same independent Maintainer support required for appointment; disputed cases go to the TSC. TSC removal
requires the governance threshold. The subject is recused, and the continuity procedure applies when necessary.
Conduct cases follow the confidential process in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Administrators may temporarily
suspend compromised access or prevent ongoing harm, recording the action and obtaining independent review within two
business days; a temporary suspension is not a permanent role-removal decision.

## Permissions and Administration

Project roles, GitHub permissions, and review routing are related but distinct. Grant only the access needed for an
accepted responsibility, and verify the actual permissions after a change. Repository administrators manage settings
and access as an operational responsibility, not a higher community rank. GitHub write access is repository-wide.
TSC membership and officer titles do not automatically require write, maintain, or admin access.

[OWNERS](OWNERS) records the repository-wide technical review mapping. All current Maintainers appear in both its
`reviewers` and `approvers` lists; these are automation responsibilities, not additional community ranks.
GitHub Actions checks the mapping against the Maintainer roster and requests review when needed.
[`.github/CODEOWNERS`](.github/CODEOWNERS) is generated from the approver list and connects it to GitHub's native
code-owner review requirement. Approvals are submitted GitHub Reviews; comments such as `/lgtm` or `/approve` and
labels do not authorize a merge. Dayu uses no self-hosted Prow service or automatic merger.

Ownership changes are reviewed under the target branch's existing owners. They take effect after merge and do not
appoint people or grant access. Use the [OWNERS guide](docs/development/owners.md) and
[administration guide](docs/development/community-administration.md) to align the roster, actual access, generated
code owners, and branch rules without leaving authors unable to obtain independent review.

People approve and edit role changes. Automation may check document consistency, request reviews, label pull requests, and propose
dependency updates, but does not appoint members, grant or revoke access, approve its own changes, or merge PRs.

Use [SUPPORT.md](SUPPORT.md) to find public discussion, private security reporting, and conduct-reporting channels.
