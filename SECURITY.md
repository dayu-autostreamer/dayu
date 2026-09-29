# Security

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.** Instead, use one of the following methods:

- **Email**: [whzhou@smail.nju.edu.cn](mailto:whzhou@smail.nju.edu.cn) (include "[SECURITY]" in the subject line)
- **Alternative contact**: [shyshy@smail.nju.edu.cn](mailto:shyshy@smail.nju.edu.cn) (Haoyang Su, Maintainer)
- **Private Advisory**: [GitHub Security Advisory](https://github.com/dayu-autostreamer/dayu/security/advisories/new) (for GitHub users)

If the report involves one of the contacts, use the other contact and explain the conflict privately. For other
questions, use [SUPPORT.md](SUPPORT.md).

**Include in your report**:  
- Detailed description of the vulnerability
- Steps to reproduce
- Potential impact
- Affected versions (if known)

We will acknowledge your report within **3 business days** and provide a timeline for resolution.

## Response Responsibility

A non-conflicted [Maintainer](MAINTAINERS.md#maintainers) coordinates triage, private technical review, release work,
and communication with the reporter, involving additional qualified contributors only as needed. The listed reporting
contacts route reports to that coordinator; there is no separate standing security-team appointment implied here.
If both listed Maintainers are involved in the report, contact a non-conflicted TSC member using the roster to
arrange independent technical handling. Do not use a repository advisory if its viewers include a conflicted person.
Keep access to the report and unreleased fix limited to the people handling it, and respect requests for anonymity.

## Security Update Process

1. **Confirmation**:  
   The coordinating Maintainer and qualified reviewers will verify the vulnerability and affected versions.
2. **Patch Development**:  
   Develop and review the fix in a [temporary private fork](https://docs.github.com/en/code-security/tutorials/fix-reported-vulnerabilities/collaborate-in-a-fork)
   associated with a GitHub security advisory, or another
   access-restricted repository. A branch in a public repository is not private. Apply the technical review rules in
   [GOVERNANCE.md](GOVERNANCE.md#technical-decisions-and-pull-requests), keeping design and review records private
   until disclosure; use its bounded emergency procedure only when normal review cannot wait.
   GitHub does not run status checks in temporary private forks or enforce the target branch's protection rules
   when merging through an advisory. Run the relevant checks in a trusted private environment, keep their results
   with the restricted review record, and have the release coordinator verify the required independent approvals
   and validation before merging. Do not expose an unreleased fix through public CI logs or artifacts.
3. **Release**:  
   Patches are released within **7 days** of confirmation via:  
   - [GitHub Releases](https://github.com/dayu-autostreamer/dayu/releases)
4. **CVE Assignment**:  
   Critical vulnerabilities will receive a CVE identifier (if applicable).

## Supported Versions

Dayu currently commits to supporting the n-1 version minor version of the current major release;
as well as the last minor version of the previous major release.


## Disclosure Policy

- **Coordinated Disclosure**:  
  Vulnerabilities are disclosed publicly **after** a patch is released.
- **Timeline Transparency**:  
  Major vulnerabilities will have a public timeline in [GitHub Security Advisories](https://github.com/dayu-autostreamer/dayu/security/advisories).


## Acknowledgments

We credit security researchers who follow responsible disclosure practices. If you wish to be acknowledged, please specify your preference (name/handle or anonymous).
