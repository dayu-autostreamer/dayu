# Before you get started

## Code of Conduct

Please make sure to read and observe our [Code of Conduct](CODE_OF_CONDUCT.md).

# Getting started

- Fork the [repository](https://github.com/dayu-autostreamer/dayu/) on GitHub
- Read the [quick start](https://dayu-autostreamer.github.io/docs/getting-started/) for deployment.
- Read the [Developer Guide](https://dayu-autostreamer.github.io/docs/developer-guide/) for development guide.


# Your First Contribution

We will help you to contribute in different areas like filing issues, developing features, fixing critical bugs and getting your work reviewed and merged.

If you have questions about the development process, feel free to [contact us](README.md#contact).

## Find something to work on

We are always in need of help, be it fixing documentation, reporting bugs or writing some code.
Look at places where you feel best coding practices aren't followed, code refactoring is needed or tests are missing.
Here is how you get started.

### Find a good first topic

Dayu system focused on cloud-edge collaborative stream data analysis, and is flexible to develop based on hook functions.

You can either expand the core functions of the whole system or expand the applicable research topics. 

Another good way to contribute is to find a documentation improvement, such as a missing/broken link. Please see [Contributing](#contributor-workflow) below for the workflow.

#### Work on an issue

When you are willing to take on an issue, you can assign it to yourself. Just reply with `/assign` or `/assign @yourself` on an issue,
then the robot will assign the issue to you and your name will present at `Assignees` list.

### File an Issue

While we encourage everyone to contribute code, it is also appreciated when someone reports an issue.
Issues should be filed under the [dayu repository](https://github.com/dayu-autostreamer/dayu/issues).

Please follow the prompted submission guidelines while opening an issue.

# Contributor Workflow

Please do not ever hesitate to ask a question or send a pull request.

This is a rough outline of what a contributor's workflow looks like:

- Create a topic branch from where to base the contribution. This is usually master.
- Make commits of logical units.
- Make sure commit messages are in the proper format (see below).
- Push changes in a topic branch to a personal fork of the repository.
- Submit a pull request to [dayu-autostreamer/dayu](https://github.com/dayu-autostreamer/dayu).
- The PR must receive an approval from two maintainers.

## Creating Pull Requests

Pull requests are often called simply "PR".
Dayu generally follows the standard [GitHub pull request](https://help.github.com/articles/about-pull-requests/) process.
To submit a proposed change, please develop the code/fix and add new test cases.


## Code Review

To make it easier for your PR to receive reviews, consider the reviewers will need you to:

* follow [good coding guidelines](https://pep8.org/) for code formatting.
* write [good commit messages](https://chris.beams.io/posts/git-commit/).
* break large changes into a logical series of smaller patches which individually make easily understandable changes, and in aggregate solve a broader issue.

### Format of the commit message

We follow a rough convention for commit messages that is designed to answer two questions: what changed and why.
The subject line should feature the what and the body of the commit should describe the why.

```
scripts: add test codes for metamanager

this add some unit test codes to improve code coverage for metamanager

Fixes #12
```

The format can be described more formally as follows:

```
<subsystem>: <what changed>
<BLANK LINE>
<why this change was made>
<BLANK LINE>
<footer>
```

The first line is the subject and should be no longer than 70 characters, the second line is always blank, and other lines should be wrapped at 80 characters. This allows the message to be easier to read on GitHub as well as in various git tools.

Note: if your pull request isn't getting enough attention, you can use the reach out to get help finding reviewers.
