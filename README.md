# 🦜️🔗 LangChain Baseten

This repository contains 1 package with Baseten integrations with LangChain:

- [langchain-baseten](https://pypi.org/project/langchain-baseten/)

## Initial Repo Checklist (Remove this section after completing)

Welcome to the LangChain Partner Integration Repository! This checklist will help you get started with your new repository.

After creating your repo from the integration-repo-template, we'll go through how to
set up your new repository in Github.

This setup assumes that the partner package is already split. For those instructions,
see [these docs](https://docs.langchain.com/oss/python/contributing/integrations-langchain).

> [!NOTE]
> Integration packages can be managed in your own Github organization.

Code (auto ecli)

- [x] Fill out the readme above (for folks that follow pypi link)
- [x] Copy package into /libs folder
- [x] Update `"Source Code"` and `repository` under `[project.urls]` in /libs/*/pyproject.toml

Workflow code (auto ecli)

- [x] Populate .github/workflows/_release.yml with `on.workflow_dispatch.inputs.working-directory.default`
- [x] Configure `LIB_DIRS` in .github/scripts/check_diff.py

Workflow code (manual)

- [x] Add secrets as env vars in .github/workflows/_release.yml

Monorepo workflow code (manual)

- [x] Pull in new code location, remove old in .github/workflows/api_doc_build.yml

In github (manual)

- [ ] Add any required integration testing secrets in Github
- [x] Add any required partner collaborators in Github
- [x] "Allow auto-merge" in General Settings (recommended)
- [x] Only "Allow squash merging" in General Settings (recommended)
- [x] Set up ruleset matching CI build (recommended)
    - name: ci build
    - enforcement: active
    - bypass: write
    - target: default branch
    - rules: restrict deletions, require status checks ("CI Success"), block force pushes
- [x] Set up ruleset (recommended)
    - name: require prs
    - enforcement: active
    - bypass: none
    - target: default branch
    - rules: restrict deletions, require a pull request before merging (0 approvals, no boxes), block force pushes

Pypi (manual)

- [ ] Add new repo to test-pypi and pypi trusted publishing

> [!NOTE]
> Tag [@ccurme](https://github.com/ccurme) if you have questions on any step.