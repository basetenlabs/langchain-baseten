# 🦜️🔗 LangChain Baseten

This repository contains 1 package with Baseten integrations with LangChain:

- [langchain-baseten](https://pypi.org/project/langchain-baseten/)

## Initial Repo Checklist (Remove this section after completing)

Welcome to the LangChain Partner Integration Repository! This checklist will help you get started with your new repository.

After creating your repo from the integration-repo-template, we'll go through the following steps:

1. Setting up your new repository in GitHub
2. 

This setup assumes that the partner package is already split. For those instructions,
see [these docs](https://python.langchain.com/docs/contributing/integrations#partner-packages).

Code (auto ecli)

- [x] Fill out the readme above (for folks that follow pypi link)
- [x] Copy package into /libs folder
- [x] Update `"Source Code"` and `repository` under `[project.urls]` in /libs/*/pyproject.toml (needs changing if we make langchain-ai the parent org)

Workflow code (auto ecli)

- [x] Populate .github/workflows/_release.yml with `on.workflow_dispatch.inputs.working-directory.default`
- [x] Configure `LIB_DIRS` in .github/scripts/check_diff.py

Workflow code (manual)

- [x] Add secrets as env vars in .github/workflows/_release.yml

Monorepo workflow code (manual)

- [?] Pull in new code location, remove old in .github/workflows/api_doc_build.yml (couldn't find relevant lines in monorepo)

In github (manual)

- [ ] Add integration testing secrets in Github (ask Chester for help)
- [ ] Add partner collaborators in Github (ask Chester for help)
- [ ] "Allow auto-merge" in General Settings 
- [ ] Only "Allow squash merging" in General Settings
- [ ] Set up ruleset matching CI build (ask Chester for help)
    - name: ci build
    - enforcement: active
    - bypass: write
    - target: default branch
    - rules: restrict deletions, require status checks ("CI Success"), block force pushes
- [ ] Set up ruleset
    - name: require prs
    - enforcement: active
    - bypass: none
    - target: default branch
    - rules: restrict deletions, require a pull request before merging (0 approvals, no boxes), block force pushes

Pypi (manual)

- [ ] Add new repo to test-pypi and pypi trusted publishing (ask Chester for help)

Slack

- [ ] Set up release alerting in Slack (ask Chester for help)

release:
/github subscribe langchain-ai/langchain-{partner_lower} releases workflows:{name:"release"}
/github unsubscribe langchain-ai/langchain-{partner_lower} issues pulls commits deployments
