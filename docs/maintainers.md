# Maintainer notes

How the automation in this repository fits together, what to do on release day,
and the things deliberately left for later. Contributor-facing instructions
belong in `CONTRIBUTING.md`; this file is for whoever has admin on the repo.

## The checks, and where each one is configured

Every check runs the same way locally and in CI. The command is the source of
truth; the workflow only calls it.

| Check | Command | Configured in |
| --- | --- | --- |
| Lint | `ruff check src tests examples` | `[tool.ruff.lint]` in `pyproject.toml` |
| Formatting | `ruff format --check src tests examples` | `[tool.ruff]` in `pyproject.toml` |
| Types | `mypy` | `[tool.mypy]` in `pyproject.toml` (strict, covers `src` and `tests`) |
| Tests + coverage | `pytest --cov` | `[tool.pytest.ini_options]`, `[tool.coverage.*]` in `pyproject.toml` |
| Package builds | `python -m build && twine check --strict dist/*` | `[build-system]`, `[project]` in `pyproject.toml` |
| Citation file | `cffconvert --validate` | `CITATION.cff` |

`make check` runs all of them (`make help` lists the rest; the Makefile creates
`.venv/` on first use). Without make: `pip install -e ".[dev]"`, and
`pre-commit install` wires the same checks into `git commit` (see
`.pre-commit-config.yaml`).

### Coverage threshold

`fail_under` in `[tool.coverage.report]` is the only place the minimum lives.
It is set very low (5%) on purpose: a floor that stops the suite from silently
becoming a no-op, not a target. Actual coverage is around 85%, so raising the
floor to 80 would already pass; do that once the test suite has settled.

CI runs `pytest --cov --cov-report=xml` and stores `coverage.xml` as a run
artifact (Actions tab, open a run, "Artifacts"). No external service is
involved.

### Version pins that must move together

- `ruff` and `mypy`: the ranges in `[project.optional-dependencies].dev` and the
  `rev:` lines in `.pre-commit-config.yaml`. New ruff minors add rules, so a
  mismatch means "passes locally, fails in CI" or vice versa. Bump both in one
  commit (`pre-commit autoupdate` updates the hook side).
- `version` in `pyproject.toml`, the git tag, `version` / `date-released` in
  `CITATION.cff`, and the release heading + compare links in `CHANGELOG.md`.
  The release workflow refuses a tag that does not match `pyproject.toml`.

## How the workflows work

A workflow is a YAML file under `.github/workflows/`. GitHub reads every file
there on every event in the repository and runs the ones whose `on:` block
matches. Nothing else registers them; adding the file is enough.

Reading `ci.yml` top to bottom:

1. **`on:`** decides *when* it runs. `push` to `main`, any `pull_request`, and
   `workflow_dispatch` (a "Run workflow" button in the Actions tab). A PR from a
   fork also triggers `pull_request`, but with a read-only token and, for
   first-time contributors, only after a maintainer approves the run.
2. **`permissions:`** sets what the automatically-minted `GITHUB_TOKEN` may do.
   `contents: read` at the top means every job is read-only unless it asks for
   more itself (the release workflow's jobs do, one at a time).
3. **`concurrency:`** groups runs by workflow + branch; a new push cancels the
   previous run of the same PR instead of queueing behind it.
4. **`jobs:`** is a set of independent jobs. Each job gets a fresh virtual
   machine (`runs-on`), so nothing is shared between jobs except what you upload
   with `actions/upload-artifact` and download again. Jobs run in parallel
   unless they declare `needs:`.
5. Inside a job, **`steps:`** run in order on that machine. A step is either
   `uses:` (a reusable action, e.g. `actions/checkout` clones the repo,
   `actions/setup-python` installs an interpreter) or `run:` (a shell command).
   The first failing step fails the job.
6. **`strategy.matrix`** clones a job once per combination. The `test` job is
   five machines: three Pythons on Ubuntu plus one Python each on macOS and
   Windows. `fail-fast: false` lets them all finish so one red cell does not
   hide another.
7. **`${{ ... }}`** is the expression language: `matrix.python-version`,
   `env.PYTHON_DEFAULT`, `needs.lint.result`, `github.ref_name`, and so on, are
   substituted before the step runs.
8. The last job, **`ci`**, has `needs: [lint, typecheck, test, build]` and
   `if: always()`, so it runs after the others whatever happened and then fails
   if any of them did. Branch protection should require this one check, named
   "CI passed". The matrix can then change freely without touching repository
   settings.

`release.yml` follows the same pattern with a different trigger: a pushed tag
matching `v1.2.3`. Its jobs chain with `needs:` so that nothing is published
unless the build verified, and the GitHub Release is created only after PyPI
accepted the upload.

`dependabot.yml` is not a workflow. It is read by GitHub's Dependabot service,
which opens pull requests when a dependency or an action has a newer version.
Those PRs run `ci.yml` like any other.

### What a "stale bot" is, and why there is none

`actions/stale` is a scheduled workflow that comments on issues and PRs with no
activity for N days and closes them after N more. It keeps the issue list
short on large projects, at the cost of closing things that were merely
waiting on a maintainer. It is not worth the friction at this project's size;
revisit if open issues ever pass a few dozen.

## Cutting a release

1. Update `version` in `pyproject.toml`, and `version` + `date-released` in
   `CITATION.cff`. Add the release section to `CHANGELOG.md`.
2. Merge that to `main` and wait for CI to pass.
3. `git tag v0.2.0 && git push origin v0.2.0`.
4. The `Release` workflow builds, then pauses at the `pypi` environment for
   your approval (Actions tab → the run → "Review deployments").
5. After approval it publishes to PyPI using Trusted Publishing (OIDC; there is
   no token stored in the repository) and creates the GitHub Release with
   auto-generated notes from merged PR titles.

### One-time setup before the first release

Items 5 (the `pypi` environment) and 6 (PyPI Trusted Publisher) of the
settings checklist below. Nothing else is needed; there are no tokens to store.

## Repository settings checklist (admin, one-time)

None of these are files; they are clicks in **Settings**. Do them in this order:
the later items (branch protection, environments) reference things the earlier
ones create. Menu names are GitHub's as of writing and drift occasionally.

### 1. Before the first push

- [ ] Replace every `OWNER` and `CONDUCT_EMAIL` placeholder
      (`grep -rn "OWNER\|CONDUCT_EMAIL" . --exclude-dir=.git`).
- [ ] Default branch is `main` (`git branch -m main` if it is still `master`;
      `ci.yml`, the CHANGELOG links, and `pyproject.toml` URLs assume `main`).
- [ ] History is clean: `git log -p | grep -iE "api[_-]?key|sk-[A-Za-z0-9]{10}"`
      returns nothing you would not publish. Squash to a single commit if in doubt.
- [ ] Create the repository as **public**, *without* letting GitHub add a
      README/.gitignore/LICENSE (this repo has them). Push.

### 2. General (Settings → General)

- [ ] **Description** and **Website** (PyPI page once published) at the top of
      the repo page; add **Topics**: `llm`, `rag`, `python`, `knowledge-base`,
      `openai`, `anthropic`, `xai`, `grounded-generation`.
- [ ] **Features**: enable *Discussions*; disable *Wikis* and *Projects*
      (nothing links to them; fewer empty tabs). Keep *Issues* on.
- [ ] **Discussions**: create the **Q&A** category (the "Announcements" and
      "Ideas" defaults are fine). `SUPPORT.md` and the issue chooser link to
      `discussions/categories/q-a`.
- [ ] **Pull Requests**: allow *squash merging* only; default commit message =
      *Pull request title and description*. Tick *Always suggest updating pull
      request branches* and *Automatically delete head branches*.
- [ ] **Issues → Labels**: create `needs-triage`, `dependencies`,
      `github-actions`, `python`. Keep the defaults `bug`, `enhancement`,
      `documentation`, `good first issue`, `help wanted`.

### 3. Actions (Settings → Actions → General)

- [ ] **Actions permissions**: *Allow OWNER, and select non-OWNER, actions and
      reusable workflows* → allow *Actions created by GitHub* and *Marketplace
      verified creators*. (Everything in `.github/workflows/` qualifies:
      `actions/*`, `pypa/gh-action-pypi-publish`.)
- [ ] **Fork pull request workflows from outside collaborators**: *Require
      approval for first-time contributors* (the default; confirm).
- [ ] **Workflow permissions**: *Read repository contents and packages
      permissions*. Leave *Allow GitHub Actions to create and approve pull
      requests* **off**. (The workflows request `contents: write` /
      `id-token: write` per job where needed.)

### 4. Code security (Settings → Code security)

- [ ] **Private vulnerability reporting**: enable. `SECURITY.md` points here.
- [ ] **Dependency graph**: on (default for public repos).
- [ ] **Dependabot alerts** and **Dependabot security updates**: enable.
      (`dependabot.yml` handles *version* updates; these handle CVEs.)
- [ ] **Grouped security updates**: enable, so one PR covers related bumps.
- [ ] **Code scanning → CodeQL analysis**: *Default setup*, languages: Python.
      Do **not** add a `codeql.yml` workflow as well; GitHub refuses to run both.
- [ ] **Secret scanning** and **Push protection**: enable. Push protection
      blocks a push that contains a recognised provider key — useful for a
      project whose users set `XAI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

### 5. Environments (Settings → Environments)

- [ ] **New environment** named exactly `pypi` (must match `release.yml`).
- [ ] **Deployment protection rules**: *Required reviewers* → add yourself.
      The release workflow will pause here and wait for approval.
- [ ] **Deployment branches and tags**: *Selected branches and tags* → add
      tag rule `v*`. Only tagged releases can use the environment.
- [ ] No secrets needed: publishing uses OIDC Trusted Publishing.

### 6. PyPI (outside GitHub)

- [ ] Create a PyPI account with 2FA; confirm `loop-kit` is free.
- [ ] **Your projects → Publishing → Add a new pending publisher**:
      PyPI project name `loop-kit`, owner `OWNER`, repository `loop-kit`,
      workflow `release.yml`, environment `pypi`. The first tag push creates
      the project.
- [ ] (Optional) do the same on test.pypi.org and run one dry release there
      first by pointing a copy of the publish job at `repository-url:
      https://test.pypi.org/legacy/`.

### 7. Branch protection (Settings → Rules → Rulesets → New branch ruleset)

Do this **after** the first CI run on `main`, so the check name exists.

- [ ] Name `main`, enforcement *Active*, target *Default branch*.
- [ ] Rules: *Restrict deletions*, *Block force pushes*, *Require linear
      history* (squash-merge already guarantees this).
- [ ] *Require a pull request before merging*: 1 approval is unrealistic for a
      solo maintainer — set **0** required approvals but keep *Dismiss stale
      approvals* and *Require conversation resolution before merging*. Raise to
      1 when there is a second maintainer.
- [ ] *Require status checks to pass*: add exactly **`CI passed`** (the
      aggregator job) and tick *Require branches to be up to date before
      merging*. Do not add the matrix jobs individually.
- [ ] Leave *Bypass list* empty. Yes, this applies to you too; it is the point.
- [ ] (Optional) a second ruleset targeting tags `v*`: *Restrict creations* to
      repository admins, so only you can cut a release.

### 8. Collaborators and moderation (Settings → Collaborators; Moderation options)

- [ ] Nothing to add for a solo project. When a second maintainer joins, give
      *Maintain*, not *Admin*, unless they need settings access; update
      `.github/CODEOWNERS`.
- [ ] **Moderation options → Interaction limits**: leave off; know it is there
      for the day a thread goes bad.
- [ ] **Code review limits**: *Limit to users explicitly granted read or
      higher access* keeps drive-by "approvals" from counting.

### 9. After the first push (repo page, not Settings)

- [ ] **Insights → Community Standards** shows 100 % (README, CoC,
      Contributing, License, Security policy, Issue templates, PR template).
- [ ] **About** sidebar (gear icon): tick *Releases* and *Packages* off until
      they exist; keep *Discussions* visible.
- [ ] **Pin** the first Discussion ("Roadmap and scope") and seed 3–5 issues
      labelled `good first issue`.
- [ ] Enable **Watching → All activity** for yourself so Dependabot PRs and
      first issues are not missed.

## Deferred, with how to pick them up

- **Codecov (or Coveralls)**: free for public repositories, gives PR comments
  with per-line coverage diff and a badge. To adopt: sign in with GitHub, add
  the repository, then in `ci.yml` after the pytest step add
  `uses: codecov/codecov-action@v5` with `files: coverage.xml` (public repos
  need no token for uploads from GitHub-hosted runners, but adding the
  `CODECOV_TOKEN` secret is recommended for reliability). Move the threshold
  from `fail_under` to Codecov's `codecov.yml` `coverage.status.project.target`
  if you want the gate enforced by the service rather than pytest.
- **pre-commit.ci**: runs the hooks on every PR and pushes fixes back to the
  branch; also auto-updates hook revs weekly. Free for public repos; enable by
  installing the GitHub app. Until then, `pre-commit` is local-only and CI is
  the gate.
- **Raise the coverage floor** to something that reflects the suite (80 today).
- **setuptools-scm / hatch-vcs** so the version comes from the tag and the
  "tag matches pyproject" step in `release.yml` becomes unnecessary.
- **Stale bot** (see above), **labeler**, **release-drafter**: process tooling
  for when volume justifies it.
