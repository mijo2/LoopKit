<!--
Thanks for the PR. Fill in what applies and delete what does not.
The title becomes the commit subject on main (we squash-merge), so write it
like one: imperative, under 72 characters.
-->

## Summary

<!-- What changes, and why. One or two paragraphs. Link the issue if there is one. -->

Closes #

## How was this tested?

<!-- New or updated tests, and anything you checked by hand. If you ran it
against a real provider, say which one (no keys, please). -->

## Checklist

- [ ] `ruff check`, `ruff format --check`, `mypy`, and `pytest` pass locally (or `pre-commit run --all-files`)
- [ ] Tests cover the change and do not call a real API
- [ ] `CHANGELOG.md` has a line under **Unreleased** (user-visible changes only)
- [ ] README / docstrings updated if behaviour, the CLI, or `loopkit.toml` changed
- [ ] No API keys, private documents, or generated files are included
