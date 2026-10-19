# Security policy

## Reporting a vulnerability

**Please do not report security problems through public issues, discussions,
or pull requests.**

Use GitHub's private reporting form instead:
<https://github.com/OWNER/loop-kit/security/advisories/new>
(the "Report a vulnerability" button on the Security tab). It creates a
private thread that only the maintainers can see. If you cannot use that form,
email CONDUCT_EMAIL with "loop-kit security" in the subject.

Include what you can of:

- the version (`pip show loop-kit` or `loopkit --version`) and provider in use
- steps or a minimal script that reproduces the problem
- what an attacker gains, as you understand it

You will get an acknowledgement within 5 working days. We will agree a
disclosure timeline with you; the default is to publish a fix and an advisory
together, and to credit you unless you prefer otherwise.

## Supported versions

loop-kit is pre-1.0. Fixes land on `main` and ship in the next release; older
releases are not patched separately.

| Version | Supported |
| --- | --- |
| latest `0.x` release | yes |
| anything older | no — please upgrade |

## What is in scope

Things we consider security bugs in loop-kit itself:

- API keys or knowledge-base contents being sent anywhere other than the
  configured provider, or being written to disk, logs, or the HTTP response
- `loopkit serve` accepting a request it should reject (for example the body
  size limit, or an unexpected error leaking internals in the response)
- a document or a task being able to alter the *instructions* the checker
  enforces, beyond what the README describes as expected behaviour
- path handling in knowledge loading that reads files outside the given paths

## What is not a loop-kit vulnerability

- **Model output quality.** The loop checks grounding and instructions with a
  model; a model can still be wrong. Treat answers as drafts, as the README says.
- **`loopkit serve` exposed to a network.** It binds `127.0.0.1`, has no
  authentication, and is documented as a local try-it tool. Putting it behind
  a public interface is not a supported configuration.
- **Provider-side behaviour** (rate limits, data retention, outages). Report
  those to the provider.
- **Vulnerabilities in dependencies** that loop-kit does not trigger. Dependabot
  alerts cover these; a PR bumping the dependency is welcome.

## Design notes relevant to security

- Keys are read only from the environment or the `api_key=` argument, never
  from `loopkit.toml`, so a project file is safe to commit.
- Stored prompts are disabled for xAI and OpenAI requests.
- Retrieved passages are treated as evidence. Text in a document that tries to
  issue instructions is not followed, and a citation to a passage id that was
  not provided fails the check.
