# AGENTS.md

## 0. Purpose

These rules guide AI coding agents working in this repository.

Primary goals:

- Keep changes safe, focused, and reviewable.
- Protect secrets, data, files, and existing behavior.
- Avoid junk code, unnecessary files, unnecessary dependencies, and token waste.
- Prefer small, verified changes over broad rewrites.
- Never fake test results, logs, benchmarks, or execution.

---

## 1. Operating mode

Use this workflow for coding tasks:

1. Explore only what is needed.
2. Identify relevant files.
3. Plan the smallest safe change.
4. Implement only the requested scope.
5. Verify with the smallest relevant check.
6. Report changes, risks, and verification status.

For complex tasks, separate planning from implementation.

Do not modify code before understanding the relevant structure.

---

## 2. Scope control

- Stay within the user’s requested task.
- Do not change unrelated files.
- Do not rewrite architecture unless explicitly requested.
- Prefer minimal safe edits over large refactors.
- Preserve existing behavior unless the task requires changing it.
- Preserve public APIs, function names, config keys, file paths, CLI commands, and environment variables unless explicitly asked to change them.
- Do not remove existing features, tests, logs, scripts, comments with operational value, or compatibility code without clear reason.
- Do not mix unrelated refactor, formatting, dependency upgrades, and feature work in one change.
- If requirements are ambiguous, choose the safest practical assumption and state it briefly.

---

## 3. Security rules

- Never hardcode secrets, passwords, API keys, tokens, private keys, cookies, credentials, or connection strings.
- Never print, expose, commit, or log secrets or sensitive personal data.
- Use environment variables, config files excluded from Git, or secret managers for credentials.
- Validate and sanitize all external input.
- Treat file paths, URLs, uploads, user input, shell arguments, database queries, and network responses as untrusted.
- Avoid unsafe `eval`, `exec`, dynamic imports, unsafe deserialization, and untrusted shell execution.
- Prevent common vulnerabilities:
  - path traversal
  - command injection
  - SQL injection
  - XSS
  - SSRF
  - insecure file upload
  - insecure temporary files
  - broken authentication
  - broken authorization
  - insecure direct object access
- Do not weaken authentication, authorization, permission checks, encryption, validation, or safety checks unless explicitly requested.
- Do not add telemetry, tracking, analytics, external callbacks, or hidden network calls without explicit approval.
- Do not introduce code that silently sends project data, logs, source code, user data, or environment data to external services.
- Security-related events should be logged safely when relevant:
  - failed login
  - permission change
  - config change
  - token refresh failure
  - suspicious input rejection
  - access denied
- Security logs must not contain secrets or sensitive payloads.

---

## 4. File safety

- Do not create files unless needed for the task.
- Do not scatter helper scripts across random folders.
- Place new files in existing appropriate directories.
- Do not overwrite user data, databases, uploaded files, generated assets, model files, backups, or production configs without explicit approval.
- Do not delete files unless explicitly requested.
- Do not perform broad find-and-replace without explaining the scope.
- Do not reformat unrelated files.
- Do not change line endings, encoding, or formatting across the repo unless required.
- Do not create fake placeholder implementations when real implementations already exist.
- Do not create duplicate modules that compete with existing code.
- Before adding a new file, check whether an existing file should be updated instead.

---

## 5. Code quality

- Write simple, readable, maintainable code.
- Follow the existing project style.
- Prefer clear names over clever names.
- Avoid vague names such as `data`, `tmp`, `foo`, `bar`, `stuff`, `thing`, `obj`, unless the scope is extremely small and obvious.
- Keep functions focused.
- Avoid unnecessary abstraction.
- Avoid duplicate logic.
- Reuse existing utilities where appropriate.
- Do not introduce dead code.
- Do not leave unused imports, unused variables, debug prints, commented-out blocks, or fake TODOs.
- Do not add comments that merely repeat the code.
- Add comments only for non-obvious logic, constraints, trade-offs, safety decisions, or operational warnings.
- Prefer explicit error handling over hidden failure.
- Avoid silent fallback behavior that hides real problems.
- Keep config values centralized and easy to find.
- Do not hardcode paths, ports, thresholds, credentials, or environment-specific values when they should be configurable.

---

## 6. Dependency rules

- Do not add new dependencies unless necessary.
- Prefer the standard library or existing dependencies.
- If adding a dependency, explain:
  - why it is needed
  - where it is used
  - why existing options are insufficient
- Do not upgrade unrelated packages.
- Do not change lock files unless dependency changes require it.
- Avoid large, abandoned, unclear, unnecessary, or risky packages.
- Do not add runtime internet dependency unless explicitly required.
- Avoid packages with known security issues.
- Do not add a dependency only to save a few lines of simple code.

---

## 7. Token and context efficiency

- Keep responses focused on the current task.
- Do not repeat project history unless it affects the current change.
- Read only files needed for the task.
- Summarize large files instead of pasting unchanged content.
- Do not paste huge unchanged files unless the user explicitly asks for full-file output.
- Avoid duplicated explanations.
- Prefer concise checklists and exact commands.
- Do not include irrelevant best-practice essays.
- Do not ask the user to provide information already present in the repo or conversation.
- Do not generate multiple alternative solutions unless comparison is useful.
- Keep instructions short, concrete, and verifiable.

---

## 8. Testing and verification

- Run or propose the smallest relevant verification for the change.
- Prefer targeted tests before full test suites.
- Do not claim tests passed unless they actually ran.
- If tests were not run, clearly say they were not run and why.
- Provide exact commands for the user to verify.
- Do not fake logs, screenshots, benchmark results, terminal output, coverage, or test results.
- Do not remove failing tests to make the build pass.
- Do not weaken assertions to make tests pass.
- Add or update tests when fixing bugs or changing behavior, when practical.
- Verify imports, syntax, startup path, and changed behavior when possible.
- For bug fixes, include the failure cause and the verification that proves the fix.

---

## 9. Error handling

- Handle expected errors explicitly.
- Avoid silent failures.
- Do not swallow exceptions without logging, returning, or surfacing meaningful information.
- Error messages should be useful for debugging.
- User-facing errors must not leak secrets, credentials, internal tokens, or sensitive paths.
- Use timeouts for network calls, subprocesses, external services, file locks, and device operations.
- Include recovery or fallback only when it is safe and obvious.
- Avoid infinite retry loops.
- Use bounded retries with delay/backoff when appropriate.

---

## 10. Logging

- Use consistent logging instead of scattered prints.
- Do not spam logs inside tight loops.
- Do not log secrets, tokens, cookies, private data, raw credentials, or sensitive headers.
- Log enough context to troubleshoot:
  - operation
  - file/module
  - status
  - error type
  - safe identifier
- Use appropriate log levels:
  - debug for detailed diagnostic info
  - info for normal lifecycle events
  - warning for recoverable problems
  - error for failed operations
- Remove temporary debug prints before final output.
- Avoid logging huge payloads.
- Prefer throttled logs for repeated events.

---

## 11. Performance

- Do not introduce unnecessary blocking operations.
- Avoid repeated expensive work inside loops.
- Avoid unbounded memory growth.
- Avoid unbounded queues.
- Avoid infinite loops without sleep, timeout, cancellation, or exit condition.
- Cache or reuse resources when safe.
- Close files, sockets, database connections, subprocesses, and handles correctly.
- Use streaming or chunking for large files when appropriate.
- Explain performance trade-offs when changing algorithms, thresholds, concurrency, batching, or caching.
- Do not reduce correctness or accuracy for speed unless explicitly requested or clearly justified.
- Keep startup time, runtime stability, and resource cleanup in mind.

---

## 12. Concurrency and async safety

- Avoid race conditions when adding threads, async tasks, queues, timers, or background workers.
- Use locks, queues, cancellation flags, or task groups where appropriate.
- Do not share mutable state across workers without protection.
- Avoid fire-and-forget tasks unless failures are handled.
- Ensure background workers can stop cleanly.
- Do not block event loops with long synchronous work.
- Use bounded queues for producer/consumer pipelines.

---

## 13. Data and database safety

- Do not run destructive migrations without explicit approval.
- Do not drop, truncate, overwrite, or mass-update data without explicit approval.
- Back up or provide rollback steps for risky data changes.
- Use parameterized queries.
- Do not build SQL by string-concatenating untrusted input.
- Validate schema assumptions before changing data access code.
- Do not log full records containing sensitive data.

---

## 14. API and network safety

- Use explicit timeouts for HTTP and network calls.
- Handle non-200 responses.
- Handle retries carefully and avoid retry storms.
- Validate response shape before using it.
- Do not assume external services are always available.
- Do not add new external calls without clear need.
- Avoid leaking internal errors to API clients.
- Preserve backward-compatible response formats unless a breaking change is required.

---

## 15. Documentation

- Update documentation only when behavior, setup, commands, config, or usage changes.
- Keep docs short, accurate, and current.
- Do not document features that do not exist.
- Include copy-ready commands when useful.
- Do not add long theory sections unless requested.
- Prefer examples that match the actual repository.
- Mention limitations and known risks when relevant.

---

## 16. Git and review hygiene

- Keep changes focused and reviewable.
- Do not mix unrelated changes.
- Do not hide breaking changes.
- Explain what changed and why.
- Mention verification performed.
- Mention verification not performed.
- Mention risks or follow-up work.
- Do not fabricate commit history, PR status, CI results, or reviewer approval.

---

## 17. Prohibited behaviors

Do not:

- hardcode secrets
- fake test results
- fake logs
- fake benchmark numbers
- claim execution without execution
- delete user data without approval
- modify unrelated files
- add unnecessary dependencies
- create duplicate competing implementations
- hide errors silently
- weaken security checks
- introduce hidden telemetry
- generate large irrelevant explanations
- rewrite architecture for a small fix
- remove tests to pass builds
- use placeholder code as final code
- leave debug junk behind

---

## 18. Good rule examples

Prefer concrete rules:

- Never hardcode secrets; use environment variables.
- Do not modify unrelated files.
- Do not claim tests passed unless they actually ran.
- Use timeouts for external calls.
- Avoid logging sensitive data.
- Prefer existing dependencies before adding new ones.
- Keep changes small and reviewable.

Avoid vague rules:

- Write perfect code.
- Use best practices.
- Be careful.
- Make it production ready.
- Optimize everything.
- Refactor as needed.
- Read the entire repository every time.
- Always explain everything in detail.

---

## 19. Output format for coding tasks

When making or proposing code changes, respond with:

1. Summary
2. Files changed
3. Key decisions
4. Verification
5. Risks or notes

For each changed file, include the exact path.

If code was not executed, say so clearly.

If tests were not run, say so clearly and provide the command to run them.

If the change is risky, explain the risk before or beside the implementation.
