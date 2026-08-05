# Security Policy

## Supported Use

The repository is intended for local or operator-controlled deployments. Treat
model-provider credentials, Feishu credentials, and any third-party search
credential as secrets.

## Reporting A Vulnerability

Do not post secrets or exploit details in a public issue. Use the repository
owner's GitHub security contact or a private contact channel and include:

- a concise impact description,
- affected file or command,
- minimal reproduction steps,
- whether any credential, write action, or private data is involved.

## Security Boundaries

- `.env.local`, `.local-data/`, SQLite databases, generated reports, and test
  output are ignored by Git.
- Default MCP bindings are read-only retrieval tools.
- Public-platform retrieval does not use cookies, browser profiles, or login
  state.
- Login pages and search-result wrappers are filtered from daily briefing
  evidence.

Before publishing a fork, run a secret scan and verify that no local `.env`
file or database is tracked.
