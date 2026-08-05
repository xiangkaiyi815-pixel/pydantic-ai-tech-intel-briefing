# Contributing

Thanks for improving the assistant.

1. Open an issue or describe the problem and its evidence.
2. Create a focused branch.
3. Add or update tests for behavior changes.
4. Run `py -3.12 -m pytest -q`.
5. Keep credentials, local databases, generated reports, cookies, and private
   content out of commits.
6. Preserve original source URLs and state public-access limitations honestly.

Changes to platform retrieval must document whether a route is public,
login-free, read-only, and stable enough for the claimed coverage. Do not add
browser automation, credential injection, or write-capable MCP tools to the
default retrieval path.
