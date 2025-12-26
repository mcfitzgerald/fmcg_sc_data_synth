# PRIME DIRECTIVE

## "Specflow" Spec-Driven Development

We are using intent and roadmap spec-driven development as desribed by the specflow standard: https://www.specflow.com

Key reference documents are located in `./docs/planning/`

Always read `./docs/planning/intent.md` and `./docs/planning/roadmap.md`

### Note

We have included the code base from a previous simulation in `reference/fmcg_example_OLD/` that used stochastic techniques and was not properly implemented via physics, but has potentially useful code and should be consulted when needed, BUT WE SHOULD SCRUTINIZE BEFORE LIFTING OR SHIFTING ANYTHING OVER

## Coding and Engineering Standards

Build modularly, separating concerns and first-principle components for reusabiltiy and extensibility

When running validations and tests, if there is a bug or something seems of, don't just skip or assume it's fine. Alert me and we will determine course of action, which typicall means we must find root cause.

Do not hardcode variables or values. Always use a config paradigm (a config file or files so variables can be easily located and changed)

Use `ruff` and `mypy` for linting, formatting, and typing

Use `semgrep` to find hardcodes

Employ a judicious but robust testing strategy, and prefer integration tests versus unit tests unless unit test is critical

Always use context7 when I need code generation, setup or configuration steps, or
library/API documentation. This means you should automatically use the Context7 MCP
tools to resolve library id and get library docs without me having to explicitly ask.

Don't reinvent the wheel. Search web for robust libraries that are fit for the purpse, and always opt for simple (but complete and correct) implementation vs complex. Don't over-engineer.

Commit and push changes with git after new code, code changes, bug fixes, new features. 

Update `CHANGELOG.md`, `README.md`, `docs/`, and `pyproject.toml` when committing changes. Use semantic versioning

Unless noted otherwise, do not plan or code for backwards compatibility

## Python environment and dependencies managed by poetry

Always use poetry to run python for this project
