# Contribution guide

Follow the rules below to contribute.

## Commits and branches

### Prefixes

| Prefix | Description |
|--|--|
| project | Changes in the project's structure |
| package | Changes in the requirements |
| config | Changes in the configuration |
| data | Changes in the data |
| feature | Creation of new features |
| fix | Fixing bugs |
| refactor | Refactoring code |
| style | Changes in code-style |
| research | Research and experimental tasks |
| report | Creation of reports |
| test | Creation of tests |
| docs | Creation of technical or user documentation |

### Commit messages
A commit message must follow the format below:

> prefix(context)!: short description

:warning: No commas, no periods.

:warning: Start with imperative verb.

:warning: Symbol ! is for breaking changes only.

:warning: No more than 50 characters.

:warning: Only lowercase letters.

:warning: Description is not required.

Examples:

> research(notebooks): create data pipeline

> docs(readme): add project description

### Branch names
A branch name must follow the format below:

> prefix-short-description

:warning: No more than 4 words.

:warning: Description is not required.

Examples:

> report-models-performance

> research-exploratory-analysis
