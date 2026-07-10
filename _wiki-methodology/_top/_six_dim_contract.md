# Six-Dim Contract - <PROJECT_NAME>

> Purpose: keep the six dimensions thick enough to be useful, but prevent them from competing for the same source of truth.

## Core Rules

- One fact type has one owner; other columns may link or derive, never duplicate body text.
- Current phase, next action, blockers and pause list only live in `_wiki-methodology/_top/_task_plan.md`.
- Code facts live in `_codemap.md`, subproject codemaps and `_meta/_registry.yml`.
- Reusable tests and coverage gaps live in `_wiki-methodology/_tests/`.
- Data/model/sample assets live in `_meta/_data_registry.yml`.
- File lifecycle rules live in `_meta/_naming.md`, `_sandbox/`, `_tmp/` and `_legacy/`.
- Whole Picture is a derived user-facing view, generated only when the user explicitly asks for webpage/HTML.
- Cross-column refs use owner-native keys such as `code:<id>`, `data:<id>`, `test:<id>`, `finding:<id>`; reverse refs are derived, not hand-maintained.

## Boundary Table

| Dimension | Owns | Does Not Own |
|---|---|---|
| COL1 Wiki | Stable knowledge, methodology, entities, source notes | Current task queue |
| COL2 Plan/TOP | Current plan, roadmap, decisions, findings, logs | Code facts, data asset inventory |
| COL3 Tests | Test map, reusable commands, coverage gaps, run ledger | Product backlog unrelated to test coverage |
| COL4 Codemap/Registry | Code domains, entrypoints, tool/script registry, call boundaries | Current phase or next action |
| COL5 File/Data Lifecycle | Naming, archive/tmp/sandbox rules, non-code asset registry | Code-domain registry |
| COL6 Whole Picture | Derived HTML/user overview | Hand-maintained current state |

## Validation

```bash
python3 ~/.claude/skills/share-top/scripts/top-lint.py .
python3 ~/.claude/skills/share-top/scripts/topic-doctor.py . --links
```

Add project-specific YAML/path/current-state drift checks here once the project has real registries and tests.
