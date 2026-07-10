# _refs/_literature/ — Literature Acquisition Assets

This directory stores literature assets acquired by the academic-search workflow.

## Layout

| Directory | Contents |
|---|---|
| `_search/` | Scopus/CrossRef/Semantic Scholar search payloads and tabular exports |
| `_markdown/` | AI-friendly full text or abstract Markdown; primary reading asset |
| `_pdf/` | PDF fallback only when explicitly requested |
| `_bib/` | BibTeX/RIS/reference exports |
| `_meta/` | quota logs, acquisition summaries, source/permission traces |
| `_tmp/` | transient workflow state |

## Policy

- Prefer structured XML/JATS/HTML/LaTeX converted to Markdown.
- Do not download or preserve PDFs by default.
- Keep API credentials outside the project. This folder stores only results and provenance.
- Promote digested methodology notes to `_wiki-methodology/_wiki/`; promote cited references to `_paper/bib/`.
