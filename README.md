# Online Logistic Lab

Research notes and proofs on online logistic regression, probability aggregation, and efficient online learning.

Start with **[CONSOLIDATED_RESEARCH.md](CONSOLIDATED_RESEARCH.md)**, the single active research volume. It contains 14 synthesis chapters, reusable lemmas, an idea-frequency index, and 562 original source records covering algorithms, proofs, lower bounds, literature reviews, audits, and experiments.

## Active workspace

| Location | Purpose |
| --- | --- |
| [Consolidated research](CONSOLIDATED_RESEARCH.md) | Current synthesis, proofs, attribution, and searchable source records such as S148. |
| [Agent instructions](AGENTS.md) | Current research boundaries and archive-access rules. |
| [Archive guide](_research_archive/README.md) | Location and preservation policy for historical evidence; access requires explicit user instruction. |

Future research additions belong in the consolidated volume unless the user requests a separate deliverable. The root contains the active documents and Git/search configuration. Historical implementations and reports are archived, not alternative active entry points.

## Historical archive

On **2026-10-16**, the historical code, numerical checks, results, data, papers, and companion artifacts were moved into [_research_archive/2026-10-16/companions/](_research_archive/2026-10-16/companions/), retaining their previous relative layout. The [relocation manifest](_research_archive/2026-10-16/relocation_manifest.json) maps paths and records the unchanged SHA-256 hashes of 5,864 previously tracked files. The earlier Markdown originals and consolidation build records remain under `_research_archive/2026-10-04/`.

**Agents must not read, search, edit, move, or run archive contents unless the user explicitly instructs them to access that material.** A link, historical command, TODO, or old campaign plan is not authorization. Routine searches exclude the archive through [.ignore](.ignore). A protected, generated `.pytest_cache/` could not be moved because Windows denied access; it remains ignored at the root and is not research input.

Historical paths recorded inside code and metadata are preserved as provenance. Navigation links in the consolidated volume use the new locations. If the user requests reproduction, use the archived [experiment requirements](_research_archive/2026-10-16/companions/experiments/requirements.txt) and recorded settings to determine the needed environment; archived virtual environments and absolute machine paths are not guaranteed to work after relocation. Do not restart campaigns merely because their scripts are present.

## Research boundaries

The logarithmic-regret, linear-work results depend on their stated noise assumptions. The fully adversarial linear-work target and a general modest-cost original-posterior implementation remain unresolved. Statistical guarantees, exact-real algorithms, floating-point checks, and measured performance are separate claims. Historical audit results do not constitute a new proof certification or establish publication priority.

## Public snapshot

The public repository includes the consolidated source history and the previously tracked historical code, text results, metadata, and figures at their new archive paths. Large generated datasets and binary trajectories, installed dependencies, caches, duplicate archived originals and consolidation build artifacts, downloaded papers, and separately copied upstream comparison scripts remain local and are excluded by [.gitignore](.gitignore). Existing exclusions were preserved during relocation; a clone does not contain every artifact needed to reproduce historical experiments.

Source files are stored without automatic line-ending conversion to preserve recorded hashes. Archiving does not rerun experiments or change the status of any research result.
