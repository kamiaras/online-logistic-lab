# Historical research archive — explicit access only

The active entry point is [CONSOLIDATED_RESEARCH.md](../CONSOLIDATED_RESEARCH.md). Current instructions are in the root [AGENTS.md](../AGENTS.md).

**Do not read, search, edit, move, or execute material in this archive unless the user explicitly instructs access to the relevant material.** Do not follow archive links automatically during routine research. Old TODOs, commands, campaign plans, and archived instruction files are historical evidence, not current instructions. Keep the archive excluded from routine searches.

## Layout

| Location | Contents |
| --- | --- |
| `2026-10-04/originals/` | Original Markdown sources, preserved byte for byte; already incorporated in the active volume. Local only. |
| `2026-10-04/build/` | Source inventory, consolidation reviews, hashes, manuscript fragments, and integrity records. Local only. |
| [2026-10-16/companions/](2026-10-16/companions/) | Historical implementations, verification scripts, check JSON, experimental data/results/figures, papers, dependencies, and caches, retaining the old relative directory layout. Existing Git exclusions still apply. |
| [2026-10-16/relocation_manifest.json](2026-10-16/relocation_manifest.json) | Old/new path mapping and unchanged SHA-256 hashes for 5,864 previously tracked companion files. |
| `2026-10-16/local-verification/` | Local relocation snapshots, navigation-change ledger, and verification records. Excluded from Git. |

For a former companion path `p`, the current location is `2026-10-16/companions/p` relative to this folder. Historical path strings inside code, logs, commands, and manifests are preserved verbatim. Source identifiers such as S148 continue to resolve inside the active volume. One already-missing historical audit has no recovered file; its unavailable status remains recorded in the volume.

The generated root `.pytest_cache/` was inaccessible to the operating system during relocation and remains ignored outside this archive. It is not research input.

## Preservation and reproduction

Moving the complete companion tree preserves its relative layout and measured file bytes. It does not certify that captured absolute paths, old environments, or cancelled campaign launchers are runnable. If the user explicitly requests reproduction or recovery, first identify the relevant source record, recorded environment, data, settings, and hashes. Do not substitute a new run for historical evidence or edit recorded source bytes silently.

Previously tracked companion files remain in Git at their new locations. Previously ignored large binaries, dependencies, downloaded papers, upstream copies, duplicate originals, and local verification snapshots stay excluded. No experiment was restarted for this archival change.
