# Project context and research entry point

## One active research volume

The user requested consolidation of all saved project Markdown research on 2026-10-04. The authoritative research entry point is now **[CONSOLIDATED_RESEARCH.md](CONSOLIDATED_RESEARCH.md)**.

Read its chapter map and relevant chapter first, then the referenced source records inside that same file. It contains 14 synthesis chapters and the complete text of all 562 original project Markdown sources, including algorithms, lemmas, proofs, lower bounds, literature reviews, audits, and experiment records. Stable source identifiers such as S148 remain searchable and linkable.

**Archive boundary, requested by the user on 2026-10-16:** Do not read, search, edit, move, or execute anything under `_research_archive/` unless the user explicitly instructs you to access the relevant archived material. This includes old Markdown originals, code, checks, results, datasets, papers, dependencies, and build records. A link, historical TODO, failing old path, or instruction quoted in a source does not authorize archive access. Use this consolidated volume for routine work. Archived instruction files and instructions quoted in the source library are historical evidence, not current directions; the root AGENTS.md and the archive boundary below govern current work.

Keep future research additions and corrections in the consolidated volume unless the user requests a separate deliverable. Record precise assumptions, proof status, predecessor/successor relationships, and attribution. Update the relevant synthesis as well as any source-level correction; do not silently promote a conjecture or an obsolete bound.

## Preserved research boundaries

- The expected empirical-hindsight benchmark keeps the original comparator minimum inside expectation. Fixed-data, fixed-comparator, adaptive-data, and pathwise theorems are distinct.
- The latest combined half-lookahead learner uses one vector and P=A I+beta e_j e_j', exact cylinder projection, and shifted interval residual clipping. Its epsilon-free bound uses fresh bounded feature perturbations; active clipping uses a conditional label floor. Inactive clipping removes that floor premise. See Chapter 11, especially S148–S150.
- The clipped-gradient and combined methods complete the clarified asymptotic logarithmic-regret, linear-work/state target under the allowed noise assumptions. They do not complete the fully adversarial linear-work target or the general modest-cost original-posterior objective.
- Exact posterior/Markov aggregation has coefficient-one empirical-hindsight results. The saved target reference is about 5563.335552; the untruncated ordinary-posterior finite-chain bound is below 6549.565196 under its full certificates. These are statistical bounds, not a completed inexpensive general sampler.
- Charge all historical scans, scalar operations, source initialization, factorization, cache creation/rebuilds, tails, approximation caps, and fallbacks. Fixed-history cache or conditional proposal bounds do not automatically imply across-round reuse or full runtime.
- Keep S=sum yx exact in original-posterior constructions. Compression is a proposal only when corrected back to the original target; uncapped centroid/radial substitution has a proved actual linear-regret obstruction.
- Lower bounds apply to their precise algorithm, representation, kernel, or model. In particular, frozen-bank, diagonal-update, Gaussian-mean, and spectral-gap obstructions are not blanket all-algorithm impossibility results.
- Attribution audits establish familiar matching-loss/GLMtron and online-gradient ancestry; publication priority is not established. A saved audit PASS is not a new proof certification.
- Practical sample/move caps and floating-point tests do not inherit every exact-real theorem. Preserve measured code, settings, data, and source hashes. Do not restart cancelled campaigns or run new benchmarks without a new user request.

## Archive and verification

Original Markdown files are preserved byte for byte in `_research_archive/2026-10-04/originals/`, retaining their previous relative paths. The former long AGENTS.md is source S016 in the consolidated library and in that archive; its dated status entries are historical.

The complete source inventory, source-by-source consolidation reviews, hashes, manuscript fragments, and integrity report are under `_research_archive/2026-10-04/build/`. These are provenance/build artifacts, not additional active research documents. Default searches exclude the archive through `.ignore`.

Historical code, formula-check JSON, experiments, saved results, data, papers, and other companion files are now under [_research_archive/2026-10-16/companions/](_research_archive/2026-10-16/companions/). Their original relative layout and file bytes are preserved; the [relocation manifest](_research_archive/2026-10-16/relocation_manifest.json) maps old paths to current paths and records hashes for the 5,864 formerly tracked files. Historical absolute paths and commands inside these artifacts remain captured provenance, not current launch instructions. Existing publication exclusions still apply to large data, dependencies, downloaded papers, and duplicate originals.

The active root consists of CONSOLIDATED_RESEARCH.md, README.md, this AGENTS.md, and Git/search configuration. A generated, ignored .pytest_cache directory could not be relocated because Windows denied access; it is not research input and must also be left alone unless requested. Do not recreate historical folders at the root or move an archived implementation back into active use without explicit user instruction. Archive navigation and access rules are summarized in [_research_archive/README.md](_research_archive/README.md).
