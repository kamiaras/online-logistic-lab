# Online Logistic Lab

Research notes, proofs, and experiments on online logistic regression, probability aggregation, and efficient online learning.

Start with **[CONSOLIDATED_RESEARCH.md](CONSOLIDATED_RESEARCH.md)**, the authoritative research volume. It brings together 14 synthesis chapters and 562 original source records, including algorithm specifications, proofs, lower bounds, literature reviews, audits, and experiment reports.

## Research directions

- Exact Gaussian probability aggregation and original-posterior forecasts.
- Clipped-gradient and half-lookahead learners with linear work and state.
- Posterior sampling, compression, cache reuse, and full computational accounting.
- Lower bounds, counterexamples, formula checks, and measured implementations.

The logarithmic-regret, linear-work results depend on their stated noise assumptions. The fully adversarial linear-work target and a general modest-cost original-posterior implementation remain unresolved. Statistical guarantees, exact-real algorithms, floating-point checks, and measured performance are separate claims. Historical audit results do not constitute a new proof certification or establish publication priority.

## Repository guide

| Location | Contents |
| --- | --- |
| [Consolidated research](CONSOLIDATED_RESEARCH.md) | Chapter synthesis, proofs, attribution, and searchable source records such as S148. |
| [Research implementations](research/) | Algorithm implementations and focused formula checks. |
| [Experiments](experiments/) | Experimental code, analysis tools, saved JSON results, progress records, figures, and source snapshots. |
| [Research checks](research_checks/) | Additional verification code. |
| [Brainstorm checks](brainstorming_100/) | Candidate-method checks and historical research state. |
| Root-level verification scripts and JSON files | Saved algebraic and numerical checks. |

See [AGENTS.md](AGENTS.md) for the project's research boundaries and evidence-handling conventions. Future research additions belong in the consolidated volume.

## Working with the code

This is a research workspace rather than a packaged library. The experiment environment is recorded in [experiments/requirements.txt](experiments/requirements.txt); the author-code comparison has its own requirements and provenance manifest. Inspect each script's settings and data requirements before running it. Historical campaign scripts can launch long computations, and their presence does not mean those campaigns completed.

Existing code, saved results, settings, and source hashes are retained as research evidence. Some records contain original local paths. Numerical checks support the derivations but do not replace proofs.

## Public snapshot

The public repository includes the consolidated source history, project code, saved text results, metadata, and figures. Large generated datasets and binary trajectories, installed dependencies, caches, duplicate archived originals and consolidation build artifacts, downloaded papers, and separately copied upstream comparison scripts remain local and are excluded by [.gitignore](.gitignore). Their available provenance manifests remain included. A clone therefore does not contain every artifact needed to replay historical experiments; references to excluded local files are retained as provenance.

Source files are stored without automatic line-ending conversion to preserve recorded file hashes. This initial publication does not rerun experiments or change the status of any research result.
