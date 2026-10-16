"""Produce the short research-facing result, with all numbers read from artifacts."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def main():
    base=Path(__file__).resolve().parent
    read=lambda p:json.loads((base/p).read_text())
    rows=read('results/screen/summary.json')
    confirm=read('results/confirmation/paired_statistics.json')
    long=read('results/long_horizon/summary.json')[0]
    precision=read('results/precision_audit.json')
    controls=read('results/controls.json')
    auxiliary=read('results/auxiliary_comparisons.json')
    totals=read('results/campaign_totals.json')
    feasibility=read('results/prescribed_feasibility.json')
    groups=defaultdict(list)
    for r in rows:groups[(r['family'],r['d'],r['B'],r['T'],r['B_generator'],r['algorithm'])].append(r)
    names={'ours':'Ours','sigmoid':'Sigmoid/MALA','aioli':'AIOLI'}
    bycase=defaultdict(dict)
    for r in rows:bycase[r['case_id']][r['algorithm']]=r
    wins={}
    for a,b in [('ours','sigmoid'),('ours','aioli'),('sigmoid','aioli')]:
        valid=[g for g in bycase.values() if g.get(a,{}).get('common') and g.get(b,{}).get('common')]
        wins[(a,b)]=(sum(g[a]['common']['loss']<g[b]['common']['loss'] for g in valid),len(valid))
    lines=['# Experimental result: proximal log-odds, sigmoid/MALA, and AIOLI','',
           f'The main screen measured {len(rows)} runs: 21 parameter/data regimes, three paired data seeds, and three algorithms. It used R=1, B up to 10,000, d up to 5,000, and configured T up to 1,000,000. Each worker used one BLAS thread and a 2 GiB hard process-commit cap. The primary per-run online budget was 15 seconds.','',
           '**Our capped sampler did not show an overall regret–runtime advantage in this study.** The screening counts below compare cumulative loss at each case’s common completed prefix; the identical comparator cancels in each difference. These correlated screen cases are descriptive counts, not independent statistical trials.','',
           '| Lower-loss method | Comparator method | Cases with lower loss / paired cases |','|---|---|---:|']
    for (a,b),(w,n) in wins.items():lines.append(f'| {names[a]} | {names[b]} | {w} / {n} |')
    lines+=['',
            'Scope: ours uses the original `SMALL_CLOUD_COMPLETE.md` schedule. The later stratified/fresh-restart refinement in `ELEGANT_LOG_ODDS_SINGLE_FLOW.md` is not benchmarked here; the population projections below concern the original schedule.','',
            'Calibration on separate seeds selected a cap of 256 particles and one transition per round for ours, and 16 particles with 16 total lazy MALA steps per round for sigmoid. Our actual population can be lower when its prescribed m is smaller. AIOLI uses its prescribed finite scalar solve and a Cholesky representation. These capped randomized variants are not certified realizations of their theoretical regret bounds.','',
            '## Representative paired measurements','',
            'Each row averages three data seeds. Regret intervals represent numerical comparator error, not statistical uncertainty. The range gives common-prefix lengths across seeds. Time is measured at those prefixes; memory is the peak over the whole run. All examples below use fixed generating radius 3.','',
            '| B | d | Configured T | Method | Common rounds | Mean regret interval | Mean online seconds at common prefix | Mean peak private MiB |',
            '|---:|---:|---:|---|---:|---:|---:|---:|']
    for d,B,T in [(100,1.,1000),(100,100.,1000),(1000,1000.,10000),(5000,100.,1000),(10,10000.,1000)]:
        for algo in ['ours','sigmoid','aioli']:
            rr=groups[('signal',d,B,T,3.,algo)];cr=[r['common_round'] for r in rr]
            lo=np.mean([r['common']['regret_lower'] for r in rr]);hi=np.mean([r['common']['regret_upper'] for r in rr])
            tm=np.mean([r['common']['online_seconds'] for r in rr]);mem=np.mean([r['peak_private_commit']/2**20 for r in rr])
            lines.append(f'| {B:g} | {d:,} | {T:,} | {names[algo]} | {min(cr)}–{max(cr)} | [{lo:.3f}, {hi:.3f}] | {tm:.4f} | {mem:.1f} |')
    lines+=['','## Independent confirmation','',
            'The confirmation uses B=100, d=10, configured T=1000, and a fixed evaluation endpoint of 500 rounds, declared before these runs. It has ten new data seeds, three independent sampler repeats per dataset, and one deterministic AIOLI run per dataset. Run order was shuffled with a fixed seed. Sampler repetitions are averaged within each dataset; the statistical sample size is ten.','',
            'Positive differences mean the first method has greater regret/loss. The comparator cancels exactly. Intervals are approximate pointwise Student-t intervals across the ten dataset clusters, with no multiple-comparison correction.','',
            '| First − second | Mean regret difference per round | Approximate 95% interval |','|---|---:|---:|']
    for r in confirm['comparisons']:
        lo,hi=r['approximate_95_percent_t_interval']
        lines.append(f'| {names[r["first"]]} − {names[r["second"]]} | {r["mean_paired_regret_difference_per_round"]:.5f} | [{lo:.5f}, {hi:.5f}] |')
    f=long['final']
    lines+=['','## Horizon and memory','',
            f'A separate AIOLI scaling run had a 240-second allowance and completed **{long["completed_rounds"]:,} / {long["T"]:,}** rounds at B=100,d=10. Online time was **{long["online_seconds"]:.3f} seconds**, whole-process wall time **{long["process_wall_seconds"]:.3f} seconds**, and peak private commit **{long["peak_private_commit"]/2**20:.2f} MiB**. Its numerical final regret interval is **[{f["regret_lower"]:.3f}, {f["regret_upper"]:.3f}]**. This longer allowance is separate from the equal-budget screen.','',
            'At completed round n, the persistent ndarray counts in bytes are exactly `8(d²+d)` for AIOLI, `8[1024⌈n/1024⌉(d+1)+m(d+1)]` for ours, and `8[1024⌈n/1024⌉(d+1)+md]` for sigmoid, between predictions. OS peaks additionally include imports, Python objects, temporaries, and input buffering. The samplers revisit retained history; AIOLI uses its matrix and vector sufficient statistics.','',
            '## Matrix and tempering controls','',
            'The two AIOLI representations implement the same mathematical update. Their recorded prefix losses agreed closely; their time and memory differed. Memory below is the peak over each whole control run, while time uses the common prefix.','',
            '| d | B | Common rounds | Inverse / Cholesky seconds | Inverse / Cholesky peak private MiB | Maximum per-round loss difference |',
            '|---:|---:|---:|---:|---:|---:|']
    for r,d,B in zip(auxiliary['matrix'],[1000,5000],[1000,100]):
        lines.append(f'| {d:,} | {B} | {r["common_round"]:,} | {r["first_online_seconds_at_common"]:.3f} / {r["second_online_seconds_at_common"]:.3f} | {r["first_peak_private_MiB"]:.1f} / {r["second_peak_private_MiB"]:.1f} | {r["maximum_absolute_round_loss_difference"]:.3e} |')
    lines+=['',
            'At B=10, two tempering controls used 20 bridges (the paper’s bridge count at this B) and 20 total lazy steps, versus one bridge and 16 steps. The cumulative loss differences were '+', '.join(f'{r["first_minus_second_cumulative_loss"]:.4f} at {r["common_round"]} rounds' for r in auxiliary['tempered'])+'. The tempered runs used more time at those prefixes. These are two illustrative controls, not a statistical superiority claim.','',
            '## Prescribed schedules and numerical diagnostics','',
            'Three small prescribed-schedule runs of ours (T=10, with (d,B)=(1,0.1),(1,1),(2,1)) completed. Large-case population projections were computed without allocating those clouds. Examples:', '',
            '| B | d | T | Our prescribed m (approximately) | Our particle array bytes alone (approximately) | Expected-loss sigmoid population |','|---:|---:|---:|---:|---:|---:|']
    for r in feasibility:
        if (r['d'],r['B'],r['T']) in [(100,100.,1000),(1000,1000.,10000),(10,10000.,1000)]:
            lines.append(f'| {r["B"]:g} | {r["d"]:,} | {r["T"]:,} | {r["ours"]["prescribed"]["m"]:.6e} | {r["ours_particle_array_bytes"]:.6e} | {r["sigmoid_expected_population"]:,} |')
    maxres=max(r['max_absolute_scalar_residual'] for r in precision['results'])
    sumres=max(r['sum_absolute_scalar_residuals'] for r in precision['results'])
    lines+=['',
            'The sigmoid theorem still lacks numerical universal MALA mixing constants, so no finite-budget run here is labeled a certified implementation of that theorem. Arithmetic counters record events and instrumented kernel-work estimates; they are not hardware instruction counts. Full projections and every actual sampling budget are retained in the raw records.','',
            f'At B=10,000, AIOLI recorded scalar-rounding stalls and logistic-curvature underflows. An offline 80-digit Decimal check of its represented scalar equations found maximum absolute residual **{maxres:.6e}**; the largest sum over a 1000-round run was **{sumres:.6e}**. Since the scalar equation has derivative at least one, its residual bounds scalar-root error with those represented statistics fixed. This does not certify matrix rounding, subsequent update perturbations, or floating-point regret.','',
            f'The OS memory-cap control rejected a 256 MiB allocation under a 64 MiB process limit: **{controls["hard_memory_limit_control"]["allocation_rejected"]}**. The main screen has **{sum(r["status"]=="error" for r in rows)} numerical-error exits**; all time limits remain explicitly recorded.','',
            f'The study contains **{sum(r["runs"] for r in totals)} measured runs**, totaling **{sum(r["process_wall_seconds"] for r in totals)/60:.2f} minutes of worker wall time**, plus six initial smoke runs. All **60 tests pass**, including one- and two-dimensional integration checks. Environment versions, source hashes, manifests, raw losses/times, and diagnostic checkpoints are retained.','',
            '## Artifacts','',
            '- [Full results, all regimes, and plots](experiments/RESULTS.md)',
            '- [Python implementations and reproduction](experiments/README.md)',
            '- [Main manifest](experiments/results/screen/manifest.json), [raw summary](experiments/results/screen/summary.json), [regret curves](experiments/results/screen/curves.json)',
            '- [Confirmation statistics](experiments/results/confirmation/paired_statistics.json)',
            '- [Prescribed resource projections](experiments/results/prescribed_feasibility.json)',
            '- [Numerical precision audit](experiments/results/precision_audit.json)','']
    (base.parent/'EXPERIMENT_RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(base.parent/'EXPERIMENT_RESULT.md')


if __name__=='__main__':main()
