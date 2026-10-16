$ErrorActionPreference = 'Stop'
$fused = Get-Content -Raw -LiteralPath 'FUSED_SWEEP_ALGORITHM.md'
$elegant = Get-Content -Raw -LiteralPath 'ELEGANT_LOG_ODDS_SINGLE_FLOW.md'
$sweep = Get-Content -Raw -LiteralPath 'INTEGRATION_TILT_INVESTIGATION.md'

function Part([string]$source, [string]$start, [string]$end) {
    $first = $source.IndexOf($start, [StringComparison]::Ordinal)
    if ($first -lt 0) { throw "Missing start marker: $start" }
    if ($end -eq '') { return $source.Substring($first) }
    $last = $source.IndexOf($end, $first + $start.Length, [StringComparison]::Ordinal)
    if ($last -lt 0) { throw "Missing end marker: $end" }
    return $source.Substring($first, $last - $first)
}
function Scope([string]$source, [string]$prefix) {
    $source = [regex]::Replace($source, '\\tag\{(\d+(?:\.\d+)?)\}', { param($match) '\tag{' + $prefix + $match.Groups[1].Value + '}' })
    return [regex]::Replace($source, '(?<![\p{L}\p{N}_\\])\((\d+(?:\.\d+)?)\)', { param($match) '(' + $prefix + $match.Groups[1].Value + ')' })
}

$intro = @'
# A single posterior sweep: algorithm, complete regret proof, and explicit computation

This manuscript gives the complete revised algorithm and its proof in one flow. The algorithm stores the features and one accumulated label vector, follows one chain through deterministic midpoint tilts, and uses a single scalar random routine throughout. Rejected vector proposals improve their own tangent point; there is no separate optimizer or implemented fitting-accuracy schedule.

Every specialized comparator, variance, entropy, mixing, correlated-sampling, and rejection inequality used below is proved here. The continuous-time evolution appearing in the entropy argument is only a proof device. The implemented algorithm uses exact real arithmetic, elementary functions, and independent uniform draws.

'@
$setting = Scope (Part $fused '## Setting and constants' '## One scalar helper') 'F'
$setting = $setting.Replace('## Setting and constants', '## 1. Setting and guarantee')
$setting = $setting.Replace('\pi_0(d\theta)\propto e^{-\lambda\|\theta\|^2/2}', '\pi_0(d\theta)=Z_0^{-1}e^{-\lambda\|\theta\|^2/2}')
$setting = $setting.Replace('Set' + [Environment]::NewLine + '\[', 'Set' + [Environment]::NewLine + '\[')
$setting = $setting.Replace('The statistically checked guarantee, whose full proof is pending insertion here, is', 'The algorithm below satisfies, for every comparator in the ball,')
$setting = $setting.Replace('\log\left(1+\frac{B^2}{4}\sum_{t=1}^T x_{t,i}^2\right).', '\log\left(1+\frac{B^2}{4}\sum_{t=1}^T x_{t,i}^2\right)' + "`n" + '\le2+\frac d2\log\left(1+\frac{B^2}{4d}\sum_t\|x_t\|^2\right).')
$setting = $setting.Replace('The expectation concerns only algorithmic randomness. The prior and feature history are retained; past labels are compressed into one vector.', 'Here \(Z_0=\int_K e^{-\lambda\|z\|^2/2}dz\) is a proof normalizer that the algorithm never computes. The expectation concerns only algorithmic randomness. The prior is unchanged and the feature history is retained; past labels are compressed into one vector.')

$scalar = Scope (Part $fused '## One scalar helper' '## The complete online algorithm') 'F'
$scalar = $scalar.Replace('## One scalar helper', '## 2. One scalar random routine')
$scalar = $scalar.Replace('A=1-e^{r(\alpha-c)},\quad B=1-e^{r(c-\beta)}.', 'M_-=1-e^{r(\alpha-c)},\quad M_+=1-e^{r(c-\beta)}.')
$scalar = $scalar.Replace('c=0,r=1,A=B=1', 'c=0,r=1,M_-=M_+=1')
$scalar = $scalar.Replace('A/(A+B)', 'M_-/(M_-+M_+)')

$algorithm = Scope (Part $fused '## The complete online algorithm' '## Exactness and cost of the Gaussian proposals') 'F'
$algorithm = $algorithm.Replace('## The complete online algorithm', '## 3. The online algorithm')

$evidence = Scope (Part $elegant 'Write \(h(z)=\log(2\cosh(z/2))\).' 'We now control the fluctuation in (20).') 'E'
$evidence = $evidence.Replace('in (E12)', 'in the online algorithm')
$evidence = $evidence.Replace('Stratification will estimate this integral without bias.', 'The deterministic midpoint sweep will approximate this integral; its quadrature bias and its dependent sample fluctuations will both be bounded explicitly.')
$evidence = $evidence.Replace('then the fluctuation of that stratified estimate, then the effect of obtaining each score with the stated finite chain.', 'then prove the scalar and vector sampling routines and the chain contraction, and finally bound the entire dependent midpoint sweep.')
$evidence = $evidence.Replace('We first bound the Bayesian comparator term, then prove', 'We first bound the Bayesian comparator term, then prove')

$functional = Scope (Part $elegant 'We now control the fluctuation in (20).' 'Return to the score at round \(t\).') 'E'
$functional = $functional.Replace('We now control the fluctuation in (E20).', 'To control both sampling fluctuations and mixing, we next derive the required functional inequalities directly.')

$exact = Scope (Part $fused '## Exactness and cost of the Gaussian proposals' '## Explicit operations and storage') 'F'
$exact = $exact.Replace('## Exactness and cost of the Gaussian proposals', 'We now verify that the implemented random routines generate exactly the required transitions, despite changing their proposal after a rejection.')
$exact = $exact.Replace('r e^{-rz}/(A+B)', 'r e^{-rz}/(M_-+M_+)')
$exact = $exact.Replace('with the same proximal kernel as before.', 'with the exact proximal kernel consisting of a Gaussian displacement followed by a draw from its conditional distribution on the box.')
$exact = $exact.Replace('This change of variables identifies the resulting move with the same proximal kernel as before.', 'This change of variables identifies the resulting move with the exact proximal kernel consisting of a Gaussian displacement followed by a draw from its conditional distribution on the box.')

$mixing = Scope (Part $elegant 'We next quantify the convergence of the exact conditional transition just verified.' 'We apply this convergence result directly from a fresh prior.') 'E'

$prior = Scope (Part $elegant 'We apply this convergence result directly from a fresh prior.' 'A function with range length \(c\) has expectation difference at most') 'E'
$prior = $prior.Replace('and \(\Delta=W-V\) is \(c\)-Lipschitz, direct normalization', '\(\nabla^2V\succeq\lambda I\), and \(\Delta=W-V\) is \(c\)-Lipschitz, direct normalization')
$prior = $prior.Replace("The target's difference from the prior is \(C_{t,s}\), whose gradient norm is at most \(tR\). Consequently", "The target's difference from the prior is \(\Phi_t(z)-(a_{\rm obs}+sx_t)^\top z\). Since \(|h'|\le1/2\), the gradient of \(\Phi_t\) has norm at most \(tR/2\); also \(\|a_{\rm obs}+sx_t\|\le tR/2\) for the algorithmic tilts. The full difference is therefore \(tR\)-Lipschitz. Consequently")
$prior = $prior.Replace('There is no inherited approximate initial law.', 'This is the accuracy used only for the first midpoint of each round. Later midpoints retain the state and will be controlled by a different argument.')

$moving = Scope (Part $sweep 'Fix a round and suppress its index.' '### One burn-in error controls the entire recorded path') 'I'
$moving = [regex]::Replace($moving, '(?m)^### [^\r\n]+\r?\n\r?\n', '')
$moving = $moving.Replace('Fix a round and suppress its index.', 'We can now analyze all the correlated scores in one round. Fix that round and suppress its index.')
$moving = $moving.Replace('all the targets in (I2)', 'all targets \(q_{t,s}\) defined by the online potential')
$moving = $moving.Replace('the algorithm itself uses only (I3).', 'the algorithm itself uses only the specified midpoint tilts.')
$moving = $moving.Replace('together with the explicitly proved curvature-to-variance inequality.', 'together with the curvature-to-variance inequality proved above.')
$moving = $moving.Replace('the proved box variance inequality', 'the box variance inequality (E40)')
$moving = $moving.Replace('The exact Bayesian log odds are \(D_t=\int_{-1/2}^{1/2}f(s)ds\).', 'The exact Bayesian log odds are \(D_t=\int_{-1/2}^{1/2}f(s)ds\), by (E20). Here \(b\) is a scalar variance bound, distinct from the label-and-tilt vector \(b_j\).')
$moving = $moving.Replace('The entropy contraction already proved for this kernel is', 'The entropy contraction (E54) proved above gives')
$moving = $moving.Replace('The entropy contraction already proved for this kernel', 'The entropy contraction (E54)')
$moving = $moving.Replace('The entropy contraction already proved', 'The entropy contraction proved above')
$moving = $moving.Replace('The entropy contraction already proved for this kernel is', 'The entropy contraction (E54) gives')
$moving = $moving.Replace('Equation (I5)', 'The block count in (F3)')
$moving = $moving.Replace('by (I1)', 'by (F1)')
$moving = $moving.Replace('required by (I1)', 'required by (F1)')
$moving = $moving.Replace('The existing rank-one calculation, reproduced briefly here, gives the required total budget.', 'Reinsert time indices and put \(A_0=\lambda I\). The following elementary rank-one calculation gives the total budget.')

$completion = @'
The remaining approximation is the first burn-in. At its end the actual state has law within (delta) of (q_{t,s_1}), by (E58). Generate the complete remainder of the sweep from those two possible initial laws using the same prescribed sequence of kernels. This is one Markov operation on the initial state, so it cannot increase total variation. To verify this fact directly, the conditional probability of any output-path event is a function taking values in ([0,1]); write that function as the integral of its superlevel indicators and apply the definition of TV.

Every score average lies in ([-L,L]), and either logistic label loss has range exactly (L) on that interval, since (log(1+e^L)-log(1+e^{-L})=L). The same level-set integration shows that the expectation of a function of range length (L) changes by at most (L) times TV. Therefore the single burn-in error contributes

[
sum_tleft[mathbb Eell_{y_t}(widehat D_t)-mathbb Eell_{y_t}(widehat D_t^{m ideal})ight]
le LTdelta=rac{L}{4(1+L)}lerac14.
	ag{Z1}
]

The proof now closes in one chain of inequalities. The comparator calculation (E39) contributes at most (3/4+rac12sum_ilogalpha_i); the midpoint, moving-target bias, and covariance calculation (I19) contributes at most one over all rounds; and (Z1) contributes at most one quarter for the actual first burn-in. Since (ell_{y_t}(D_t)=-log p_t^{m B}(y_t)), their sum proves (F4). The stronger version retains (sum_i I(u_i/B)), the nonpositive truncation correction in (E39), and the exact final factor (L/[4(1+L)]).

The expectation is only over the implemented uniform draws for a fixed data sequence. The scores within a round are deliberately dependent, and the deterministic midpoint rule is deliberately allowed a controlled bias. Neither independence nor exact midpoint unbiasedness was used.

'@

$cost = Scope (Part $fused '## Explicit operations and storage' '') 'F'
$cost = $cost.Replace('## Explicit operations and storage', '## 5. Explicit operations and storage')
$cost = $cost.Replace('The statistical theorem (F4) has been independently checked, but this draft remains incomplete until its statistical proof is inserted.', 'The regret theorem is proved above for fixed oblivious data and a known horizon. No finite-bit, adaptive-feedback, high-probability sampled-regret, empirical-speed, or literature-novelty claim follows from these real-arithmetic bounds.')
$cost = $cost.Replace('Substitution into (F17) exposes polynomial dependence on every parameter, up to the displayed logarithms.', 'For a fully explicit uniform substitution, set \(\bar m=1+\max\{L,3\kappa T/8\}\) and \(\overline{\cal A}=H+2(\bar m-1)\log32\). Every factor on the right of (F17) is nonnegative and nondecreasing in \(m,{\cal A}\), so replacing them by \(\bar m,\overline{\cal A}\) gives a valid bound polynomial in every problem parameter, up to the displayed logarithms.')

$document = $intro + $setting + $scalar + $algorithm + "## 4. One proof, from evidence to the implemented sweep`n`n" + $evidence + $functional + $exact + $mixing + $prior + $moving + $completion + $cost
$document = [regex]::Replace($document, '\([EFIZ]0\)', '(0)')
$document = $document.Replace('(E1)', '(F1)')
$safeCompletion = Scope (Part $sweep '### One burn-in error controls the entire recorded path' '## Explicit computation and parameter dependence') 'I'
$safeCompletion = $safeCompletion.Replace('### One burn-in error controls the entire recorded path', 'One burn-in error now controls the entire recorded path.')
$safeCompletion = $safeCompletion.Replace('the unchanged comparator bound \(3/4\)', 'the comparator bound (E39), namely \(3/4\)')
$safeCompletion = $safeCompletion.Replace('proves (I6)', 'proves (F4)')
$document = $document.Replace($completion, $safeCompletion)

# Resolve all source-local equation labels only after assembling the final flow.
$labels = [ordered]@{}
$counter = 0
foreach ($match in [regex]::Matches($document, '\\tag\{([^}]+)\}')) {
    $key = $match.Groups[1].Value
    if ($labels.Contains($key)) { throw "Duplicate source equation label: $key" }
    $counter++
    $labels[$key] = $counter
}
$document = [regex]::Replace($document, '\\tag\{([^}]+)\}', { param($match) '\tag{' + $labels[$match.Groups[1].Value] + '}' })
$document = [regex]::Replace($document, '\(([EFIZ]\d+(?:\.\d+)?)\)', {
    param($match)
    $key = $match.Groups[1].Value
    if (-not $labels.Contains($key)) { throw "Unresolved source equation reference: $key" }
    '(' + $labels[$key] + ')'
})
Set-Content -LiteralPath 'FUSED_LOG_ODDS_COMPLETE.md' -Value $document -Encoding utf8
Write-Output "Saved complete manuscript with $counter sequential equations and $($document.Length) characters."
