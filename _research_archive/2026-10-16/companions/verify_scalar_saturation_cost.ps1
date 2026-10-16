# Formula arithmetic only for SCALAR_FULL_MODE_COST_RESEARCH.md.
# This does not run a learning algorithm, Monte Carlo study, or timing test.
$satLambda = .001
$satNu = .00016
$satTrace = .1
$satDim = 999.0
$satN = 100000.0
$satEll = 6.1
$satG = .996
$satX = 40.0
$satGradient = [Math]::Sqrt(.0008) / 4
$satMode = 7.0
$satH = { param($z) $z / 2 + [Math]::Log(1 + [Math]::Exp(-$z)) - [Math]::Log(2) }
$satSigmoid = { param($z) 1 / (1 + [Math]::Exp(-$z)) }
$satJ0 = 2 / [Math]::Sqrt(25000.001) * [Math]::Exp(-.5 - $satGradient / [Math]::Sqrt(25000.001))
$satGap = $satLambda * [Math]::Pow($satMode - $satEll, 2) / 2 + $satN * .998 * .998 * ((& $satH $satEll) - (& $satH $satMode) - ((& $satSigmoid $satMode) - .5) * ($satEll - $satMode)) - $satGradient * ($satMode - $satEll)
$satSlope = $satLambda * ($satMode - $satEll) + $satN * .998 * .998 * ((& $satSigmoid $satMode) - (& $satSigmoid $satEll)) - $satGradient
$satLabelBound = ($satN * [Math]::Exp(-.998 * $satMode) + $satGradient) / $satG
$satBp = [Math]::Sqrt($satNu * 93)
$satMean = $satBp / $satLambda + 1 + $satNu / (4 * $satG * $satLambda)
$satNormCap = $satMean + [Math]::Sqrt($satDim / $satLambda) + [Math]::Sqrt(2 * $satX / $satLambda)
$satZeroQ = .5 * [Math]::Exp(-$satG * $satEll)
$satZeroK = $satZeroQ * $satNu * $satEll * $satEll
$satZeroProposals = [Math]::Exp($satZeroK)
$satZeroRows = $satZeroQ * $satTrace * $satEll * $satEll * $satZeroProposals
$satTable = foreach ($satA in @(.002, .001004)) {
    $satQ = .5 * [Math]::Exp($satA * $satNormCap - $satG * $satEll)
    $satKmax = $satQ * $satNu * [Math]::Pow($satNormCap + $satEll, 2)
    $satS = $satQ * 1.01
    $satOffset = $satQ * 101 * $satNu * $satEll * $satEll
    $satDenominator = 1 - 2 * $satS * $satNu / $satLambda
    $satEnergy = $satNu * $satMean * $satMean + $satTrace / $satLambda
    $satProposals = 6 * [Math]::Exp($satOffset + $satS * $satEnergy / $satDenominator)
    $satRows = $satQ * $satTrace * $satProposals * ($satDim / ($satLambda * $satDenominator) + $satMean * $satMean / ($satDenominator * $satDenominator) + $satEll * $satEll)
    [ordered]@{ a_star = $satA; Q = $satQ; K_uniform = $satKmax; moment_s = $satS; moment_offset = $satOffset; gaussian_denominator = $satDenominator; proposals_with_cap_1e_minus_9 = $satProposals + 1e-9 * $satZeroProposals; rows_with_cap_1e_minus_9 = $satRows + 1e-9 * $satZeroRows }
}
[ordered]@{
    lower_gap = $satGap; lower_slope = $satSlope; J0 = $satJ0
    original_tail = [Math]::Exp(10 - $satGap) / ($satSlope * $satJ0)
    negative_label_bound = $satLabelBound; b_perp_bound = $satBp
    source_mean = $satMean; norm_cap = $satNormCap
    fallback_probability = 6 * [Math]::Exp(-$satX)
    zero_cap_proposals = $satZeroProposals; zero_cap_rows = $satZeroRows
    table = @($satTable)
} | ConvertTo-Json -Depth 4
