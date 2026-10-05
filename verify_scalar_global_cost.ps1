# Formula-only arithmetic for SCALAR_GLOBAL_COST_RESEARCH.md, Section 8.
# No learning run, sampling experiment, or timing measurement is performed.
$scalarN = 100000.0
$scalarK = 999.0
$scalarLambda = .001
$scalarNu = .00016
$scalarTrace = .1
$scalarOmega = .00016
$scalarRadius = .07
$scalarModeBound = .1
$scalarKmax = 10.0
$scalarH2 = .25
$scalarH3 = 1 / (6 * [Math]::Sqrt(3))
$scalarH4 = .125
$scalarYoung = .0005
$scalarC = $scalarH3 / 2 + $scalarH2 * $scalarYoung / 2
$scalarA = $scalarH2 / (2 * $scalarYoung)
$scalarGradient = [Math]::Sqrt(.0008) / 4
$scalarLc = 24000.0
$scalarL3 = 2125.0
$scalarL = $scalarLambda + $scalarN / 4
$scalarHm = $scalarLambda + $scalarN * .998 * .998 * (.25 - .1 * .1 / 16)
$scalarJ0 = 2 / [Math]::Sqrt($scalarL) * [Math]::Exp(-.5 - $scalarGradient / [Math]::Sqrt($scalarL))
$scalarEndpoint = $scalarLc * $scalarRadius * $scalarRadius / 2 - $scalarGradient * $scalarRadius
$scalarBoundary = 2 * [Math]::Exp(-$scalarEndpoint) / $scalarJ0
$scalarM2computed = [Math]::Pow(($scalarGradient + [Math]::Sqrt($scalarGradient * $scalarGradient + 4 * $scalarLc)) / (2 * $scalarLc), 2)
$scalarM1computed = ($scalarGradient + $scalarBoundary + $scalarL3 * $scalarM2computed / 2) / $scalarHm
# The slightly rounded-up constants used in the manuscript.
$scalarM2 = 4.166856852451e-5
$scalarM1 = 2.067164441318e-6
$scalarK1 = $scalarRadius * $scalarC * 1.001
$scalarK0 = $scalarRadius * $scalarA * $scalarOmega + $scalarRadius * $scalarC * $scalarOmega * (1001 * [Math]::Pow($scalarModeBound + $scalarRadius / 2, 2) + $scalarRadius * $scalarRadius / 12)
$scalarSconstant = 1001 * $scalarOmega * $scalarRadius * $scalarRadius
$scalarUcoefficient = $scalarM1 * ($scalarH3 / 2 + $scalarH2 * .01) + $scalarM2 / 2 * ($scalarH4 / 2 + 2 * $scalarH3 * .01) * 1.001
$scalarUconstant = $scalarM1 * $scalarH2 * $scalarOmega / .04 + $scalarM2 / 2 * (($scalarH4 / 2 + 2 * $scalarH3 * .01) * $scalarSconstant + 2 * $scalarH3 * $scalarOmega / .04 + $scalarH2 * $scalarOmega)
$scalarU1 = $scalarUcoefficient * 1.001
$scalarU0 = $scalarUcoefficient * 1001 * $scalarOmega * $scalarModeBound * $scalarModeBound + $scalarUconstant
$scalarExponent = $scalarK1 + $scalarU1
$scalarOffset = $scalarK0 + $scalarU0
$scalarDenominator = 1 - 2 * $scalarExponent * $scalarNu / $scalarLambda
$scalarSlope = 3 * $scalarLc * $scalarRadius / 4 - $scalarGradient - 4 * $scalarKmax / $scalarRadius
$scalarTail = [Math]::Exp(2 * $scalarKmax) / $scalarJ0 * 2 * [Math]::Exp(-$scalarEndpoint) / $scalarSlope
$scalarSourceFactor = 6.0
$scalarCapTolerance = 1e-9
$scalarAtomProposals = [Math]::Exp($scalarOffset)
$scalarAtomRows = 2 * $scalarRadius * $scalarTrace * ($scalarC * ($scalarModeBound * $scalarModeBound + $scalarModeBound * $scalarRadius + $scalarRadius * $scalarRadius / 3) + $scalarA) * $scalarAtomProposals
$scalarRows = foreach ($scalarBp in @(0.0, .5, 1.0, 1.5, 2.0)) {
    $scalarMean = $scalarBp / $scalarLambda + 1 + $scalarNu / (4 * .996 * $scalarLambda)
    $scalarMeanEnergy = $scalarNu * $scalarMean * $scalarMean + $scalarTrace / $scalarLambda
    $scalarPsi = [Math]::Exp($scalarExponent * $scalarMeanEnergy / $scalarDenominator)
    $scalarProposals = $scalarSourceFactor * [Math]::Exp($scalarOffset) * $scalarPsi
    $scalarQueries = 2 * $scalarRadius * $scalarTrace * $scalarProposals * ($scalarC * ($scalarK / ($scalarLambda * $scalarDenominator) + $scalarMean * $scalarMean / ($scalarDenominator * $scalarDenominator) + $scalarModeBound * $scalarModeBound + $scalarModeBound * $scalarRadius + $scalarRadius * $scalarRadius / 3) + $scalarA)
    $scalarLogFailure = [Math]::Log($scalarSourceFactor) - ($scalarKmax - $scalarK0) / $scalarK1 + $scalarMeanEnergy / (1 - 2 * $scalarNu / $scalarLambda)
    [ordered]@{ b_perp = $scalarBp; mean_bound = $scalarMean; proposals = $scalarProposals + $scalarCapTolerance * $scalarAtomProposals; rows = $scalarQueries + $scalarCapTolerance * $scalarAtomRows; log_K_failure = $scalarLogFailure }
}
[ordered]@{
    C = $scalarC; A = $scalarA; M2_computed = $scalarM2computed; M1_computed = $scalarM1computed
    k1 = $scalarK1; k0 = $scalarK0; u1 = $scalarU1; u0 = $scalarU0
    gaussian_denominator = $scalarDenominator; scalar_tail = $scalarTail
    cap_atom_proposals = $scalarAtomProposals; cap_atom_rows = $scalarAtomRows
    table_including_cap_1e_minus_9 = @($scalarRows)
} | ConvertTo-Json -Depth 4
