$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$taskLaunchFolder = Join-Path $PSScriptRoot 'results/author_baselines_launch'
if (Test-Path -LiteralPath $taskLaunchFolder) {
    throw 'The author-baseline launch folder already exists; refusing a duplicate launch.'
}
New-Item -ItemType Directory -Path $taskLaunchFolder | Out-Null
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-m', 'experiments.author_baseline_campaign', '--seconds', '43200') -WorkingDirectory $taskWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLaunchFolder 'stdout.txt') -RedirectStandardError (Join-Path $taskLaunchFolder 'stderr.txt')
@{
    launched_utc = (Get-Date).ToUniversalTime().ToString('o')
    launcher_pid = $taskProcess.Id
    output = 'experiments/results/author_baselines_d1000_B1000_T100000'
    algorithms = @('author_aioli', 'author_sigmoid')
    reused_new16_run = 'experiments/results/computation_first_overnight_d1000_B1000_T100000'
    target_rounds = 100000
    per_method_online_seconds = 43200
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunchFolder 'launch.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $taskLaunchFolder 'launch.json')
