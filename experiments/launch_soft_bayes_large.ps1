$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$taskOutput = Join-Path $PSScriptRoot 'results/soft_bayes_full_d1000_B1000_T100000'
$taskLaunchRecord = Join-Path $taskOutput 'launch.json'
if (-not (Test-Path -LiteralPath (Join-Path $taskOutput 'manifest.json'))) {
    throw 'Prepare the large comparison manifest first.'
}
if (Test-Path -LiteralPath $taskLaunchRecord) {
    throw 'The comparison was already launched; refusing a duplicate.'
}
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-m', 'experiments.soft_bayes_large_campaign') -WorkingDirectory $taskWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'campaign_stdout.txt') -RedirectStandardError (Join-Path $taskOutput 'campaign_stderr.txt')
@{
    launched_utc = (Get-Date).ToUniversalTime().ToString('o')
    launcher_pid = $taskProcess.Id
    methods = @('soft_bayes', 'author_aioli')
    target_rounds = 100000
    online_seconds_per_method = 10800
    automatic_report = 'REPORT.md'
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskLaunchRecord -Encoding utf8
Get-Content -LiteralPath $taskLaunchRecord
