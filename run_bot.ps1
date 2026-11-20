# Keeps the NBA prop bot running. Starts full_run.bat hidden with output in
# nba-prop-bot\logs, and restarts it after any exit (crash, kill, update):
# 30s later normally, 10 min later if it died within 5 min of starting so a
# startup crash can't loop. Exits at once if a bot is already running, so it
# is safe to launch at every logon (Task Scheduler: "NBA Prop Bot").
#
# To restart the bot onto new code: stop the python main.py full_run process
# only; this script brings it back up.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = Join-Path $root 'nba-prop-bot\logs'
$supervisorLog = Join-Path $logs 'supervisor.log'

function Test-BotRunning {
    @(Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'main\.py\s+full_run' }).Count -gt 0
}

function Write-SupervisorLog($msg) {
    Add-Content -Path $supervisorLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg" -Encoding utf8
}

New-Item -ItemType Directory -Force $logs | Out-Null
if (Test-BotRunning) {
    Write-SupervisorLog "bot already running; supervisor not started"
    exit 0
}

while ($true) {
    foreach ($name in 'full_run.log', 'full_run.err.log') {
        $path = Join-Path $logs $name
        if (Test-Path $path) {
            Move-Item $path (Join-Path $logs ($name -replace '^full_run', 'full_run.prev')) -Force
        }
    }
    $started = Get-Date
    $p = Start-Process cmd -ArgumentList '/c', "`"$root\full_run.bat`"" -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs 'full_run.log') `
        -RedirectStandardError (Join-Path $logs 'full_run.err.log') -PassThru
    Write-SupervisorLog "bot started (cmd pid $($p.Id))"
    $p.WaitForExit()

    $ranMin = ((Get-Date) - $started).TotalMinutes
    $wait = if ($ranMin -lt 5) { 600 } else { 30 }
    Write-SupervisorLog ("bot exited after {0:N1} min (code {1}); restarting in {2}s" -f $ranMin, $p.ExitCode, $wait)
    Start-Sleep -Seconds $wait
    if (Test-BotRunning) {
        Write-SupervisorLog "another bot instance is running; supervisor exiting"
        exit 0
    }
}
