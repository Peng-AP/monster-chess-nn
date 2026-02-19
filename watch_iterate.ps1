param(
    [int]$TargetMaxGen = 37,
    [int]$IterationsPerRun = 2,
    [int]$PollSeconds = 90,
    [int]$StallMinutes = 30,
    [int]$SeedBase = 20260220,
    [int]$CurrentPid = 0
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ModelsDir = Join-Path $ProjectRoot "models"
$RawDir = Join-Path $ProjectRoot "data\raw"
$SupervisorLog = Join-Path $ModelsDir "watchdog_supervisor.log"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Output $line
    Add-Content -Path $SupervisorLog -Value $line
}

function Get-MaxGeneration {
    if (-not (Test-Path $RawDir)) {
        return 0
    }
    $gens = Get-ChildItem $RawDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^nn_gen(\d+)$' } |
        ForEach-Object { [int]([regex]::Match($_.Name, '^nn_gen(\d+)$').Groups[1].Value) }
    if (-not $gens) {
        return 0
    }
    return ($gens | Measure-Object -Maximum).Maximum
}

function Start-OneIteration {
    param([int]$Seed)
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $ModelsDir "watchdog_run_${ts}.log"
    $err = Join-Path $ModelsDir "watchdog_run_${ts}.err.log"

    $args = @(
        "-3", "src/iterate.py",
        "--iterations", "$IterationsPerRun",
        "--games", "180",
        "--curriculum-games", "220",
        "--black-focus-games", "260",
        "--simulations", "120",
        "--curriculum-simulations", "50",
        "--black-focus-simulations", "100",
        "--epochs", "12",
        "--warmup-epochs", "3",
        "--warmup-start-factor", "0.1",
        "--position-budget", "40000",
        "--alternating",
        "--opponent-sims", "140",
        "--pool-size", "6",
        "--arena-games", "80",
        "--arena-sims", "80",
        "--arena-workers", "4",
        "--gate-threshold", "0.54",
        "--gate-min-other-side", "0.42",
        "--seed", "$Seed",
        "--human-eval"
    )

    $proc = Start-Process -FilePath "py" -ArgumentList $args -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru

    return [pscustomobject]@{
        Process = $proc
        OutLog = $out
        ErrLog = $err
        Started = Get-Date
        Mode = "watchdog-run"
    }
}

function Attach-CurrentProcess {
    param([int]$PidToAttach)
    $proc = Get-Process -Id $PidToAttach -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $null
    }
    $existingOut = Get-ChildItem $ModelsDir -Filter "overnight_*.log" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*.err.log" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $existingErr = Get-ChildItem $ModelsDir -Filter "overnight_*.err.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1

    return [pscustomobject]@{
        Process = $proc
        OutLog = if ($existingOut) { $existingOut.FullName } else { $null }
        ErrLog = if ($existingErr) { $existingErr.FullName } else { $null }
        Started = Get-Date
        Mode = "attached"
    }
}

function Get-LatestLogTimestampUtc {
    param([string]$OutLog, [string]$ErrLog)
    $times = @()
    if ($OutLog -and (Test-Path $OutLog)) {
        $times += (Get-Item $OutLog).LastWriteTimeUtc
    }
    if ($ErrLog -and (Test-Path $ErrLog)) {
        $times += (Get-Item $ErrLog).LastWriteTimeUtc
    }
    if (-not $times -or $times.Count -eq 0) {
        return $null
    }
    return ($times | Sort-Object -Descending | Select-Object -First 1)
}

New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
Write-Log "Watchdog started. TargetMaxGen=$TargetMaxGen IterationsPerRun=$IterationsPerRun Poll=${PollSeconds}s Stall=${StallMinutes}m SeedBase=$SeedBase"

$run = $null
if ($CurrentPid -gt 0) {
    $run = Attach-CurrentProcess -PidToAttach $CurrentPid
    if ($run) {
        Write-Log "Attached to existing PID=$CurrentPid OutLog=$($run.OutLog) ErrLog=$($run.ErrLog)"
    } else {
        Write-Log "Could not attach to PID=$CurrentPid (not running)"
    }
}

while ($true) {
    $maxGen = Get-MaxGeneration
    if ($TargetMaxGen -gt 0 -and $maxGen -ge $TargetMaxGen) {
        Write-Log "Target reached (max_gen=$maxGen >= $TargetMaxGen). Exiting watchdog."
        break
    }

    if (-not $run) {
        $seed = $SeedBase + $maxGen + 1
        $run = Start-OneIteration -Seed $seed
        Write-Log "Started watchdog run: PID=$($run.Process.Id) seed=$seed iterations=$IterationsPerRun out=$($run.OutLog) err=$($run.ErrLog)"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $procAlive = Get-Process -Id $run.Process.Id -ErrorAction SilentlyContinue
    if ($procAlive) {
        $latestUtc = Get-LatestLogTimestampUtc -OutLog $run.OutLog -ErrLog $run.ErrLog
        if ($latestUtc) {
            $idleMinutes = ((Get-Date).ToUniversalTime() - $latestUtc).TotalMinutes
            if ($idleMinutes -gt $StallMinutes) {
                Write-Log "Stall detected for PID=$($run.Process.Id): no log update for $([math]::Round($idleMinutes,1))m. Killing process."
                Stop-Process -Id $run.Process.Id -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 5
                $run = $null
                continue
            }
        }
    } else {
        $outTail = ""
        if ($run.OutLog -and (Test-Path $run.OutLog)) {
            $outTail = (Get-Content $run.OutLog -Tail 80 -ErrorAction SilentlyContinue) -join "`n"
        }
        if ($outTail -match "Iteration metadata saved") {
            Write-Log "Run PID=$($run.Process.Id) finished (normal completion marker found)."
        } else {
            Write-Log "Run PID=$($run.Process.Id) ended without completion marker. Restarting next loop."
        }
        $run = $null
        continue
    }

    Start-Sleep -Seconds $PollSeconds
}
