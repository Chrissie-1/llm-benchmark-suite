# Waits for enough commit charge to memory-map Phi-3's safetensors shards, then
# evaluates it, regenerates the report, and pushes the result.
# The blocker is commit charge, not VRAM: this machine has a fixed pagefile, so a
# concurrent process holding commit makes the load fail with os error 1455.

$ErrorActionPreference = 'Stop'
$root = 'C:\Users\ASUS\benchmark'
$python = Join-Path $root '.venv\Scripts\python.exe'
$needGB = 10          # largest shard is 4.97 GB; this leaves real headroom
$deadline = (Get-Date).AddHours(6)

Set-Location $root

# Run a native exe without the PowerShell pipeline. Piping or *>&1 in PS 5.1 wraps
# each stderr line in an ErrorRecord, which under ErrorActionPreference='Stop'
# turns a harmless warning (e.g. the HF "unauthenticated requests" notice) into a
# terminating error. Start-Process keeps stderr as plain text in a file.
function Invoke-Native {
    # Not $Args: that is a PowerShell automatic variable, and a parameter of that
    # name arrives empty.
    param([string]$Exe, [string[]]$ExeArgs, [string]$LogPath)

    # Start-Process joins the array with spaces and does no quoting of its own, so
    # any argument containing a space would be split into two.
    $quoted = $ExeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }

    $errPath = "$LogPath.err"
    $proc = Start-Process -FilePath $Exe -ArgumentList $quoted -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $LogPath -RedirectStandardError $errPath
    if (Test-Path $errPath) {
        Get-Content $errPath | Add-Content -Path $LogPath -Encoding utf8
        Remove-Item $errPath -Force
    }
    return $proc.ExitCode
}

while ($true) {
    $freeGB = (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -like '*crucible*' })

    if (($freeGB -ge $needGB) -and ($busy.Count -eq 0)) {
        "READY: commit free {0:N1} GB, no crucible process. Starting Phi-3." -f $freeGB
        break
    }
    if ((Get-Date) -gt $deadline) {
        "GAVE UP after 6h: commit free {0:N1} GB, crucible processes {1}" -f $freeGB, $busy.Count
        exit 2
    }
    Start-Sleep -Seconds 120
}

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$code = Invoke-Native $python @('src/run.py', '--models', 'Phi-3-mini') (Join-Path $root 'run_phi3.log')
if ($code -ne 0) {
    "PHI3 FAILED with exit code $code - see run_phi3.log"
    exit 1
}

# Only claim success if both rows actually landed in the CSV.
$phi = @(Import-Csv (Join-Path $root 'results\benchmark.csv') |
         Where-Object { $_.model -eq 'Phi-3-mini' })
if ($phi.Count -lt 2) {
    "PHI3 INCOMPLETE: only $($phi.Count) row(s) in benchmark.csv"
    exit 1
}

$code = Invoke-Native $python @('src/report.py') (Join-Path $root 'run_report.log')
if ($code -ne 0) {
    "REPORT FAILED with exit code $code - see run_report.log"
    exit 1
}

$msg = @'
Add Phi-3-mini's rows to the benchmark

The evaluation itself was never the blocker: loading Phi-3-mini needs about
7.6 GB of commit charge to memory-map its safetensors shards, and this machine
has a fixed pagefile, so the load failed whenever another process held commit.
Re-run once that cleared.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
'@
$msgFile = Join-Path $env:TEMP 'phi3_commit_msg.txt'
Set-Content -Path $msgFile -Value $msg -Encoding utf8

$code = Invoke-Native 'git' @('add', '-A') (Join-Path $root 'run_git.log')
if ($code -eq 0) {
    $code = Invoke-Native 'git' @('commit', '-F', $msgFile) (Join-Path $root 'run_git.log')
}
if ($code -eq 0) {
    $code = Invoke-Native 'git' @('push', 'origin', 'master') (Join-Path $root 'run_git.log')
}

if ($code -eq 0) { 'PHI3 COMPLETE AND PUSHED' } else { "PHI3 DONE BUT GIT STEP FAILED ($code) - see run_git.log" }
