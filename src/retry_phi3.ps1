# Waits for enough commit charge to memory-map Phi-3's safetensors shards, then
# evaluates it and merges the rows into results/benchmark.csv.
# The blocker is commit charge, not VRAM: this machine has a fixed pagefile, so a
# concurrent process holding commit makes the load fail with os error 1455.

$ErrorActionPreference = 'Stop'
$root = 'C:\Users\ASUS\benchmark'
$python = Join-Path $root '.venv\Scripts\python.exe'
$needGB = 10          # largest shard is 4.97 GB; this leaves real headroom
$deadline = (Get-Date).AddHours(6)

Set-Location $root

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
& $python 'src/run.py' '--models' 'Phi-3-mini' *>&1 |
    Out-File -FilePath (Join-Path $root 'run_phi3.log') -Encoding utf8

if ($LASTEXITCODE -ne 0) {
    "PHI3 FAILED with exit code $LASTEXITCODE - see run_phi3.log"
    exit 1
}

# Only claim success if the rows actually landed in the CSV.
$csv = Import-Csv (Join-Path $root 'results\benchmark.csv')
$phi = @($csv | Where-Object { $_.model -eq 'Phi-3-mini' })
if ($phi.Count -lt 2) {
    "PHI3 INCOMPLETE: only $($phi.Count) row(s) in benchmark.csv"
    exit 1
}

& $python 'src/report.py' *>&1 |
    Out-File -FilePath (Join-Path $root 'run_report.log') -Encoding utf8

& git add -A
& git commit -m @'
Add Phi-3-mini's rows to the benchmark

The evaluation itself was never the blocker: loading Phi-3-mini needs about
7.6 GB of commit charge to memory-map its safetensors shards, and this machine
has a fixed pagefile, so the load failed whenever another process held commit.
Re-run once that cleared.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
'@
& git push origin master

if ($LASTEXITCODE -eq 0) { 'PHI3 COMPLETE AND PUSHED' } else { 'PHI3 DONE BUT PUSH FAILED' }
