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

if ($LASTEXITCODE -eq 0) {
    & $python 'src/report.py' *>&1 |
        Out-File -FilePath (Join-Path $root 'run_report.log') -Encoding utf8
    'PHI3 COMPLETE'
} else {
    "PHI3 FAILED with exit code $LASTEXITCODE - see run_phi3.log"
}
