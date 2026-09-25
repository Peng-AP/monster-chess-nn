param([switch]$Apply, [switch]$Restore)
$ErrorActionPreference = 'Stop'
$repoPath = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$activeRoot = Join-Path $repoPath 'models/candidates'
$archiveRoot = Join-Path $repoPath 'models/archive/cleanup_gen47_20260907/candidates'
$manifestPath = Join-Path $repoPath 'benchmarks/cleanup_candidates_gen47_20260907.json'
# These checkpoints are referenced by the live generation, its source manifests,
# its queued transfer checks, or its resumable rehearsal. Releases live elsewhere.
$keep = @('bootstrap_main_gen_0042', 'bootstrap_main_gen_0044',
          'bootstrap_main_gen_0045', 'bootstrap_main_gen_0046',
          'bootstrap_main_gen_0047', 'bootstrap_rehearsal_stateful_gen47_20260907_gen_0001')
if ($Restore) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $entries = @($manifest.entries)
} else {
    if (Test-Path -LiteralPath $manifestPath) { throw 'Archive manifest already exists; use -Restore to undo it.' }
    $entries = @(Get-ChildItem -LiteralPath $activeRoot -Directory | Where-Object { $_.Name -notin $keep } | ForEach-Object {
        $sourceItem = $_
        if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing linked directory' }
        $children = @(Get-ChildItem -LiteralPath $sourceItem.FullName -Recurse -Force)
        if ($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }) { throw 'Refusing linked descendant' }
        $files = @($children | Where-Object { -not $_.PSIsContainer })
        [pscustomobject]@{name=$sourceItem.Name; files=$files.Count; bytes=($files|Measure-Object Length -Sum).Sum}
    })
}
$moves = @()
foreach ($entry in $entries) {
    if ($entry.name -ne [IO.Path]::GetFileName($entry.name)) { throw 'Invalid manifest directory name' }
    $source = [IO.Path]::GetFullPath((Join-Path $activeRoot $entry.name))
    $destination = [IO.Path]::GetFullPath((Join-Path $archiveRoot $entry.name))
    if ((Split-Path -Parent $source) -ne [IO.Path]::GetFullPath($activeRoot) -or
        (Split-Path -Parent $destination) -ne [IO.Path]::GetFullPath($archiveRoot)) { throw 'Target escaped declared roots' }
    if ($Restore) { $swap=$source; $source=$destination; $destination=$swap }
    if ((Resolve-Path -LiteralPath $source).Path -ne $source) { throw 'Unexpected source resolution' }
    if (Test-Path -LiteralPath $destination) { throw "Refusing overwrite: $destination" }
    $files = @(Get-ChildItem -LiteralPath $source -Recurse -File -Force)
    if ($files.Count -ne $entry.files -or ($files|Measure-Object Length -Sum).Sum -ne $entry.bytes) { throw "Source inventory mismatch: $source" }
    $moves += [pscustomobject]@{source=$source;destination=$destination;entry=$entry}
}
Write-Output "Validated $($moves.Count) candidate-directory moves; $((($entries|Measure-Object bytes -Sum).Sum / 1GB).ToString('F2')) GiB. Apply=$Apply Restore=$Restore"
if (-not $Apply) { $entries | Select-Object name,files,bytes; return }
if (-not $Restore) {
    [pscustomobject]@{schema_version=1; created=(Get-Date -Format o); active_root=$activeRoot; archive_root=$archiveRoot; retained=$keep; entries=$entries} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}
foreach ($move in $moves) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $move.destination) -Force | Out-Null
    Move-Item -LiteralPath $move.source -Destination $move.destination
    $files = @(Get-ChildItem -LiteralPath $move.destination -Recurse -File -Force)
    if ($files.Count -ne $move.entry.files -or ($files|Measure-Object Length -Sum).Sum -ne $move.entry.bytes) { throw 'Post-move inventory mismatch' }
    Write-Output "Moved and verified $($move.entry.name)"
}
