param([switch]$Apply, [switch]$Restore)
$ErrorActionPreference = 'Stop'
$repoPath = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$manifestPath = Join-Path $PSScriptRoot 'cleanup_models_books_20260906.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$moves = @()
foreach ($entry in $manifest.entries) {
    $source = [IO.Path]::GetFullPath((Join-Path $repoPath $entry.source))
    $destination = [IO.Path]::GetFullPath((Join-Path $repoPath $entry.destination))
    $activeRoot = if ($entry.kind -eq 'model') { Join-Path $repoPath 'models/candidates' } else { Join-Path $repoPath 'books' }
    $archiveRoot = if ($entry.kind -eq 'model') { Join-Path $repoPath 'models/archive/cleanup_20260906/candidates' } else { Join-Path $repoPath 'books/archive/cleanup_20260906' }
    # Exact direct-child checks, not just a broad workspace-prefix test.
    if ((Split-Path -Parent $source) -ne [IO.Path]::GetFullPath($activeRoot) -or
        (Split-Path -Parent $destination) -ne [IO.Path]::GetFullPath($archiveRoot)) {
        throw "Manifest target escaped its declared directory: $source"
    }
    if ($Restore) { $swap = $source; $source = $destination; $destination = $swap }
    $resolvedSource = (Resolve-Path -LiteralPath $source).Path
    if ($resolvedSource -ne $source) { throw "Unexpected source resolution: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Refusing to overwrite: $destination" }
    $sourceItem = Get-Item -LiteralPath $source
    if ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing reparse point: $source" }
    $files = @(if ($sourceItem.PSIsContainer) { Get-ChildItem -LiteralPath $source -Recurse -File } else { $sourceItem })
    $bytes = ($files | Measure-Object Length -Sum).Sum
    if ($files.Count -ne $entry.files -or $bytes -ne $entry.bytes) { throw "Source changed since inventory: $source" }
    $moves += [pscustomobject]@{Source=$source;Destination=$destination;Entry=$entry}
}
Write-Output "Validated $($moves.Count) exact moves. Apply=$Apply Restore=$Restore"
if (-not $Apply) { return }
foreach ($move in $moves) {
    $parent = Split-Path -Parent $move.Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Move-Item -LiteralPath $move.Source -Destination $move.Destination
    $item = Get-Item -LiteralPath $move.Destination
    $files = @(if ($item.PSIsContainer) { Get-ChildItem -LiteralPath $item.FullName -Recurse -File } else { $item })
    if ($files.Count -ne $move.Entry.files -or ($files | Measure-Object Length -Sum).Sum -ne $move.Entry.bytes) {
        throw "Archive verification failed: $($move.Destination)"
    }
    Write-Output "Moved and verified: $($move.Entry.source)"
}
