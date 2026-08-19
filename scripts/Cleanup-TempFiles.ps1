param(
    [switch]$Quiet
)

$root = Split-Path -Parent $PSScriptRoot

$fixedTargets = @(
    ".logs",
    ".npm-cache",
    ".npm-tmp",
    ".tmp-pip",
    ".tmp_12306_pkg",
    ".vendor12306",
    ".venv12306"
)

$patternTargets = @(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache"
)

function Remove-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (!(Test-Path -LiteralPath $Path)) {
        return $false
    }

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $normalizedRoot = [System.IO.Path]::GetFullPath($root)

    if (!$resolved.StartsWith($normalizedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete path outside repository: $resolved"
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop
    return $true
}

$removed = New-Object System.Collections.Generic.List[string]
$failed = New-Object System.Collections.Generic.List[string]

foreach ($target in $fixedTargets) {
    $fullPath = Join-Path $root $target
    try {
        if (Remove-RepoPath -Path $fullPath) {
            $removed.Add($target) | Out-Null
        }
    }
    catch {
        $failed.Add("$target :: $($_.Exception.Message)") | Out-Null
    }
}

$patternMatches = Get-ChildItem -Path $root -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $patternTargets -contains $_.Name }

foreach ($match in $patternMatches) {
    $relative = $match.FullName.Substring($root.Length).TrimStart("\", "/")
    try {
        if (Remove-RepoPath -Path $match.FullName) {
            $removed.Add($relative) | Out-Null
        }
    }
    catch {
        $failed.Add("$relative :: $($_.Exception.Message)") | Out-Null
    }
}

if (-not $Quiet) {
    if ($removed.Count -eq 0) {
        Write-Output "未发现需要清理的临时文件或缓存目录。"
    }
    else {
        Write-Output "已清理以下临时文件或缓存目录："
        $removed | Sort-Object -Unique | ForEach-Object { Write-Output "- $_" }
    }

    if ($failed.Count -gt 0) {
        Write-Output "以下目录未能清理："
        $failed | Sort-Object -Unique | ForEach-Object { Write-Output "- $_" }
    }
}
