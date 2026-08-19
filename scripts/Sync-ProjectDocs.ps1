param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("立项创建", "需求变更", "任务状态变化", "技术决策", "风险识别", "问题关闭", "版本发布")]
    [string]$Event,
    [string]$Detail = ""
)

$root = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$eventMap = @{
    "立项创建"   = @("项目需求说明.md", "任务清单.md", "项目路线图.md", "风险登记.md", "决策记录.md", "项目文档索引.md", "智能旅行管家-项目大纲.md", "文档自动更新规则.md")
    "需求变更"   = @("项目需求说明.md", "任务清单.md", "风险登记.md", "智能旅行管家-项目大纲.md")
    "任务状态变化" = @("任务清单.md", "项目路线图.md")
    "技术决策"   = @("决策记录.md", "风险登记.md")
    "风险识别"   = @("风险登记.md", "任务清单.md")
    "问题关闭"   = @("问题记录.md", "经验教训.md")
    "版本发布"   = @("项目路线图.md", "README.md", "项目文档索引.md", "backend/README.md")
}

$targets = $eventMap[$Event]

foreach ($file in $targets) {
    $path = Join-Path $root $file
    if (!(Test-Path $path)) { continue }

    $content = Get-Content -Path $path -Raw -Encoding utf8
    $updateLine = "- 最后自动更新：$timestamp（事件：$Event）"

    if ($content -match "(?m)^- 最后自动更新：.*$") {
        $newContent = [regex]::Replace($content, "(?m)^- 最后自动更新：.*$", $updateLine)
    }
    elseif ($content -match "(?m)^# .+$") {
        $newContent = [regex]::Replace($content, "(?m)^(# .+)$", "`$1`r`n`r`n$updateLine", 1)
    }
    else {
        $newContent = "$updateLine`r`n`r`n$content"
    }

    Set-Content -Path $path -Value $newContent -Encoding utf8
}

$logFile = Join-Path $root "自动更新事件日志.md"
if (!(Test-Path $logFile)) {
    @"
# 自动更新事件日志

| 时间 | 事件 | 详情 | 更新文件 |
|---|---|---|---|
"@ | Set-Content -Path $logFile -Encoding utf8
}

$detailText = if ([string]::IsNullOrWhiteSpace($Detail)) { "-" } else { $Detail }
$targetText = ($targets -join "、")
"| $timestamp | $Event | $detailText | $targetText |" | Add-Content -Path $logFile -Encoding utf8

Write-Output "已处理事件：$Event"
Write-Output "更新文件：$targetText"
