param(
    [int[]]$ImageCounts = @(1, 3, 10),
    [ValidateSet("cold", "warm", "both")]
    [string]$StartupMode = "both",
    [ValidateSet("auto", "parallel", "sequential")]
    [string]$ChannelExecutionMode = "auto",
    [int]$BootstrapSlots = 0,
    [int]$BootstrapCorrectionFactor = 0,
    [string]$CacheDir = "D:\Manipal\Research\Implementation\final\cache\ckks_rgb",
    [switch]$ResetCache,
    [string]$OpenFheInstall = "D:\Manipal\Research\openfhe-install",
    [string]$Msys2Bin = "C:\msys64\mingw64\bin",
    [string]$BuildDir = "D:\Manipal\Research\Implementation\final\build_bench_msys",
    [string]$OutputDir = "",
    [int]$BuildJobs = 8,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = "D:\Manipal\Research\Implementation"
$finalDir = Join-Path $repoRoot "final"
$gccPath = Join-Path $Msys2Bin "gcc.exe"
$gxxPath = Join-Path $Msys2Bin "g++.exe"

if (-not (Test-Path $Msys2Bin)) {
    throw "MSYS2 bin path not found: $Msys2Bin"
}
if (-not (Test-Path (Join-Path $OpenFheInstall "include\openfhe"))) {
    throw "OpenFHE install not found: $OpenFheInstall"
}
if (-not (Test-Path $gccPath) -or -not (Test-Path $gxxPath)) {
    throw "MSYS2 GCC/G++ not found under $Msys2Bin"
}

$env:Path = "$OpenFheInstall\lib;$Msys2Bin;$env:Path"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $finalDir "benchmarks\$stamp"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

if (-not $SkipBuild) {
    Write-Host "Configuring benchmark build (MSYS2 GCC, Release)..." -ForegroundColor Cyan
    cmake -S $finalDir -B $BuildDir -G Ninja `
        -DOPENFHE_INSTALL_DIR="$OpenFheInstall" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_C_COMPILER="$gccPath" `
        -DCMAKE_CXX_COMPILER="$gxxPath"

    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed"
    }

    Write-Host "Building ckks_medmnist_rgb target..." -ForegroundColor Cyan
    cmake --build $BuildDir --target ckks_medmnist_rgb --parallel $BuildJobs
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
}

$exePath = Join-Path $BuildDir "bin\ckks_medmnist_rgb.exe"
if (-not (Test-Path $exePath)) {
    throw "Benchmark executable not found: $exePath"
}

$env:CKKS_CHANNEL_EXECUTION_MODE = $ChannelExecutionMode
Write-Host "CKKS_CHANNEL_EXECUTION_MODE=$ChannelExecutionMode" -ForegroundColor Yellow
if ($BootstrapSlots -gt 0) {
    $env:CKKS_BOOTSTRAP_SLOTS = [string]$BootstrapSlots
    Write-Host "CKKS_BOOTSTRAP_SLOTS=$BootstrapSlots" -ForegroundColor Yellow
} else {
    Remove-Item Env:CKKS_BOOTSTRAP_SLOTS -ErrorAction SilentlyContinue
}

if ($BootstrapCorrectionFactor -gt 0) {
    $env:CKKS_BOOTSTRAP_CORRECTION_FACTOR = [string]$BootstrapCorrectionFactor
    Write-Host "CKKS_BOOTSTRAP_CORRECTION_FACTOR=$BootstrapCorrectionFactor" -ForegroundColor Yellow
} else {
    Remove-Item Env:CKKS_BOOTSTRAP_CORRECTION_FACTOR -ErrorAction SilentlyContinue
}

$env:CKKS_CACHE_DIR = $CacheDir
if ($ResetCache -and (Test-Path $CacheDir)) {
    Remove-Item -Path $CacheDir -Recurse -Force
}

$startupModes = @()
if ($StartupMode -eq "both") {
    $startupModes = @("cold", "warm")
} else {
    $startupModes = @($StartupMode)
}

function Get-MetricValue {
    param(
        [string]$Text,
        [string]$Pattern,
        [bool]$Required = $true
    )

    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $m.Success) {
        if ($Required) {
            throw "Could not parse metric using pattern: $Pattern"
        }
        return [double]::NaN
    }

    return [double]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-IntValue {
    param(
        [string]$Text,
        [string]$Pattern,
        [bool]$Required = $true
    )

    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $m.Success) {
        if ($Required) {
            throw "Could not parse integer metric using pattern: $Pattern"
        }
        return [int]::MinValue
    }

    return [int]::Parse($m.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

$rows = @()

foreach ($count in $ImageCounts) {
    foreach ($mode in $startupModes) {
        if ($mode -eq "warm") {
            $env:CKKS_WARM_START = "1"
        } else {
            $env:CKKS_WARM_START = "0"
        }

        $logPath = Join-Path $OutputDir ("rgb_{0}images_{1}.log" -f $count, $mode)

        Write-Host "Running benchmark for $count image(s), mode=$mode..." -ForegroundColor Cyan
        & $exePath $count *> $logPath
        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark run failed for $count image(s), mode=$mode. See: $logPath"
        }

        $text = Get-Content -Path $logPath -Raw
        $channelCount = Get-IntValue $text 'Color mode:\s+.*\((\d+) channel' $false
        $effectiveBootstrapSlots = Get-IntValue $text 'Bootstrap slots:\s+([0-9]+)' $false
        $effectiveCorrectionFactor = Get-IntValue $text 'Bootstrap correction factor:\s+([0-9]+)' $false
        $warmStartUsedMatch = [regex]::Match($text, 'Warm-start used:\s+(yes|no)', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        $warmStartUsed = if ($warmStartUsedMatch.Success) { $warmStartUsedMatch.Groups[1].Value.ToLowerInvariant() } else { "unknown" }
        $effectiveChannelCount = if ($channelCount -ne [int]::MinValue) { [double]$channelCount } else { 3.0 }
        $bootstrappingAvgPerChannelMs = [math]::Round((Get-MetricValue $text 'Bootstrapping \(sum ch\):\s+([0-9.]+)\s+ms') / ($effectiveChannelCount * [double]$count), 2)

        $row = [ordered]@{
            Images = $count
            StartupMode = $mode
            WarmStartRequested = if ($mode -eq "warm") { "yes" } else { "no" }
            WarmStartUsed = $warmStartUsed
            ChannelExecutionMode = $ChannelExecutionMode
            CacheDir = $CacheDir
            BootstrapSlotsRequested = if ($BootstrapSlots -gt 0) { $BootstrapSlots } else { "auto" }
            BootstrapCorrectionFactorRequested = if ($BootstrapCorrectionFactor -gt 0) { $BootstrapCorrectionFactor } else { "auto" }
            BootstrapSlotsEffective = if ($effectiveBootstrapSlots -ne [int]::MinValue) { $effectiveBootstrapSlots } else { "unknown" }
            BootstrapCorrectionFactorEffective = if ($effectiveCorrectionFactor -ne [int]::MinValue) { $effectiveCorrectionFactor } else { "unknown" }
            WeightLoadingMs = Get-MetricValue $text 'Weight Loading:\s+([0-9.]+)\s+ms'
            ContextSetupMs = Get-MetricValue $text 'Context Setup:\s+([0-9.]+)\s+ms'
            BootstrapSetupMs = Get-MetricValue $text 'Bootstrap Setup:\s+([0-9.]+)\s+ms'
            KeyGenerationMs = Get-MetricValue $text 'Key Generation:\s+([0-9.]+)\s+ms'
            MultKeyGenMs = Get-MetricValue $text 'Mult Key Gen:\s+([0-9.]+)\s+ms'
            BootstrapKeyGenMs = Get-MetricValue $text 'Bootstrap Key Gen:\s+([0-9.]+)\s+ms'
            EncryptionSumChMs = Get-MetricValue $text 'Encryption \(sum ch\):\s+([0-9.]+)\s+ms'
            Layers1to3SumChMs = Get-MetricValue $text 'Layers 1-3 \(sum ch\):\s+([0-9.]+)\s+ms'
            BootstrappingSumChMs = Get-MetricValue $text 'Bootstrapping \(sum ch\):\s+([0-9.]+)\s+ms'
            BootstrappingAvgPerChannelMs = $bootstrappingAvgPerChannelMs
            Layers4to5SumChMs = Get-MetricValue $text 'Layers 4-5 \(sum ch\):\s+([0-9.]+)\s+ms'
            DecryptionSumChMs = Get-MetricValue $text 'Decryption \(sum ch\):\s+([0-9.]+)\s+ms'
            ChannelPipelineWallMs = Get-MetricValue $text 'Channel Pipeline \(wall\):\s+([0-9.]+)\s+ms'
            ClassificationMs = Get-MetricValue $text 'Classification:\s+([0-9.]+)\s+ms'
            PlaintextReferenceMs = Get-MetricValue $text 'Plaintext Reference:\s+([0-9.]+)\s+ms'
            SecurityMetricsMs = Get-MetricValue $text 'Security Metrics:\s+([0-9.]+)\s+ms'
            TotalMs = Get-MetricValue $text 'TOTAL:\s+([0-9.]+)\s+ms'
            PerImageTotalMs = [math]::Round((Get-MetricValue $text 'TOTAL:\s+([0-9.]+)\s+ms') / [double]$count, 2)
            LogFile = $logPath
        }

        $rows += [pscustomobject]$row
    }
}

$csvPath = Join-Path $OutputDir "summary.csv"
$rows | Export-Csv -Path $csvPath -NoTypeInformation

Write-Host "`nBenchmark complete." -ForegroundColor Green
Write-Host "Summary CSV: $csvPath" -ForegroundColor Green
Write-Host "Logs directory: $OutputDir" -ForegroundColor Green
Write-Host ""
$rows | Format-Table -AutoSize
