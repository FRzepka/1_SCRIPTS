param(
    [string]$Port = "COM7",
    [string]$Cell = "C09",
    [int]$MaxRows = 2025,
    [string[]]$Models = @("DM", "HDM", "HECM", "DD", "DDS", "DDP"),
    [string]$ProbeSerial = "004000283234510E33353533"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $Root "../..")).Path
$Programmer = "C:/Program Files/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI.exe"
$Vectors = Join-Path $Root "test_vectors/multicell/jes2_nominal_${Cell}_vectors.csv"
$OutRoot = Join-Path $Root "results/runtime_memory"
$AllowedModels = @("DM", "HDM", "HECM", "DD", "DDS", "DDP")

if (-not (Test-Path -LiteralPath $Programmer)) {
    throw "STM32CubeProgrammer not found: $Programmer"
}
if (-not (Test-Path -LiteralPath $Vectors)) {
    throw "Test vectors not found: $Vectors"
}
foreach ($Model in $Models) {
    if ($AllowedModels -notcontains $Model) {
        throw "Unknown model: $Model"
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$Rows = @()

foreach ($Model in $Models) {
    $Project = Join-Path $Root "firmware/JES2_HW_$Model"
    $Elf = Join-Path $Project "build/Release/JES2_HW_$Model.elf"
    $OutDir = Join-Path $OutRoot $Model

    Write-Host "[$Model] Building profiling firmware"
    & (Join-Path $Project "build.ps1") -Configuration Release
    if ($LASTEXITCODE -ne 0) { throw "$Model build failed" }

    Write-Host "[$Model] Flashing STM32"
    & (Join-Path $Project "flash.ps1") -Configuration Release -ProbeSerial $ProbeSerial
    if ($LASTEXITCODE -ne 0) { throw "$Model flash failed" }
    Start-Sleep -Seconds 1
    & $Programmer -c "port=SWD" "sn=$ProbeSerial" -run | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not start $Model firmware" }
    Start-Sleep -Milliseconds 500

    Write-Host "[$Model] Running $MaxRows samples from $Cell"
    python (Join-Path $Root "scripts/collect_serial_benchmark.py") `
        --port $Port `
        --model $Model `
        --vectors $Vectors `
        --out-dir $OutDir `
        --rounds 1 `
        --max-rows $MaxRows `
        --reset-wait-s 0.5
    if ($LASTEXITCODE -ne 0) { throw "$Model runtime-memory benchmark failed" }

    $Summary = Get-Content (Join-Path $OutDir "summary.json") -Raw | ConvertFrom-Json
    if ($null -eq $Summary.runtime_memory) {
        throw "$Model did not return a runtime-memory profile"
    }
    $Rows += [pscustomobject]@{
        model = $Model
        cell = $Cell
        samples = $Summary.rows_total
        valid_inferences = $Summary.rows_ok
        firmware_revision = $Summary.device.firmware_revision
        elf_sha256 = (Get-FileHash -LiteralPath $Elf -Algorithm SHA256).Hash.ToLowerInvariant()
        data_bytes = $Summary.runtime_memory.data_bytes
        bss_bytes = $Summary.runtime_memory.bss_bytes
        static_bytes = $Summary.runtime_memory.static_bytes
        heap_peak_bytes = $Summary.runtime_memory.heap_peak_bytes
        stack_peak_bytes = $Summary.runtime_memory.stack_peak_bytes
        total_peak_bytes = $Summary.runtime_memory.total_peak_bytes
    }
}

$CsvPath = Join-Path $OutRoot "runtime_memory_summary.csv"
$JsonPath = Join-Path $OutRoot "runtime_memory_measurements.json"
$Rows | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding UTF8
$Payload = [ordered]@{
    schema_version = 1
    created_utc = [DateTime]::UtcNow.ToString("o")
    method = "On-device stack watermark plus _sbrk heap high-water mark; .data and .bss from linker symbols"
    cell = $Cell
    max_rows = $MaxRows
    git_revision = (& git -C $RepoRoot rev-parse HEAD).Trim()
    models = $Rows
}
$JsonText = $Payload | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($JsonPath, $JsonText, [Text.UTF8Encoding]::new($false))

Write-Host "Runtime-memory results written to $OutRoot"
$Rows | Format-Table model, static_bytes, heap_peak_bytes, stack_peak_bytes, total_peak_bytes -AutoSize
