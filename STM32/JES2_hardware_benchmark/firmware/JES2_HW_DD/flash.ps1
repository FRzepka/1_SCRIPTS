param(
    [string]$Configuration = "Release",
    [string]$ProbeSerial = "004000283234510E33353533"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Programmer = "C:/Program Files/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI.exe"
$Elf = Join-Path $ProjectRoot "build/$Configuration/JES2_HW_DD.elf"

if (-not (Test-Path -LiteralPath $Elf)) {
    throw "Firmware image not found: $Elf. Run build.ps1 first."
}

& $Programmer -c "port=SWD" "sn=$ProbeSerial" "freq=4000" -w $Elf -v -rst
if ($LASTEXITCODE -ne 0) {
    throw "STM32 flashing failed"
}
