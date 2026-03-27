param(
  [string]$CubeIdeRoot = "C:\\ST\\STM32CubeIDE_1.17.0\\STM32CubeIDE",
  [string]$ProgrammerCli = "C:\\Program Files\\STMicroelectronics\\STM32Cube\\STM32CubeProgrammer\\bin\\STM32_Programmer_CLI.exe",
  [int]$SwdFreqKHz = 1000,
  [string]$StLinkSerial = ""
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $here "..\\..\\..\\..\\..\\..")  # .../1_Scripts
$fwRoot = Join-Path $repoRoot "STM32DC\\mg-farm-4s-bms-PA--mg-farm-4s-bms\\Firmware\\EET_MG_FARM_BMS"

$cm7Debug = Join-Path $fwRoot "CM7\\Debug"
$cm4Debug = Join-Path $fwRoot "CM4\\Debug"

if (!(Test-Path $cm7Debug)) { throw "Missing: $cm7Debug" }
if (!(Test-Path $cm4Debug)) { throw "Missing: $cm4Debug" }
if (!(Test-Path $ProgrammerCli)) { throw "Missing STM32CubeProgrammer CLI: $ProgrammerCli" }

# Resolve make + GCC from CubeIDE installation.
$makeExe = Join-Path $CubeIdeRoot "plugins\\com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.0.202409170845\\tools\\bin\\make.exe"
if (!(Test-Path $makeExe)) { throw "Missing make.exe: $makeExe" }

$gnuPlugin = Get-ChildItem -Path (Join-Path $CubeIdeRoot "plugins") -Filter "com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*" | Select-Object -First 1
if ($null -eq $gnuPlugin) { throw "Could not find GNU tools plugin under: $CubeIdeRoot\\plugins" }
$gnuBin = Join-Path $gnuPlugin.FullName "tools\\bin"
if (!(Test-Path (Join-Path $gnuBin "arm-none-eabi-gcc.exe"))) { throw "Missing arm-none-eabi-gcc.exe under: $gnuBin" }

# Put toolchain on PATH for the makefiles.
$env:PATH = ($gnuBin + ";" + (Split-Path -Parent $makeExe) + ";" + $env:PATH)

Push-Location $repoRoot

try {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $logRoot = Join-Path $repoRoot "DL_Models\\LFP_SOH_Optimization_Study\\6_test\\STM32DC\\LSTM_0.1.2.3"
  New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
  $logFile = Join-Path $logRoot ("BUILD_FLASH_" + $ts + ".log")

  "[$ts] build+flash started" | Out-File -FilePath $logFile -Encoding utf8
  "CubeIDE: $CubeIdeRoot" | Out-File -FilePath $logFile -Append -Encoding utf8
  "GNU: $gnuBin" | Out-File -FilePath $logFile -Append -Encoding utf8
  "Programmer: $ProgrammerCli" | Out-File -FilePath $logFile -Append -Encoding utf8
  "" | Out-File -FilePath $logFile -Append -Encoding utf8

  Write-Host "Clean/build CM4..."
  # IMPORTANT: run clean and build as separate invocations; `make clean all -j`
  # can race (clean deleting .o while link runs).
  $cmd = '"' + $makeExe + '" -C "' + $cm4Debug + '" clean >> "' + $logFile + '" 2>&1'
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "CM4 build failed (see $logFile)" }
  $cmd = '"' + $makeExe + '" -C "' + $cm4Debug + '" all -j8 >> "' + $logFile + '" 2>&1'
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "CM4 build failed (see $logFile)" }

  Write-Host "Clean/build CM7..."
  $cmd = '"' + $makeExe + '" -C "' + $cm7Debug + '" clean >> "' + $logFile + '" 2>&1'
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "CM7 build failed (see $logFile)" }
  $cmd = '"' + $makeExe + '" -C "' + $cm7Debug + '" all -j8 >> "' + $logFile + '" 2>&1'
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "CM7 build failed (see $logFile)" }

  $cm4Elf = Join-Path $cm4Debug "EET_MG_FARM_BMS_CM4.elf"
  $cm7Elf = Join-Path $cm7Debug "EET_MG_FARM_BMS_CM7.elf"
  if (!(Test-Path $cm4Elf)) { throw "Missing build output: $cm4Elf" }
  if (!(Test-Path $cm7Elf)) { throw "Missing build output: $cm7Elf" }

  $connect = @("port=SWD", "freq=$SwdFreqKHz", "mode=UR", "reset=HWrst")
  if ($StLinkSerial) { $connect += @("sn=$StLinkSerial") }

  Write-Host "Flash CM7+CM4..."
  $connStr = ($connect -join " ")
  $cmd = '"' + $ProgrammerCli + '" -c ' + $connStr + ' -d "' + $cm7Elf + '" -d "' + $cm4Elf + '" -rst >> "' + $logFile + '" 2>&1'
  cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Flash failed (see $logFile)" }

  Write-Host "OK. Log: $logFile"
} finally {
  Pop-Location
}
