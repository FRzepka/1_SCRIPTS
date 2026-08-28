param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ProjectRoot "../../../../")).Path
$ToolRoot = "C:/ST/STM32CubeIDE_1.17.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.win32_1.0.0.202411081344/tools/bin"
$Gcc = Join-Path $ToolRoot "arm-none-eabi-gcc.exe"
$Objcopy = Join-Path $ToolRoot "arm-none-eabi-objcopy.exe"
$Size = Join-Path $ToolRoot "arm-none-eabi-size.exe"

if (-not (Test-Path -LiteralPath $Gcc)) {
    throw "STM32 GCC toolchain not found at $Gcc"
}

$BuildDir = Join-Path $ProjectRoot "build/$Configuration"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$GitSha = (& git -C $RepoRoot rev-parse --short=8 HEAD).Trim()
$FirmwarePath = "STM32/JES2_hardware_benchmark/firmware/JES2_HW_DDS"
$FirmwareStatus = & git -C $RepoRoot status --porcelain -- $FirmwarePath
if ($FirmwareStatus) {
    $GitSha += "-dirty"
}
$Optimization = if ($Configuration -eq "Release") { "-O3" } else { "-Og" }
$Defines = @(
    "-DUSE_PWR_LDO_SUPPLY",
    "-DUSE_HAL_DRIVER",
    "-DSTM32H753xx",
    ('-DFIRMWARE_GIT_SHA=\"' + $GitSha + '\"')
)
$Includes = @(
    "-I$ProjectRoot/Core/Inc",
    "-I$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Inc",
    "-I$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Inc/Legacy",
    "-I$ProjectRoot/Drivers/CMSIS/Device/ST/STM32H7xx/Include",
    "-I$ProjectRoot/Drivers/CMSIS/Include"
)
$Common = @(
    "-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard",
    "-std=gnu11", $Optimization, "-g3", "-ffunction-sections", "-fdata-sections",
    "-Wall", "-Wextra", "-Werror=implicit-function-declaration", "--specs=nano.specs"
) + $Defines + $Includes

$Sources = @(
    "$ProjectRoot/Core/Src/main.c",
    "$ProjectRoot/Core/Src/dd_model.c",
    "$ProjectRoot/Core/Src/dd_weights.c",
    "$ProjectRoot/Core/Src/stm32h7xx_it.c",
    "$ProjectRoot/Core/Src/syscalls.c",
    "$ProjectRoot/Core/Src/sysmem.c",
    "$ProjectRoot/Core/Src/system_stm32h7xx.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_cortex.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_exti.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_flash.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_flash_ex.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_gpio.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pwr.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_pwr_ex.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_rcc.c",
    "$ProjectRoot/Drivers/STM32H7xx_HAL_Driver/Src/stm32h7xx_hal_rcc_ex.c"
)

$Objects = @()
foreach ($Source in $Sources) {
    $Object = Join-Path $BuildDir ([System.IO.Path]::GetFileNameWithoutExtension($Source) + ".o")
    & $Gcc @Common -c $Source -o $Object
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $Source" }
    $Objects += $Object
}

$StartupObject = Join-Path $BuildDir "startup_stm32h753zitx.o"
& $Gcc "-mcpu=cortex-m7" "-mthumb" "-mfpu=fpv5-d16" "-mfloat-abi=hard" -c "$ProjectRoot/Core/Startup/startup_stm32h753zitx.s" -o $StartupObject
if ($LASTEXITCODE -ne 0) { throw "Startup assembly failed" }
$Objects += $StartupObject

$Elf = Join-Path $BuildDir "JES2_HW_DDS.elf"
$Map = Join-Path $BuildDir "JES2_HW_DDS.map"
& $Gcc @Objects "-mcpu=cortex-m7" "-mthumb" "-mfpu=fpv5-d16" "-mfloat-abi=hard" "-T$ProjectRoot/STM32H753ZITX_FLASH.ld" "--specs=nosys.specs" "--specs=nano.specs" "-Wl,-Map=$Map" "-Wl,--gc-sections" "-Wl,-u,_printf_float" "-Wl,--start-group" "-lc" "-lm" "-Wl,--end-group" -o $Elf
if ($LASTEXITCODE -ne 0) { throw "Linking failed" }

$Bin = Join-Path $BuildDir "JES2_HW_DDS.bin"
& $Objcopy -O binary $Elf $Bin
if ($LASTEXITCODE -ne 0) { throw "Binary export failed" }
& $Size $Elf
Write-Output "Built $Elf"
