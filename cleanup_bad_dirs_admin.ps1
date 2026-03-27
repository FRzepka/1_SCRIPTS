<# 
Run this script from an *elevated* PowerShell (Run as Administrator).

Purpose: remove a few leftover temp folders that ended up with broken ACLs:
  - `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\_chmod_test`
  - `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\_mkdir_mode_700`
  - `C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\_pytemp`
  - `...\STM32DC\...\tools\pc_sim\.tmp`

If you don't see those folders, you can ignore this script.
#>

$ErrorActionPreference = "Stop"

$root = "C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts"
$paths = @(
  (Join-Path $root "_chmod_test"),
  (Join-Path $root "_mkdir_mode_700"),
  (Join-Path $root "_pytemp"),
  (Join-Path $root "STM32DC\mg-farm-4s-bms-PA--mg-farm-4s-bms\Firmware\EET_MG_FARM_BMS\tools\pc_sim\.tmp")
)

foreach ($p in $paths) {
  if (!(Test-Path $p)) { continue }
  Write-Host "Fixing ACL + deleting: $p"

  # TAKEOWN is localized: /D expects "J" (Ja) or "N" (Nein) on German Windows.
  cmd /c "takeown /f `"$p`" /r /d J" | Out-Host
  cmd /c "icacls `"$p`" /grant %USERNAME%:F /t /c" | Out-Host
  Remove-Item -Recurse -Force $p
}

Write-Host "Done."

