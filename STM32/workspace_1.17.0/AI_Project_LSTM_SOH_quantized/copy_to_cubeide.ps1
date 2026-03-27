# Copy Hybrid INT8 files to CubeIDE project
# Run this after making changes to hybrid_int8 project

$source = "C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\1_Scripts\STM32\workspace_1.17.0\AI_Project_LSTM_hybrid_int8"
$dest = "C:\Users\Florian\STM32CubeIDE\workspace_1.17.0\AI_Project_LSTM_v2"

Write-Host "Copying Hybrid INT8 files to CubeIDE project..." -ForegroundColor Cyan

# Core/Src files
Write-Host "  Copying Core/Src..." -ForegroundColor Yellow
Copy-Item "$source\Core\Src\main.c" "$dest\Core\Src\" -Force
Copy-Item "$source\Core\Src\lstm_model_hybrid_int8.c" "$dest\Core\Src\" -Force
Copy-Item "$source\Core\Src\model_weights_hybrid_int8.c" "$dest\Core\Src\" -Force

# Core/Inc files
Write-Host "  Copying Core/Inc..." -ForegroundColor Yellow
Copy-Item "$source\Core\Inc\lstm_model_hybrid_int8.h" "$dest\Core\Inc\" -Force
Copy-Item "$source\Core\Inc\model_weights_hybrid_int8.h" "$dest\Core\Inc\" -Force
Copy-Item "$source\Core\Inc\scaler_params.h" "$dest\Core\Inc\" -Force

Write-Host "`n✓ All files copied successfully!" -ForegroundColor Green
Write-Host "`nNow in STM32CubeIDE:" -ForegroundColor Cyan
Write-Host "  1. Project -> Clean" -ForegroundColor White
Write-Host "  2. Project -> Build (Ctrl+B)" -ForegroundColor White
Write-Host "  3. Flash to STM32" -ForegroundColor White
