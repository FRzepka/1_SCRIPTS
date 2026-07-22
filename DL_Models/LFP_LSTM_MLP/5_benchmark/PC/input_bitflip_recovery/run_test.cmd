@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..\..\..\..") do set "WORKSPACE=%%~fI"
set "PYTHON=%WORKSPACE%\LATEX\EAAI\elsarticle\elsarticle\review_1\review_analysis\runtime\python\python.exe"

if not exist "%PYTHON%" (
    echo Bundled review Python not found: %PYTHON%
    exit /b 1
)

"%PYTHON%" "%SCRIPT_DIR%run_input_bitflip_test.py" %*
exit /b %ERRORLEVEL%
