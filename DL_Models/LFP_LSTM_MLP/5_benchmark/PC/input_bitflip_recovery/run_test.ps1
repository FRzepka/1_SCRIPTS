param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TestArguments
)

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "../../../../..")).Path
$python = Join-Path $workspace "LATEX/EAAI/elsarticle/elsarticle/review_1/review_analysis/runtime/python/python.exe"
$script = Join-Path $PSScriptRoot "run_input_bitflip_test.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Bundled review Python not found: $python"
}

& $python $script @TestArguments
exit $LASTEXITCODE
