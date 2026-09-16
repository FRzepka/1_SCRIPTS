param([string]$Pattern = '*.png')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime] | Out-Null
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.IsGenericMethod })[0]
function Await-WinRT($Operation, [Type]$ResultType) {
    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$out = Join-Path $root 'archive/tnr_qa/ocr'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('en-US'))
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $root 'Bilder_Vortrag') -Filter $Pattern -File) {
    $storage = Await-WinRT ([Windows.Storage.StorageFile]::GetFileFromPathAsync($file.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await-WinRT ($storage.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
    $decoder = Await-WinRT ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $transform = [Windows.Graphics.Imaging.BitmapTransform]::new()
    $scale = [Math]::Min(1.0, 3600.0 / [Math]::Max($decoder.PixelWidth, $decoder.PixelHeight))
    $transform.ScaledWidth = [uint32]($decoder.PixelWidth * $scale)
    $transform.ScaledHeight = [uint32]($decoder.PixelHeight * $scale)
    $bitmap = Await-WinRT ($decoder.GetSoftwareBitmapAsync([Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8, [Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied, $transform, [Windows.Graphics.Imaging.ExifOrientationMode]::IgnoreExifOrientation, [Windows.Graphics.Imaging.ColorManagementMode]::DoNotColorManage)) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await-WinRT ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $lines = @()
    foreach ($line in $result.Lines) {
        $words = @()
        foreach ($word in $line.Words) {
            $b = $word.BoundingRect
            $words += @{ text=$word.Text; x=$b.X/$scale; y=$b.Y/$scale; w=$b.Width/$scale; h=$b.Height/$scale }
        }
        $lines += @{text=$line.Text; words=$words}
    }
    @{file=$file.Name;angle=$result.TextAngle;lines=$lines} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out ($file.BaseName + '.json')) -Encoding UTF8
    $bitmap.Dispose()
    $stream.Dispose()
    Write-Output "$($file.Name): $($lines.Count) text lines"
}
