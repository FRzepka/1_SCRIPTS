param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$archive = Join-Path $root 'archive'
$latest = Join-Path $root 'Diss_Presentation_FR_v1.5.pptx'
$expectedHash = 'BD5615DD4078D5EECF71FF817AF72ABBBAF0F27F291B1623E4040A929DE499F1'
if ((Get-FileHash -LiteralPath $latest).Hash -ne $expectedHash) {
    throw 'The protected v1.5 deck has changed. Review before cleanup.'
}
function Checked-Path([string]$relative) {
    $path = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (!$path.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside presentation folder: $path"
    }
    if (Test-Path -LiteralPath $path) {
        $resolved = (Resolve-Path -LiteralPath $path).Path
        if (!$resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Resolved outside presentation folder: $resolved"
        }
    }
    return $path
}
$delete = @(
    'Diss_FR.pptx', 'Diss_FR_Presentation_v1.pptx',
    'Diss_FR_Dissertation_30min.pptx', 'Diss_FR_Dissertation_30min.pdf',
    'Diss_FR_Dissertation_30min_v2.pptx', 'Diss_FR_Dissertation_30min_v2.pdf',
    'Diss_FR_Dissertation_30min_v3.pptx', 'Diss_FR_Dissertation_30min_v3.pdf',
    'Diss_FR_Dissertation_30min_v4.pptx',
    'Diss_FR_Dissertation_30min_v5.pptx', 'Diss_FR_Dissertation_30min_v5.pdf',
    'Diss_FR_Dissertation_30min_v6.pptx', 'Diss_FR_Dissertation_30min_v6.pdf',
    'Diss_FR.pptx.parts', '_appendix_preview', '_render_v3_194759', '_render_v3_check_195046',
    'Vortrag_Erzaehlfassung_30min_v2.pdf', 'Vortrag_Erzaehlfassung_30min_v3.pdf'
)
$move = @(
    'add_model_appendix.ps1', 'build_dissertation_defense.ps1',
    'build_dissertation_defense_v2.ps1', 'restore_Diss_FR.ps1',
    'PowerPoint_Auftrag_v7.txt', 'shape_inventory.txt',
    '_build_v5', '_build_v6', '_build_erzaehlfassung'
)
$records = @()
foreach ($name in $delete + $move) {
    $path = Checked-Path $name
    if (!(Test-Path -LiteralPath $path)) { continue }
    $item = Get-Item -LiteralPath $path
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $path -Recurse -File) } else { @($item) }
    $records += [pscustomobject]@{
        Action = $(if ($name -in $delete) { 'Delete' } else { 'Archive' })
        Path = $path
        Bytes = ($files | Measure-Object Length -Sum).Sum
        Files = $files.Count
    }
}
$records | Format-Table Action,Files,Bytes,Path -AutoSize
if (!$Apply) { return }
New-Item -ItemType Directory -Path $archive -Force | Out-Null
$log = Join-Path $archive 'cleanup_2026-09-13.csv'
if (Test-Path -LiteralPath $log) { throw 'Cleanup already recorded. Review before repeating.' }
$records | Export-Csv -LiteralPath $log -NoTypeInformation -Encoding UTF8

$narrative = Checked-Path 'Vortrag_Erzaehlfassung_30min_v3.pdf'
$canonical = Checked-Path 'Vortrag_Erzaehlfassung_30min.pdf'
$narrativeHash = (Get-FileHash -LiteralPath $narrative).Hash
Copy-Item -LiteralPath $canonical -Destination (Join-Path $archive 'Vortrag_Erzaehlfassung_30min_previous.pdf')
Copy-Item -LiteralPath $narrative -Destination $canonical -Force
if ((Get-FileHash -LiteralPath $canonical).Hash -ne $narrativeHash) { throw 'Narrative copy mismatch.' }

foreach ($record in $records) {
    $path = Checked-Path ($record.Path.Substring($root.Length + 1))
    if ($record.Action -eq 'Delete') {
        Remove-Item -LiteralPath $path -Recurse -Force
    } else {
        $target = Checked-Path ('archive/' + (Split-Path $path -Leaf))
        if (Test-Path -LiteralPath $target) { throw "Archive destination exists: $target" }
        Move-Item -LiteralPath $path -Destination $target
    }
}
if ((Get-FileHash -LiteralPath $latest).Hash -ne $expectedHash) { throw 'Protected deck changed.' }
Write-Output 'Cleanup complete. Current presentation, narrative sources, images and tools retained.'
