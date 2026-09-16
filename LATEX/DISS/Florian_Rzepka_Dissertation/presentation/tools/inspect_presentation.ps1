param(
    [Parameter(Mandatory=$true)][string]$Deck,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][string]$PictureRoot
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$output = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $output -Force | Out-Null
$mediaDir = Join-Path $output 'media'
New-Item -ItemType Directory -Path $mediaDir -Force | Out-Null
$sourceHashes = @{}
Get-ChildItem -LiteralPath $PictureRoot -Recurse -File | Where-Object Extension -In '.png','.jpg','.jpeg','.svg','.emf' | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    if (!$sourceHashes.ContainsKey($hash)) { $sourceHashes[$hash] = @() }
    $sourceHashes[$hash] += $_.FullName
}
$zip = [IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $Deck))
function Read-XmlEntry([string]$name) {
    $entry = $zip.GetEntry($name)
    if (!$entry) { throw "Missing ZIP entry: $name" }
    $reader = [IO.StreamReader]::new($entry.Open())
    try { return [xml]$reader.ReadToEnd() } finally { $reader.Dispose() }
}
function Resolve-Part([string]$source, [string]$target) {
    return ([uri]::new([uri]('http://package/' + $source), $target)).AbsolutePath.TrimStart('/')
}
function Get-Namespace($doc) {
    $ns = [Xml.XmlNamespaceManager]::new($doc.NameTable)
    $ns.AddNamespace('a','http://schemas.openxmlformats.org/drawingml/2006/main')
    $ns.AddNamespace('p','http://schemas.openxmlformats.org/presentationml/2006/main')
    $ns.AddNamespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
    return ,$ns
}
try {
    $p = Read-XmlEntry 'ppt/presentation.xml'
    $rels = Read-XmlEntry 'ppt/_rels/presentation.xml.rels'
    $nsp = Get-Namespace $p
    $slides = @()
    $index = 0
    foreach ($slideId in $p.SelectNodes('//p:sldId', $nsp)) {
        $index++
        $rid = $slideId.GetAttribute('id','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
        $rel = $rels.Relationships.Relationship | Where-Object Id -EQ $rid
        $part = Resolve-Part 'ppt/presentation.xml' $rel.Target
        $doc = Read-XmlEntry $part
        $ns = Get-Namespace $doc
        $partDir = $part.Substring(0,$part.LastIndexOf('/'))
        $partName = $part.Substring($part.LastIndexOf('/')+1)
        $sr = Read-XmlEntry "$partDir/_rels/$partName.rels"
        $pictures = @()
        foreach ($pic in $doc.SelectNodes('//p:pic', $ns)) {
            $blip = $pic.SelectSingleNode('.//a:blip',$ns)
            if (!$blip) { continue }
            $embed = $blip.GetAttribute('embed','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
            $r = $sr.Relationships.Relationship | Where-Object Id -EQ $embed
            if (!$r -or $r.TargetMode -eq 'External') { continue }
            $target = Resolve-Part $part $r.Target
            $file = Join-Path $mediaDir ($target.Split('/')[-1])
            if (!(Test-Path -LiteralPath $file)) {
                [IO.Compression.ZipFileExtensions]::ExtractToFile($zip.GetEntry($target),$file,$false)
            }
            $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
            $xfrm = $pic.SelectSingleNode('./p:spPr/a:xfrm',$ns)
            $props = $pic.SelectSingleNode('./p:nvPicPr/p:cNvPr',$ns)
            $pictures += [pscustomobject]@{
                Name=$props.name;Description=$props.descr;Part=$target;File=$file;Hash=$hash
                Sources=@($sourceHashes[$hash]);Geometry=$xfrm.OuterXml
                Crop=$pic.SelectSingleNode('.//a:srcRect',$ns).OuterXml
            }
        }
        $texts = @($doc.SelectNodes('//a:t',$ns) | ForEach-Object InnerText)
        $slides += [pscustomobject]@{Slide=$index;Part=$part;Title=$texts[0];Text=$texts;Pictures=$pictures}
    }
    [pscustomobject]@{Deck=(Resolve-Path -LiteralPath $Deck).Path;SlideSize=$p.presentation.sldSz;Slides=$slides} |
        ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'inventory.json') -Encoding UTF8
    $slides | Select-Object Slide,Title,@{Name='Pictures';Expression={$_.Pictures.Count}} | Format-Table -AutoSize -Wrap
} finally { $zip.Dispose() }
