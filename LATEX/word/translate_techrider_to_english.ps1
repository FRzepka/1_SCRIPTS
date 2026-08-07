$ErrorActionPreference = 'Stop'

$documentPath = Join-Path $PSScriptRoot 'Techrider_WERKSTATTBUEHNE_neu_EN.docx'

$translations = @{
    2 = 'WORKSHOP STAGE'
    6 = 'Werkstattbuehne duty mobile'
    10 = 'Head of Transport & Logistics: Frank Flohe'
    14 = 'Production Management: Louisa Kaspar'
    19 = 'An advance visit to the venue is possible and encouraged at any time by prior arrangement with the Technical Directorate.'
    21 = 'GENERAL INFORMATION'
    23 = 'Preliminary technical meeting'
    24 = 'One week before the design concept is submitted, the draft will be reviewed in a small-group online preliminary meeting to assess its feasibility and financial viability. Initial drafts must be sent in advance or shared on screen.'
    26 = 'Design concept submission'
    27 = 'The design concept submission and preliminary stage mock-up meeting take place by prior telephone appointment at least three weeks before the stage mock-up. Design concept meetings are generally held online.'
    28 = 'The following documents must be submitted:'
    29 = 'Photographs or renderings of the scale model'
    30 = 'Stage floor plan and section drawing'
    31 = 'Design drawings of the construction elements'
    32 = 'Bill of materials'
    34 = 'The drawings should be sent as CAD files (DWG or PRT) three to four days before the meeting. Initial ideas and sketches should ideally be discussed with the Technical Directorate before the design concept is submitted so that their feasibility and financial viability can be assessed.'
    35 = 'Stage mock-up'
    36 = 'The date of the stage mock-up is set by Theatre Management and is generally at least six months before the premiere.'
    37 = 'The following documents must be brought to the meeting:'
    38 = 'Scale model of the set at 1:20, 1:25 or 1:33'
    40 = 'The theatre will prepare and distribute a report documenting the stage mock-up and its outcomes.'
    41 = 'Drawing submission'
    42 = 'The drawings are submitted after a processing period of approximately two weeks. The documents to be submitted must be agreed with Production Management.'
    43 = 'Workshop meeting'
    44 = 'The workshop meeting takes place approximately four weeks after the drawings have been submitted and is held at the Production Management offices in Bonn-Beuel. The exact date is set during the stage mock-up.'
    45 = 'The following documents are required for the workshop meeting:1'
    46 = 'Set design model'
    47 = 'Material and colour samples'
    48 = 'Print-ready artwork'
    50 = 'The workshops can produce digital prints up to 160 cm wide on a range of materials. A final print file, preferably a PDF, must be provided. If no digital original is available, the pattern can be scanned in the workshops. If necessary, the set designer must then prepare the file for printing and approve it.'
    51 = 'Start of rehearsals'
    52 = 'The rehearsal-room setup should be discussed with the technical departments during the stage mock-up. No later than four weeks before rehearsals begin, we require:'
    53 = 'Details of the required rehearsal set for all scenes'
    54 = 'A list of all rehearsal props'
    56 = 'The stage supervisor will conduct the safety briefing before the first rehearsal. If original set pieces are required or requested when rehearsals begin, this must be communicated no later than the workshop meeting.'
    57 = 'Rehearsal support'
    58 = 'Due to limited staffing, the theatre can provide a design assistant for Werkstattbuehne productions only by prior arrangement.'
    59 = 'Under the new NV Buehne Solo provisions, assistants must be granted one full weekday off and one half-day off per week. At Theater Bonn, these are generally Saturday and Sunday. If rehearsal requirements necessitate a different arrangement, this must be agreed directly with the relevant assistant, and the full day off must be granted on another day.'
    60 = 'Rehearsals and performances at the Werkstattbuehne are staffed by two stage technicians on the early shift and two event technicians responsible for lighting, audio and video on the late shift. The technical crew therefore cannot carry out scene changes. Rehearsal support at the Werkstattbuehne must be scheduled in coordination with these staff members.'
    61 = 'No lighting or audio technical support is provided on the rehearsal stages. Adjustable lighting states are not available there.'
    62 = 'For unattended rehearsals, the key can be collected from the gatehouse. Werkstattbuehne staff will provide an initial briefing. Transport of the rehearsal set and props must be coordinated independently with the Transport Department.'
    63 = 'Set volume'
    64 = 'The following conditions apply to every production:'
    65 = 'Due to limited storage space and difficult access routes, the set should be limited to 16.5 m2 or 50 m3.'
    66 = 'The set must be capable of being assembled and dismantled by two technicians within three hours.'
    67 = 'A performance can be supported by no more than one stage operator.'
    68 = 'Maximum dimensions for scenic flats: 300 cm x 160 cm x 25 cm (h x w x d)'
    70 = 'Venue-specific conditions'
    71 = 'Set pieces can be suspended only to a limited extent because the Helm rails used as attachment points can carry only light loads. When planning the set, note that luminaires are attached to these rails and may obstruct sightlines.'
    72 = 'Set pieces are delivered via a small transport platform. The specified dimensions must therefore be observed without exception.'
    73 = 'The following safety regulations must also be observed:'
    74 = 'Pyrotechnic effects cannot be used because of the low ceiling height.'
    75 = 'The performance area must remain visible and accessible to the fire brigade at all times.'
    76 = 'Access to safety equipment must not be obstructed.'
    77 = 'All escape and rescue routes must be kept clear.'
    78 = 'The transport platform cannot be used as part of the performance.'
    79 = 'Attention is drawn to the director''s responsibility and particular duty of care for the performers during rehearsals.'
    81 = 'Budget'
    82 = 'The design budget must cover all costs for producing the stage set, producing and procuring props, production-related materials and equipment for audio, video and lighting, and producing the costumes, including wigs and make-up.'
    83 = 'Production Management calculates the production costs after the required documents have been submitted. If the cost estimate exceeds the budget, the set and costume designer must revise the design and adapt it to the available financial and staffing resources.'
    84 = 'Changes to the set arising during rehearsals must be submitted by the set designer, together with the relevant documents, to the Production Manager and Technical Director for approval.'
    85 = 'Parking'
    87 = 'Cars cannot be parked on the loading ramp directly in front of the Werkstattbuehne because this area is used for deliveries and designated as a fire brigade access and staging area. Paid parking is available in the Operngarage underground car park. Bicycles must be parked in the designated areas, for example by the stage door.'
    89 = 'STAGE TECHNICAL EQUIPMENT'
    91 = 'Audio equipment'
    92 = 'Loudspeakers'
    93 = "Front:`t`tCoda Audio`t 4 x HOPS8"
    95 = "Stage:`t`tCoda Audio`t2 x N-APS"
    96 = "`t`tCoda Audio`t2 x N-SUB"
    97 = "Surround:`tCoda Audio`t12 x D5"
    98 = "Mobile:`t`t4 x d&b E9"
    99 = "`t`t2 x d&b E3"
    101 = "Mixing console:`tFOH YAMAHA CL5 mixing console"
    102 = "Microphones:`tWireless microphones: 8 x Shure UHF-R (8 bodypack transmitters, 6 handheld transmitters)"
    103 = "Accessories:`tYamaha RIO 3224-D2 mobile stage box"
    104 = 'Audio/video playback computer: Mac Studio with QLab and Ableton'
    106 = 'Video equipment'
    107 = 'The use of video on stage is not always possible without restrictions and requires individual coordination.'
    108 = 'No content archive is available and no in-house content creation is provided.'
    109 = 'No video support is provided on the rehearsal stages.'
    110 = 'Operation: Audio and video are operated by one person.'
    112 = 'Equipment:'
    116 = 'Fog machines (use is possible only to a limited and controlled extent)'
    117 = '2 x Safex Twinfog fog machines'
    118 = '4 x Safex Accu F 2010 fog machines'
    119 = '2 x Martin Glaciator X-Stream JEM low-fog machines'
    121 = '3 x Look Solutions Unique 2.1 hazers'
    122 = '1 x MDG Haze Generator Touring Atmosphere without DMX'
    124 = 'Lighting equipment'
    125 = 'Due to repertoire operation and the staffing situation, the luminaires in the standard rig shown in the lighting plan cannot be changed.'
    126 = 'In addition to the luminaires shown in the rigging plan, the Werkstattbuehne has four ballet towers, each equipped with:'
    127 = '4 x Source 4 WRD LED (2 x Daylight and 2 x Color)'
    133 = 'Lighting console'
    135 = 'Software'
    136 = 'grandMA3 (please enquire about the installed software version)'
    137 = 'Lighting equipment'
    139 = 'Werkstattbuehne lighting rig plan'
}

function Get-ContentEnd {
    param([object]$Range)

    $text = [string]$Range.Text
    $trailingControls = 0
    for ($i = $text.Length - 1; $i -ge 0; $i--) {
        if ([int][char]$text[$i] -le 31) {
            $trailingControls++
        }
        else {
            break
        }
    }

    return $Range.End - $trailingControls
}

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $document = $word.Documents.Open($documentPath, $false, $false)
    $document.TrackRevisions = $false

    foreach ($index in ($translations.Keys | Sort-Object)) {
        if ($index -gt $document.Paragraphs.Count) {
            throw "Paragraph $index does not exist."
        }

        $paragraph = $document.Paragraphs.Item($index)
        $range = $paragraph.Range.Duplicate
        $contentEnd = Get-ContentEnd -Range $range
        $range.End = $contentEnd
        $range.Text = $translations[$index]

        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($range) | Out-Null
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($paragraph) | Out-Null
    }

    $section = $document.Sections.Item(1)
    $header = $section.Headers.Item(1)
    $find = $header.Range.Find
    $find.ClearFormatting()
    $find.Replacement.ClearFormatting()
    $find.Text = 'Stand Juli 2026'
    $find.Replacement.Text = 'As of July 2026'
    [void]$find.Execute(
        $find.Text,
        $false,
        $false,
        $false,
        $false,
        $false,
        $true,
        1,
        $false,
        $find.Replacement.Text,
        2
    )

    $footer = $section.Footers.Item(1)
    for ($i = 1; $i -le $footer.Shapes.Count; $i++) {
        $shape = $footer.Shapes.Item($i)
        if ($shape.TextFrame.HasText) {
            $shape.TextFrame.TextRange.Text = "Document class B (for internal use only)`r"
        }
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($shape) | Out-Null
    }

    $document.Content.LanguageID = 2057
    $header.Range.LanguageID = 2057
    $footer.Range.LanguageID = 2057

    $document.Save()

    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($footer) | Out-Null
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($header) | Out-Null
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($section) | Out-Null
}
finally {
    if ($document) {
        $document.Close($false)
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) | Out-Null
    }
    if ($word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zip = [System.IO.Compression.ZipFile]::Open(
    $documentPath,
    [System.IO.Compression.ZipArchiveMode]::Update
)

try {
    $entry = $zip.GetEntry('word/document.xml')
    $reader = New-Object System.IO.StreamReader($entry.Open())
    try {
        [xml]$xml = $reader.ReadToEnd()
    }
    finally {
        $reader.Dispose()
    }

    $namespaceManager = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
    $namespaceManager.AddNamespace(
        'w',
        'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    )

    $paragraphs = $xml.SelectNodes('//w:p', $namespaceManager)
    foreach ($paragraphIndex in 6, 9, 12) {
        $textNodes = $paragraphs[$paragraphIndex - 1].SelectNodes(
            './/w:t',
            $namespaceManager
        )
        if ($textNodes.Count -gt 1) {
            $textNodes[$textNodes.Count - 1].InnerText = ''
        }
    }

    $uUmlaut = [char]0x00FC
    foreach ($textNode in $xml.SelectNodes('//w:t', $namespaceManager)) {
        $textNode.InnerText = $textNode.InnerText.Replace(
            'Werkstattbuehne',
            "Werkstattb${uUmlaut}hne"
        )
        $textNode.InnerText = $textNode.InnerText.Replace(
            'NV Buehne Solo',
            "NV B${uUmlaut}hne Solo"
        )
    }

    $stream = $entry.Open()
    try {
        $stream.SetLength(0)
        $settings = New-Object System.Xml.XmlWriterSettings
        $settings.Encoding = New-Object System.Text.UTF8Encoding($false)
        $settings.Indent = $false
        $settings.CloseOutput = $false
        $writer = [System.Xml.XmlWriter]::Create($stream, $settings)
        try {
            $xml.Save($writer)
        }
        finally {
            $writer.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}
finally {
    $zip.Dispose()
}
