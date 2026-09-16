# Presentation Figure Set

Output: `../Bilder_Vortrag/Times_New_Roman/`.

The output contains exactly the 32 PNG filenames from the parent folder, at the
same pixel dimensions. Legends and subpanels are part of their respective PNG.
Original PNGs, paper figures, data files, presentation and narrative are not
overwritten.

Run `./replot_presentation_figures.ps1` to rebuild the complete set.
Run `build_times_new_roman.py --verify-only` in the `ml1` environment to check
filenames, dimensions, PNG integrity and original-file SHA-256 hashes.

## Sources

- Eight schematics are rendered from archived editable SVG sources with Times
  New Roman text styles.
- Fifteen existing chart compositions receive reviewed Times New Roman SVG
  labels. Their original raster plots remain the background, without a new
  inference run or new histogram bins. The initialization figure's three red
  time annotations are removed. Its connected curve pixels are retained beneath
  the old label areas.
- The synthesis radar and bars are replotted from the saved decision-dimension
  and priority-profile CSVs. Both panels and the shared legend stay in one PNG.
- Seven existing serif figures are retained after visual inspection. The PCB
  photograph is unchanged.

All chart text uses regular weight. Panel letters are removed. The disturbance
taxonomy and initialization/recovery figure use larger labels. The latter has
one integrated model/style legend, rather than repeated small model legends.
Figures 16 and 17 place all legends below the three panels. Panel placement is
recorded in the label metadata. Figure 16's source PNG already has text painted
over its first SOC trough. Existing coloured pixels are retained there, undoing
the source script's known 0.82 white-box opacity, without inventing curve samples.
Pixels fully hidden by the original text cannot be recovered from this PNG.
Long narrative headings and annotations are omitted or shortened to labels.
The previous complete set is preserved in
`../archive/tnr_before_type_cleanup_20260913/`.

Editable generated SVGs: `../archive/tnr_build/`.
Source snapshots: `../archive/tnr_sources/` and `../archive/plot_sources/`.
Original/output hash inventories and visual QA sheets: `../archive/tnr_qa/`.
Superseded partial charts and standalone legends:
`../archive/tnr_old_partial_outputs/`.

`check_label_coverage.py` checks the reviewed overlay masks against the original
OCR bounding boxes. Plot markers and legend line samples can be misread as text,
so the remaining reports require visual interpretation.
