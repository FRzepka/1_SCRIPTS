# Calibri Presentation Figures

Output: `../Bilder_Vortrag/Calibri/`, 32 PNGs with the same filenames and pixel
dimensions as the approved `Times_New_Roman` set. All legends remain integrated.
The PCB photograph is copied unchanged. Original sources and the Times New Roman
set are not overwritten.

Run `build_calibri.py` in the `ml1` Python environment to rebuild the set.
Run `build_calibri.py --verify-only` for source hashes, filenames, dimensions,
font declarations, and a strict font-only SVG comparison.

Twenty-two approved SVG compositions are reused with only the font family
replaced. Their font sizes, positions, colours, plot backgrounds, data, and
clipping masks remain unchanged. The methodology figure uses the approved
drawing bounds as a fixed export viewport.

Nine figures have rasterized source typography. `calibri_raster_labels.py`
replaces their text locally while keeping their raster plots and diagrams.
Type sizes are taken from the existing SVG headings where available, or matched
to the original raster lettering. Label coordinates and masks are recorded in
`../archive/calibri_qa/`. No model inference, metric calculation, resampling,
smoothing, or new histogram binning is performed.

Editable output SVGs: `../archive/calibri_build/`.
Source/output hashes and white-composited contact sheets: `../archive/calibri_qa/`.
