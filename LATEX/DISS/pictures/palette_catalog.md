# Dissertation color palette tests

Base color: `#b6302d`.

Recommended starting point: `07_tu_red_muted_reordered` if the current red-muted look should be preserved but M1/M2 need stronger separation. Use `12_tu_red_colorblind_safe` when many lines overlap or when maximum distinguishability is more important than a red-dominant appearance. `05_tu_red_monochrome` should only be used when the four series are ordered, not when they are nominal model classes.

| Palette | Colors | Use case |
|---|---|---|
| `01_tu_red_muted` | `#b6302d` `#d1887e` `#8b6763` `#566b78` | Main recommendation for four estimator classes: TU red, muted rose, desaturated brown-gray, and blue-gray. |
| `02_tu_red_scientific` | `#b6302d` `#6b7f8f` `#b88a5a` `#6f857d` | Still red-led, but with cool and ochre accents for stronger distinction in line and scatter plots. |
| `03_tu_red_pastel` | `#b6302d` `#e0a39b` `#9f7d7a` `#7f9caf` | Softest palette. Good for filled bars and boxes, slightly weaker for thin lines. |
| `04_tu_red_earth` | `#b6302d` `#c27a54` `#8b806b` `#596b76` | Warm, calm print palette with red, clay, olive-gray, and slate. |
| `05_tu_red_monochrome` | `#7c1f1d` `#b6302d` `#d6766d` `#edbdb8` | Useful when the four categories are ordered. Not ideal when categories must be equally distinct. |
| `06_tu_red_print_safe` | `#9e2a2b` `#6c757d` `#b06d47` `#4f6d7a` | More conservative and robust in grayscale or print. Less pastel, but still not poppy. |
| `07_tu_red_muted_reordered` | `#b6302d` `#566b78` `#d1887e` `#8b6763` | Same family as the current red-muted palette, but M2 is moved to blue-gray. This is the quickest fix when M1 and M2 must be separated clearly. |
| `08_tu_red_cool_muted` | `#b6302d` `#2f6f88` `#7f8f6b` `#8b6763` | Red-led but with blue, olive, and brown-gray accents. Good for nominal model classes because the four hues are not just red shades. |
| `09_tu_red_neutral_highcontrast` | `#b6302d` `#434343` `#6f8290` `#b2b2b2` | Very restrained palette for print-heavy figures. Red marks the main model, while the other classes use graphite, blue-gray, and light gray. |
| `10_user_bright_direct` | `#c40d1e` `#9013fe` `#1f90cc` `#49cb40`<br>Neutral: `#000000` `#434343` `#b2b2b2` | Uses the requested saturated red, purple, blue, and green exactly. Strong separation, but noticeably more colorful than the dissertation style. |
| `11_user_bright_tempered` | `#c40d1e` `#6f4a8e` `#2f7898` `#6f9a63`<br>Neutral: `#000000` `#434343` `#b2b2b2` | A calmer version inspired by the requested colors. It keeps the red-purple-blue-green structure but lowers saturation for a less poppy scientific look. |
| `12_tu_red_colorblind_safe` | `#b6302d` `#0072b2` `#009e73` `#e69f00`<br>Neutral: `#000000` `#434343` `#b2b2b2` | Scientific/high-contrast option: TU red combined with blue, green, and ochre accents. Best candidate when lines overlap heavily. |
| `13_tu_red_muted_purple_green` | `#b6302d` `#7f5f9f` `#8fbf8a` `#566b78`<br>Neutral: `#000000` `#434343` `#b2b2b2` | Close to the current red-muted style, but the rose and brown-gray slots are replaced by muted purple and a softer light green for clearer M1/M2/M3 separation. |
| `14_tu_red_muted_purple_m2` | `#b6302d` `#7f5f9f` `#8b6763` `#566b78`<br>Neutral: `#000000` `#434343` `#b2b2b2` | Minimal variant of the current red-muted palette: M1, M3, and M4 stay unchanged, while the rose M2 is replaced by muted purple. |
| `15_default_cycler_red_green_orange_cyan` | `#cc0000` `#22a15c` `#ff8000` `#00a6b3`<br>Neutral: `#000000` `#434343` `#b2b2b2` | Direct test of the proposed Matplotlib cycler colors. Strongly separated and readable, but more saturated than the muted dissertation palettes. |
| `16_eaai_extracted_light` | `#2ca02c` `#d62728` `#1f77b4` `#9467bd`<br>Fill: `#a6d7a6` `#eea4a5` `#a1c6e0` `#d2bfe3`<br>Neutral: `#000000` `#434343` `#b2b2b2` | Palette extracted from the EAAI figures: green, red, and blue as the main colours plus matching purple for a fourth class. Use the main colours for lines and outlines; use the fill colours for bars, boxes, and histograms. |

Generated files:

- `palette_preview_all.svg` / `palette_preview_all.png`: compact overview of all palettes
- `testplot_01_tu_red_muted.svg`: line, bar, boxplot, and scatter examples
- `testplot_02_tu_red_scientific.svg`: line, bar, boxplot, and scatter examples
- `testplot_03_tu_red_pastel.svg`: line, bar, boxplot, and scatter examples
- `testplot_04_tu_red_earth.svg`: line, bar, boxplot, and scatter examples
- `testplot_05_tu_red_monochrome.svg`: line, bar, boxplot, and scatter examples
- `testplot_06_tu_red_print_safe.svg`: line, bar, boxplot, and scatter examples
- `testplot_07_tu_red_muted_reordered.svg`: line, bar, boxplot, and scatter examples
- `testplot_08_tu_red_cool_muted.svg`: line, bar, boxplot, and scatter examples
- `testplot_09_tu_red_neutral_highcontrast.svg`: line, bar, boxplot, and scatter examples
- `testplot_10_user_bright_direct.svg`: line, bar, boxplot, and scatter examples
- `testplot_11_user_bright_tempered.svg`: line, bar, boxplot, and scatter examples
- `testplot_12_tu_red_colorblind_safe.svg`: line, bar, boxplot, and scatter examples
- `testplot_13_tu_red_muted_purple_green.svg`: line, bar, boxplot, and scatter examples
- `testplot_14_tu_red_muted_purple_m2.svg`: line, bar, boxplot, and scatter examples
- `testplot_15_default_cycler_red_green_orange_cyan.svg`: line, bar, boxplot, and scatter examples
- `testplot_16_eaai_extracted_light.svg` / `testplot_16_eaai_extracted_light.png`: real-figure contact sheet using EAAI/JES dissertation plots
