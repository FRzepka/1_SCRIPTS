# ADC Quantization Extension (v4)

This extension uses the already available `adc_quantization` runs from the v4 benchmark setup.

Applied quantization steps:
- Current: `0.01 A`
- Voltage: `0.005 V`
- Temperature: `0.5 °C`

Main observation:
- Under these quantization levels, the global SOC error changes only marginally for all four estimator classes.
- Largest MAE increase: `HDM` with `ΔMAE = +0.000462`.
- Smallest / slightly improved case: `HECM` with `ΔMAE = -0.000275`.

Interpretation:
- ADC quantization appears to be much less discriminative than current bias, voltage noise, missing samples, or initialization mismatch for the present benchmark configuration.
- This makes it a reasonable appendix or supporting scenario, but not necessarily a core main-text figure unless embedded sensor resolution is a central discussion point.
