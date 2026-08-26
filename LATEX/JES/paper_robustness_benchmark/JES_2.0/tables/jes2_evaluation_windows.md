# JES2 evaluation windows

Windows start at a measured full-charge anchor (SOC >= 0.98 and voltage >= 3.58 V).
Selection is the multivariate medoid of measured operating features within each available SOH state; estimator outputs are not used.
Primary scenarios use 24 h (86400 rows); the 1 h missing-gap scenario uses 48 h (172800 rows).

| window_id | cell_load_class | soh_state | start_row | primary_rows | event_rows | soh_median | temperature_mean_c | abs_c_rate_p95 | throughput_ah | candidate_count_in_state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C09_fresh | middle | fresh | 1690346 | 86400 | 172800 | 0.9587 | 28.4775 | 2.5083 | 35.1712 | 29 |
| C09_mid_life | middle | mid_life | 4256212 | 86400 | 172800 | 0.8898 | 26.4152 | 1.0031 | 4.5535 | 11 |
| C09_aged | middle | aged | 6252455 | 86400 | 172800 | 0.7523 | 29.2064 | 2.5083 | 34.505 | 18 |
| C13_fresh | middle | fresh | 943790 | 86400 | 172800 | 0.9695 | 30.5839 | 2.4955 | 55.1747 | 19 |
| C13_mid_life | middle | mid_life | 2475157 | 86400 | 172800 | 0.8785 | 31.3133 | 2.4957 | 54.6253 | 4 |
| C13_aged | middle | aged | 3255078 | 86400 | 172800 | 0.6836 | 32.4047 | 2.4955 | 54.7259 | 9 |
| C15_fresh | high | fresh | 1071509 | 86400 | 172800 | 0.9669 | 30.8725 | 2.5266 | 54.9269 | 17 |
| C15_mid_life | high | mid_life | 2623747 | 86400 | 172800 | 0.8595 | 31.7627 | 2.5268 | 54.786 | 12 |
| C15_aged | high | aged | 4154449 | 86400 | 172800 | 0.731 | 32.454 | 2.5268 | 54.7178 | 16 |
| C25_fresh | low | fresh | 1060045 | 86400 | 172800 | 0.9748 | 29.8367 | 1.7612 | 54.5148 | 42 |
| C25_mid_life | low | mid_life | 6161570 | 86400 | 172800 | 0.853 | 30.0598 | 1.7612 | 54.4852 | 19 |
| C25_aged | low | aged | 9060003 | 86400 | 172800 | 0.7385 | 31.0499 | 1.7612 | 54.5922 | 31 |
| C27_fresh | low | fresh | 5427475 | 86400 | 172800 | 0.9629 | 27.6281 | 0.6962 | 24.7009 | 246 |
| C29_fresh | high | fresh | 917433 | 86400 | 172800 | 0.9699 | 28.862 | 3.0035 | 47.4354 | 23 |
| C29_mid_life | high | mid_life | 2712150 | 86400 | 172800 | 0.824 | 29.5979 | 3.0035 | 47.2834 | 5 |
| C29_aged | high | aged | 3626725 | 86400 | 172800 | 0.7118 | 30.0323 | 3.0035 | 46.7584 | 17 |
