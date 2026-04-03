figure,global_evidence,local_evidence,note
Figure_1_baseline_performance,Baseline MAE and RMSE on the complete C07 run,none,pure global ranking figure
Figure_2_current_bias,MAE vs current-bias level (0.5 / 1.5 / 3.0 % of |I|max),30 min current-input zoom plus first-12h SOC divergence at 3.0 % bias,baseline accuracy and bias robustness differ strongly; HECM remains the most controlled under bias
Figure_3_noise_robustness,"Delta MAE under current, voltage, and temperature noise",rolling-MAE drift and late-minus-early rolling MAE,use local metrics because global noise penalties are small
Figure_4_signal_integrity,"Delta MAE for missing samples, irregular sampling, and burst dropout",recovery time after burst dropout,mixed global/local summary figure
Figure_5_missing_gap_transition,none,sensor and SOC behaviour around the dropout window,pure local transition figure
Figure_6_missing_gap_recovery,none,post-gap absolute-error recovery over time,pure local recovery figure
Figure_7_initial_state_recovery,global MAE listed in key-results table,CV-free initial-state recovery against the fair baseline band,local interpretation is the primary evidence
Figure_8_spike_response,Delta MAE under voltage spikes,aligned per-spike excess-error response,"single spikes are globally weak, so local response matters"
