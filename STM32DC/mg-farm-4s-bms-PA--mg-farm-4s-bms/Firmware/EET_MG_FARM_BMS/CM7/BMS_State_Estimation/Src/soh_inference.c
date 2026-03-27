#include "soh_inference.h"

#include <math.h>
#include <string.h>

#include "Build_Config.h"

#if SOH_USE_MODEL
#if __has_include("soh_0_1_2_3_weights.h")
#include "soh_0_1_2_3_weights.h"
#include "soh_0_1_2_3_scaler_params.h"
#else
#error "Missing generated 0.1.2.3 headers. Run: DL_Models/LFP_SOH_Optimization_Study/2_models/LSTM/Base/0.1.2.3/c_implementation/export_0_1_2_3_to_stm32_c.py"
#endif
#endif

#ifndef SOH_0123_EPS
#define SOH_0123_EPS 1.0e-5f
#endif

/* Hourly binning */
#define SOH_0123_HOUR_MS (3600UL * 1000UL)

typedef struct
{
	uint32_t n;
	float sum;
	float sumsq;
	float minv;
	float maxv;
} StatsAcc;

static inline void stats_reset(StatsAcc *s)
{
	s->n = 0;
	s->sum = 0.0f;
	s->sumsq = 0.0f;
	s->minv = 0.0f;
	s->maxv = 0.0f;
}

static inline void stats_push(StatsAcc *s, float x)
{
	if (s->n == 0)
	{
		s->minv = x;
		s->maxv = x;
	}
	else
	{
		if (x < s->minv) s->minv = x;
		if (x > s->maxv) s->maxv = x;
	}
	s->n++;
	s->sum += x;
	s->sumsq += x * x;
}

static inline float stats_mean(const StatsAcc *s)
{
	return (s->n > 0) ? (s->sum / (float)s->n) : 0.0f;
}

static inline float stats_std(const StatsAcc *s)
{
	if (s->n == 0) return 0.0f;
	const float mean = stats_mean(s);
	float var = (s->sumsq / (float)s->n) - mean * mean;
	if (var < 0.0f) var = 0.0f;
	return sqrtf(var);
}

static inline float sigmoidf_fast(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

/* GELU approximation (tanh-based) */
static inline float geluf(float x)
{
	const float k0 = 0.7978845608f; /* sqrt(2/pi) */
	const float k1 = 0.044715f;
	const float x3 = x * x * x;
	return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

static void layernorm_f32(const float *x, float *y, const float *gamma, const float *beta, int n)
{
	float mean = 0.0f;
	float var = 0.0f;
	for (int i = 0; i < n; i++)
	{
		mean += x[i];
	}
	mean /= (float)n;
	for (int i = 0; i < n; i++)
	{
		const float d = x[i] - mean;
		var += d * d;
	}
	var /= (float)n;
	const float inv = 1.0f / sqrtf(var + SOH_0123_EPS);
	for (int i = 0; i < n; i++)
	{
		y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
	}
}

static void linear_q8_f32(
		const int8_t *w_q, const float *w_s, const float *b,
		const float *x, int out_dim, int in_dim,
		float *y)
{
	for (int o = 0; o < out_dim; o++)
	{
		const int8_t *row = &w_q[o * in_dim];
		float acc = 0.0f;
		for (int i = 0; i < in_dim; i++)
		{
			acc += ((float)row[i]) * x[i];
		}
		y[o] = acc * w_s[o] + b[o];
	}
}

typedef struct
{
	float h[SOH_0123_LAYERS][SOH_0123_HIDDEN];
	float c[SOH_0123_LAYERS][SOH_0123_HIDDEN];
} LstmState0123;

static uint8_t inited = 0;
static float last_soh = 1.0f;
static uint32_t last_pred_ts_ms = 0;

static uint32_t current_hour_bin = 0;
static uint32_t last_sample_ts_ms = 0;
static StatsAcc acc_v, acc_i, acc_t, acc_efc, acc_qc;

static LstmState0123 st;

static inline float cell_raw_to_v(uint16_t raw)
{
	return ((float)raw) * 5.0f / 65536.0f;
}

static void reset_hour_accumulators(void)
{
	stats_reset(&acc_v);
	stats_reset(&acc_i);
	stats_reset(&acc_t);
	stats_reset(&acc_efc);
	stats_reset(&acc_qc);
}

static void reset_state(void)
{
	memset(&st, 0, sizeof(st));
	last_soh = 1.0f;
	last_pred_ts_ms = 0;
	current_hour_bin = 0;
	last_sample_ts_ms = 0;
	reset_hour_accumulators();
}

static void lstm_step_0123(const float *x_embed, float *out_h)
{
	float x_in[SOH_0123_HIDDEN];
	float gate[4 * SOH_0123_HIDDEN];

	/* Layer 0 input is embed. */
	{
		for (int j = 0; j < 4 * SOH_0123_HIDDEN; j++)
		{
			const int8_t *row_ih = &SOH0123_L0_WIH_Q[j * SOH_0123_EMBED];
			const int8_t *row_hh = &SOH0123_L0_WHH_Q[j * SOH_0123_HIDDEN];
			float acc_ih = 0.0f;
			for (int i = 0; i < SOH_0123_EMBED; i++) acc_ih += ((float)row_ih[i]) * x_embed[i];
			float acc_hh = 0.0f;
			for (int i = 0; i < SOH_0123_HIDDEN; i++) acc_hh += ((float)row_hh[i]) * st.h[0][i];
			gate[j] = acc_ih * SOH0123_L0_WIH_S[j] + acc_hh * SOH0123_L0_WHH_S[j] + SOH0123_L0_B[j];
		}
		for (int h = 0; h < SOH_0123_HIDDEN; h++)
		{
			const float i_gate = sigmoidf_fast(gate[0 * SOH_0123_HIDDEN + h]);
			const float f_gate = sigmoidf_fast(gate[1 * SOH_0123_HIDDEN + h]);
			const float g_gate = tanhf(gate[2 * SOH_0123_HIDDEN + h]);
			const float o_gate = sigmoidf_fast(gate[3 * SOH_0123_HIDDEN + h]);
			st.c[0][h] = f_gate * st.c[0][h] + i_gate * g_gate;
			st.h[0][h] = o_gate * tanhf(st.c[0][h]);
		}
		memcpy(x_in, st.h[0], sizeof(x_in));
	}

	/* Layers 1..2 input is hidden. */
	for (int li = 1; li < SOH_0123_LAYERS; li++)
	{
		const int8_t *wih_q = (li == 1) ? SOH0123_L1_WIH_Q : SOH0123_L2_WIH_Q;
		const float  *wih_s = (li == 1) ? SOH0123_L1_WIH_S : SOH0123_L2_WIH_S;
		const int8_t *whh_q = (li == 1) ? SOH0123_L1_WHH_Q : SOH0123_L2_WHH_Q;
		const float  *whh_s = (li == 1) ? SOH0123_L1_WHH_S : SOH0123_L2_WHH_S;
		const float  *b     = (li == 1) ? SOH0123_L1_B : SOH0123_L2_B;

		for (int j = 0; j < 4 * SOH_0123_HIDDEN; j++)
		{
			const int8_t *row_ih = &wih_q[j * SOH_0123_HIDDEN];
			const int8_t *row_hh = &whh_q[j * SOH_0123_HIDDEN];
			float acc_ih = 0.0f;
			for (int i = 0; i < SOH_0123_HIDDEN; i++) acc_ih += ((float)row_ih[i]) * x_in[i];
			float acc_hh = 0.0f;
			for (int i = 0; i < SOH_0123_HIDDEN; i++) acc_hh += ((float)row_hh[i]) * st.h[li][i];
			gate[j] = acc_ih * wih_s[j] + acc_hh * whh_s[j] + b[j];
		}
		for (int h = 0; h < SOH_0123_HIDDEN; h++)
		{
			const float i_gate = sigmoidf_fast(gate[0 * SOH_0123_HIDDEN + h]);
			const float f_gate = sigmoidf_fast(gate[1 * SOH_0123_HIDDEN + h]);
			const float g_gate = tanhf(gate[2 * SOH_0123_HIDDEN + h]);
			const float o_gate = sigmoidf_fast(gate[3 * SOH_0123_HIDDEN + h]);
			st.c[li][h] = f_gate * st.c[li][h] + i_gate * g_gate;
			st.h[li][h] = o_gate * tanhf(st.c[li][h]);
		}
		memcpy(x_in, st.h[li], sizeof(x_in));
	}

	memcpy(out_h, st.h[SOH_0123_LAYERS - 1], sizeof(float) * SOH_0123_HIDDEN);
}

static float infer_hourly_0123(const float *x_features)
{
	float x_scaled[SOH_0123_IN_FEATURES];
	for (int i = 0; i < SOH_0123_IN_FEATURES; i++)
	{
		const float sc = SOH0123_SCALER_SCALE[i];
		const float denom = (fabsf(sc) > 1e-12f) ? sc : 1.0f;
		x_scaled[i] = (x_features[i] - SOH0123_SCALER_CENTER[i]) / denom;
	}

	/* feature_proj: Linear -> LN -> GELU -> Linear -> GELU */
	float e0[SOH_0123_EMBED];
	float e0n[SOH_0123_EMBED];
	float e1[SOH_0123_EMBED];

	linear_q8_f32(SOH0123_FP0_W_Q, SOH0123_FP0_W_S, SOH0123_FP0_B, x_scaled, SOH_0123_EMBED, SOH_0123_IN_FEATURES, e0);
	layernorm_f32(e0, e0n, SOH0123_LN0_W, SOH0123_LN0_B, SOH_0123_EMBED);
	for (int i = 0; i < SOH_0123_EMBED; i++) e0n[i] = geluf(e0n[i]);

	linear_q8_f32(SOH0123_FP1_W_Q, SOH0123_FP1_W_S, SOH0123_FP1_B, e0n, SOH_0123_EMBED, SOH_0123_EMBED, e1);
	for (int i = 0; i < SOH_0123_EMBED; i++) e1[i] = geluf(e1[i]);

	/* LSTM */
	float h_last[SOH_0123_HIDDEN];
	lstm_step_0123(e1, h_last);

	/* post_norm */
	float x0[SOH_0123_HIDDEN];
	layernorm_f32(h_last, x0, SOH0123_POSTLN_W, SOH0123_POSTLN_B, SOH_0123_HIDDEN);

	/* Residual block 0 */
	{
		float t1[SOH_0123_MLP_HIDDEN];
		float t2[SOH_0123_HIDDEN];
		linear_q8_f32(SOH0123_RB0_FC1_W_Q, SOH0123_RB0_FC1_W_S, SOH0123_RB0_FC1_B, x0, SOH_0123_MLP_HIDDEN, SOH_0123_HIDDEN, t1);
		for (int i = 0; i < SOH_0123_MLP_HIDDEN; i++) t1[i] = geluf(t1[i]);
		linear_q8_f32(SOH0123_RB0_FC2_W_Q, SOH0123_RB0_FC2_W_S, SOH0123_RB0_FC2_B, t1, SOH_0123_HIDDEN, SOH_0123_MLP_HIDDEN, t2);
		for (int i = 0; i < SOH_0123_HIDDEN; i++) t2[i] = t2[i] + x0[i];
		layernorm_f32(t2, x0, SOH0123_RB0_LN_W, SOH0123_RB0_LN_B, SOH_0123_HIDDEN);
	}

	/* Residual block 1 */
	{
		float t1[SOH_0123_MLP_HIDDEN];
		float t2[SOH_0123_HIDDEN];
		linear_q8_f32(SOH0123_RB1_FC1_W_Q, SOH0123_RB1_FC1_W_S, SOH0123_RB1_FC1_B, x0, SOH_0123_MLP_HIDDEN, SOH_0123_HIDDEN, t1);
		for (int i = 0; i < SOH_0123_MLP_HIDDEN; i++) t1[i] = geluf(t1[i]);
		linear_q8_f32(SOH0123_RB1_FC2_W_Q, SOH0123_RB1_FC2_W_S, SOH0123_RB1_FC2_B, t1, SOH_0123_HIDDEN, SOH_0123_MLP_HIDDEN, t2);
		for (int i = 0; i < SOH_0123_HIDDEN; i++) t2[i] = t2[i] + x0[i];
		layernorm_f32(t2, x0, SOH0123_RB1_LN_W, SOH0123_RB1_LN_B, SOH_0123_HIDDEN);
	}

	/* Head */
	float h0[SOH_0123_MLP_HIDDEN];
	float h1[SOH_0123_MLP_HIDDEN];
	linear_q8_f32(SOH0123_HEAD0_W_Q, SOH0123_HEAD0_W_S, SOH0123_HEAD0_B, x0, SOH_0123_MLP_HIDDEN, SOH_0123_HIDDEN, h0);
	for (int i = 0; i < SOH_0123_MLP_HIDDEN; i++) h0[i] = geluf(h0[i]);
	linear_q8_f32(SOH0123_HEAD1_W_Q, SOH0123_HEAD1_W_S, SOH0123_HEAD1_B, h0, SOH_0123_MLP_HIDDEN, SOH_0123_MLP_HIDDEN, h1);
	for (int i = 0; i < SOH_0123_MLP_HIDDEN; i++) h1[i] = geluf(h1[i]);

	float out1[1];
	linear_q8_f32(SOH0123_HEAD2_W_Q, SOH0123_HEAD2_W_S, SOH0123_HEAD2_B, h1, 1, SOH_0123_MLP_HIDDEN, out1);
	return out1[0];
}

void soh_inference_init(void)
{
	inited = 1;
	reset_state();
}

uint32_t soh_inference_last_timestamp_ms(void)
{
	return last_pred_ts_ms;
}

float soh_inference_step(const Cell_Module_t *module)
{
	if (!inited)
	{
		soh_inference_init();
	}

	const uint32_t ts = module->timestamp;

	/* If the timestamp goes backwards (typical for PC replay starting at 0), reset state so
	 * repeated playbacks don't depend on a manual board reset. */
	if (last_sample_ts_ms != 0U && ts < last_sample_ts_ms)
	{
		reset_state();
	}
	const uint32_t bin = ts / SOH_0123_HOUR_MS;

	/* Per-second feature stream */
	const float v = cell_raw_to_v(module->cellVoltages[0]);
	const float i = module->current;
	const float t = module->Temperatures[0];
	const float efc = module->EFC;
	const float qc = module->Q_c;

	if (acc_v.n == 0 && last_sample_ts_ms == 0)
	{
		current_hour_bin = bin;
	}

	/* Hour boundary: compute on previous bin, then reset and start accumulating new bin. */
	if (bin != current_hour_bin)
	{
		if (acc_v.n > 0)
		{
			float feat[SOH_0123_IN_FEATURES];
			int k = 0;

			feat[k++] = stats_mean(&acc_v);
			feat[k++] = stats_std(&acc_v);
			feat[k++] = acc_v.minv;
			feat[k++] = acc_v.maxv;

			feat[k++] = stats_mean(&acc_i);
			feat[k++] = stats_std(&acc_i);
			feat[k++] = acc_i.minv;
			feat[k++] = acc_i.maxv;

			feat[k++] = stats_mean(&acc_t);
			feat[k++] = stats_std(&acc_t);
			feat[k++] = acc_t.minv;
			feat[k++] = acc_t.maxv;

			feat[k++] = stats_mean(&acc_efc);
			feat[k++] = stats_std(&acc_efc);
			feat[k++] = acc_efc.minv;
			feat[k++] = acc_efc.maxv;

			feat[k++] = stats_mean(&acc_qc);
			feat[k++] = stats_std(&acc_qc);
			feat[k++] = acc_qc.minv;
			feat[k++] = acc_qc.maxv;

			last_soh = infer_hourly_0123(feat);
			last_pred_ts_ms = last_sample_ts_ms; /* end of previous hour bin */
		}

		current_hour_bin = bin;
		reset_hour_accumulators();
	}

	stats_push(&acc_v, v);
	stats_push(&acc_i, i);
	stats_push(&acc_t, t);
	stats_push(&acc_efc, efc);
	stats_push(&acc_qc, qc);
	last_sample_ts_ms = ts;

	return last_soh;
}
