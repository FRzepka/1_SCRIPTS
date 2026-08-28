#include "dd_model.h"
#include "dd_weights.h"

#include <math.h>
#include <string.h>

static float hidden[DD_HIDDEN_SIZE];
static float hidden_previous[DD_HIDDEN_SIZE];
static float gate_z[DD_HIDDEN_SIZE];
static float gate_r[DD_HIDDEN_SIZE];
static float recurrent_candidate[DD_HIDDEN_SIZE];
static float mlp_hidden[DD_MLP_SIZE];
static uint32_t input_count;

static float sigmoidf_stable(float value)
{
  if (value >= 0.0f) {
    float exponential = expf(-value);
    return 1.0f / (1.0f + exponential);
  }
  float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

static float dot(const float *weights, const float *values, uint32_t length)
{
  float sum = 0.0f;
  for (uint32_t index = 0U; index < length; ++index) {
    sum += weights[index] * values[index];
  }
  return sum;
}

static void gru_step(const float input[DD_INPUT_SIZE])
{
  memcpy(hidden_previous, hidden, sizeof(hidden));

  for (uint32_t unit = 0U; unit < DD_HIDDEN_SIZE; ++unit) {
    uint32_t z_offset = unit;
    uint32_t r_offset = DD_HIDDEN_SIZE + unit;
    uint32_t h_offset = 2U * DD_HIDDEN_SIZE + unit;

    float z = dd_gru_b[z_offset] + dd_gru_b[3U * DD_HIDDEN_SIZE + z_offset];
    z += dot(&dd_gru_w[z_offset * DD_INPUT_SIZE], input, DD_INPUT_SIZE);
    z += dot(&dd_gru_r[z_offset * DD_HIDDEN_SIZE], hidden_previous, DD_HIDDEN_SIZE);
    gate_z[unit] = sigmoidf_stable(z);

    float r = dd_gru_b[r_offset] + dd_gru_b[3U * DD_HIDDEN_SIZE + r_offset];
    r += dot(&dd_gru_w[r_offset * DD_INPUT_SIZE], input, DD_INPUT_SIZE);
    r += dot(&dd_gru_r[r_offset * DD_HIDDEN_SIZE], hidden_previous, DD_HIDDEN_SIZE);
    gate_r[unit] = sigmoidf_stable(r);

    recurrent_candidate[unit] = dd_gru_b[3U * DD_HIDDEN_SIZE + h_offset];
    recurrent_candidate[unit] += dot(&dd_gru_r[h_offset * DD_HIDDEN_SIZE],
                                     hidden_previous, DD_HIDDEN_SIZE);
  }

  for (uint32_t unit = 0U; unit < DD_HIDDEN_SIZE; ++unit) {
    uint32_t h_offset = 2U * DD_HIDDEN_SIZE + unit;
    float candidate = dd_gru_b[h_offset];
    candidate += dot(&dd_gru_w[h_offset * DD_INPUT_SIZE], input, DD_INPUT_SIZE);
    candidate += gate_r[unit] * recurrent_candidate[unit];
    candidate = tanhf(candidate);
    hidden[unit] = (1.0f - gate_z[unit]) * candidate + gate_z[unit] * hidden_previous[unit];
  }
}

static float mlp_forward(void)
{
  for (uint32_t unit = 0U; unit < DD_MLP_SIZE; ++unit) {
    float value = dd_mlp_b1[unit];
    value += dot(&dd_mlp_w1[unit * DD_HIDDEN_SIZE], hidden, DD_HIDDEN_SIZE);
    mlp_hidden[unit] = value > 0.0f ? value : 0.0f;
  }
  float output = dd_mlp_b2 + dot(dd_mlp_w2, mlp_hidden, DD_MLP_SIZE);
  return sigmoidf_stable(output);
}

void DD_ModelReset(void)
{
  memset(hidden, 0, sizeof(hidden));
  input_count = 0U;
}

uint8_t DD_ModelPush(const float input[DD_INPUT_SIZE], float *soc)
{
  float scaled[DD_INPUT_SIZE];
  for (uint32_t feature = 0U; feature < DD_INPUT_SIZE; ++feature) {
    scaled[feature] = (input[feature] - dd_scaler_center[feature]) / dd_scaler_scale[feature];
  }
  gru_step(scaled);
  ++input_count;
  *soc = mlp_forward();
  return 1U;
}
