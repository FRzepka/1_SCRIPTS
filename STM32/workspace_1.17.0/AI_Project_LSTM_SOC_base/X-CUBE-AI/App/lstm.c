/**
  ******************************************************************************
  * @file    lstm.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-10-08T21:47:54+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "lstm.h"
#include "lstm_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_lstm
 
#undef AI_LSTM_MODEL_SIGNATURE
#define AI_LSTM_MODEL_SIGNATURE     "0xe6e375ad30c0da71847972e6c1b6c291"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-10-08T21:47:54+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_LSTM_N_BATCHES
#define AI_LSTM_N_BATCHES         (1)

static ai_ptr g_lstm_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_lstm_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_Transpose_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_output2_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_Transpose_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_1_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_4_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_initial_h_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_initial_c_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_peepholes_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 448, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_core_lstm_LSTM_output_0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_initial_c, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_lstm_LSTM_output_0_initial_c_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_initial_h, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_lstm_LSTM_output_0_initial_h_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_kernel, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 6, 256, 1, 1), AI_STRIDE_INIT(4, 4, 24, 6144, 6144),
  1, &_core_lstm_LSTM_output_0_kernel_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_output0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_lstm_LSTM_output_0_output0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_output1, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_lstm_LSTM_output_0_output1_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_output2, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_lstm_LSTM_output_0_output2_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_peepholes, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 192), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &_core_lstm_LSTM_output_0_peepholes_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_recurrent, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 64, 256, 1, 1), AI_STRIDE_INIT(4, 4, 256, 65536, 65536),
  1, &_core_lstm_LSTM_output_0_recurrent_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_scratch0, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 448, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1792, 1792),
  1, &_core_lstm_LSTM_output_0_scratch0_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_Transpose_1_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_lstm_Transpose_1_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _core_lstm_Transpose_output_0_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &_core_lstm_Transpose_output_0_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_mlp_mlp_0_Gemm_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_mlp_mlp_0_Gemm_output_0_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_weights, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_core_mlp_mlp_0_Gemm_output_0_weights_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_1_Relu_output_0_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_core_mlp_mlp_1_Relu_output_0_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_core_mlp_mlp_3_Gemm_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_core_mlp_mlp_3_Gemm_output_0_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_weights, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 4, 256, 256, 256),
  1, &_core_mlp_mlp_3_Gemm_output_0_weights_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _core_mlp_mlp_4_Sigmoid_output_0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_core_mlp_mlp_4_Sigmoid_output_0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &input_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_mlp_mlp_4_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_3_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_4_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _core_mlp_mlp_4_Sigmoid_output_0_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_core_mlp_mlp_4_Sigmoid_output_0_chain,
  NULL, &_core_mlp_mlp_4_Sigmoid_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_3_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_core_mlp_mlp_3_Gemm_output_0_weights, &_core_mlp_mlp_3_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _core_mlp_mlp_3_Gemm_output_0_layer, 23,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_core_mlp_mlp_3_Gemm_output_0_chain,
  NULL, &_core_mlp_mlp_4_Sigmoid_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_mlp_mlp_1_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_1_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _core_mlp_mlp_1_Relu_output_0_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_core_mlp_mlp_1_Relu_output_0_chain,
  NULL, &_core_mlp_mlp_3_Gemm_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_lstm_Transpose_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_mlp_mlp_0_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_core_mlp_mlp_0_Gemm_output_0_weights, &_core_mlp_mlp_0_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _core_mlp_mlp_0_Gemm_output_0_layer, 21,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_core_mlp_mlp_0_Gemm_output_0_chain,
  NULL, &_core_mlp_mlp_1_Relu_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_lstm_Transpose_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_lstm_LSTM_output_0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_lstm_Transpose_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _core_lstm_Transpose_1_output_0_layer, 5,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_core_lstm_Transpose_1_output_0_chain,
  NULL, &_core_mlp_mlp_0_Gemm_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_BATCH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_lstm_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_core_lstm_LSTM_output_0_output0, &_core_lstm_LSTM_output_0_output1, &_core_lstm_LSTM_output_0_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 9, &_core_lstm_LSTM_output_0_kernel, &_core_lstm_LSTM_output_0_recurrent, NULL, NULL, NULL, &_core_lstm_LSTM_output_0_bias, &_core_lstm_LSTM_output_0_initial_h, &_core_lstm_LSTM_output_0_initial_c, &_core_lstm_LSTM_output_0_peepholes),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_lstm_LSTM_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _core_lstm_LSTM_output_0_layer, 2,
  LSTM_TYPE, 0x0, NULL,
  lstm, forward_lstm,
  &_core_lstm_LSTM_output_0_chain,
  NULL, &_core_lstm_Transpose_1_output_0_layer, AI_STATIC, 
  .n_units = 64, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = true, 
  .out_nl = nl_func_tanh_array_f32, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .cell_clip = 3e+38, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _core_lstm_Transpose_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_core_lstm_Transpose_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _core_lstm_Transpose_output_0_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_core_lstm_Transpose_output_0_chain,
  NULL, &_core_lstm_LSTM_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_BATCH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 90884, 1, 1),
    90884, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2584, 1, 1),
    2584, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_LSTM_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_LSTM_OUT_NUM, &_core_mlp_mlp_4_Sigmoid_output_0_output),
  &_core_lstm_Transpose_output_0_layer, 0xdbde6175, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 90884, 1, 1),
      90884, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2584, 1, 1),
      2584, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_LSTM_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_LSTM_OUT_NUM, &_core_mlp_mlp_4_Sigmoid_output_0_output),
  &_core_lstm_Transpose_output_0_layer, 0xdbde6175, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool lstm_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_lstm_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_lstm_activations_map[0] + 1768);
    input_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 1768);
    _core_lstm_Transpose_output_0_output_array.data = AI_PTR(g_lstm_activations_map[0] + 1792);
    _core_lstm_Transpose_output_0_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 1792);
    _core_lstm_LSTM_output_0_scratch0_array.data = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_lstm_LSTM_output_0_scratch0_array.data_start = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_lstm_LSTM_output_0_output0_array.data = AI_PTR(g_lstm_activations_map[0] + 1816);
    _core_lstm_LSTM_output_0_output0_array.data_start = AI_PTR(g_lstm_activations_map[0] + 1816);
    _core_lstm_LSTM_output_0_output1_array.data = AI_PTR(g_lstm_activations_map[0] + 2072);
    _core_lstm_LSTM_output_0_output1_array.data_start = AI_PTR(g_lstm_activations_map[0] + 2072);
    _core_lstm_LSTM_output_0_output2_array.data = AI_PTR(g_lstm_activations_map[0] + 2328);
    _core_lstm_LSTM_output_0_output2_array.data_start = AI_PTR(g_lstm_activations_map[0] + 2328);
    _core_lstm_Transpose_1_output_0_output_array.data = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_lstm_Transpose_1_output_0_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_mlp_mlp_0_Gemm_output_0_output_array.data = AI_PTR(g_lstm_activations_map[0] + 256);
    _core_mlp_mlp_0_Gemm_output_0_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 256);
    _core_mlp_mlp_1_Relu_output_0_output_array.data = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_mlp_mlp_1_Relu_output_0_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_mlp_mlp_3_Gemm_output_0_output_array.data = AI_PTR(g_lstm_activations_map[0] + 256);
    _core_mlp_mlp_3_Gemm_output_0_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 256);
    _core_mlp_mlp_4_Sigmoid_output_0_output_array.data = AI_PTR(g_lstm_activations_map[0] + 0);
    _core_mlp_mlp_4_Sigmoid_output_0_output_array.data_start = AI_PTR(g_lstm_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool lstm_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_lstm_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _core_lstm_LSTM_output_0_kernel_array.format |= AI_FMT_FLAG_CONST;
    _core_lstm_LSTM_output_0_kernel_array.data = AI_PTR(g_lstm_weights_map[0] + 0);
    _core_lstm_LSTM_output_0_kernel_array.data_start = AI_PTR(g_lstm_weights_map[0] + 0);
    _core_lstm_LSTM_output_0_recurrent_array.format |= AI_FMT_FLAG_CONST;
    _core_lstm_LSTM_output_0_recurrent_array.data = AI_PTR(g_lstm_weights_map[0] + 6144);
    _core_lstm_LSTM_output_0_recurrent_array.data_start = AI_PTR(g_lstm_weights_map[0] + 6144);
    _core_lstm_LSTM_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _core_lstm_LSTM_output_0_bias_array.data = AI_PTR(g_lstm_weights_map[0] + 71680);
    _core_lstm_LSTM_output_0_bias_array.data_start = AI_PTR(g_lstm_weights_map[0] + 71680);
    _core_lstm_LSTM_output_0_initial_h_array.format |= AI_FMT_FLAG_CONST;
    _core_lstm_LSTM_output_0_initial_h_array.data = AI_PTR(g_lstm_weights_map[0] + 72704);
    _core_lstm_LSTM_output_0_initial_h_array.data_start = AI_PTR(g_lstm_weights_map[0] + 72704);
    _core_lstm_LSTM_output_0_initial_c_array.format |= AI_FMT_FLAG_CONST;
    _core_lstm_LSTM_output_0_initial_c_array.data = AI_PTR(g_lstm_weights_map[0] + 72960);
    _core_lstm_LSTM_output_0_initial_c_array.data_start = AI_PTR(g_lstm_weights_map[0] + 72960);
    _core_lstm_LSTM_output_0_peepholes_array.format |= AI_FMT_FLAG_CONST;
    _core_lstm_LSTM_output_0_peepholes_array.data = AI_PTR(g_lstm_weights_map[0] + 73216);
    _core_lstm_LSTM_output_0_peepholes_array.data_start = AI_PTR(g_lstm_weights_map[0] + 73216);
    _core_mlp_mlp_0_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _core_mlp_mlp_0_Gemm_output_0_weights_array.data = AI_PTR(g_lstm_weights_map[0] + 73984);
    _core_mlp_mlp_0_Gemm_output_0_weights_array.data_start = AI_PTR(g_lstm_weights_map[0] + 73984);
    _core_mlp_mlp_0_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _core_mlp_mlp_0_Gemm_output_0_bias_array.data = AI_PTR(g_lstm_weights_map[0] + 90368);
    _core_mlp_mlp_0_Gemm_output_0_bias_array.data_start = AI_PTR(g_lstm_weights_map[0] + 90368);
    _core_mlp_mlp_3_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _core_mlp_mlp_3_Gemm_output_0_weights_array.data = AI_PTR(g_lstm_weights_map[0] + 90624);
    _core_mlp_mlp_3_Gemm_output_0_weights_array.data_start = AI_PTR(g_lstm_weights_map[0] + 90624);
    _core_mlp_mlp_3_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _core_mlp_mlp_3_Gemm_output_0_bias_array.data = AI_PTR(g_lstm_weights_map[0] + 90880);
    _core_mlp_mlp_3_Gemm_output_0_bias_array.data_start = AI_PTR(g_lstm_weights_map[0] + 90880);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_lstm_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_LSTM_MODEL_NAME,
      .model_signature   = AI_LSTM_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 22574,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xdbde6175,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_lstm_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_LSTM_MODEL_NAME,
      .model_signature   = AI_LSTM_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 22574,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xdbde6175,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_lstm_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_lstm_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_lstm_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_lstm_create(network, AI_LSTM_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_lstm_data_params_get(&params) != true) {
    err = ai_lstm_get_error(*network);
    return err;
  }
#if defined(AI_LSTM_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_LSTM_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_lstm_init(*network, &params) != true) {
    err = ai_lstm_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_lstm_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_lstm_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_lstm_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_lstm_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= lstm_configure_weights(net_ctx, params);
  ok &= lstm_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_lstm_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_lstm_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_LSTM_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

