/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-19T08:43:57+0000
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x60ceb04812f35cec7b27dcb14adf84ee"
#define _STAI_NETWORK_DATETIME            "2026-05-19T08:43:57+0000"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 120, 20),
      STAI_DECLARE_ARRAY(float, 1, 0.10564419627189636f),
      STAI_DECLARE_ARRAY(int16_t, 1, 45)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 120, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.0050134118646383286f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 121888),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_2_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_3_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_4_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_4_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_5_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_5_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_6_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_6_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_7_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_7_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_8_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_8_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 96),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_9_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_9_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 1920),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_10_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_10_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 384),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_11_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_11_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 9600),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_12_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_12_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 384),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_13_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_13_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 46080),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_14_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_14_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 384),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_15_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_15_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_16_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_16_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 384),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_17_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_17_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_18_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_18_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_19_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_19_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_20_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_20_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_21_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_21_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_22_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_22_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_23_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_23_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 184320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_24_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_24_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 12288),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_25_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_25_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 512),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_26_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_26_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 128),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_27_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_27_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 4),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_tfl_pseudo_qconst18_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst16_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst14_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst12_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst10_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst8_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst6_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst4_4D_array,(stai_ptr)g_network_conv2d_22_weights_array,(stai_ptr)g_network_conv2d_22_bias_array,(stai_ptr)g_network_conv2d_5_weights_array,(stai_ptr)g_network_conv2d_5_bias_array,(stai_ptr)g_network_conv2d_16_weights_array,(stai_ptr)g_network_conv2d_16_bias_array,(stai_ptr)g_network_conv2d_34_weights_array,(stai_ptr)g_network_conv2d_34_bias_array,(stai_ptr)g_network_conv2d_48_weights_array,(stai_ptr)g_network_conv2d_63_weights_array,(stai_ptr)g_network_conv2d_77_weights_array,(stai_ptr)g_network_conv2d_92_weights_array,(stai_ptr)g_network_conv2d_106_weights_array,(stai_ptr)g_network_conv2d_121_weights_array,(stai_ptr)g_network_conv2d_135_weights_array,(stai_ptr)g_network_conv2d_148_weights_array,(stai_ptr)g_network_conv2d_148_bias_array,(stai_ptr)g_network_conv2d_158_weights_array,(stai_ptr)g_network_conv2d_158_bias_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041041214019060135f),
    AI_PACK_INTQ_ZP(41)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013773981481790543f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02312243916094303f),
    AI_PACK_INTQ_ZP(31)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_20_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00866710301488638f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_22_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05705142021179199f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_27_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05705142021179199f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_34_0_0_eltwise_36_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05669168010354042f),
    AI_PACK_INTQ_ZP(79)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_36_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05652033910155296f),
    AI_PACK_INTQ_ZP(79)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst18_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005197778809815645f),
    AI_PACK_INTQ_ZP(45)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_40_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010546959936618805f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_48_0_0_eltwise_50_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029165739193558693f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_50_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029433224350214005f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst16_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008359195198863745f),
    AI_PACK_INTQ_ZP(-34)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_54_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013859591446816921f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_56_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05705142021179199f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_63_0_0_eltwise_65_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.058796655386686325f),
    AI_PACK_INTQ_ZP(29)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_65_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05876212194561958f),
    AI_PACK_INTQ_ZP(28)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst14_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00034713049535639584f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_69_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022744812071323395f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_77_0_0_eltwise_79_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030611073598265648f),
    AI_PACK_INTQ_ZP(19)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_79_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030895592644810677f),
    AI_PACK_INTQ_ZP(20)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst12_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010852782288566232f),
    AI_PACK_INTQ_ZP(-53)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_83_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013009907677769661f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_85_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05705142021179199f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_92_0_0_eltwise_94_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07281295955181122f),
    AI_PACK_INTQ_ZP(43)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_94_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0728796049952507f),
    AI_PACK_INTQ_ZP(43)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst10_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0003900982264894992f),
    AI_PACK_INTQ_ZP(-26)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_98_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.024060485884547234f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_106_0_0_eltwise_108_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029464116320014f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_108_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.029880983754992485f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst8_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006146567175164819f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_112_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01472543366253376f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_114_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06278161704540253f),
    AI_PACK_INTQ_ZP(-29)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_121_0_0_eltwise_123_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10135211050510406f),
    AI_PACK_INTQ_ZP(16)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_123_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07416337728500366f),
    AI_PACK_INTQ_ZP(69)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst6_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007110435399226844f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_127_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.016977179795503616f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_135_0_0_eltwise_137_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01921113394200802f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_137_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01973133161664009f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst4_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007557491189800203f),
    AI_PACK_INTQ_ZP(-26)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_141_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009253321215510368f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_143_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07079243659973145f),
    AI_PACK_INTQ_ZP(-41)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_148_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.08966033160686493f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_152_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04055215045809746f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_158_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0050134118646383286f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_158_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.022365882992744446f),
    AI_PACK_INTQ_ZP(0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  nl_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  nl_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_34_0_0_eltwise_36_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst18_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  nl_40_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_48_0_0_eltwise_50_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst16_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  nl_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_56_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_63_0_0_eltwise_65_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_65_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst14_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  nl_69_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_77_0_0_eltwise_79_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_79_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst12_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  nl_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_85_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_92_0_0_eltwise_94_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_94_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst10_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  nl_98_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_106_0_0_eltwise_108_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_108_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst8_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  nl_112_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_114_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_121_0_0_eltwise_123_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_123_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst6_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  nl_127_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_135_0_0_eltwise_137_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_137_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst4_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 96, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  nl_141_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_143_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 11520, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_148_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 15360, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  nl_152_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 15360, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_158_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 120, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_158_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_158_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_158_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_5_output_array, &conv2d_5_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  nl_9_output, AI_STATIC,
  68, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_9_output_array, &nl_9_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_16_output_array, &conv2d_16_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  nl_20_output, AI_STATIC,
  62, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_20_output_array, &nl_20_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_22_output_array, &conv2d_22_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_27_output, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_27_output_array, &eltwise_27_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_34_0_0_eltwise_36_conversion_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_34_0_0_eltwise_36_conversion_output_array, &conv2d_34_0_0_eltwise_36_conversion_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_36_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_36_output_array, &eltwise_36_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst18_4D, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst18_4D_array, &tfl_pseudo_qconst18_4D_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  nl_40_output, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_40_output_array, &nl_40_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_48_0_0_eltwise_50_conversion_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_48_0_0_eltwise_50_conversion_output_array, &conv2d_48_0_0_eltwise_50_conversion_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_50_output, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_50_output_array, &eltwise_50_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst16_4D, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst16_4D_array, &tfl_pseudo_qconst16_4D_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  nl_54_output, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_54_output_array, &nl_54_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_56_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_56_output_array, &eltwise_56_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_63_0_0_eltwise_65_conversion_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_63_0_0_eltwise_65_conversion_output_array, &conv2d_63_0_0_eltwise_65_conversion_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_65_output, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_65_output_array, &eltwise_65_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst14_4D, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst14_4D_array, &tfl_pseudo_qconst14_4D_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  nl_69_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_69_output_array, &nl_69_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_77_0_0_eltwise_79_conversion_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_77_0_0_eltwise_79_conversion_output_array, &conv2d_77_0_0_eltwise_79_conversion_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_79_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_79_output_array, &eltwise_79_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst12_4D, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst12_4D_array, &tfl_pseudo_qconst12_4D_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  nl_83_output, AI_STATIC,
  66, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_83_output_array, &nl_83_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_85_output, AI_STATIC,
  56, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_85_output_array, &eltwise_85_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_92_0_0_eltwise_94_conversion_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_92_0_0_eltwise_94_conversion_output_array, &conv2d_92_0_0_eltwise_94_conversion_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_94_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_94_output_array, &eltwise_94_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst10_4D, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst10_4D_array, &tfl_pseudo_qconst10_4D_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  nl_98_output, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_98_output_array, &nl_98_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_106_0_0_eltwise_108_conversion_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_106_0_0_eltwise_108_conversion_output_array, &conv2d_106_0_0_eltwise_108_conversion_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_108_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_108_output_array, &eltwise_108_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst8_4D, AI_STATIC,
  95, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst8_4D_array, &tfl_pseudo_qconst8_4D_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  nl_112_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_112_output_array, &nl_112_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_114_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_114_output_array, &eltwise_114_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_121_0_0_eltwise_123_conversion_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_121_0_0_eltwise_123_conversion_output_array, &conv2d_121_0_0_eltwise_123_conversion_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_123_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_123_output_array, &eltwise_123_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst6_4D, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst6_4D_array, &tfl_pseudo_qconst6_4D_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  nl_127_output, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_127_output_array, &nl_127_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_135_0_0_eltwise_137_conversion_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &conv2d_135_0_0_eltwise_137_conversion_output_array, &conv2d_135_0_0_eltwise_137_conversion_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_137_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_137_output_array, &eltwise_137_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst4_4D, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &tfl_pseudo_qconst4_4D_array, &tfl_pseudo_qconst4_4D_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  nl_141_output, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &nl_141_output_array, &nl_141_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_143_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 120), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &eltwise_143_output_array, &eltwise_143_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_148_output, AI_STATIC,
  10, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 120), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_148_output_array, &conv2d_148_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  nl_152_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 120), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_152_output_array, &nl_152_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_158_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_158_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_158_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 120), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &conv2d_158_output_array, &conv2d_158_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_158_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_158_scratch0_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_158_weights, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 128, 1, 1, 1), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_158_weights_array, &conv2d_158_weights_array_intq)



AI_STATIC_CONST ai_i8 nl_9_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -122, -119, -116, -113, -110, -107, -104, -101, -98, -95, -92, -89, -86, -83, -80, -77, -74, -71, -68, -65, -62, -59, -56, -54, -51, -48, -45, -42, -39, -36, -33, -30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_9_nl_params, AI_ARRAY_FORMAT_S8,
    nl_9_nl_params_data, nl_9_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_9_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_9_chain,
  NULL, &nl_9_layer, AI_STATIC, 
  .nl_params = &nl_9_nl_params, 
)


AI_STATIC_CONST ai_i8 nl_20_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -123, -120, -117, -115, -112, -109, -107, -104, -101, -99, -96, -93, -91, -88, -85, -83, -80, -77, -75, -72, -69, -67, -64, -61, -59, -56, -53, -51, -48, -45, -43, -40, -37, -35, -32, -29, -27, -24, -21, -19, -16, -13, -11, -8, -5, -3, 0, 3, 5, 8, 11, 13, 16, 19, 21, 24, 27, 29, 32, 35, 37, 40, 43, 45, 48, 51, 53, 56, 59, 61, 64, 67, 69, 72, 75, 77, 80, 83, 85, 88, 91, 93, 96, 99, 101, 104, 107, 109, 112, 115, 117, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_20_nl_params, AI_ARRAY_FORMAT_S8,
    nl_20_nl_params_data, nl_20_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_20_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_20_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_20_chain,
  NULL, &nl_20_layer, AI_STATIC, 
  .nl_params = &nl_20_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_20_output, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_27_layer, 27,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_27_chain,
  NULL, &eltwise_27_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_34_0_0_eltwise_36_conversion_output, &tfl_pseudo_qconst18_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_36_layer, 36,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_36_chain,
  NULL, &eltwise_36_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_40_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -123, -117, -112, -107, -101, -96, -90, -85, -80, -74, -69, -64, -58, -53, -48, -42, -37, -32, -26, -21, -15, -10, -5, 1, 6, 11, 17, 22, 27, 33, 38, 43, 49, 54, 60, 65, 70, 76, 81, 86, 92, 97, 102, 108, 113, 119, 124, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_40_nl_params, AI_ARRAY_FORMAT_S8,
    nl_40_nl_params_data, nl_40_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_40_layer, 40,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_40_chain,
  NULL, &nl_40_layer, AI_STATIC, 
  .nl_params = &nl_40_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_50_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_48_0_0_eltwise_50_conversion_output, &tfl_pseudo_qconst16_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_50_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_50_layer, 50,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_50_chain,
  NULL, &eltwise_50_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_54_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -122, -120, -117, -115, -113, -111, -109, -107, -105, -103, -100, -98, -96, -94, -92, -90, -88, -86, -83, -81, -79, -77, -75, -73, -71, -69, -66, -64, -62, -60, -58, -56, -54, -52, -49, -47, -45, -43, -41, -39, -37, -35, -32, -30, -28, -26, -24, -22, -20, -18, -15, -13, -11, -9, -7, -5, -3, -1, 2, 4, 6, 8, 10, 12, 14, 16, 19, 21, 23, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 46, 48, 50, 53, 55, 57, 59, 61, 63, 65, 67, 70, 72, 74, 76, 78, 80, 82, 84, 86, 89, 91, 93, 95, 97, 99, 101, 103, 106, 108, 110, 112, 114, 116, 118, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_54_nl_params, AI_ARRAY_FORMAT_S8,
    nl_54_nl_params_data, nl_54_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_50_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_54_layer, 54,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_54_chain,
  NULL, &nl_54_layer, AI_STATIC, 
  .nl_params = &nl_54_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_54_output, &eltwise_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_56_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_56_layer, 56,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_56_chain,
  NULL, &eltwise_56_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_63_0_0_eltwise_65_conversion_output, &tfl_pseudo_qconst14_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_65_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_65_layer, 65,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_65_chain,
  NULL, &eltwise_65_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_69_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -123, -120, -118, -115, -112, -110, -107, -105, -102, -100, -97, -94, -92, -89, -87, -84, -81, -79, -76, -74, -71, -69, -66, -63, -61, -58, -56, -53, -50, -48, -45, -43, -40, -38, -35, -32, -30, -27, -25, -22, -19, -17, -14, -12, -9, -7, -4, -1, 1, 4, 6, 9, 12, 14, 17, 19, 22, 24, 27, 30, 32, 35, 37, 40, 43, 45, 48, 50, 53, 55, 58, 61, 63, 66, 68, 71, 74, 76, 79, 81, 84, 86, 89, 92, 94, 97, 99, 102, 105, 107, 110, 112, 115, 117, 120, 123, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_69_nl_params, AI_ARRAY_FORMAT_S8,
    nl_69_nl_params_data, nl_69_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_69_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_69_layer, 69,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_69_chain,
  NULL, &nl_69_layer, AI_STATIC, 
  .nl_params = &nl_69_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_79_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_77_0_0_eltwise_79_conversion_output, &tfl_pseudo_qconst12_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_79_layer, 79,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_79_chain,
  NULL, &eltwise_79_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_83_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -123, -121, -119, -116, -114, -111, -109, -107, -104, -102, -100, -97, -95, -92, -90, -88, -85, -83, -81, -78, -76, -73, -71, -69, -66, -64, -62, -59, -57, -54, -52, -50, -47, -45, -43, -40, -38, -35, -33, -31, -28, -26, -24, -21, -19, -16, -14, -12, -9, -7, -5, -2, 0, 3, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38, 41, 43, 45, 48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 71, 74, 76, 79, 81, 83, 86, 88, 90, 93, 95, 98, 100, 102, 105, 107, 109, 112, 114, 117, 119, 121, 124, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_83_nl_params, AI_ARRAY_FORMAT_S8,
    nl_83_nl_params_data, nl_83_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_83_layer, 83,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_83_chain,
  NULL, &nl_83_layer, AI_STATIC, 
  .nl_params = &nl_83_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_85_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_83_output, &eltwise_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_85_layer, 85,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_85_chain,
  NULL, &eltwise_85_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_94_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_92_0_0_eltwise_94_conversion_output, &tfl_pseudo_qconst10_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_94_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_94_layer, 94,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_94_chain,
  NULL, &eltwise_94_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_98_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -122, -119, -116, -113, -110, -107, -104, -101, -98, -95, -92, -89, -86, -83, -80, -77, -73, -70, -67, -64, -61, -58, -55, -52, -49, -46, -43, -40, -37, -34, -31, -28, -25, -22, -19, -16, -13, -10, -7, -4, -1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_98_nl_params, AI_ARRAY_FORMAT_S8,
    nl_98_nl_params_data, nl_98_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_98_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_94_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_98_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_98_layer, 98,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_98_chain,
  NULL, &nl_98_layer, AI_STATIC, 
  .nl_params = &nl_98_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_108_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_106_0_0_eltwise_108_conversion_output, &tfl_pseudo_qconst8_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_108_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_108_layer, 108,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_108_chain,
  NULL, &eltwise_108_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_112_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -122, -120, -118, -116, -114, -112, -110, -108, -106, -104, -102, -100, -98, -96, -94, -91, -89, -87, -85, -83, -81, -79, -77, -75, -73, -71, -69, -67, -65, -63, -61, -59, -57, -55, -53, -51, -49, -47, -45, -43, -41, -39, -37, -35, -33, -31, -29, -27, -25, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 116, 118, 120, 122, 124, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_112_nl_params, AI_ARRAY_FORMAT_S8,
    nl_112_nl_params_data, nl_112_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_112_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_108_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_112_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_112_layer, 112,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_112_chain,
  NULL, &nl_112_layer, AI_STATIC, 
  .nl_params = &nl_112_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_114_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_112_output, &eltwise_85_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_114_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_114_layer, 114,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_114_chain,
  NULL, &eltwise_114_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_123_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_121_0_0_eltwise_123_conversion_output, &tfl_pseudo_qconst6_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_123_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_123_layer, 123,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_123_chain,
  NULL, &eltwise_123_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_127_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -124, -119, -115, -111, -106, -102, -97, -93, -89, -84, -80, -76, -71, -67, -62, -58, -54, -49, -45, -41, -36, -32, -28, -23, -19, -14, -10, -6, -1, 3, 7, 12, 16, 21, 25, 29, 34, 38, 42, 47, 51, 55, 60, 64, 69, 73, 77, 82, 86, 90, 95, 99, 104, 108, 112, 117, 121, 125 };
AI_ARRAY_OBJ_DECLARE(
    nl_127_nl_params, AI_ARRAY_FORMAT_S8,
    nl_127_nl_params_data, nl_127_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_127_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_123_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_127_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_127_layer, 127,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_127_chain,
  NULL, &nl_127_layer, AI_STATIC, 
  .nl_params = &nl_127_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_137_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_135_0_0_eltwise_137_conversion_output, &tfl_pseudo_qconst4_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_137_layer, 137,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_137_chain,
  NULL, &eltwise_137_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_141_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -122, -119, -117, -115, -113, -111, -109, -107, -105, -102, -100, -98, -96, -94, -92, -90, -87, -85, -83, -81, -79, -77, -75, -73, -70, -68, -66, -64, -62, -60, -58, -56, -53, -51, -49, -47, -45, -43, -41, -38, -36, -34, -32, -30, -28, -26, -24, -21, -19, -17, -15, -13, -11, -9, -6, -4, -2, 0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 26, 28, 30, 32, 34, 36, 38, 40, 43, 45, 47, 49, 51, 53, 55, 58, 60, 62, 64, 66, 68, 70, 72, 75, 77, 79, 81, 83, 85, 87, 89, 92, 94, 96, 98, 100, 102, 104, 107, 109, 111, 113, 115, 117, 119, 121, 124, 126, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_141_nl_params, AI_ARRAY_FORMAT_S8,
    nl_141_nl_params_data, nl_141_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_141_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_137_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_141_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_141_layer, 141,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_141_chain,
  NULL, &nl_141_layer, AI_STATIC, 
  .nl_params = &nl_141_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_143_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_141_output, &eltwise_114_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_143_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_143_layer, 143,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_143_chain,
  NULL, &eltwise_143_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_152_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -124, -121, -119, -117, -115, -113, -110, -108, -106, -104, -101, -99, -97, -95, -93, -90, -88, -86, -84, -82, -79, -77, -75, -73, -71, -68, -66, -64, -62, -59, -57, -55, -53, -51, -48, -46, -44, -42, -40, -37, -35, -33, -31, -29, -26, -24, -22, -20, -17, -15, -13, -11, -9, -6, -4, -2, 0, 2, 5, 7, 9, 11, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 47, 49, 51, 53, 56, 58, 60, 62, 64, 67, 69, 71, 73, 75, 78, 80, 82, 84, 86, 89, 91, 93, 95, 98, 100, 102, 104, 106, 109, 111, 113, 115, 117, 120, 122, 124, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_152_nl_params, AI_ARRAY_FORMAT_S8,
    nl_152_nl_params_data, nl_152_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_152_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_148_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_152_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_152_layer, 152,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_152_chain,
  NULL, &nl_152_layer, AI_STATIC, 
  .nl_params = &nl_152_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_158_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_152_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_158_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_158_weights, &conv2d_158_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_158_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_158_layer, 158,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_158_chain,
  NULL, &conv2d_158_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_nl_integer_nl_9(_stai_network_context* net_ctx)
{
  conv2d_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 87328);
  conv2d_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87328);
  nl_9_output_array.data = AI_PTR(net_ctx->_activations[0] + 87328);
  nl_9_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87328);
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 1, { conv2d_5_output.data->data});
  forward_nl_integer(&nl_9_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, { nl_9_output.data->data});
}
void forward_lite_nl_integer_nl_20(_stai_network_context* net_ctx)
{
  conv2d_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 86848);
  conv2d_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 86848);
  nl_20_output_array.data = AI_PTR(net_ctx->_activations[0] + 86848);
  nl_20_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 86848);
  _STAI_NETWORK_EVENT_NODE_START_CB(20, 1, { conv2d_16_output.data->data});
  forward_nl_integer(&nl_20_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(20, 1, { nl_20_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_27(_stai_network_context* net_ctx)
{
  nl_20_output_array.data = AI_PTR(net_ctx->_activations[0] + 86848);
  nl_20_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 86848);
  conv2d_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  conv2d_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  eltwise_27_output_array.data = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_27_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 110368);
  _STAI_NETWORK_EVENT_NODE_START_CB(27, 2, { nl_20_output.data->data,conv2d_22_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_27_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(27, 1, { eltwise_27_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_36(_stai_network_context* net_ctx)
{
  conv2d_34_0_0_eltwise_36_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  conv2d_34_0_0_eltwise_36_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  tfl_pseudo_qconst18_4D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  tfl_pseudo_qconst18_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 2, { conv2d_34_0_0_eltwise_36_conversion_output.data->data,tfl_pseudo_qconst18_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_36_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, { eltwise_36_output.data->data});
}
void forward_lite_nl_integer_nl_40(_stai_network_context* net_ctx)
{
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  nl_40_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  nl_40_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, { eltwise_36_output.data->data});
  forward_nl_integer(&nl_40_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, { nl_40_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_50(_stai_network_context* net_ctx)
{
  conv2d_48_0_0_eltwise_50_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  conv2d_48_0_0_eltwise_50_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  tfl_pseudo_qconst16_4D_array.data = AI_PTR(net_ctx->_weights[1] + 0);
  tfl_pseudo_qconst16_4D_array.data_start = AI_PTR(net_ctx->_weights[1] + 0);
  eltwise_50_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  eltwise_50_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  _STAI_NETWORK_EVENT_NODE_START_CB(50, 2, { conv2d_48_0_0_eltwise_50_conversion_output.data->data,tfl_pseudo_qconst16_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_50_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(50, 1, { eltwise_50_output.data->data});
}
void forward_lite_nl_integer_nl_54(_stai_network_context* net_ctx)
{
  eltwise_50_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  eltwise_50_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  nl_54_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  nl_54_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, { eltwise_50_output.data->data});
  forward_nl_integer(&nl_54_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { nl_54_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_56(_stai_network_context* net_ctx)
{
  nl_54_output_array.data = AI_PTR(net_ctx->_activations[0] + 60832);
  nl_54_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 60832);
  eltwise_27_output_array.data = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_27_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_56_output_array.data = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_56_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 110368);
  _STAI_NETWORK_EVENT_NODE_START_CB(56, 2, { nl_54_output.data->data,eltwise_27_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_56_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(56, 1, { eltwise_56_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_65(_stai_network_context* net_ctx)
{
  conv2d_63_0_0_eltwise_65_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  conv2d_63_0_0_eltwise_65_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  tfl_pseudo_qconst14_4D_array.data = AI_PTR(net_ctx->_weights[2] + 0);
  tfl_pseudo_qconst14_4D_array.data_start = AI_PTR(net_ctx->_weights[2] + 0);
  eltwise_65_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  eltwise_65_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  _STAI_NETWORK_EVENT_NODE_START_CB(65, 2, { conv2d_63_0_0_eltwise_65_conversion_output.data->data,tfl_pseudo_qconst14_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_65_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(65, 1, { eltwise_65_output.data->data});
}
void forward_lite_nl_integer_nl_69(_stai_network_context* net_ctx)
{
  eltwise_65_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  eltwise_65_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  nl_69_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  nl_69_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  _STAI_NETWORK_EVENT_NODE_START_CB(69, 1, { eltwise_65_output.data->data});
  forward_nl_integer(&nl_69_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(69, 1, { nl_69_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_79(_stai_network_context* net_ctx)
{
  conv2d_77_0_0_eltwise_79_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  conv2d_77_0_0_eltwise_79_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  tfl_pseudo_qconst12_4D_array.data = AI_PTR(net_ctx->_weights[3] + 0);
  tfl_pseudo_qconst12_4D_array.data_start = AI_PTR(net_ctx->_weights[3] + 0);
  eltwise_79_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  eltwise_79_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  _STAI_NETWORK_EVENT_NODE_START_CB(79, 2, { conv2d_77_0_0_eltwise_79_conversion_output.data->data,tfl_pseudo_qconst12_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_79_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(79, 1, { eltwise_79_output.data->data});
}
void forward_lite_nl_integer_nl_83(_stai_network_context* net_ctx)
{
  eltwise_79_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  eltwise_79_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  nl_83_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  nl_83_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  _STAI_NETWORK_EVENT_NODE_START_CB(83, 1, { eltwise_79_output.data->data});
  forward_nl_integer(&nl_83_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(83, 1, { nl_83_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_85(_stai_network_context* net_ctx)
{
  nl_83_output_array.data = AI_PTR(net_ctx->_activations[0] + 57760);
  nl_83_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 57760);
  eltwise_56_output_array.data = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_56_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_85_output_array.data = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_85_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 110368);
  _STAI_NETWORK_EVENT_NODE_START_CB(85, 2, { nl_83_output.data->data,eltwise_56_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_85_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(85, 1, { eltwise_85_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_94(_stai_network_context* net_ctx)
{
  conv2d_92_0_0_eltwise_94_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  conv2d_92_0_0_eltwise_94_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  tfl_pseudo_qconst10_4D_array.data = AI_PTR(net_ctx->_weights[4] + 0);
  tfl_pseudo_qconst10_4D_array.data_start = AI_PTR(net_ctx->_weights[4] + 0);
  eltwise_94_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  eltwise_94_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  _STAI_NETWORK_EVENT_NODE_START_CB(94, 2, { conv2d_92_0_0_eltwise_94_conversion_output.data->data,tfl_pseudo_qconst10_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_94_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(94, 1, { eltwise_94_output.data->data});
}
void forward_lite_nl_integer_nl_98(_stai_network_context* net_ctx)
{
  eltwise_94_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  eltwise_94_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  nl_98_output_array.data = AI_PTR(net_ctx->_activations[0] + 51616);
  nl_98_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 51616);
  _STAI_NETWORK_EVENT_NODE_START_CB(98, 1, { eltwise_94_output.data->data});
  forward_nl_integer(&nl_98_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(98, 1, { nl_98_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_108(_stai_network_context* net_ctx)
{
  conv2d_106_0_0_eltwise_108_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  conv2d_106_0_0_eltwise_108_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  tfl_pseudo_qconst8_4D_array.data = AI_PTR(net_ctx->_weights[5] + 0);
  tfl_pseudo_qconst8_4D_array.data_start = AI_PTR(net_ctx->_weights[5] + 0);
  eltwise_108_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  eltwise_108_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  _STAI_NETWORK_EVENT_NODE_START_CB(108, 2, { conv2d_106_0_0_eltwise_108_conversion_output.data->data,tfl_pseudo_qconst8_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_108_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(108, 1, { eltwise_108_output.data->data});
}
void forward_lite_nl_integer_nl_112(_stai_network_context* net_ctx)
{
  eltwise_108_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  eltwise_108_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  nl_112_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  nl_112_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  _STAI_NETWORK_EVENT_NODE_START_CB(112, 1, { eltwise_108_output.data->data});
  forward_nl_integer(&nl_112_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(112, 1, { nl_112_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_114(_stai_network_context* net_ctx)
{
  nl_112_output_array.data = AI_PTR(net_ctx->_activations[0] + 98848);
  nl_112_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 98848);
  eltwise_85_output_array.data = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_85_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 110368);
  eltwise_114_output_array.data = AI_PTR(net_ctx->_activations[0] + 87328);
  eltwise_114_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87328);
  _STAI_NETWORK_EVENT_NODE_START_CB(114, 2, { nl_112_output.data->data,eltwise_85_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_114_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(114, 1, { eltwise_114_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_123(_stai_network_context* net_ctx)
{
  conv2d_121_0_0_eltwise_123_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 62368);
  conv2d_121_0_0_eltwise_123_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 62368);
  tfl_pseudo_qconst6_4D_array.data = AI_PTR(net_ctx->_weights[6] + 0);
  tfl_pseudo_qconst6_4D_array.data_start = AI_PTR(net_ctx->_weights[6] + 0);
  eltwise_123_output_array.data = AI_PTR(net_ctx->_activations[0] + 16288);
  eltwise_123_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16288);
  _STAI_NETWORK_EVENT_NODE_START_CB(123, 2, { conv2d_121_0_0_eltwise_123_conversion_output.data->data,tfl_pseudo_qconst6_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_123_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(123, 1, { eltwise_123_output.data->data});
}
void forward_lite_nl_integer_nl_127(_stai_network_context* net_ctx)
{
  eltwise_123_output_array.data = AI_PTR(net_ctx->_activations[0] + 16288);
  eltwise_123_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16288);
  nl_127_output_array.data = AI_PTR(net_ctx->_activations[0] + 27808);
  nl_127_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 27808);
  _STAI_NETWORK_EVENT_NODE_START_CB(127, 1, { eltwise_123_output.data->data});
  forward_nl_integer(&nl_127_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(127, 1, { nl_127_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_137(_stai_network_context* net_ctx)
{
  conv2d_135_0_0_eltwise_137_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 62368);
  conv2d_135_0_0_eltwise_137_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 62368);
  tfl_pseudo_qconst4_4D_array.data = AI_PTR(net_ctx->_weights[7] + 0);
  tfl_pseudo_qconst4_4D_array.data_start = AI_PTR(net_ctx->_weights[7] + 0);
  eltwise_137_output_array.data = AI_PTR(net_ctx->_activations[0] + 16288);
  eltwise_137_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16288);
  _STAI_NETWORK_EVENT_NODE_START_CB(137, 2, { conv2d_135_0_0_eltwise_137_conversion_output.data->data,tfl_pseudo_qconst4_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_137_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(137, 1, { eltwise_137_output.data->data});
}
void forward_lite_nl_integer_nl_141(_stai_network_context* net_ctx)
{
  eltwise_137_output_array.data = AI_PTR(net_ctx->_activations[0] + 16288);
  eltwise_137_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16288);
  nl_141_output_array.data = AI_PTR(net_ctx->_activations[0] + 27808);
  nl_141_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 27808);
  _STAI_NETWORK_EVENT_NODE_START_CB(141, 1, { eltwise_137_output.data->data});
  forward_nl_integer(&nl_141_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(141, 1, { nl_141_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_143(_stai_network_context* net_ctx)
{
  nl_141_output_array.data = AI_PTR(net_ctx->_activations[0] + 27808);
  nl_141_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 27808);
  eltwise_114_output_array.data = AI_PTR(net_ctx->_activations[0] + 87328);
  eltwise_114_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 87328);
  eltwise_143_output_array.data = AI_PTR(net_ctx->_activations[0] + 16288);
  eltwise_143_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16288);
  _STAI_NETWORK_EVENT_NODE_START_CB(143, 2, { nl_141_output.data->data,eltwise_114_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_143_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(143, 1, { eltwise_143_output.data->data});
}
void forward_lite_nl_integer_nl_152(_stai_network_context* net_ctx)
{
  conv2d_148_output_array.data = AI_PTR(net_ctx->_activations[0] + 27808);
  conv2d_148_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 27808);
  nl_152_output_array.data = AI_PTR(net_ctx->_activations[0] + 43168);
  nl_152_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 43168);
  _STAI_NETWORK_EVENT_NODE_START_CB(152, 1, { conv2d_148_output.data->data});
  forward_nl_integer(&nl_152_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(152, 1, { nl_152_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_conv2d_158(_stai_network_context* net_ctx)
{
  nl_152_output_array.data = AI_PTR(net_ctx->_activations[0] + 43168);
  nl_152_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 43168);
  conv2d_158_weights_array.data = AI_PTR(net_ctx->_weights[25] + 0);
  conv2d_158_weights_array.data_start = AI_PTR(net_ctx->_weights[25] + 0);
  conv2d_158_bias_array.data = AI_PTR(net_ctx->_weights[26] + 0);
  conv2d_158_bias_array.data_start = AI_PTR(net_ctx->_weights[26] + 0);
  conv2d_158_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_158_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_158_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  conv2d_158_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(158, 1, { nl_152_output.data->data});
  forward_conv2d_integer_SSSA(&conv2d_158_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(158, 1, { conv2d_158_output.data->data});
}

/*****************************************************************************/


static const ai_u16 conv2d_22_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_22_t_in_0_shape_h_const_u16 = 120;
static const ai_u16 conv2d_22_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_22_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_22_t_in_0_shape_ch_const_u16 = 20;
static const ai_u16 conv2d_22_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_22_t_in_0_fmt_zero_const_s8 = 45;
static const ai_i8 conv2d_22_t_out_0_fmt_zero_const_s8 = -20;
static const ai_float conv2d_22_t_in_0_fmt_scale_const_f32 = 0.10564419627189636f;
static const ai_float conv2d_22_t_out_0_fmt_scale_const_f32 = 0.05705142021179199f;
static const ai_float conv2d_22_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001527676940895617f, 0.0027895078528672457f, 0.0017365195089951158f, 0.002615047385916114f, 0.001976851373910904f, 0.0028566247783601284f, 0.0030430254992097616f, 0.0025308418553322554f, 0.0022881734184920788f, 0.0023102322593331337f, 0.0015321312239393592f, 0.0017304223729297519f, 0.0016875932924449444f, 0.0015310818562284112f, 0.00299543677829206f, 0.00209888257086277f, 0.0019565639086067677f, 0.0023308629170060158f, 0.0021916248369961977f, 0.0024325831327587366f, 0.0035041694063693285f, 0.0019487052923068404f, 0.0017137123504653573f, 0.002100614132359624f, 0.0030674953013658524f, 0.001756359706632793f, 0.002614312805235386f, 0.0019439239986240864f, 0.0018532145768404007f, 0.0024295663461089134f, 0.002378486329689622f, 0.0018679980421438813f, 0.0023476507049053907f, 0.002299643587321043f, 0.0020801685750484467f, 0.002182340482249856f, 0.0026491989847272635f, 0.0020093414932489395f, 0.0019816013518720865f, 0.0023674936965107918f, 0.002345239045098424f, 0.0017098399112001061f, 0.0029155313968658447f, 0.0017981169512495399f, 0.0024843583814799786f, 0.0019266197923570871f, 0.0021488559432327747f, 0.002096545184031129f, 0.002496168250218034f, 0.001912670093588531f, 0.002599530154839158f, 0.002419071039184928f, 0.0019263072172179818f, 0.002492844359949231f, 0.0023440064396709204f, 0.0023032622411847115f, 0.0022806369233876467f, 0.002046456327661872f, 0.002074620919302106f, 0.0016848619561642408f, 0.0019001405453309417f, 0.0018451344221830368f, 0.0031922280322760344f, 0.0011848579160869122f, 0.00234398921020329f, 0.0016920360503718257f, 0.002262757159769535f, 0.002716240705922246f, 0.0023492807522416115f, 0.0026638933923095465f, 0.0026203403249382973f, 0.0027284384705126286f, 0.002795438515022397f, 0.0031543138902634382f, 0.003081035101786256f, 0.0016920474590733647f, 0.0019107841653749347f, 0.0022751628421247005f, 0.0015835600206628442f, 0.0021956146229058504f, 0.0017584572779014707f, 0.0017497970256954432f, 0.0019628701265901327f, 0.0029256113339215517f, 0.002990777138620615f, 0.0021055827382951975f, 0.0026833321899175644f, 0.0017237140564247966f, 0.002322562737390399f, 0.002390276873484254f, 0.0020872161258012056f, 0.0019180014496669173f, 0.002569878241047263f, 0.0029465630650520325f, 0.0023675374686717987f, 0.0021585116628557444f);
static const ai_layer_format_type conv2d_22_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 pad_4_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(45);
static const ai_i16 pad_4_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_4_t_in_0_shape_h_const_u32 = 120;

static const ai_u16 conv2d_5_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_5_t_in_0_shape_h_const_u16 = 124;
static const ai_u16 conv2d_5_t_in_0_shape_ch_const_u16 = 20;
static const ai_u16 conv2d_5_t_out_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_5_t_weight_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_5_t_weight_0_shape_h_const_u16 = 5;
static const ai_u16 conv2d_5_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_5_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_5_t_in_0_fmt_zero_const_s8 = 45;
static const ai_i8 conv2d_5_t_out_0_fmt_zero_const_s8 = 41;
static const ai_float conv2d_5_t_in_0_fmt_scale_const_f32 = 0.10564419627189636f;
static const ai_float conv2d_5_t_out_0_fmt_scale_const_f32 = 0.041041214019060135f;
static const ai_float conv2d_5_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0018004437442868948f, 0.0014112226199358702f, 0.0016210871981456876f, 0.0012572294799610972f, 0.0010787826031446457f, 0.0014088416937738657f, 0.0012148019159212708f, 0.0010846738005056977f, 0.0014922614209353924f, 0.0012908143689855933f, 0.0012135099386796355f, 0.001525200204923749f, 0.0013079188065603375f, 0.0012307483702898026f, 0.0011242699110880494f, 0.0017273218836635351f, 0.0012673899764195085f, 0.0013900677440688014f, 0.0015785390278324485f, 0.0020042494870722294f, 0.001017778879031539f, 0.001311834785155952f, 0.0015833564102649689f, 0.0011165639152750373f, 0.0013523520901799202f, 0.0017765286611393094f, 0.0012542945332825184f, 0.0015255706384778023f, 0.0012530067469924688f, 0.0013948888517916203f, 0.001700958120636642f, 0.00141875387635082f, 0.001226833206601441f, 0.0015072901733219624f, 0.0011121390853077173f, 0.001184859313070774f, 0.001436328748241067f, 0.0012815056834369898f, 0.0012385874288156629f, 0.0017690603854134679f, 0.0014915043720975518f, 0.0015760347014293075f, 0.001096232794225216f, 0.001288924366235733f, 0.0012461695587262511f, 0.0016412845579907298f, 0.0012928012292832136f, 0.001345304073765874f, 0.0014030355960130692f, 0.001286167185753584f, 0.001603335258550942f, 0.0016229964094236493f, 0.001336575485765934f, 0.0016514837043359876f, 0.0013538715429604053f, 0.0013481731293722987f, 0.0015102883335202932f, 0.0013588431756943464f, 0.0012461221776902676f, 0.0012206463143229485f, 0.0012167297536507249f, 0.001579358708113432f, 0.0016196563374251127f, 0.0013369631487876177f, 0.0009307998698204756f, 0.001303300028666854f, 0.0012090073432773352f, 0.001354034524410963f, 0.0014341010246425867f, 0.0014163758605718613f, 0.0014012247556820512f, 0.0013988419668748975f, 0.0015268884599208832f, 0.001108732889406383f, 0.0013904788065701723f, 0.0012581582413986325f, 0.0013533455785363913f, 0.0011332046706229448f, 0.0014799335040152073f, 0.001168233691714704f, 0.0013415795983746648f, 0.0015783662674948573f, 0.001327096950262785f, 0.0013346591731533408f, 0.0010572177125141025f, 0.0017980183474719524f, 0.0011915976647287607f, 0.0014153546653687954f, 0.0015182849019765854f, 0.001499462639912963f, 0.001280264463275671f, 0.0016001195181161165f, 0.0012112209806218743f, 0.0014275218127295375f, 0.0013358534779399633f, 0.000970762106589973f);
static const ai_layer_format_type conv2d_5_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_5_t_out_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_5_t_out_0_shape_h_const_u16 = 120;


static const ai_i8 pad_15_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_15_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_15_t_in_0_shape_h_const_u32 = 120;

static const ai_u16 conv2d_16_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_16_t_in_0_shape_h_const_u16 = 124;
static const ai_u16 conv2d_16_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_16_t_out_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_16_t_weight_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_16_t_weight_0_shape_h_const_u16 = 5;
static const ai_u16 conv2d_16_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_16_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_16_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_16_t_out_0_fmt_zero_const_s8 = 31;
static const ai_float conv2d_16_t_in_0_fmt_scale_const_f32 = 0.013773981481790543f;
static const ai_float conv2d_16_t_out_0_fmt_scale_const_f32 = 0.02312243916094303f;
static const ai_float conv2d_16_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017389509594067931f, 0.0016696358798071742f, 0.0015883732121437788f, 0.001508953864686191f, 0.0015684613026678562f, 0.0017554224468767643f, 0.0016641413094475865f, 0.0015206205425783992f, 0.0017824367387220263f, 0.0016325396718457341f, 0.0016752470983192325f, 0.0017702379263937473f, 0.0014175475807860494f, 0.0014116693055257201f, 0.0016515410970896482f, 0.0020857159979641438f, 0.0015106756472960114f, 0.0015241713263094425f, 0.0016203706618398428f, 0.0018514270195737481f, 0.001476367935538292f, 0.0015272225718945265f, 0.002068336121737957f, 0.0015976378927007318f, 0.0014297496527433395f, 0.0015271863667294383f, 0.0017098421230912209f, 0.0015777201624587178f, 0.00199591345153749f, 0.0014550359919667244f, 0.0014652274549007416f, 0.0016141918022185564f, 0.0015809645410627127f, 0.0015290066367015243f, 0.001695537124760449f, 0.001587200677022338f, 0.001757485093548894f, 0.0015181144699454308f, 0.0017229478107765317f, 0.0013313752133399248f, 0.0018580557079985738f, 0.0016338105779141188f, 0.0015602834755554795f, 0.0017266427166759968f, 0.001517362310551107f, 0.0015168485697358847f, 0.0015729814767837524f, 0.0015003118896856904f, 0.0014897469663992524f, 0.0015106587670743465f, 0.0014540775446221232f, 0.0022381271701306105f, 0.0015573985874652863f, 0.001520079793408513f, 0.0017217654967680573f, 0.0014093673089519143f, 0.0018166513182222843f, 0.00189167947974056f, 0.0015653680311515927f, 0.0015519189182668924f, 0.001486414228565991f, 0.0012274769833311439f, 0.001825683400966227f, 0.0017876686761155725f, 0.0012351800687611103f, 0.0015112538821995258f, 0.0013661278644576669f, 0.002479500835761428f, 0.0025288627948611975f, 0.001816566800698638f, 0.002003762871026993f, 0.001928474404849112f, 0.0015807644231244922f, 0.0019174482440575957f, 0.001780800987035036f, 0.0018335076747462153f, 0.0015283995307981968f, 0.001589313498698175f, 0.0014857319183647633f, 0.0014158787671476603f, 0.0015954027185216546f, 0.0016887927195057273f, 0.0017758460016921163f, 0.0023184819146990776f, 0.002169344574213028f, 0.0015999219613149762f, 0.0016060253838077188f, 0.001473244046792388f, 0.0017490285681560636f, 0.0018678937340155244f, 0.0018395914230495691f, 0.0014297673478722572f, 0.0015536764403805137f, 0.0018077569548040628f, 0.0013918024487793446f, 0.0014507779851555824f);
static const ai_layer_format_type conv2d_16_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_16_t_out_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_16_t_out_0_shape_h_const_u16 = 120;



static const ai_i8 pad_32_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-20);
static const ai_i16 pad_32_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_32_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_32_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12288;
static const ai_float pad_32_0_conversion_t_in_0_fmt_scale_const_f32 = 0.05705142021179199f;
static const ai_i8 pad_32_0_conversion_t_in_0_fmt_zero_const_s8 = -20;

static const ai_u32 conv2d_34_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_34_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_34_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_34_t_in_0_shape_h_const_u32 = 128;
static const ai_u32 conv2d_34_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_34_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_34_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_34_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_34_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_34_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_34_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_34_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_34_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_34_l_dilation_H_const_u16 = 2;
static const ai_size conv2d_34_v_n_groups_const_size = 1;

static const ai_u32 conv2d_34_0_0_eltwise_36_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_34_0_0_eltwise_36_conversion_t_out_0_fmt_scale_const_f32 = 0.05669168010354042f;
static const ai_i8 conv2d_34_0_0_eltwise_36_conversion_t_out_0_fmt_zero_const_s8 = 79;



static const ai_i8 pad_46_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_46_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_46_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_46_0_0_conv2d_48_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12288;
static const ai_float pad_46_0_0_conv2d_48_conversion_t_in_0_fmt_scale_const_f32 = 0.010546959936618805f;
static const ai_i8 pad_46_0_0_conv2d_48_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 conv2d_48_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_48_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_48_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_48_t_in_0_shape_h_const_u32 = 128;
static const ai_u32 conv2d_48_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_48_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_48_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_48_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_48_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_48_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_48_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_48_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_48_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_48_l_dilation_H_const_u16 = 2;
static const ai_size conv2d_48_v_n_groups_const_size = 1;

static const ai_u32 conv2d_48_0_0_eltwise_50_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_48_0_0_eltwise_50_conversion_t_out_0_fmt_scale_const_f32 = 0.029165739193558693f;
static const ai_i8 conv2d_48_0_0_eltwise_50_conversion_t_out_0_fmt_zero_const_s8 = 8;




static const ai_i8 pad_61_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-20);
static const ai_i16 pad_61_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_61_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_61_0_0_conv2d_63_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 13056;
static const ai_float pad_61_0_0_conv2d_63_conversion_t_in_0_fmt_scale_const_f32 = 0.05705142021179199f;
static const ai_i8 pad_61_0_0_conv2d_63_conversion_t_in_0_fmt_zero_const_s8 = -20;

static const ai_u32 conv2d_63_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_63_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_63_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_63_t_in_0_shape_h_const_u32 = 136;
static const ai_u32 conv2d_63_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_63_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_63_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_63_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_63_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_63_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_63_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_63_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_63_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_63_l_dilation_H_const_u16 = 4;
static const ai_size conv2d_63_v_n_groups_const_size = 1;

static const ai_u32 conv2d_63_0_0_eltwise_65_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_63_0_0_eltwise_65_conversion_t_out_0_fmt_scale_const_f32 = 0.058796655386686325f;
static const ai_i8 conv2d_63_0_0_eltwise_65_conversion_t_out_0_fmt_zero_const_s8 = 29;



static const ai_i8 pad_75_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_75_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_75_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_75_0_0_conv2d_77_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 13056;
static const ai_float pad_75_0_0_conv2d_77_conversion_t_in_0_fmt_scale_const_f32 = 0.022744812071323395f;
static const ai_i8 pad_75_0_0_conv2d_77_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 conv2d_77_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_77_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_77_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_77_t_in_0_shape_h_const_u32 = 136;
static const ai_u32 conv2d_77_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_77_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_77_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_77_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_77_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_77_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_77_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_77_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_77_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_77_l_dilation_H_const_u16 = 4;
static const ai_size conv2d_77_v_n_groups_const_size = 1;

static const ai_u32 conv2d_77_0_0_eltwise_79_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_77_0_0_eltwise_79_conversion_t_out_0_fmt_scale_const_f32 = 0.030611073598265648f;
static const ai_i8 conv2d_77_0_0_eltwise_79_conversion_t_out_0_fmt_zero_const_s8 = 19;




static const ai_i8 pad_90_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-20);
static const ai_i16 pad_90_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_90_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_90_0_0_conv2d_92_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 14592;
static const ai_float pad_90_0_0_conv2d_92_conversion_t_in_0_fmt_scale_const_f32 = 0.05705142021179199f;
static const ai_i8 pad_90_0_0_conv2d_92_conversion_t_in_0_fmt_zero_const_s8 = -20;

static const ai_u32 conv2d_92_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_92_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_92_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_92_t_in_0_shape_h_const_u32 = 152;
static const ai_u32 conv2d_92_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_92_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_92_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_92_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_92_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_92_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_92_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_92_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_92_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_92_l_dilation_H_const_u16 = 8;
static const ai_size conv2d_92_v_n_groups_const_size = 1;

static const ai_u32 conv2d_92_0_0_eltwise_94_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_92_0_0_eltwise_94_conversion_t_out_0_fmt_scale_const_f32 = 0.07281295955181122f;
static const ai_i8 conv2d_92_0_0_eltwise_94_conversion_t_out_0_fmt_zero_const_s8 = 43;



static const ai_i8 pad_104_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_104_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_104_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_104_0_0_conv2d_106_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 14592;
static const ai_float pad_104_0_0_conv2d_106_conversion_t_in_0_fmt_scale_const_f32 = 0.024060485884547234f;
static const ai_i8 pad_104_0_0_conv2d_106_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 conv2d_106_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_106_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_106_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_106_t_in_0_shape_h_const_u32 = 152;
static const ai_u32 conv2d_106_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_106_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_106_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_106_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_106_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_106_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_106_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_106_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_106_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_106_l_dilation_H_const_u16 = 8;
static const ai_size conv2d_106_v_n_groups_const_size = 1;

static const ai_u32 conv2d_106_0_0_eltwise_108_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_106_0_0_eltwise_108_conversion_t_out_0_fmt_scale_const_f32 = 0.029464116320014f;
static const ai_i8 conv2d_106_0_0_eltwise_108_conversion_t_out_0_fmt_zero_const_s8 = 2;




static const ai_i8 pad_119_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-29);
static const ai_i16 pad_119_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_119_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_119_0_0_conv2d_121_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 17664;
static const ai_float pad_119_0_0_conv2d_121_conversion_t_in_0_fmt_scale_const_f32 = 0.06278161704540253f;
static const ai_i8 pad_119_0_0_conv2d_121_conversion_t_in_0_fmt_zero_const_s8 = -29;

static const ai_u32 conv2d_121_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_121_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_121_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_121_t_in_0_shape_h_const_u32 = 184;
static const ai_u32 conv2d_121_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_121_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_121_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_121_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_121_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_121_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_121_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_121_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_121_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_121_l_dilation_H_const_u16 = 16;
static const ai_size conv2d_121_v_n_groups_const_size = 1;

static const ai_u32 conv2d_121_0_0_eltwise_123_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_121_0_0_eltwise_123_conversion_t_out_0_fmt_scale_const_f32 = 0.10135211050510406f;
static const ai_i8 conv2d_121_0_0_eltwise_123_conversion_t_out_0_fmt_zero_const_s8 = 16;



static const ai_i8 pad_133_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_133_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_133_t_in_0_shape_h_const_u32 = 120;

static const ai_u32 pad_133_0_0_conv2d_135_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 17664;
static const ai_float pad_133_0_0_conv2d_135_conversion_t_in_0_fmt_scale_const_f32 = 0.016977179795503616f;
static const ai_i8 pad_133_0_0_conv2d_135_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 conv2d_135_t_in_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_135_t_out_0_shape_ch_const_u32 = 96;
static const ai_u32 conv2d_135_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_135_t_in_0_shape_h_const_u32 = 184;
static const ai_u32 conv2d_135_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_135_t_out_0_shape_h_const_u32 = 120;
static const ai_u32 conv2d_135_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_135_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_135_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_135_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_135_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_135_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_135_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_135_l_dilation_H_const_u16 = 16;
static const ai_size conv2d_135_v_n_groups_const_size = 1;

static const ai_u32 conv2d_135_0_0_eltwise_137_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 11520;
static const ai_float conv2d_135_0_0_eltwise_137_conversion_t_out_0_fmt_scale_const_f32 = 0.01921113394200802f;
static const ai_i8 conv2d_135_0_0_eltwise_137_conversion_t_out_0_fmt_zero_const_s8 = 8;




static const ai_u16 conv2d_148_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_148_t_in_0_shape_h_const_u16 = 120;
static const ai_u16 conv2d_148_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_148_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_148_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_148_t_out_0_shape_ch_const_u16 = 128;
static const ai_i8 conv2d_148_t_in_0_fmt_zero_const_s8 = -41;
static const ai_i8 conv2d_148_t_out_0_fmt_zero_const_s8 = 12;
static const ai_float conv2d_148_t_in_0_fmt_scale_const_f32 = 0.07079243659973145f;
static const ai_float conv2d_148_t_out_0_fmt_scale_const_f32 = 0.08966033160686493f;
static const ai_float conv2d_148_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0021659352350980043f, 0.0019248401513323188f, 0.0027148418594151735f, 0.0021256371401250362f, 0.002352086128666997f, 0.002528394339606166f, 0.003511179704219103f, 0.0031100320629775524f, 0.002586992923170328f, 0.002303843153640628f, 0.0023591110948473215f, 0.0020529807079583406f, 0.0032524324487894773f, 0.00265489611774683f, 0.0024077745620161295f, 0.0029844066593796015f, 0.003441672306507826f, 0.002329643350094557f, 0.002415456110611558f, 0.003190734889358282f, 0.0025146708358079195f, 0.002500481903553009f, 0.0030888577457517385f, 0.0023394119925796986f, 0.0020798323675990105f, 0.00231350539252162f, 0.00263007590547204f, 0.0022589750587940216f, 0.002624915447086096f, 0.0022977793123573065f, 0.002801262540742755f, 0.0035144290886819363f, 0.0025222531985491514f, 0.0024261546786874533f, 0.002641452243551612f, 0.003056131536141038f, 0.002483244286850095f, 0.0031630583107471466f, 0.0031925560906529427f, 0.002651600167155266f, 0.002763134427368641f, 0.002949229208752513f, 0.002379296813160181f, 0.003087678924202919f, 0.003146864240989089f, 0.002885130001232028f, 0.002581611042842269f, 0.002309482078999281f, 0.00238056224770844f, 0.002443654229864478f, 0.002405791077762842f, 0.002453182591125369f, 0.0020193257369101048f, 0.0022949923295527697f, 0.0027214395813643932f, 0.0023769764229655266f, 0.0025550639256834984f, 0.0027278068009763956f, 0.002233279636129737f, 0.0023954773787409067f, 0.003006406594067812f, 0.002359798876568675f, 0.0019735645037144423f, 0.002243252005428076f, 0.0032328860834240913f, 0.00300804921425879f, 0.0025373948737978935f, 0.0024333924520760775f, 0.0022910237312316895f, 0.0036148771177977324f, 0.0024528594221919775f, 0.002277710707858205f, 0.0025483001954853535f, 0.002528092823922634f, 0.0027579343877732754f, 0.0030396580696105957f, 0.0024774090852588415f, 0.0035683244932442904f, 0.0022557764314115047f, 0.0033411062322556973f, 0.0026739435270428658f, 0.0019279962871223688f, 0.004076970275491476f, 0.0022430114913731813f, 0.0026338358875364065f, 0.0024384423159062862f, 0.0026132736820727587f, 0.0026822250802069902f, 0.0030554502736777067f, 0.0024432451464235783f, 0.002968085464090109f, 0.002436433220282197f, 0.0026390061248093843f, 0.002894932171329856f, 0.0022244222927838564f, 0.0028044304344803095f, 0.002120570046827197f, 0.002506555989384651f, 0.0027298331260681152f, 0.0024150474928319454f, 0.0032006544061005116f, 0.001974400831386447f, 0.0023547159507870674f, 0.0033317659981548786f, 0.0026417765766382217f, 0.003616574453189969f, 0.003212094074115157f, 0.002609512535855174f, 0.0028207609429955482f, 0.002497483743354678f, 0.00224015093408525f, 0.0036790750455111265f, 0.002699136734008789f, 0.0026228250935673714f, 0.0024396951775997877f, 0.0024482232984155416f, 0.0025633613113313913f, 0.0034177240449935198f, 0.002488186815753579f, 0.0031159466598182917f, 0.0021620497573167086f, 0.0035598287358880043f, 0.0032316274009644985f, 0.0025182708632200956f, 0.0024191015399992466f, 0.003982840571552515f, 0.0030266311950981617f, 0.0035443983506411314f);
static const ai_layer_format_type conv2d_148_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN conv2d_22 */
  {
      const ai_i8* conv2d_22_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* conv2d_22_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[8] + 0);
    const ai_i32* conv2d_22_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[9] + 0);
    ai_i8* conv2d_22_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 98848);
    ai_i16* conv2d_22_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 15248);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, {(stai_ptr) conv2d_22_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_22_t_in_0_ptr_const_s8, conv2d_22_t_in_0_shape_w_const_u16, conv2d_22_t_in_0_shape_h_const_u16, conv2d_22_l_stride_1_const_u16, conv2d_22_l_stride_0_const_u16, conv2d_22_t_in_0_shape_ch_const_u16, conv2d_22_t_weight_0_ptr_const_s8, conv2d_22_t_out_0_shape_ch_const_u16, conv2d_22_t_weight_1_ptr_const_s32, conv2d_22_t_in_0_fmt_zero_const_s8, conv2d_22_t_out_0_fmt_zero_const_s8, conv2d_22_t_in_0_fmt_scale_const_f32, conv2d_22_t_out_0_fmt_scale_const_f32, conv2d_22_t_weight_0_fmt_scale_const_f32, conv2d_22_l_out_ch_format_const_layer_format_type, conv2d_22_t_out_0_ptr_s8, 1, 1040, conv2d_22_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, {(stai_ptr) conv2d_22_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_22 */
  /* LITE_KERNEL_SECTION BEGIN pad_4 */
  {
      const ai_ptr pad_4_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_inputs[0] + 0);
    ai_ptr pad_4_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 110368);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 1, {(stai_ptr) pad_4_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_4_t_in_0_ptr_const_ptr, pad_4_t_out_0_ptr_ptr, (ai_handle)(pad_4_v_pad_constant_value_const_s8), pad_4_t_in_0_fmt_bitsize_const_s16, pad_4_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(20), (ai_i32)(80), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) pad_4_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_4 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_5 */
  {
      const ai_i8* conv2d_5_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 110368);
    const ai_i8* conv2d_5_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[10] + 0);
    const ai_i32* conv2d_5_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[11] + 0);
    ai_i8* conv2d_5_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 87328);
    ai_i16* conv2d_5_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8384);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, {(stai_ptr) conv2d_5_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(conv2d_5_t_in_0_ptr_const_s8, conv2d_5_t_in_0_shape_w_const_u16, conv2d_5_t_in_0_shape_h_const_u16, conv2d_5_t_in_0_shape_ch_const_u16, conv2d_5_t_weight_0_ptr_const_s8, conv2d_5_t_out_0_shape_ch_const_u16, conv2d_5_t_weight_0_shape_w_const_u16, conv2d_5_t_weight_0_shape_h_const_u16, conv2d_5_l_stride_1_const_u16, conv2d_5_l_stride_0_const_u16, conv2d_5_t_weight_1_ptr_const_s32, conv2d_5_t_in_0_fmt_zero_const_s8, conv2d_5_t_out_0_fmt_zero_const_s8, conv2d_5_t_in_0_fmt_scale_const_f32, conv2d_5_t_out_0_fmt_scale_const_f32, conv2d_5_t_weight_0_fmt_scale_const_f32, conv2d_5_l_out_ch_format_const_layer_format_type, conv2d_5_t_out_0_ptr_s8, conv2d_5_t_out_0_shape_w_const_u16, conv2d_5_t_out_0_shape_h_const_u16, 1, 1, 6864, conv2d_5_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) conv2d_5_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_5 */
  /* LITE_KERNEL_SECTION BEGIN nl_9 */
  {
    
  forward_lite_nl_integer_nl_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_9 */
  /* LITE_KERNEL_SECTION BEGIN pad_15 */
  {
      const ai_ptr pad_15_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 87328);
    ai_ptr pad_15_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 86944);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) pad_15_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_15_t_in_0_ptr_const_ptr, pad_15_t_out_0_ptr_ptr, (ai_handle)(pad_15_v_pad_constant_value_const_s8), pad_15_t_in_0_fmt_bitsize_const_s16, pad_15_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(384), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) pad_15_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_15 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_16 */
  {
      const ai_i8* conv2d_16_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 86944);
    const ai_i8* conv2d_16_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[12] + 0);
    const ai_i32* conv2d_16_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[13] + 0);
    ai_i8* conv2d_16_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 86848);
    ai_i16* conv2d_16_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) conv2d_16_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(conv2d_16_t_in_0_ptr_const_s8, conv2d_16_t_in_0_shape_w_const_u16, conv2d_16_t_in_0_shape_h_const_u16, conv2d_16_t_in_0_shape_ch_const_u16, conv2d_16_t_weight_0_ptr_const_s8, conv2d_16_t_out_0_shape_ch_const_u16, conv2d_16_t_weight_0_shape_w_const_u16, conv2d_16_t_weight_0_shape_h_const_u16, conv2d_16_l_stride_1_const_u16, conv2d_16_l_stride_0_const_u16, conv2d_16_t_weight_1_ptr_const_s32, conv2d_16_t_in_0_fmt_zero_const_s8, conv2d_16_t_out_0_fmt_zero_const_s8, conv2d_16_t_in_0_fmt_scale_const_f32, conv2d_16_t_out_0_fmt_scale_const_f32, conv2d_16_t_weight_0_fmt_scale_const_f32, conv2d_16_l_out_ch_format_const_layer_format_type, conv2d_16_t_out_0_ptr_s8, conv2d_16_t_out_0_shape_w_const_u16, conv2d_16_t_out_0_shape_h_const_u16, 1, 1, 8384, conv2d_16_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) conv2d_16_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_16 */
  /* LITE_KERNEL_SECTION BEGIN nl_20 */
  {
    
  forward_lite_nl_integer_nl_20(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_20 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_27 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_27(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_27 */
  /* LITE_KERNEL_SECTION BEGIN pad_32 */
  {
      const ai_ptr pad_32_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 110368);
    ai_ptr pad_32_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 98080);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 1, {(stai_ptr) pad_32_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_32_t_in_0_ptr_const_ptr, pad_32_t_out_0_ptr_ptr, (ai_handle)(pad_32_v_pad_constant_value_const_s8), pad_32_t_in_0_fmt_bitsize_const_s16, pad_32_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(768), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, {(stai_ptr) pad_32_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_32 */
  /* LITE_KERNEL_SECTION BEGIN pad_32_0_conversion */
  {
      const ai_i8* pad_32_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 98080);
    ai_float* pad_32_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 61216);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 1, {(stai_ptr) pad_32_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_32_0_conversion_t_in_0_ptr_const_s8, pad_32_0_conversion_t_out_0_ptr_f32, pad_32_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_32_0_conversion_t_in_0_fmt_scale_const_f32, pad_32_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, {(stai_ptr) pad_32_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_32_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_34 */
  {
      const ai_float* conv2d_34_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 61216);
    ai_float* conv2d_34_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 60832);
    const ai_u8* conv2d_34_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[14] + 0);
    const ai_u8* conv2d_34_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, {(stai_ptr) conv2d_34_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_34_t_in_0_ptr_const_f32, conv2d_34_t_out_0_ptr_f32, conv2d_34_t_weight_0_ptr_const_u8, conv2d_34_t_weight_1_ptr_const_u8, conv2d_34_t_in_0_shape_ch_const_u32, conv2d_34_t_out_0_shape_ch_const_u32, conv2d_34_t_in_0_shape_w_const_u32, conv2d_34_t_in_0_shape_h_const_u32, conv2d_34_t_out_0_shape_w_const_u32, conv2d_34_t_out_0_shape_h_const_u32, conv2d_34_t_weight_0_shape_w_const_u32, conv2d_34_t_weight_0_shape_h_const_u32, conv2d_34_l_pad_W_0_const_s32, conv2d_34_l_pad_H_0_const_s32, conv2d_34_l_stride_1_const_u16, conv2d_34_l_stride_0_const_u16, 9, 1, conv2d_34_l_dilation_W_const_u16, conv2d_34_l_dilation_H_const_u16, conv2d_34_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, {(stai_ptr) conv2d_34_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_34 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_34_0_0_eltwise_36_conversion */
  {
      const ai_float* conv2d_34_0_0_eltwise_36_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 60832);
    ai_i8* conv2d_34_0_0_eltwise_36_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 60832);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, {(stai_ptr) conv2d_34_0_0_eltwise_36_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_34_0_0_eltwise_36_conversion_t_in_0_ptr_const_f32, conv2d_34_0_0_eltwise_36_conversion_t_out_0_ptr_s8, conv2d_34_0_0_eltwise_36_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_34_0_0_eltwise_36_conversion_t_out_0_fmt_scale_const_f32, conv2d_34_0_0_eltwise_36_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, {(stai_ptr) conv2d_34_0_0_eltwise_36_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_34_0_0_eltwise_36_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_36 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_36(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_36 */
  /* LITE_KERNEL_SECTION BEGIN nl_40 */
  {
    
  forward_lite_nl_integer_nl_40(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_40 */
  /* LITE_KERNEL_SECTION BEGIN pad_46 */
  {
      const ai_ptr pad_46_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 60832);
    ai_ptr pad_46_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 98080);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, {(stai_ptr) pad_46_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_46_t_in_0_ptr_const_ptr, pad_46_t_out_0_ptr_ptr, (ai_handle)(pad_46_v_pad_constant_value_const_s8), pad_46_t_in_0_fmt_bitsize_const_s16, pad_46_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(768), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, {(stai_ptr) pad_46_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_46 */
  /* LITE_KERNEL_SECTION BEGIN pad_46_0_0_conv2d_48_conversion */
  {
      const ai_i8* pad_46_0_0_conv2d_48_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 98080);
    ai_float* pad_46_0_0_conv2d_48_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 61216);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, {(stai_ptr) pad_46_0_0_conv2d_48_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_46_0_0_conv2d_48_conversion_t_in_0_ptr_const_s8, pad_46_0_0_conv2d_48_conversion_t_out_0_ptr_f32, pad_46_0_0_conv2d_48_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_46_0_0_conv2d_48_conversion_t_in_0_fmt_scale_const_f32, pad_46_0_0_conv2d_48_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, {(stai_ptr) pad_46_0_0_conv2d_48_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_46_0_0_conv2d_48_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_48 */
  {
      const ai_float* conv2d_48_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 61216);
    ai_float* conv2d_48_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 60832);
    const ai_u8* conv2d_48_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[16] + 0);
    const ai_u8* conv2d_48_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 1, {(stai_ptr) conv2d_48_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_48_t_in_0_ptr_const_f32, conv2d_48_t_out_0_ptr_f32, conv2d_48_t_weight_0_ptr_const_u8, conv2d_48_t_weight_1_ptr_const_u8, conv2d_48_t_in_0_shape_ch_const_u32, conv2d_48_t_out_0_shape_ch_const_u32, conv2d_48_t_in_0_shape_w_const_u32, conv2d_48_t_in_0_shape_h_const_u32, conv2d_48_t_out_0_shape_w_const_u32, conv2d_48_t_out_0_shape_h_const_u32, conv2d_48_t_weight_0_shape_w_const_u32, conv2d_48_t_weight_0_shape_h_const_u32, conv2d_48_l_pad_W_0_const_s32, conv2d_48_l_pad_H_0_const_s32, conv2d_48_l_stride_1_const_u16, conv2d_48_l_stride_0_const_u16, 9, 1, conv2d_48_l_dilation_W_const_u16, conv2d_48_l_dilation_H_const_u16, conv2d_48_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, {(stai_ptr) conv2d_48_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_48 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_48_0_0_eltwise_50_conversion */
  {
      const ai_float* conv2d_48_0_0_eltwise_50_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 60832);
    ai_i8* conv2d_48_0_0_eltwise_50_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 60832);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 1, {(stai_ptr) conv2d_48_0_0_eltwise_50_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_48_0_0_eltwise_50_conversion_t_in_0_ptr_const_f32, conv2d_48_0_0_eltwise_50_conversion_t_out_0_ptr_s8, conv2d_48_0_0_eltwise_50_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_48_0_0_eltwise_50_conversion_t_out_0_fmt_scale_const_f32, conv2d_48_0_0_eltwise_50_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, {(stai_ptr) conv2d_48_0_0_eltwise_50_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_48_0_0_eltwise_50_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_50 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_50(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_50 */
  /* LITE_KERNEL_SECTION BEGIN nl_54 */
  {
    
  forward_lite_nl_integer_nl_54(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_54 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_56 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_56(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_56 */
  /* LITE_KERNEL_SECTION BEGIN pad_61 */
  {
      const ai_ptr pad_61_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 110368);
    ai_ptr pad_61_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 97312);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(61, 1, {(stai_ptr) pad_61_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_61_t_in_0_ptr_const_ptr, pad_61_t_out_0_ptr_ptr, (ai_handle)(pad_61_v_pad_constant_value_const_s8), pad_61_t_in_0_fmt_bitsize_const_s16, pad_61_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(61, 1, {(stai_ptr) pad_61_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_61 */
  /* LITE_KERNEL_SECTION BEGIN pad_61_0_0_conv2d_63_conversion */
  {
      const ai_i8* pad_61_0_0_conv2d_63_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 97312);
    ai_float* pad_61_0_0_conv2d_63_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 58144);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(61, 1, {(stai_ptr) pad_61_0_0_conv2d_63_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_61_0_0_conv2d_63_conversion_t_in_0_ptr_const_s8, pad_61_0_0_conv2d_63_conversion_t_out_0_ptr_f32, pad_61_0_0_conv2d_63_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_61_0_0_conv2d_63_conversion_t_in_0_fmt_scale_const_f32, pad_61_0_0_conv2d_63_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(61, 1, {(stai_ptr) pad_61_0_0_conv2d_63_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_61_0_0_conv2d_63_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_63 */
  {
      const ai_float* conv2d_63_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 58144);
    ai_float* conv2d_63_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 57760);
    const ai_u8* conv2d_63_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[17] + 0);
    const ai_u8* conv2d_63_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(64, 1, {(stai_ptr) conv2d_63_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_63_t_in_0_ptr_const_f32, conv2d_63_t_out_0_ptr_f32, conv2d_63_t_weight_0_ptr_const_u8, conv2d_63_t_weight_1_ptr_const_u8, conv2d_63_t_in_0_shape_ch_const_u32, conv2d_63_t_out_0_shape_ch_const_u32, conv2d_63_t_in_0_shape_w_const_u32, conv2d_63_t_in_0_shape_h_const_u32, conv2d_63_t_out_0_shape_w_const_u32, conv2d_63_t_out_0_shape_h_const_u32, conv2d_63_t_weight_0_shape_w_const_u32, conv2d_63_t_weight_0_shape_h_const_u32, conv2d_63_l_pad_W_0_const_s32, conv2d_63_l_pad_H_0_const_s32, conv2d_63_l_stride_1_const_u16, conv2d_63_l_stride_0_const_u16, 17, 1, conv2d_63_l_dilation_W_const_u16, conv2d_63_l_dilation_H_const_u16, conv2d_63_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(64, 1, {(stai_ptr) conv2d_63_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_63 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_63_0_0_eltwise_65_conversion */
  {
      const ai_float* conv2d_63_0_0_eltwise_65_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 57760);
    ai_i8* conv2d_63_0_0_eltwise_65_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 57760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(64, 1, {(stai_ptr) conv2d_63_0_0_eltwise_65_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_63_0_0_eltwise_65_conversion_t_in_0_ptr_const_f32, conv2d_63_0_0_eltwise_65_conversion_t_out_0_ptr_s8, conv2d_63_0_0_eltwise_65_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_63_0_0_eltwise_65_conversion_t_out_0_fmt_scale_const_f32, conv2d_63_0_0_eltwise_65_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(64, 1, {(stai_ptr) conv2d_63_0_0_eltwise_65_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_63_0_0_eltwise_65_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_65 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_65(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_65 */
  /* LITE_KERNEL_SECTION BEGIN nl_69 */
  {
    
  forward_lite_nl_integer_nl_69(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_69 */
  /* LITE_KERNEL_SECTION BEGIN pad_75 */
  {
      const ai_ptr pad_75_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 57760);
    ai_ptr pad_75_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 97312);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(75, 1, {(stai_ptr) pad_75_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_75_t_in_0_ptr_const_ptr, pad_75_t_out_0_ptr_ptr, (ai_handle)(pad_75_v_pad_constant_value_const_s8), pad_75_t_in_0_fmt_bitsize_const_s16, pad_75_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(75, 1, {(stai_ptr) pad_75_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_75 */
  /* LITE_KERNEL_SECTION BEGIN pad_75_0_0_conv2d_77_conversion */
  {
      const ai_i8* pad_75_0_0_conv2d_77_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 97312);
    ai_float* pad_75_0_0_conv2d_77_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 58144);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(75, 1, {(stai_ptr) pad_75_0_0_conv2d_77_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_75_0_0_conv2d_77_conversion_t_in_0_ptr_const_s8, pad_75_0_0_conv2d_77_conversion_t_out_0_ptr_f32, pad_75_0_0_conv2d_77_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_75_0_0_conv2d_77_conversion_t_in_0_fmt_scale_const_f32, pad_75_0_0_conv2d_77_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(75, 1, {(stai_ptr) pad_75_0_0_conv2d_77_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_75_0_0_conv2d_77_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_77 */
  {
      const ai_float* conv2d_77_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 58144);
    ai_float* conv2d_77_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 57760);
    const ai_u8* conv2d_77_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[18] + 0);
    const ai_u8* conv2d_77_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(78, 1, {(stai_ptr) conv2d_77_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_77_t_in_0_ptr_const_f32, conv2d_77_t_out_0_ptr_f32, conv2d_77_t_weight_0_ptr_const_u8, conv2d_77_t_weight_1_ptr_const_u8, conv2d_77_t_in_0_shape_ch_const_u32, conv2d_77_t_out_0_shape_ch_const_u32, conv2d_77_t_in_0_shape_w_const_u32, conv2d_77_t_in_0_shape_h_const_u32, conv2d_77_t_out_0_shape_w_const_u32, conv2d_77_t_out_0_shape_h_const_u32, conv2d_77_t_weight_0_shape_w_const_u32, conv2d_77_t_weight_0_shape_h_const_u32, conv2d_77_l_pad_W_0_const_s32, conv2d_77_l_pad_H_0_const_s32, conv2d_77_l_stride_1_const_u16, conv2d_77_l_stride_0_const_u16, 17, 1, conv2d_77_l_dilation_W_const_u16, conv2d_77_l_dilation_H_const_u16, conv2d_77_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(78, 1, {(stai_ptr) conv2d_77_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_77 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_77_0_0_eltwise_79_conversion */
  {
      const ai_float* conv2d_77_0_0_eltwise_79_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 57760);
    ai_i8* conv2d_77_0_0_eltwise_79_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 57760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(78, 1, {(stai_ptr) conv2d_77_0_0_eltwise_79_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_77_0_0_eltwise_79_conversion_t_in_0_ptr_const_f32, conv2d_77_0_0_eltwise_79_conversion_t_out_0_ptr_s8, conv2d_77_0_0_eltwise_79_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_77_0_0_eltwise_79_conversion_t_out_0_fmt_scale_const_f32, conv2d_77_0_0_eltwise_79_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(78, 1, {(stai_ptr) conv2d_77_0_0_eltwise_79_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_77_0_0_eltwise_79_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_79 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_79(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_79 */
  /* LITE_KERNEL_SECTION BEGIN nl_83 */
  {
    
  forward_lite_nl_integer_nl_83(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_83 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_85 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_85(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_85 */
  /* LITE_KERNEL_SECTION BEGIN pad_90 */
  {
      const ai_ptr pad_90_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 110368);
    ai_ptr pad_90_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 95776);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(90, 1, {(stai_ptr) pad_90_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_90_t_in_0_ptr_const_ptr, pad_90_t_out_0_ptr_ptr, (ai_handle)(pad_90_v_pad_constant_value_const_s8), pad_90_t_in_0_fmt_bitsize_const_s16, pad_90_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(3072), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(90, 1, {(stai_ptr) pad_90_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_90 */
  /* LITE_KERNEL_SECTION BEGIN pad_90_0_0_conv2d_92_conversion */
  {
      const ai_i8* pad_90_0_0_conv2d_92_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 95776);
    ai_float* pad_90_0_0_conv2d_92_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 52000);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(90, 1, {(stai_ptr) pad_90_0_0_conv2d_92_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_90_0_0_conv2d_92_conversion_t_in_0_ptr_const_s8, pad_90_0_0_conv2d_92_conversion_t_out_0_ptr_f32, pad_90_0_0_conv2d_92_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_90_0_0_conv2d_92_conversion_t_in_0_fmt_scale_const_f32, pad_90_0_0_conv2d_92_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(90, 1, {(stai_ptr) pad_90_0_0_conv2d_92_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_90_0_0_conv2d_92_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_92 */
  {
      const ai_float* conv2d_92_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 52000);
    ai_float* conv2d_92_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51616);
    const ai_u8* conv2d_92_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[19] + 0);
    const ai_u8* conv2d_92_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(93, 1, {(stai_ptr) conv2d_92_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_92_t_in_0_ptr_const_f32, conv2d_92_t_out_0_ptr_f32, conv2d_92_t_weight_0_ptr_const_u8, conv2d_92_t_weight_1_ptr_const_u8, conv2d_92_t_in_0_shape_ch_const_u32, conv2d_92_t_out_0_shape_ch_const_u32, conv2d_92_t_in_0_shape_w_const_u32, conv2d_92_t_in_0_shape_h_const_u32, conv2d_92_t_out_0_shape_w_const_u32, conv2d_92_t_out_0_shape_h_const_u32, conv2d_92_t_weight_0_shape_w_const_u32, conv2d_92_t_weight_0_shape_h_const_u32, conv2d_92_l_pad_W_0_const_s32, conv2d_92_l_pad_H_0_const_s32, conv2d_92_l_stride_1_const_u16, conv2d_92_l_stride_0_const_u16, 33, 1, conv2d_92_l_dilation_W_const_u16, conv2d_92_l_dilation_H_const_u16, conv2d_92_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(93, 1, {(stai_ptr) conv2d_92_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_92 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_92_0_0_eltwise_94_conversion */
  {
      const ai_float* conv2d_92_0_0_eltwise_94_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 51616);
    ai_i8* conv2d_92_0_0_eltwise_94_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 98848);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(93, 1, {(stai_ptr) conv2d_92_0_0_eltwise_94_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_92_0_0_eltwise_94_conversion_t_in_0_ptr_const_f32, conv2d_92_0_0_eltwise_94_conversion_t_out_0_ptr_s8, conv2d_92_0_0_eltwise_94_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_92_0_0_eltwise_94_conversion_t_out_0_fmt_scale_const_f32, conv2d_92_0_0_eltwise_94_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(93, 1, {(stai_ptr) conv2d_92_0_0_eltwise_94_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_92_0_0_eltwise_94_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_94 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_94(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_94 */
  /* LITE_KERNEL_SECTION BEGIN nl_98 */
  {
    
  forward_lite_nl_integer_nl_98(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_98 */
  /* LITE_KERNEL_SECTION BEGIN pad_104 */
  {
      const ai_ptr pad_104_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 51616);
    ai_ptr pad_104_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 95776);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(104, 1, {(stai_ptr) pad_104_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_104_t_in_0_ptr_const_ptr, pad_104_t_out_0_ptr_ptr, (ai_handle)(pad_104_v_pad_constant_value_const_s8), pad_104_t_in_0_fmt_bitsize_const_s16, pad_104_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(3072), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(104, 1, {(stai_ptr) pad_104_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_104 */
  /* LITE_KERNEL_SECTION BEGIN pad_104_0_0_conv2d_106_conversion */
  {
      const ai_i8* pad_104_0_0_conv2d_106_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 95776);
    ai_float* pad_104_0_0_conv2d_106_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 52000);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(104, 1, {(stai_ptr) pad_104_0_0_conv2d_106_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_104_0_0_conv2d_106_conversion_t_in_0_ptr_const_s8, pad_104_0_0_conv2d_106_conversion_t_out_0_ptr_f32, pad_104_0_0_conv2d_106_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_104_0_0_conv2d_106_conversion_t_in_0_fmt_scale_const_f32, pad_104_0_0_conv2d_106_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(104, 1, {(stai_ptr) pad_104_0_0_conv2d_106_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_104_0_0_conv2d_106_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_106 */
  {
      const ai_float* conv2d_106_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 52000);
    ai_float* conv2d_106_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51616);
    const ai_u8* conv2d_106_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[20] + 0);
    const ai_u8* conv2d_106_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(107, 1, {(stai_ptr) conv2d_106_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_106_t_in_0_ptr_const_f32, conv2d_106_t_out_0_ptr_f32, conv2d_106_t_weight_0_ptr_const_u8, conv2d_106_t_weight_1_ptr_const_u8, conv2d_106_t_in_0_shape_ch_const_u32, conv2d_106_t_out_0_shape_ch_const_u32, conv2d_106_t_in_0_shape_w_const_u32, conv2d_106_t_in_0_shape_h_const_u32, conv2d_106_t_out_0_shape_w_const_u32, conv2d_106_t_out_0_shape_h_const_u32, conv2d_106_t_weight_0_shape_w_const_u32, conv2d_106_t_weight_0_shape_h_const_u32, conv2d_106_l_pad_W_0_const_s32, conv2d_106_l_pad_H_0_const_s32, conv2d_106_l_stride_1_const_u16, conv2d_106_l_stride_0_const_u16, 33, 1, conv2d_106_l_dilation_W_const_u16, conv2d_106_l_dilation_H_const_u16, conv2d_106_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(107, 1, {(stai_ptr) conv2d_106_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_106 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_106_0_0_eltwise_108_conversion */
  {
      const ai_float* conv2d_106_0_0_eltwise_108_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 51616);
    ai_i8* conv2d_106_0_0_eltwise_108_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 98848);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(107, 1, {(stai_ptr) conv2d_106_0_0_eltwise_108_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_106_0_0_eltwise_108_conversion_t_in_0_ptr_const_f32, conv2d_106_0_0_eltwise_108_conversion_t_out_0_ptr_s8, conv2d_106_0_0_eltwise_108_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_106_0_0_eltwise_108_conversion_t_out_0_fmt_scale_const_f32, conv2d_106_0_0_eltwise_108_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(107, 1, {(stai_ptr) conv2d_106_0_0_eltwise_108_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_106_0_0_eltwise_108_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_108 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_108(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_108 */
  /* LITE_KERNEL_SECTION BEGIN nl_112 */
  {
    
  forward_lite_nl_integer_nl_112(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_112 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_114 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_114(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_114 */
  /* LITE_KERNEL_SECTION BEGIN pad_119 */
  {
      const ai_ptr pad_119_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 87328);
    ai_ptr pad_119_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 104224);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(119, 1, {(stai_ptr) pad_119_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_119_t_in_0_ptr_const_ptr, pad_119_t_out_0_ptr_ptr, (ai_handle)(pad_119_v_pad_constant_value_const_s8), pad_119_t_in_0_fmt_bitsize_const_s16, pad_119_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(6144), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(119, 1, {(stai_ptr) pad_119_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_119 */
  /* LITE_KERNEL_SECTION BEGIN pad_119_0_0_conv2d_121_conversion */
  {
      const ai_i8* pad_119_0_0_conv2d_121_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 104224);
    ai_float* pad_119_0_0_conv2d_121_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 16672);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(119, 1, {(stai_ptr) pad_119_0_0_conv2d_121_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_119_0_0_conv2d_121_conversion_t_in_0_ptr_const_s8, pad_119_0_0_conv2d_121_conversion_t_out_0_ptr_f32, pad_119_0_0_conv2d_121_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_119_0_0_conv2d_121_conversion_t_in_0_fmt_scale_const_f32, pad_119_0_0_conv2d_121_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(119, 1, {(stai_ptr) pad_119_0_0_conv2d_121_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_119_0_0_conv2d_121_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_121 */
  {
      const ai_float* conv2d_121_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 16672);
    ai_float* conv2d_121_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 16288);
    const ai_u8* conv2d_121_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[21] + 0);
    const ai_u8* conv2d_121_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(122, 1, {(stai_ptr) conv2d_121_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_121_t_in_0_ptr_const_f32, conv2d_121_t_out_0_ptr_f32, conv2d_121_t_weight_0_ptr_const_u8, conv2d_121_t_weight_1_ptr_const_u8, conv2d_121_t_in_0_shape_ch_const_u32, conv2d_121_t_out_0_shape_ch_const_u32, conv2d_121_t_in_0_shape_w_const_u32, conv2d_121_t_in_0_shape_h_const_u32, conv2d_121_t_out_0_shape_w_const_u32, conv2d_121_t_out_0_shape_h_const_u32, conv2d_121_t_weight_0_shape_w_const_u32, conv2d_121_t_weight_0_shape_h_const_u32, conv2d_121_l_pad_W_0_const_s32, conv2d_121_l_pad_H_0_const_s32, conv2d_121_l_stride_1_const_u16, conv2d_121_l_stride_0_const_u16, 65, 1, conv2d_121_l_dilation_W_const_u16, conv2d_121_l_dilation_H_const_u16, conv2d_121_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(122, 1, {(stai_ptr) conv2d_121_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_121 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_121_0_0_eltwise_123_conversion */
  {
      const ai_float* conv2d_121_0_0_eltwise_123_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 16288);
    ai_i8* conv2d_121_0_0_eltwise_123_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 62368);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(122, 1, {(stai_ptr) conv2d_121_0_0_eltwise_123_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_121_0_0_eltwise_123_conversion_t_in_0_ptr_const_f32, conv2d_121_0_0_eltwise_123_conversion_t_out_0_ptr_s8, conv2d_121_0_0_eltwise_123_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_121_0_0_eltwise_123_conversion_t_out_0_fmt_scale_const_f32, conv2d_121_0_0_eltwise_123_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(122, 1, {(stai_ptr) conv2d_121_0_0_eltwise_123_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_121_0_0_eltwise_123_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_123 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_123(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_123 */
  /* LITE_KERNEL_SECTION BEGIN nl_127 */
  {
    
  forward_lite_nl_integer_nl_127(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_127 */
  /* LITE_KERNEL_SECTION BEGIN pad_133 */
  {
      const ai_ptr pad_133_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 27808);
    ai_ptr pad_133_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 69664);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(133, 1, {(stai_ptr) pad_133_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_133_t_in_0_ptr_const_ptr, pad_133_t_out_0_ptr_ptr, (ai_handle)(pad_133_v_pad_constant_value_const_s8), pad_133_t_in_0_fmt_bitsize_const_s16, pad_133_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(96), (ai_i32)(6144), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(133, 1, {(stai_ptr) pad_133_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_133 */
  /* LITE_KERNEL_SECTION BEGIN pad_133_0_0_conv2d_135_conversion */
  {
      const ai_i8* pad_133_0_0_conv2d_135_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 69664);
    ai_float* pad_133_0_0_conv2d_135_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 16672);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(133, 1, {(stai_ptr) pad_133_0_0_conv2d_135_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_133_0_0_conv2d_135_conversion_t_in_0_ptr_const_s8, pad_133_0_0_conv2d_135_conversion_t_out_0_ptr_f32, pad_133_0_0_conv2d_135_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_133_0_0_conv2d_135_conversion_t_in_0_fmt_scale_const_f32, pad_133_0_0_conv2d_135_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(133, 1, {(stai_ptr) pad_133_0_0_conv2d_135_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_133_0_0_conv2d_135_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_135 */
  {
      const ai_float* conv2d_135_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 16672);
    ai_float* conv2d_135_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 16288);
    const ai_u8* conv2d_135_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[22] + 0);
    const ai_u8* conv2d_135_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[15] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(136, 1, {(stai_ptr) conv2d_135_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_135_t_in_0_ptr_const_f32, conv2d_135_t_out_0_ptr_f32, conv2d_135_t_weight_0_ptr_const_u8, conv2d_135_t_weight_1_ptr_const_u8, conv2d_135_t_in_0_shape_ch_const_u32, conv2d_135_t_out_0_shape_ch_const_u32, conv2d_135_t_in_0_shape_w_const_u32, conv2d_135_t_in_0_shape_h_const_u32, conv2d_135_t_out_0_shape_w_const_u32, conv2d_135_t_out_0_shape_h_const_u32, conv2d_135_t_weight_0_shape_w_const_u32, conv2d_135_t_weight_0_shape_h_const_u32, conv2d_135_l_pad_W_0_const_s32, conv2d_135_l_pad_H_0_const_s32, conv2d_135_l_stride_1_const_u16, conv2d_135_l_stride_0_const_u16, 65, 1, conv2d_135_l_dilation_W_const_u16, conv2d_135_l_dilation_H_const_u16, conv2d_135_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(136, 1, {(stai_ptr) conv2d_135_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_135 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_135_0_0_eltwise_137_conversion */
  {
      const ai_float* conv2d_135_0_0_eltwise_137_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 16288);
    ai_i8* conv2d_135_0_0_eltwise_137_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 62368);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(136, 1, {(stai_ptr) conv2d_135_0_0_eltwise_137_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_135_0_0_eltwise_137_conversion_t_in_0_ptr_const_f32, conv2d_135_0_0_eltwise_137_conversion_t_out_0_ptr_s8, conv2d_135_0_0_eltwise_137_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_135_0_0_eltwise_137_conversion_t_out_0_fmt_scale_const_f32, conv2d_135_0_0_eltwise_137_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(136, 1, {(stai_ptr) conv2d_135_0_0_eltwise_137_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_135_0_0_eltwise_137_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_137 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_137(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_137 */
  /* LITE_KERNEL_SECTION BEGIN nl_141 */
  {
    
  forward_lite_nl_integer_nl_141(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_141 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_143 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_143(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_143 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_148 */
  {
      const ai_i8* conv2d_148_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16288);
    const ai_i8* conv2d_148_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[23] + 0);
    const ai_i32* conv2d_148_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[24] + 0);
    ai_i8* conv2d_148_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 27808);
    ai_i16* conv2d_148_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(148, 1, {(stai_ptr) conv2d_148_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_148_t_in_0_ptr_const_s8, conv2d_148_t_in_0_shape_w_const_u16, conv2d_148_t_in_0_shape_h_const_u16, conv2d_148_l_stride_1_const_u16, conv2d_148_l_stride_0_const_u16, conv2d_148_t_in_0_shape_ch_const_u16, conv2d_148_t_weight_0_ptr_const_s8, conv2d_148_t_out_0_shape_ch_const_u16, conv2d_148_t_weight_1_ptr_const_s32, conv2d_148_t_in_0_fmt_zero_const_s8, conv2d_148_t_out_0_fmt_zero_const_s8, conv2d_148_t_in_0_fmt_scale_const_f32, conv2d_148_t_out_0_fmt_scale_const_f32, conv2d_148_t_weight_0_fmt_scale_const_f32, conv2d_148_l_out_ch_format_const_layer_format_type, conv2d_148_t_out_0_ptr_s8, 1, 1664, conv2d_148_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(148, 1, {(stai_ptr) conv2d_148_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_148 */
  /* LITE_KERNEL_SECTION BEGIN nl_152 */
  {
    
  forward_lite_nl_integer_nl_152(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_152 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_158 */
  {
    
  forward_lite_conv2d_integer_SSSA_conv2d_158(net_ctx);
  }
  /* LITE_KERNEL_SECTION END conv2d_158 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 96448;

  net_ctx->_outputs[0] = activations[0] + 16288;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

