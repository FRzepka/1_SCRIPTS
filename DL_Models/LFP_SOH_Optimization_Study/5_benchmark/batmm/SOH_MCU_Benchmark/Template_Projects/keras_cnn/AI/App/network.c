/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-15T16:26:50+0000
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
#define _STAI_NETWORK_MODEL_SIGNATURE     "0xc6577e8907a23cd1253b78e85f5c87cc"
#define _STAI_NETWORK_DATETIME            "2026-05-15T16:26:50+0000"
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
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 96, 20),
      STAI_DECLARE_ARRAY(float, 1, 0.07195434719324112f),
      STAI_DECLARE_ARRAY(int16_t, 1, 40)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 96, 1, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.003846552222967148f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 90448),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 128),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_2_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 128),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_3_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 128),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_4_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_4_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 128),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_5_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_5_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 2560),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_6_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_6_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 512),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_7_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_7_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 81920),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_8_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_8_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 512),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_9_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_9_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 81920),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_10_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_10_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 512),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_11_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_11_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 327680),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_12_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_12_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 512),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_13_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_13_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 327680),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_14_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_14_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 327680),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_15_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_15_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 327680),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_16_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_16_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 8192),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_17_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_17_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 256),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_18_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_18_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 64),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_19_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_19_SIZE_BYTES,
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
      (stai_ptr)g_network_tfl_pseudo_qconst10_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst8_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst6_4D_array,(stai_ptr)g_network_tfl_pseudo_qconst4_4D_array,(stai_ptr)g_network_conv2d_4_weights_array,(stai_ptr)g_network_conv2d_4_bias_array,(stai_ptr)g_network_conv2d_14_weights_array,(stai_ptr)g_network_conv2d_14_bias_array,(stai_ptr)g_network_conv2d_25_weights_array,(stai_ptr)g_network_conv2d_25_bias_array,(stai_ptr)g_network_conv2d_38_weights_array,(stai_ptr)g_network_conv2d_38_bias_array,(stai_ptr)g_network_conv2d_52_weights_array,(stai_ptr)g_network_conv2d_67_weights_array,(stai_ptr)g_network_conv2d_81_weights_array,(stai_ptr)g_network_conv2d_94_weights_array,(stai_ptr)g_network_conv2d_94_bias_array,(stai_ptr)g_network_conv2d_104_weights_array,(stai_ptr)g_network_conv2d_104_bias_array
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
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_14_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06683990359306335f),
    AI_PACK_INTQ_ZP(66)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_18_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01600812003016472f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_25_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03006605990231037f),
    AI_PACK_INTQ_ZP(62)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_29_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007697582710534334f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03005221113562584f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_31_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030483368784189224f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_38_0_0_eltwise_40_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.041260287165641785f),
    AI_PACK_INTQ_ZP(77)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_40_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04132267087697983f),
    AI_PACK_INTQ_ZP(78)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst10_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005193492397665977f),
    AI_PACK_INTQ_ZP(1)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_44_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007900108583271503f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_52_0_0_eltwise_54_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015926923602819443f),
    AI_PACK_INTQ_ZP(43)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_54_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015453035943210125f),
    AI_PACK_INTQ_ZP(39)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst8_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0010873150313273072f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_58_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005308276973664761f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_60_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.030903693288564682f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_67_0_0_eltwise_69_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05199640244245529f),
    AI_PACK_INTQ_ZP(77)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_69_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05238831415772438f),
    AI_PACK_INTQ_ZP(76)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst6_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005255109863355756f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_73_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01041225902736187f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_81_0_0_eltwise_83_conversion_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02176525816321373f),
    AI_PACK_INTQ_ZP(97)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_83_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.021538667380809784f),
    AI_PACK_INTQ_ZP(94)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(tfl_pseudo_qconst4_4D_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0013553326716646552f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_87_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0027950801886618137f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_89_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03198061138391495f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_94_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.042549774050712585f),
    AI_PACK_INTQ_ZP(51)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_98_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012757483869791031f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_104_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003846552222967148f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_104_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02335912361741066f),
    AI_PACK_INTQ_ZP(0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  nl_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  nl_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_0_0_eltwise_40_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_40_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst10_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  nl_44_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_52_0_0_eltwise_54_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst8_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  nl_58_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_60_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_67_0_0_eltwise_69_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_69_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst6_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  nl_73_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_81_0_0_eltwise_83_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  tfl_pseudo_qconst4_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  nl_87_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_89_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12288, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_94_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  nl_98_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_104_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 96, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_104_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_104_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_104_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_14_output_array, &conv2d_14_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  nl_18_output, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_18_output_array, &nl_18_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_25_output_array, &conv2d_25_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  nl_29_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_29_output_array, &nl_29_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_31_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_31_output_array, &eltwise_31_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_0_0_eltwise_40_conversion_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_38_0_0_eltwise_40_conversion_output_array, &conv2d_38_0_0_eltwise_40_conversion_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_40_output, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_40_output_array, &eltwise_40_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst10_4D, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &tfl_pseudo_qconst10_4D_array, &tfl_pseudo_qconst10_4D_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  nl_44_output, AI_STATIC,
  42, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_44_output_array, &nl_44_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_52_0_0_eltwise_54_conversion_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_52_0_0_eltwise_54_conversion_output_array, &conv2d_52_0_0_eltwise_54_conversion_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_54_output_array, &eltwise_54_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst8_4D, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &tfl_pseudo_qconst8_4D_array, &tfl_pseudo_qconst8_4D_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  nl_58_output, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_58_output_array, &nl_58_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_60_output, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_60_output_array, &eltwise_60_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_67_0_0_eltwise_69_conversion_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_67_0_0_eltwise_69_conversion_output_array, &conv2d_67_0_0_eltwise_69_conversion_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_69_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_69_output_array, &eltwise_69_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst6_4D, AI_STATIC,
  60, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &tfl_pseudo_qconst6_4D_array, &tfl_pseudo_qconst6_4D_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  nl_73_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_73_output_array, &nl_73_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_81_0_0_eltwise_83_conversion_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_81_0_0_eltwise_83_conversion_output_array, &conv2d_81_0_0_eltwise_83_conversion_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_83_output, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_83_output_array, &eltwise_83_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  tfl_pseudo_qconst4_4D, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &tfl_pseudo_qconst4_4D_array, &tfl_pseudo_qconst4_4D_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  nl_87_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &nl_87_output_array, &nl_87_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_89_output, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 96), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &eltwise_89_output_array, &eltwise_89_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_94_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 96), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_94_output_array, &conv2d_94_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  nl_98_output, AI_STATIC,
  46, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 96), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &nl_98_output_array, &nl_98_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_104_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &conv2d_104_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_104_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 96), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &conv2d_104_output_array, &conv2d_104_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_104_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &conv2d_104_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_104_weights, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_104_weights_array, &conv2d_104_weights_array_intq)



AI_STATIC_CONST ai_i8 nl_18_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -124, -120, -115, -111, -107, -103, -99, -95, -90, -86, -82, -78, -74, -70, -65, -61, -57, -53, -49, -44, -40, -36, -32, -28, -24, -19, -15, -11, -7, -3, 1, 6, 10, 14, 18, 22, 26, 31, 35, 39, 43, 47, 52, 56, 60, 64, 68, 72, 77, 81, 85, 89, 93, 97, 102, 106, 110, 114, 118, 123, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_18_nl_params, AI_ARRAY_FORMAT_S8,
    nl_18_nl_params_data, nl_18_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_18_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_18_chain,
  NULL, &nl_18_layer, AI_STATIC, 
  .nl_params = &nl_18_nl_params, 
)


AI_STATIC_CONST ai_i8 nl_29_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -124, -120, -116, -112, -108, -105, -101, -97, -93, -89, -85, -81, -77, -73, -69, -66, -62, -58, -54, -50, -46, -42, -38, -34, -30, -26, -23, -19, -15, -11, -7, -3, 1, 5, 9, 13, 17, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 102, 106, 110, 114, 118, 122, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_29_nl_params, AI_ARRAY_FORMAT_S8,
    nl_29_nl_params_data, nl_29_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_29_layer, 29,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_29_chain,
  NULL, &nl_29_layer, AI_STATIC, 
  .nl_params = &nl_29_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_29_output, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_31_layer, 31,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_31_chain,
  NULL, &eltwise_31_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_38_0_0_eltwise_40_conversion_output, &tfl_pseudo_qconst10_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_40_layer, 40,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_40_chain,
  NULL, &eltwise_40_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_44_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -123, -118, -112, -107, -102, -97, -91, -86, -81, -76, -70, -65, -60, -55, -50, -44, -39, -34, -29, -23, -18, -13, -8, -2, 3, 8, 13, 18, 24, 29, 34, 39, 45, 50, 55, 60, 66, 71, 76, 81, 86, 92, 97, 102, 107, 113, 118, 123, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_44_nl_params, AI_ARRAY_FORMAT_S8,
    nl_44_nl_params_data, nl_44_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_44_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_44_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_44_layer, 44,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_44_chain,
  NULL, &nl_44_layer, AI_STATIC, 
  .nl_params = &nl_44_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_52_0_0_eltwise_54_conversion_output, &tfl_pseudo_qconst8_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_54_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_58_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -122, -119, -116, -113, -111, -108, -105, -102, -99, -96, -93, -90, -87, -84, -81, -79, -76, -73, -70, -67, -64, -61, -58, -55, -52, -49, -46, -44, -41, -38, -35, -32, -29, -26, -23, -20, -17, -14, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 119, 122, 125, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_58_nl_params, AI_ARRAY_FORMAT_S8,
    nl_58_nl_params_data, nl_58_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_58_layer, 58,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_58_chain,
  NULL, &nl_58_layer, AI_STATIC, 
  .nl_params = &nl_58_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_60_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_58_output, &eltwise_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_60_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_60_layer, 60,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_60_chain,
  NULL, &eltwise_60_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_69_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_67_0_0_eltwise_69_conversion_output, &tfl_pseudo_qconst6_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_69_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_69_layer, 69,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_69_chain,
  NULL, &eltwise_69_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_73_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -123, -118, -113, -108, -103, -98, -93, -88, -83, -78, -73, -68, -63, -58, -53, -47, -42, -37, -32, -27, -22, -17, -12, -7, -2, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 114, 119, 124, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_73_nl_params, AI_ARRAY_FORMAT_S8,
    nl_73_nl_params_data, nl_73_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_73_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_73_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_73_layer, 73,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_73_chain,
  NULL, &nl_73_layer, AI_STATIC, 
  .nl_params = &nl_73_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_81_0_0_eltwise_83_conversion_output, &tfl_pseudo_qconst4_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_83_layer, 83,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_83_chain,
  NULL, &eltwise_83_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_87_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -120, -113, -105, -97, -89, -82, -74, -66, -59, -51, -43, -36, -28, -20, -12, -5, 3, 11, 18, 26, 34, 42, 49, 57, 65, 72, 80, 88, 95, 103, 111, 119, 126 };
AI_ARRAY_OBJ_DECLARE(
    nl_87_nl_params, AI_ARRAY_FORMAT_S8,
    nl_87_nl_params_data, nl_87_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_83_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_87_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_87_layer, 87,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_87_chain,
  NULL, &nl_87_layer, AI_STATIC, 
  .nl_params = &nl_87_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_89_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_87_output, &eltwise_60_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_89_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_89_layer, 89,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_89_chain,
  NULL, &eltwise_89_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_i8 nl_98_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -125, -121, -118, -115, -111, -108, -105, -101, -98, -95, -91, -88, -85, -81, -78, -75, -71, -68, -65, -61, -58, -55, -51, -48, -45, -41, -38, -35, -31, -28, -25, -21, -18, -15, -11, -8, -5, -1, 2, 5, 9, 12, 15, 19, 22, 25, 29, 32, 35, 39, 42, 45, 49, 52, 55, 59, 62, 65, 69, 72, 75, 79, 82, 85, 89, 92, 95, 99, 102, 105, 109, 112, 115, 119, 122, 125 };
AI_ARRAY_OBJ_DECLARE(
    nl_98_nl_params, AI_ARRAY_FORMAT_S8,
    nl_98_nl_params_data, nl_98_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_98_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_94_output),
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
  conv2d_104_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_98_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_104_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_104_weights, &conv2d_104_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_104_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_104_layer, 104,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &conv2d_104_chain,
  NULL, &conv2d_104_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_nl_integer_nl_18(_stai_network_context* net_ctx)
{
  conv2d_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 52944);
  conv2d_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 52944);
  nl_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 52944);
  nl_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 52944);
  _STAI_NETWORK_EVENT_NODE_START_CB(18, 1, { conv2d_14_output.data->data});
  forward_nl_integer(&nl_18_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(18, 1, { nl_18_output.data->data});
}
void forward_lite_nl_integer_nl_29(_stai_network_context* net_ctx)
{
  conv2d_25_output_array.data = AI_PTR(net_ctx->_activations[0] + 52304);
  conv2d_25_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 52304);
  nl_29_output_array.data = AI_PTR(net_ctx->_activations[0] + 52304);
  nl_29_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 52304);
  _STAI_NETWORK_EVENT_NODE_START_CB(29, 1, { conv2d_25_output.data->data});
  forward_nl_integer(&nl_29_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(29, 1, { nl_29_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_31(_stai_network_context* net_ctx)
{
  nl_29_output_array.data = AI_PTR(net_ctx->_activations[0] + 52304);
  nl_29_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 52304);
  conv2d_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 65872);
  conv2d_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 65872);
  eltwise_31_output_array.data = AI_PTR(net_ctx->_activations[0] + 78160);
  eltwise_31_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 78160);
  _STAI_NETWORK_EVENT_NODE_START_CB(31, 2, { nl_29_output.data->data,conv2d_4_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_31_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(31, 1, { eltwise_31_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_40(_stai_network_context* net_ctx)
{
  conv2d_38_0_0_eltwise_40_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  conv2d_38_0_0_eltwise_40_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  tfl_pseudo_qconst10_4D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  tfl_pseudo_qconst10_4D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  eltwise_40_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  eltwise_40_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 2, { conv2d_38_0_0_eltwise_40_conversion_output.data->data,tfl_pseudo_qconst10_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_40_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, { eltwise_40_output.data->data});
}
void forward_lite_nl_integer_nl_44(_stai_network_context* net_ctx)
{
  eltwise_40_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  eltwise_40_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  nl_44_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  nl_44_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  _STAI_NETWORK_EVENT_NODE_START_CB(44, 1, { eltwise_40_output.data->data});
  forward_nl_integer(&nl_44_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(44, 1, { nl_44_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_54(_stai_network_context* net_ctx)
{
  conv2d_52_0_0_eltwise_54_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  conv2d_52_0_0_eltwise_54_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  tfl_pseudo_qconst8_4D_array.data = AI_PTR(net_ctx->_weights[1] + 0);
  tfl_pseudo_qconst8_4D_array.data_start = AI_PTR(net_ctx->_weights[1] + 0);
  eltwise_54_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  eltwise_54_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 2, { conv2d_52_0_0_eltwise_54_conversion_output.data->data,tfl_pseudo_qconst8_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_54_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { eltwise_54_output.data->data});
}
void forward_lite_nl_integer_nl_58(_stai_network_context* net_ctx)
{
  eltwise_54_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  eltwise_54_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  nl_58_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  nl_58_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  _STAI_NETWORK_EVENT_NODE_START_CB(58, 1, { eltwise_54_output.data->data});
  forward_nl_integer(&nl_58_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(58, 1, { nl_58_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_60(_stai_network_context* net_ctx)
{
  nl_58_output_array.data = AI_PTR(net_ctx->_activations[0] + 24400);
  nl_58_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 24400);
  eltwise_31_output_array.data = AI_PTR(net_ctx->_activations[0] + 78160);
  eltwise_31_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 78160);
  eltwise_60_output_array.data = AI_PTR(net_ctx->_activations[0] + 78160);
  eltwise_60_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 78160);
  _STAI_NETWORK_EVENT_NODE_START_CB(60, 2, { nl_58_output.data->data,eltwise_31_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_60_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(60, 1, { eltwise_60_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_69(_stai_network_context* net_ctx)
{
  conv2d_67_0_0_eltwise_69_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 20304);
  conv2d_67_0_0_eltwise_69_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20304);
  tfl_pseudo_qconst6_4D_array.data = AI_PTR(net_ctx->_weights[2] + 0);
  tfl_pseudo_qconst6_4D_array.data_start = AI_PTR(net_ctx->_weights[2] + 0);
  eltwise_69_output_array.data = AI_PTR(net_ctx->_activations[0] + 32592);
  eltwise_69_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32592);
  _STAI_NETWORK_EVENT_NODE_START_CB(69, 2, { conv2d_67_0_0_eltwise_69_conversion_output.data->data,tfl_pseudo_qconst6_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_69_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(69, 1, { eltwise_69_output.data->data});
}
void forward_lite_nl_integer_nl_73(_stai_network_context* net_ctx)
{
  eltwise_69_output_array.data = AI_PTR(net_ctx->_activations[0] + 32592);
  eltwise_69_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32592);
  nl_73_output_array.data = AI_PTR(net_ctx->_activations[0] + 20304);
  nl_73_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20304);
  _STAI_NETWORK_EVENT_NODE_START_CB(73, 1, { eltwise_69_output.data->data});
  forward_nl_integer(&nl_73_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(73, 1, { nl_73_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_83(_stai_network_context* net_ctx)
{
  conv2d_81_0_0_eltwise_83_conversion_output_array.data = AI_PTR(net_ctx->_activations[0] + 20304);
  conv2d_81_0_0_eltwise_83_conversion_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20304);
  tfl_pseudo_qconst4_4D_array.data = AI_PTR(net_ctx->_weights[3] + 0);
  tfl_pseudo_qconst4_4D_array.data_start = AI_PTR(net_ctx->_weights[3] + 0);
  eltwise_83_output_array.data = AI_PTR(net_ctx->_activations[0] + 32592);
  eltwise_83_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32592);
  _STAI_NETWORK_EVENT_NODE_START_CB(83, 2, { conv2d_81_0_0_eltwise_83_conversion_output.data->data,tfl_pseudo_qconst4_4D.data->data});
  forward_eltwise_integer_INT8(&eltwise_83_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(83, 1, { eltwise_83_output.data->data});
}
void forward_lite_nl_integer_nl_87(_stai_network_context* net_ctx)
{
  eltwise_83_output_array.data = AI_PTR(net_ctx->_activations[0] + 32592);
  eltwise_83_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32592);
  nl_87_output_array.data = AI_PTR(net_ctx->_activations[0] + 20304);
  nl_87_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20304);
  _STAI_NETWORK_EVENT_NODE_START_CB(87, 1, { eltwise_83_output.data->data});
  forward_nl_integer(&nl_87_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(87, 1, { nl_87_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_89(_stai_network_context* net_ctx)
{
  nl_87_output_array.data = AI_PTR(net_ctx->_activations[0] + 20304);
  nl_87_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20304);
  eltwise_60_output_array.data = AI_PTR(net_ctx->_activations[0] + 78160);
  eltwise_60_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 78160);
  eltwise_89_output_array.data = AI_PTR(net_ctx->_activations[0] + 32592);
  eltwise_89_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 32592);
  _STAI_NETWORK_EVENT_NODE_START_CB(89, 2, { nl_87_output.data->data,eltwise_60_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_89_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(89, 1, { eltwise_89_output.data->data});
}
void forward_lite_nl_integer_nl_98(_stai_network_context* net_ctx)
{
  conv2d_94_output_array.data = AI_PTR(net_ctx->_activations[0] + 20304);
  conv2d_94_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 20304);
  nl_98_output_array.data = AI_PTR(net_ctx->_activations[0] + 26448);
  nl_98_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26448);
  _STAI_NETWORK_EVENT_NODE_START_CB(98, 1, { conv2d_94_output.data->data});
  forward_nl_integer(&nl_98_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(98, 1, { nl_98_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_conv2d_104(_stai_network_context* net_ctx)
{
  nl_98_output_array.data = AI_PTR(net_ctx->_activations[0] + 26448);
  nl_98_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 26448);
  conv2d_104_weights_array.data = AI_PTR(net_ctx->_weights[17] + 0);
  conv2d_104_weights_array.data_start = AI_PTR(net_ctx->_weights[17] + 0);
  conv2d_104_bias_array.data = AI_PTR(net_ctx->_weights[18] + 0);
  conv2d_104_bias_array.data_start = AI_PTR(net_ctx->_weights[18] + 0);
  conv2d_104_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_104_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_104_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  conv2d_104_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(104, 1, { nl_98_output.data->data});
  forward_conv2d_integer_SSSA(&conv2d_104_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(104, 1, { conv2d_104_output.data->data});
}

/*****************************************************************************/


static const ai_u16 conv2d_4_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_4_t_in_0_shape_h_const_u16 = 96;
static const ai_u16 conv2d_4_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_4_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_4_t_in_0_shape_ch_const_u16 = 20;
static const ai_u16 conv2d_4_t_out_0_shape_ch_const_u16 = 128;
static const ai_i8 conv2d_4_t_in_0_fmt_zero_const_s8 = 40;
static const ai_i8 conv2d_4_t_out_0_fmt_zero_const_s8 = 2;
static const ai_float conv2d_4_t_in_0_fmt_scale_const_f32 = 0.07195434719324112f;
static const ai_float conv2d_4_t_out_0_fmt_scale_const_f32 = 0.03005221113562584f;
static const ai_float conv2d_4_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002117775846272707f, 0.0010764488251879811f, 0.001676731277257204f, 0.0021946909837424755f, 0.0017108606407418847f, 0.0020996432285755873f, 0.002031615935266018f, 0.0023452427703887224f, 0.002129106782376766f, 0.0017137337708845735f, 0.0021731469314545393f, 0.0027580379974097013f, 0.0015618841862305999f, 0.0014902710681781173f, 0.0021276597399264574f, 0.001138764200732112f, 0.0021590208634734154f, 0.0015418528346344829f, 0.0013782777823507786f, 0.0015979019226506352f, 0.001496587530709803f, 0.0021791807375848293f, 0.001971632707864046f, 0.0017962418496608734f, 0.001762405619956553f, 0.0014211623929440975f, 0.0015579948667436838f, 0.001650723279453814f, 0.0017429118743166327f, 0.002274487167596817f, 0.001736092148348689f, 0.001900596427731216f, 0.0019751142244786024f, 0.0020102441776543856f, 0.0025090312119573355f, 0.0021488917991518974f, 0.0011467679869383574f, 0.00217076507396996f, 0.001619294285774231f, 0.003313726279884577f, 0.002497168257832527f, 0.0019632535986602306f, 0.0019779070280492306f, 0.00229179416783154f, 0.0019084792584180832f, 0.00259737903252244f, 0.001788444584235549f, 0.00197179545648396f, 0.0027292075101286173f, 0.0011696405708789825f, 0.001654367079026997f, 0.0018468705238774419f, 0.0014828331768512726f, 0.0019045529188588262f, 0.0019507051911205053f, 0.0022990060970187187f, 0.001688205637037754f, 0.0015231702709570527f, 0.0018585803918540478f, 0.00264660082757473f, 0.0016423399792984128f, 0.0011921764817088842f, 0.0016958264168351889f, 0.0017340845661237836f, 0.0019303010776638985f, 0.0015583783388137817f, 0.0011378000490367413f, 0.0019413505215197802f, 0.001826297608204186f, 0.0014299858594313264f, 0.002457977272570133f, 0.0022036510054022074f, 0.0020798982586711645f, 0.0017945660511031747f, 0.0017382834339514375f, 0.0016280199633911252f, 0.0017685039201751351f, 0.002177777700126171f, 0.0020623879972845316f, 0.002201968338340521f, 0.001555388793349266f, 0.002059526741504669f, 0.0025455739814788103f, 0.001687100506387651f, 0.002378003438934684f, 0.0017891761381179094f, 0.001613366766832769f, 0.0023197720292955637f, 0.0022536995820701122f, 0.0016738317208364606f, 0.0020617288537323475f, 0.0016287211328744888f, 0.0019919294863939285f, 0.0027051353827118874f, 0.0020911688916385174f, 0.002444448648020625f, 0.002057895762845874f, 0.0019797843415290117f, 0.0015544103225693107f, 0.0013036256423220038f, 0.0019764977041631937f, 0.0022277734242379665f, 0.0012511081295087934f, 0.0014352895086631179f, 0.0024871579371392727f, 0.0017010667361319065f, 0.0016242904821410775f, 0.0025976242031902075f, 0.0014369890559464693f, 0.002690620254725218f, 0.0012275578919798136f, 0.002059594029560685f, 0.0014047028962522745f, 0.0031433640979230404f, 0.002026368398219347f, 0.0018605171935632825f, 0.001929859514348209f, 0.002591044409200549f, 0.0027927954215556383f, 0.002067115157842636f, 0.001497834688052535f, 0.002119498560205102f, 0.0023667151108384132f, 0.0018842772115021944f, 0.001567972474731505f, 0.0016100640641525388f, 0.0013684891164302826f, 0.002595562255010009f);
static const ai_layer_format_type conv2d_4_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 pad_13_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(2);
static const ai_i16 pad_13_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_13_t_in_0_shape_h_const_u32 = 96;

static const ai_u16 conv2d_14_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_14_t_in_0_shape_h_const_u16 = 100;
static const ai_u16 conv2d_14_t_in_0_shape_ch_const_u16 = 128;
static const ai_u16 conv2d_14_t_out_0_shape_ch_const_u16 = 128;
static const ai_u16 conv2d_14_t_weight_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_14_t_weight_0_shape_h_const_u16 = 5;
static const ai_u16 conv2d_14_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_14_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_14_t_in_0_fmt_zero_const_s8 = 2;
static const ai_i8 conv2d_14_t_out_0_fmt_zero_const_s8 = 66;
static const ai_float conv2d_14_t_in_0_fmt_scale_const_f32 = 0.03005221113562584f;
static const ai_float conv2d_14_t_out_0_fmt_scale_const_f32 = 0.06683990359306335f;
static const ai_float conv2d_14_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017542792484164238f, 0.001422364846803248f, 0.0017620435683056712f, 0.0016186166321858764f, 0.001420345506630838f, 0.0018903440795838833f, 0.0016089227283373475f, 0.0017204933101311326f, 0.0016382925678044558f, 0.0016374158440157771f, 0.0016136299818754196f, 0.0019537529442459345f, 0.001386017189361155f, 0.0016172631876543164f, 0.0013793476391583681f, 0.0016267950413748622f, 0.0013860684121027589f, 0.0015425907913595438f, 0.0015758044319227338f, 0.001589362625963986f, 0.0016475007869303226f, 0.0017224923940375447f, 0.001990352524444461f, 0.0021487469784915447f, 0.0018454844830557704f, 0.0019743703305721283f, 0.0014167716726660728f, 0.0014729128452017903f, 0.0019115301547572017f, 0.001207765657454729f, 0.0018511322559788823f, 0.0016895622247830033f, 0.0015836135717108846f, 0.0014428594149649143f, 0.0016308497870340943f, 0.002110237255692482f, 0.0013776706764474511f, 0.001719180727377534f, 0.0014998774277046323f, 0.0012619567569345236f, 0.0019096247851848602f, 0.0015791574260219932f, 0.0014118654653429985f, 0.001440369407646358f, 0.0018256688490509987f, 0.001673986203968525f, 0.0014113682555034757f, 0.0014686636859551072f, 0.0015444594901055098f, 0.0013955376343801618f, 0.0019274377264082432f, 0.001530096516944468f, 0.0013461292255669832f, 0.0013328331988304853f, 0.0016638949746266007f, 0.0019492696737870574f, 0.0015033792005851865f, 0.0013896699529141188f, 0.0017067535081878304f, 0.0018000847194343805f, 0.0013879855396226048f, 0.0016450113616883755f, 0.0015972189139574766f, 0.0020533346105366945f, 0.0013448931276798248f, 0.0017967827152460814f, 0.001712194993160665f, 0.001858451054431498f, 0.0014709747629240155f, 0.0018346342258155346f, 0.0017454461194574833f, 0.0013654745416715741f, 0.0014146693283692002f, 0.0018110446399077773f, 0.0013713837834075093f, 0.0017238307045772672f, 0.0014811462024226785f, 0.0013596441131085157f, 0.002182980300858617f, 0.0014712580014020205f, 0.0016320159193128347f, 0.001900535193271935f, 0.0014813562156632543f, 0.0019469575490802526f, 0.0019099764758720994f, 0.0018829131731763482f, 0.0017180596478283405f, 0.0013573836768046021f, 0.001688978518359363f, 0.0014781038044020534f, 0.0012537800939753652f, 0.0022855771239846945f, 0.0016244739526882768f, 0.0017918419325724244f, 0.0017233950784429908f, 0.0014250774402171373f, 0.0017050639726221561f, 0.001679962850175798f, 0.0016356875421479344f, 0.0018091389210894704f, 0.0017039170488715172f, 0.0017192274099215865f, 0.0014506335137411952f, 0.0014920856337994337f, 0.0015083544421941042f, 0.0012355685466900468f, 0.0015893216477707028f, 0.0016590513987466693f, 0.0017117115203291178f, 0.0017566592432558537f, 0.0015567130176350474f, 0.0018224897794425488f, 0.0015754395863041282f, 0.0014091281918808818f, 0.001533039496280253f, 0.0015799349639564753f, 0.0019180536037310958f, 0.0013914990704506636f, 0.0013671296183019876f, 0.0017724217614158988f, 0.0015658511547371745f, 0.0021223106887191534f, 0.001481048297137022f, 0.0016016344306990504f, 0.001524648629128933f, 0.0021708663552999496f, 0.0012060892768204212f, 0.0016852285480126739f);
static const ai_layer_format_type conv2d_14_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_14_t_out_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_14_t_out_0_shape_h_const_u16 = 96;


static const ai_i8 pad_24_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_24_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_24_t_in_0_shape_h_const_u32 = 96;

static const ai_u16 conv2d_25_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_25_t_in_0_shape_h_const_u16 = 100;
static const ai_u16 conv2d_25_t_in_0_shape_ch_const_u16 = 128;
static const ai_u16 conv2d_25_t_out_0_shape_ch_const_u16 = 128;
static const ai_u16 conv2d_25_t_weight_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_25_t_weight_0_shape_h_const_u16 = 5;
static const ai_u16 conv2d_25_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_25_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_25_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_25_t_out_0_fmt_zero_const_s8 = 62;
static const ai_float conv2d_25_t_in_0_fmt_scale_const_f32 = 0.01600812003016472f;
static const ai_float conv2d_25_t_out_0_fmt_scale_const_f32 = 0.03006605990231037f;
static const ai_float conv2d_25_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0013873206917196512f, 0.0012646254617720842f, 0.0015120250172913074f, 0.0014399787178263068f, 0.001419957960024476f, 0.001703281537629664f, 0.0018460205756127834f, 0.0018979617161676288f, 0.001468751928769052f, 0.0017537003150209785f, 0.0016531962901353836f, 0.00213624513708055f, 0.0016336501576006413f, 0.002073773881420493f, 0.0020904303528368473f, 0.0014762899372726679f, 0.0014925216091796756f, 0.0014769895933568478f, 0.001685632742010057f, 0.001691328827291727f, 0.0013997027417644858f, 0.0011798064224421978f, 0.0017259916057810187f, 0.0018617215100675821f, 0.0014524785801768303f, 0.00242733396589756f, 0.0018708623247221112f, 0.0022076505701988935f, 0.0018953346880152822f, 0.0016817683354020119f, 0.002966401632875204f, 0.002065876731649041f, 0.0025125991087406874f, 0.0012992102419957519f, 0.002124291379004717f, 0.0013416492147371173f, 0.001495071337558329f, 0.0021976165007799864f, 0.0012820923002436757f, 0.0021688479464501143f, 0.0014839390059933066f, 0.002143222838640213f, 0.001633038162253797f, 0.001302610500715673f, 0.001995543483644724f, 0.00233657518401742f, 0.0016571710584685206f, 0.001569820218719542f, 0.0021191900596022606f, 0.0015222531510517001f, 0.001844649901613593f, 0.0015491486992686987f, 0.0014527598395943642f, 0.0015139353927224874f, 0.001708788680844009f, 0.0014151857467368245f, 0.0016588567523285747f, 0.0014332227874547243f, 0.0013631178298965096f, 0.0015797873493283987f, 0.0015890899812802672f, 0.0018675483297556639f, 0.001717966515570879f, 0.0017174732638522983f, 0.0016462230123579502f, 0.0018060895381495357f, 0.003332067746669054f, 0.0016053235158324242f, 0.0017040972597897053f, 0.002723077079281211f, 0.0012517248978838325f, 0.0021202736534178257f, 0.001526174833998084f, 0.0019667590968310833f, 0.002738490467891097f, 0.0014875080669298768f, 0.001979659777134657f, 0.001304160337895155f, 0.001946910284459591f, 0.0026961311232298613f, 0.001480672974139452f, 0.002701185178011656f, 0.0021946216002106667f, 0.0019520691130310297f, 0.0017249047523364425f, 0.0017484609270468354f, 0.0018351954640820622f, 0.0020590415224432945f, 0.001921854098327458f, 0.0017784836236387491f, 0.0012781015830114484f, 0.0017575530800968409f, 0.0014903040137141943f, 0.0015592395793646574f, 0.001590445637702942f, 0.0016778623685240746f, 0.0014417706988751888f, 0.0019068897236138582f, 0.0015499959699809551f, 0.0020348201505839825f, 0.0024894077796489f, 0.0014169617788866162f, 0.001786228152923286f, 0.0015419418923556805f, 0.002325774170458317f, 0.0014412828022614121f, 0.0017119088442996144f, 0.0013801755849272013f, 0.0015352548798546195f, 0.0015373495407402515f, 0.0016755281249061227f, 0.0020740532781928778f, 0.0024804044514894485f, 0.0019990515429526567f, 0.002173011191189289f, 0.0013364610495045781f, 0.0026363886427134275f, 0.002025018911808729f, 0.002146888757124543f, 0.0013470262056216598f, 0.0020390169229358435f, 0.0013222356792539358f, 0.002777751302346587f, 0.0014219062868505716f, 0.0015957839787006378f, 0.002767090918496251f, 0.0012396045494824648f, 0.001723861088976264f);
static const ai_layer_format_type conv2d_25_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_25_t_out_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_25_t_out_0_shape_h_const_u16 = 96;



static const ai_i8 pad_36_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(0);
static const ai_i16 pad_36_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_36_t_in_0_shape_h_const_u32 = 96;

static const ai_u32 pad_36_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 13312;
static const ai_float pad_36_0_conversion_t_in_0_fmt_scale_const_f32 = 0.030483368784189224f;
static const ai_i8 pad_36_0_conversion_t_in_0_fmt_zero_const_s8 = 0;

static const ai_u32 conv2d_38_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_38_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_38_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_38_t_in_0_shape_h_const_u32 = 104;
static const ai_u32 conv2d_38_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_38_t_out_0_shape_h_const_u32 = 96;
static const ai_u32 conv2d_38_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_38_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_38_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_38_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_38_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_38_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_38_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_38_l_dilation_H_const_u16 = 2;
static const ai_size conv2d_38_v_n_groups_const_size = 1;

static const ai_u32 conv2d_38_0_0_eltwise_40_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12288;
static const ai_float conv2d_38_0_0_eltwise_40_conversion_t_out_0_fmt_scale_const_f32 = 0.041260287165641785f;
static const ai_i8 conv2d_38_0_0_eltwise_40_conversion_t_out_0_fmt_zero_const_s8 = 77;



static const ai_i8 pad_50_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_50_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_50_t_in_0_shape_h_const_u32 = 96;

static const ai_u32 pad_50_0_0_conv2d_52_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 13312;
static const ai_float pad_50_0_0_conv2d_52_conversion_t_in_0_fmt_scale_const_f32 = 0.007900108583271503f;
static const ai_i8 pad_50_0_0_conv2d_52_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 conv2d_52_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_52_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_52_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_52_t_in_0_shape_h_const_u32 = 104;
static const ai_u32 conv2d_52_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_52_t_out_0_shape_h_const_u32 = 96;
static const ai_u32 conv2d_52_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_52_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_52_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_52_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_52_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_52_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_52_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_52_l_dilation_H_const_u16 = 2;
static const ai_size conv2d_52_v_n_groups_const_size = 1;

static const ai_u32 conv2d_52_0_0_eltwise_54_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12288;
static const ai_float conv2d_52_0_0_eltwise_54_conversion_t_out_0_fmt_scale_const_f32 = 0.015926923602819443f;
static const ai_i8 conv2d_52_0_0_eltwise_54_conversion_t_out_0_fmt_zero_const_s8 = 43;




static const ai_i8 pad_65_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-2);
static const ai_i16 pad_65_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_65_t_in_0_shape_h_const_u32 = 96;

static const ai_u32 pad_65_0_0_conv2d_67_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 14336;
static const ai_float pad_65_0_0_conv2d_67_conversion_t_in_0_fmt_scale_const_f32 = 0.030903693288564682f;
static const ai_i8 pad_65_0_0_conv2d_67_conversion_t_in_0_fmt_zero_const_s8 = -2;

static const ai_u32 conv2d_67_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_67_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_67_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_67_t_in_0_shape_h_const_u32 = 112;
static const ai_u32 conv2d_67_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_67_t_out_0_shape_h_const_u32 = 96;
static const ai_u32 conv2d_67_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_67_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_67_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_67_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_67_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_67_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_67_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_67_l_dilation_H_const_u16 = 4;
static const ai_size conv2d_67_v_n_groups_const_size = 1;

static const ai_u32 conv2d_67_0_0_eltwise_69_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12288;
static const ai_float conv2d_67_0_0_eltwise_69_conversion_t_out_0_fmt_scale_const_f32 = 0.05199640244245529f;
static const ai_i8 conv2d_67_0_0_eltwise_69_conversion_t_out_0_fmt_zero_const_s8 = 77;



static const ai_i8 pad_79_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 pad_79_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 pad_79_t_in_0_shape_h_const_u32 = 96;

static const ai_u32 pad_79_0_0_conv2d_81_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 14336;
static const ai_float pad_79_0_0_conv2d_81_conversion_t_in_0_fmt_scale_const_f32 = 0.01041225902736187f;
static const ai_i8 pad_79_0_0_conv2d_81_conversion_t_in_0_fmt_zero_const_s8 = -128;

static const ai_u32 conv2d_81_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_81_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv2d_81_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_81_t_in_0_shape_h_const_u32 = 112;
static const ai_u32 conv2d_81_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_81_t_out_0_shape_h_const_u32 = 96;
static const ai_u32 conv2d_81_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv2d_81_t_weight_0_shape_h_const_u32 = 5;
static const ai_i32 conv2d_81_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_81_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv2d_81_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_81_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_81_l_dilation_W_const_u16 = 1;
static const ai_u16 conv2d_81_l_dilation_H_const_u16 = 4;
static const ai_size conv2d_81_v_n_groups_const_size = 1;

static const ai_u32 conv2d_81_0_0_eltwise_83_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32 = 12288;
static const ai_float conv2d_81_0_0_eltwise_83_conversion_t_out_0_fmt_scale_const_f32 = 0.02176525816321373f;
static const ai_i8 conv2d_81_0_0_eltwise_83_conversion_t_out_0_fmt_zero_const_s8 = 97;




static const ai_u16 conv2d_94_t_in_0_shape_w_const_u16 = 1;
static const ai_u16 conv2d_94_t_in_0_shape_h_const_u16 = 96;
static const ai_u16 conv2d_94_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_94_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_94_t_in_0_shape_ch_const_u16 = 128;
static const ai_u16 conv2d_94_t_out_0_shape_ch_const_u16 = 64;
static const ai_i8 conv2d_94_t_in_0_fmt_zero_const_s8 = -6;
static const ai_i8 conv2d_94_t_out_0_fmt_zero_const_s8 = 51;
static const ai_float conv2d_94_t_in_0_fmt_scale_const_f32 = 0.03198061138391495f;
static const ai_float conv2d_94_t_out_0_fmt_scale_const_f32 = 0.042549774050712585f;
static const ai_float conv2d_94_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0035053121391683817f, 0.003660228569060564f, 0.0032936071511358023f, 0.003816637909039855f, 0.0037086354568600655f, 0.0037615501787513494f, 0.0036503246519714594f, 0.004091455135494471f, 0.003920160233974457f, 0.0035947917494922876f, 0.003168283263221383f, 0.003471744479611516f, 0.0034241406247019768f, 0.005045603960752487f, 0.0030802458059042692f, 0.0040162093937397f, 0.0045229606330394745f, 0.004071826580911875f, 0.004418472293764353f, 0.0036068358458578587f, 0.003740784013643861f, 0.004338604398071766f, 0.004197762347757816f, 0.003985375631600618f, 0.004028848838061094f, 0.00335231376811862f, 0.004453236237168312f, 0.0039688399992883205f, 0.003533621784299612f, 0.003917399328202009f, 0.004425687715411186f, 0.0036091632209718227f, 0.003604768542572856f, 0.004382882732897997f, 0.004163295961916447f, 0.004003490321338177f, 0.0039606778882443905f, 0.004203165415674448f, 0.004294476471841335f, 0.004348202608525753f, 0.0036207528319209814f, 0.004955110605806112f, 0.003265700303018093f, 0.004001866094768047f, 0.004880268592387438f, 0.0038234940730035305f, 0.003665521275252104f, 0.003665859578177333f, 0.003698447486385703f, 0.0037065723445266485f, 0.0039878664538264275f, 0.0033446764573454857f, 0.004583141766488552f, 0.0038470211438834667f, 0.003999179229140282f, 0.004638860002160072f, 0.004901114851236343f, 0.0035529774613678455f, 0.004516227636486292f, 0.0035554084461182356f, 0.003741584252566099f, 0.0033664347138255835f, 0.0048542930744588375f, 0.005050512030720711f);
static const ai_layer_format_type conv2d_94_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


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


  /* LITE_KERNEL_SECTION BEGIN conv2d_4 */
  {
      const ai_i8* conv2d_4_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* conv2d_4_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[4] + 0);
    const ai_i32* conv2d_4_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[5] + 0);
    ai_i8* conv2d_4_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 65872);
    ai_i16* conv2d_4_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 1, {(stai_ptr) conv2d_4_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_4_t_in_0_ptr_const_s8, conv2d_4_t_in_0_shape_w_const_u16, conv2d_4_t_in_0_shape_h_const_u16, conv2d_4_l_stride_1_const_u16, conv2d_4_l_stride_0_const_u16, conv2d_4_t_in_0_shape_ch_const_u16, conv2d_4_t_weight_0_ptr_const_s8, conv2d_4_t_out_0_shape_ch_const_u16, conv2d_4_t_weight_1_ptr_const_s32, conv2d_4_t_in_0_fmt_zero_const_s8, conv2d_4_t_out_0_fmt_zero_const_s8, conv2d_4_t_in_0_fmt_scale_const_f32, conv2d_4_t_out_0_fmt_scale_const_f32, conv2d_4_t_weight_0_fmt_scale_const_f32, conv2d_4_l_out_ch_format_const_layer_format_type, conv2d_4_t_out_0_ptr_s8, 1, 1360, conv2d_4_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) conv2d_4_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_4 */
  /* LITE_KERNEL_SECTION BEGIN pad_13 */
  {
      const ai_ptr pad_13_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 65872);
    ai_ptr pad_13_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 53072);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) pad_13_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_13_t_in_0_ptr_const_ptr, pad_13_t_out_0_ptr_ptr, (ai_handle)(pad_13_v_pad_constant_value_const_s8), pad_13_t_in_0_fmt_bitsize_const_s16, pad_13_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(128), (ai_i32)(512), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) pad_13_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_13 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_14 */
  {
      const ai_i8* conv2d_14_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 53072);
    const ai_i8* conv2d_14_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[6] + 0);
    const ai_i32* conv2d_14_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[7] + 0);
    ai_i8* conv2d_14_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 52944);
    ai_i16* conv2d_14_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1360);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(14, 1, {(stai_ptr) conv2d_14_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(conv2d_14_t_in_0_ptr_const_s8, conv2d_14_t_in_0_shape_w_const_u16, conv2d_14_t_in_0_shape_h_const_u16, conv2d_14_t_in_0_shape_ch_const_u16, conv2d_14_t_weight_0_ptr_const_s8, conv2d_14_t_out_0_shape_ch_const_u16, conv2d_14_t_weight_0_shape_w_const_u16, conv2d_14_t_weight_0_shape_h_const_u16, conv2d_14_l_stride_1_const_u16, conv2d_14_l_stride_0_const_u16, conv2d_14_t_weight_1_ptr_const_s32, conv2d_14_t_in_0_fmt_zero_const_s8, conv2d_14_t_out_0_fmt_zero_const_s8, conv2d_14_t_in_0_fmt_scale_const_f32, conv2d_14_t_out_0_fmt_scale_const_f32, conv2d_14_t_weight_0_fmt_scale_const_f32, conv2d_14_l_out_ch_format_const_layer_format_type, conv2d_14_t_out_0_ptr_s8, conv2d_14_t_out_0_shape_w_const_u16, conv2d_14_t_out_0_shape_h_const_u16, 1, 1, 9472, conv2d_14_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) conv2d_14_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_14 */
  /* LITE_KERNEL_SECTION BEGIN nl_18 */
  {
    
  forward_lite_nl_integer_nl_18(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_18 */
  /* LITE_KERNEL_SECTION BEGIN pad_24 */
  {
      const ai_ptr pad_24_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 52944);
    ai_ptr pad_24_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 52432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, {(stai_ptr) pad_24_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_24_t_in_0_ptr_const_ptr, pad_24_t_out_0_ptr_ptr, (ai_handle)(pad_24_v_pad_constant_value_const_s8), pad_24_t_in_0_fmt_bitsize_const_s16, pad_24_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(128), (ai_i32)(512), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, {(stai_ptr) pad_24_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_24 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_25 */
  {
      const ai_i8* conv2d_25_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 52432);
    const ai_i8* conv2d_25_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[8] + 0);
    const ai_i32* conv2d_25_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[9] + 0);
    ai_i8* conv2d_25_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 52304);
    ai_i16* conv2d_25_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 10832);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, {(stai_ptr) conv2d_25_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_deep_sssa8_ch(conv2d_25_t_in_0_ptr_const_s8, conv2d_25_t_in_0_shape_w_const_u16, conv2d_25_t_in_0_shape_h_const_u16, conv2d_25_t_in_0_shape_ch_const_u16, conv2d_25_t_weight_0_ptr_const_s8, conv2d_25_t_out_0_shape_ch_const_u16, conv2d_25_t_weight_0_shape_w_const_u16, conv2d_25_t_weight_0_shape_h_const_u16, conv2d_25_l_stride_1_const_u16, conv2d_25_l_stride_0_const_u16, conv2d_25_t_weight_1_ptr_const_s32, conv2d_25_t_in_0_fmt_zero_const_s8, conv2d_25_t_out_0_fmt_zero_const_s8, conv2d_25_t_in_0_fmt_scale_const_f32, conv2d_25_t_out_0_fmt_scale_const_f32, conv2d_25_t_weight_0_fmt_scale_const_f32, conv2d_25_l_out_ch_format_const_layer_format_type, conv2d_25_t_out_0_ptr_s8, conv2d_25_t_out_0_shape_w_const_u16, conv2d_25_t_out_0_shape_h_const_u16, 1, 1, 9472, conv2d_25_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, {(stai_ptr) conv2d_25_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_25 */
  /* LITE_KERNEL_SECTION BEGIN nl_29 */
  {
    
  forward_lite_nl_integer_nl_29(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_29 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_31 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_31(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_31 */
  /* LITE_KERNEL_SECTION BEGIN pad_36 */
  {
      const ai_ptr pad_36_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 78160);
    ai_ptr pad_36_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 64848);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 1, {(stai_ptr) pad_36_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_36_t_in_0_ptr_const_ptr, pad_36_t_out_0_ptr_ptr, (ai_handle)(pad_36_v_pad_constant_value_const_s8), pad_36_t_in_0_fmt_bitsize_const_s16, pad_36_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(128), (ai_i32)(1024), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, {(stai_ptr) pad_36_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_36 */
  /* LITE_KERNEL_SECTION BEGIN pad_36_0_conversion */
  {
      const ai_i8* pad_36_0_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 64848);
    ai_float* pad_36_0_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 24912);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 1, {(stai_ptr) pad_36_0_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_36_0_conversion_t_in_0_ptr_const_s8, pad_36_0_conversion_t_out_0_ptr_f32, pad_36_0_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_36_0_conversion_t_in_0_fmt_scale_const_f32, pad_36_0_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, {(stai_ptr) pad_36_0_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_36_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_38 */
  {
      const ai_float* conv2d_38_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 24912);
    ai_float* conv2d_38_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 24400);
    const ai_u8* conv2d_38_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[10] + 0);
    const ai_u8* conv2d_38_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[11] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(39, 1, {(stai_ptr) conv2d_38_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_38_t_in_0_ptr_const_f32, conv2d_38_t_out_0_ptr_f32, conv2d_38_t_weight_0_ptr_const_u8, conv2d_38_t_weight_1_ptr_const_u8, conv2d_38_t_in_0_shape_ch_const_u32, conv2d_38_t_out_0_shape_ch_const_u32, conv2d_38_t_in_0_shape_w_const_u32, conv2d_38_t_in_0_shape_h_const_u32, conv2d_38_t_out_0_shape_w_const_u32, conv2d_38_t_out_0_shape_h_const_u32, conv2d_38_t_weight_0_shape_w_const_u32, conv2d_38_t_weight_0_shape_h_const_u32, conv2d_38_l_pad_W_0_const_s32, conv2d_38_l_pad_H_0_const_s32, conv2d_38_l_stride_1_const_u16, conv2d_38_l_stride_0_const_u16, 9, 1, conv2d_38_l_dilation_W_const_u16, conv2d_38_l_dilation_H_const_u16, conv2d_38_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(39, 1, {(stai_ptr) conv2d_38_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_38 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_38_0_0_eltwise_40_conversion */
  {
      const ai_float* conv2d_38_0_0_eltwise_40_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 24400);
    ai_i8* conv2d_38_0_0_eltwise_40_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 24400);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(39, 1, {(stai_ptr) conv2d_38_0_0_eltwise_40_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_38_0_0_eltwise_40_conversion_t_in_0_ptr_const_f32, conv2d_38_0_0_eltwise_40_conversion_t_out_0_ptr_s8, conv2d_38_0_0_eltwise_40_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_38_0_0_eltwise_40_conversion_t_out_0_fmt_scale_const_f32, conv2d_38_0_0_eltwise_40_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(39, 1, {(stai_ptr) conv2d_38_0_0_eltwise_40_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_38_0_0_eltwise_40_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_40 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_40(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_40 */
  /* LITE_KERNEL_SECTION BEGIN nl_44 */
  {
    
  forward_lite_nl_integer_nl_44(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_44 */
  /* LITE_KERNEL_SECTION BEGIN pad_50 */
  {
      const ai_ptr pad_50_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 24400);
    ai_ptr pad_50_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 64848);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(50, 1, {(stai_ptr) pad_50_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_50_t_in_0_ptr_const_ptr, pad_50_t_out_0_ptr_ptr, (ai_handle)(pad_50_v_pad_constant_value_const_s8), pad_50_t_in_0_fmt_bitsize_const_s16, pad_50_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(128), (ai_i32)(1024), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(50, 1, {(stai_ptr) pad_50_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_50 */
  /* LITE_KERNEL_SECTION BEGIN pad_50_0_0_conv2d_52_conversion */
  {
      const ai_i8* pad_50_0_0_conv2d_52_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 64848);
    ai_float* pad_50_0_0_conv2d_52_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 24912);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(50, 1, {(stai_ptr) pad_50_0_0_conv2d_52_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_50_0_0_conv2d_52_conversion_t_in_0_ptr_const_s8, pad_50_0_0_conv2d_52_conversion_t_out_0_ptr_f32, pad_50_0_0_conv2d_52_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_50_0_0_conv2d_52_conversion_t_in_0_fmt_scale_const_f32, pad_50_0_0_conv2d_52_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(50, 1, {(stai_ptr) pad_50_0_0_conv2d_52_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_50_0_0_conv2d_52_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_52 */
  {
      const ai_float* conv2d_52_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 24912);
    ai_float* conv2d_52_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 24400);
    const ai_u8* conv2d_52_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[12] + 0);
    const ai_u8* conv2d_52_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[11] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(53, 1, {(stai_ptr) conv2d_52_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_52_t_in_0_ptr_const_f32, conv2d_52_t_out_0_ptr_f32, conv2d_52_t_weight_0_ptr_const_u8, conv2d_52_t_weight_1_ptr_const_u8, conv2d_52_t_in_0_shape_ch_const_u32, conv2d_52_t_out_0_shape_ch_const_u32, conv2d_52_t_in_0_shape_w_const_u32, conv2d_52_t_in_0_shape_h_const_u32, conv2d_52_t_out_0_shape_w_const_u32, conv2d_52_t_out_0_shape_h_const_u32, conv2d_52_t_weight_0_shape_w_const_u32, conv2d_52_t_weight_0_shape_h_const_u32, conv2d_52_l_pad_W_0_const_s32, conv2d_52_l_pad_H_0_const_s32, conv2d_52_l_stride_1_const_u16, conv2d_52_l_stride_0_const_u16, 9, 1, conv2d_52_l_dilation_W_const_u16, conv2d_52_l_dilation_H_const_u16, conv2d_52_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(53, 1, {(stai_ptr) conv2d_52_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_52 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_52_0_0_eltwise_54_conversion */
  {
      const ai_float* conv2d_52_0_0_eltwise_54_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 24400);
    ai_i8* conv2d_52_0_0_eltwise_54_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 24400);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(53, 1, {(stai_ptr) conv2d_52_0_0_eltwise_54_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_52_0_0_eltwise_54_conversion_t_in_0_ptr_const_f32, conv2d_52_0_0_eltwise_54_conversion_t_out_0_ptr_s8, conv2d_52_0_0_eltwise_54_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_52_0_0_eltwise_54_conversion_t_out_0_fmt_scale_const_f32, conv2d_52_0_0_eltwise_54_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(53, 1, {(stai_ptr) conv2d_52_0_0_eltwise_54_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_52_0_0_eltwise_54_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_54 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_54(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_54 */
  /* LITE_KERNEL_SECTION BEGIN nl_58 */
  {
    
  forward_lite_nl_integer_nl_58(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_58 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_60 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_60(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_60 */
  /* LITE_KERNEL_SECTION BEGIN pad_65 */
  {
      const ai_ptr pad_65_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 78160);
    ai_ptr pad_65_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 63824);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(65, 1, {(stai_ptr) pad_65_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_65_t_in_0_ptr_const_ptr, pad_65_t_out_0_ptr_ptr, (ai_handle)(pad_65_v_pad_constant_value_const_s8), pad_65_t_in_0_fmt_bitsize_const_s16, pad_65_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(128), (ai_i32)(2048), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(65, 1, {(stai_ptr) pad_65_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_65 */
  /* LITE_KERNEL_SECTION BEGIN pad_65_0_0_conv2d_67_conversion */
  {
      const ai_i8* pad_65_0_0_conv2d_67_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 63824);
    ai_float* pad_65_0_0_conv2d_67_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 20816);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(65, 1, {(stai_ptr) pad_65_0_0_conv2d_67_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_65_0_0_conv2d_67_conversion_t_in_0_ptr_const_s8, pad_65_0_0_conv2d_67_conversion_t_out_0_ptr_f32, pad_65_0_0_conv2d_67_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_65_0_0_conv2d_67_conversion_t_in_0_fmt_scale_const_f32, pad_65_0_0_conv2d_67_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(65, 1, {(stai_ptr) pad_65_0_0_conv2d_67_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_65_0_0_conv2d_67_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_67 */
  {
      const ai_float* conv2d_67_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 20816);
    ai_float* conv2d_67_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 20304);
    const ai_u8* conv2d_67_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[13] + 0);
    const ai_u8* conv2d_67_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[11] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(68, 1, {(stai_ptr) conv2d_67_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_67_t_in_0_ptr_const_f32, conv2d_67_t_out_0_ptr_f32, conv2d_67_t_weight_0_ptr_const_u8, conv2d_67_t_weight_1_ptr_const_u8, conv2d_67_t_in_0_shape_ch_const_u32, conv2d_67_t_out_0_shape_ch_const_u32, conv2d_67_t_in_0_shape_w_const_u32, conv2d_67_t_in_0_shape_h_const_u32, conv2d_67_t_out_0_shape_w_const_u32, conv2d_67_t_out_0_shape_h_const_u32, conv2d_67_t_weight_0_shape_w_const_u32, conv2d_67_t_weight_0_shape_h_const_u32, conv2d_67_l_pad_W_0_const_s32, conv2d_67_l_pad_H_0_const_s32, conv2d_67_l_stride_1_const_u16, conv2d_67_l_stride_0_const_u16, 17, 1, conv2d_67_l_dilation_W_const_u16, conv2d_67_l_dilation_H_const_u16, conv2d_67_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(68, 1, {(stai_ptr) conv2d_67_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_67 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_67_0_0_eltwise_69_conversion */
  {
      const ai_float* conv2d_67_0_0_eltwise_69_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 20304);
    ai_i8* conv2d_67_0_0_eltwise_69_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 20304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(68, 1, {(stai_ptr) conv2d_67_0_0_eltwise_69_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_67_0_0_eltwise_69_conversion_t_in_0_ptr_const_f32, conv2d_67_0_0_eltwise_69_conversion_t_out_0_ptr_s8, conv2d_67_0_0_eltwise_69_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_67_0_0_eltwise_69_conversion_t_out_0_fmt_scale_const_f32, conv2d_67_0_0_eltwise_69_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(68, 1, {(stai_ptr) conv2d_67_0_0_eltwise_69_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_67_0_0_eltwise_69_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_69 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_69(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_69 */
  /* LITE_KERNEL_SECTION BEGIN nl_73 */
  {
    
  forward_lite_nl_integer_nl_73(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_73 */
  /* LITE_KERNEL_SECTION BEGIN pad_79 */
  {
      const ai_ptr pad_79_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 20304);
    ai_ptr pad_79_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 63824);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(79, 1, {(stai_ptr) pad_79_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(pad_79_t_in_0_ptr_const_ptr, pad_79_t_out_0_ptr_ptr, (ai_handle)(pad_79_v_pad_constant_value_const_s8), pad_79_t_in_0_fmt_bitsize_const_s16, pad_79_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(128), (ai_i32)(2048), (ai_i32)(0), (ai_i32)(0), (ai_i32)(0));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(79, 1, {(stai_ptr) pad_79_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END pad_79 */
  /* LITE_KERNEL_SECTION BEGIN pad_79_0_0_conv2d_81_conversion */
  {
      const ai_i8* pad_79_0_0_conv2d_81_conversion_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 63824);
    ai_float* pad_79_0_0_conv2d_81_conversion_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 20816);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(79, 1, {(stai_ptr) pad_79_0_0_conv2d_81_conversion_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(pad_79_0_0_conv2d_81_conversion_t_in_0_ptr_const_s8, pad_79_0_0_conv2d_81_conversion_t_out_0_ptr_f32, pad_79_0_0_conv2d_81_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, pad_79_0_0_conv2d_81_conversion_t_in_0_fmt_scale_const_f32, pad_79_0_0_conv2d_81_conversion_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(79, 1, {(stai_ptr) pad_79_0_0_conv2d_81_conversion_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END pad_79_0_0_conv2d_81_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_81 */
  {
      const ai_float* conv2d_81_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 20816);
    ai_float* conv2d_81_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 20304);
    const ai_u8* conv2d_81_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[14] + 0);
    const ai_u8* conv2d_81_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[11] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(82, 1, {(stai_ptr) conv2d_81_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32_group(conv2d_81_t_in_0_ptr_const_f32, conv2d_81_t_out_0_ptr_f32, conv2d_81_t_weight_0_ptr_const_u8, conv2d_81_t_weight_1_ptr_const_u8, conv2d_81_t_in_0_shape_ch_const_u32, conv2d_81_t_out_0_shape_ch_const_u32, conv2d_81_t_in_0_shape_w_const_u32, conv2d_81_t_in_0_shape_h_const_u32, conv2d_81_t_out_0_shape_w_const_u32, conv2d_81_t_out_0_shape_h_const_u32, conv2d_81_t_weight_0_shape_w_const_u32, conv2d_81_t_weight_0_shape_h_const_u32, conv2d_81_l_pad_W_0_const_s32, conv2d_81_l_pad_H_0_const_s32, conv2d_81_l_stride_1_const_u16, conv2d_81_l_stride_0_const_u16, 17, 1, conv2d_81_l_dilation_W_const_u16, conv2d_81_l_dilation_H_const_u16, conv2d_81_v_n_groups_const_size);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(82, 1, {(stai_ptr) conv2d_81_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv2d_81 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_81_0_0_eltwise_83_conversion */
  {
      const ai_float* conv2d_81_0_0_eltwise_83_conversion_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 20304);
    ai_i8* conv2d_81_0_0_eltwise_83_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 20304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(82, 1, {(stai_ptr) conv2d_81_0_0_eltwise_83_conversion_t_in_0_ptr_const_f32});
    
  forward_lite_node_convert_integer_if32os8(conv2d_81_0_0_eltwise_83_conversion_t_in_0_ptr_const_f32, conv2d_81_0_0_eltwise_83_conversion_t_out_0_ptr_s8, conv2d_81_0_0_eltwise_83_conversion_t_out_0_shape_h_w_ch_d_prod_const_u32, conv2d_81_0_0_eltwise_83_conversion_t_out_0_fmt_scale_const_f32, conv2d_81_0_0_eltwise_83_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(82, 1, {(stai_ptr) conv2d_81_0_0_eltwise_83_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_81_0_0_eltwise_83_conversion */
  /* LITE_KERNEL_SECTION BEGIN eltwise_83 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_83(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_83 */
  /* LITE_KERNEL_SECTION BEGIN nl_87 */
  {
    
  forward_lite_nl_integer_nl_87(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_87 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_89 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_89(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_89 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_94 */
  {
      const ai_i8* conv2d_94_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 32592);
    const ai_i8* conv2d_94_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[15] + 0);
    const ai_i32* conv2d_94_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[16] + 0);
    ai_i8* conv2d_94_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 20304);
    ai_i16* conv2d_94_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(94, 1, {(stai_ptr) conv2d_94_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_94_t_in_0_ptr_const_s8, conv2d_94_t_in_0_shape_w_const_u16, conv2d_94_t_in_0_shape_h_const_u16, conv2d_94_l_stride_1_const_u16, conv2d_94_l_stride_0_const_u16, conv2d_94_t_in_0_shape_ch_const_u16, conv2d_94_t_weight_0_ptr_const_s8, conv2d_94_t_out_0_shape_ch_const_u16, conv2d_94_t_weight_1_ptr_const_s32, conv2d_94_t_in_0_fmt_zero_const_s8, conv2d_94_t_out_0_fmt_zero_const_s8, conv2d_94_t_in_0_fmt_scale_const_f32, conv2d_94_t_out_0_fmt_scale_const_f32, conv2d_94_t_weight_0_fmt_scale_const_f32, conv2d_94_l_out_ch_format_const_layer_format_type, conv2d_94_t_out_0_ptr_s8, 1, 1152, conv2d_94_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(94, 1, {(stai_ptr) conv2d_94_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_94 */
  /* LITE_KERNEL_SECTION BEGIN nl_98 */
  {
    
  forward_lite_nl_integer_nl_98(net_ctx);
  }
  /* LITE_KERNEL_SECTION END nl_98 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_104 */
  {
    
  forward_lite_conv2d_integer_SSSA_conv2d_104(net_ctx);
  }
  /* LITE_KERNEL_SECTION END conv2d_104 */
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
  net_ctx->_inputs[0] = activations[0] + 78160;

  net_ctx->_outputs[0] = activations[0] + 20304;
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

