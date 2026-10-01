/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-05-19T08:43:57+0000
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
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
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[64]) {
   { .size_bytes = 2400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 120, 20}}, .scale = {1, (const float[1]){0.10564419627189636}}, .zeropoint = {1, (const int16_t[1]){45}}, .name = "serving_default_input0_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "conv2d_22_output" },
   { .size_bytes = 2480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 124, 1, 20}}, .scale = {1, (const float[1]){0.10564419627189636}}, .zeropoint = {1, (const int16_t[1]){45}}, .name = "pad_4_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.041041214019060135}}, .zeropoint = {1, (const int16_t[1]){41}}, .name = "conv2d_5_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.013773981481790543}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_9_output" },
   { .size_bytes = 11904, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 124, 1, 96}}, .scale = {1, (const float[1]){0.013773981481790543}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_15_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.02312243916094303}}, .zeropoint = {1, (const int16_t[1]){31}}, .name = "conv2d_16_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.00866710301488638}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_20_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 120, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "eltwise_27_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 128, 1, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "pad_32_output" },
   { .size_bytes = 49152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 128, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_32_0_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_34_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.05669168010354042}}, .zeropoint = {1, (const int16_t[1]){79}}, .name = "conv2d_34_0_0_eltwise_36_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.05652033910155296}}, .zeropoint = {1, (const int16_t[1]){79}}, .name = "eltwise_36_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.010546959936618805}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_40_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 128, 1, 96}}, .scale = {1, (const float[1]){0.010546959936618805}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_46_output" },
   { .size_bytes = 49152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 128, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_46_0_0_conv2d_48_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_48_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.029165739193558693}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_48_0_0_eltwise_50_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.029433224350214005}}, .zeropoint = {1, (const int16_t[1]){7}}, .name = "eltwise_50_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.013859591446816921}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_54_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 120, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "eltwise_56_output" },
   { .size_bytes = 13056, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 136, 1, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "pad_61_output" },
   { .size_bytes = 52224, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 136, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_61_0_0_conv2d_63_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_63_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.058796655386686325}}, .zeropoint = {1, (const int16_t[1]){29}}, .name = "conv2d_63_0_0_eltwise_65_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.05876212194561958}}, .zeropoint = {1, (const int16_t[1]){28}}, .name = "eltwise_65_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.022744812071323395}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_69_output" },
   { .size_bytes = 13056, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 136, 1, 96}}, .scale = {1, (const float[1]){0.022744812071323395}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_75_output" },
   { .size_bytes = 52224, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 136, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_75_0_0_conv2d_77_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_77_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.030611073598265648}}, .zeropoint = {1, (const int16_t[1]){19}}, .name = "conv2d_77_0_0_eltwise_79_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.030895592644810677}}, .zeropoint = {1, (const int16_t[1]){20}}, .name = "eltwise_79_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.013009907677769661}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_83_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 120, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "eltwise_85_output" },
   { .size_bytes = 14592, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 152, 1, 96}}, .scale = {1, (const float[1]){0.05705142021179199}}, .zeropoint = {1, (const int16_t[1]){-20}}, .name = "pad_90_output" },
   { .size_bytes = 58368, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 152, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_90_0_0_conv2d_92_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_92_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.07281295955181122}}, .zeropoint = {1, (const int16_t[1]){43}}, .name = "conv2d_92_0_0_eltwise_94_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.0728796049952507}}, .zeropoint = {1, (const int16_t[1]){43}}, .name = "eltwise_94_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.024060485884547234}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_98_output" },
   { .size_bytes = 14592, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 152, 1, 96}}, .scale = {1, (const float[1]){0.024060485884547234}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_104_output" },
   { .size_bytes = 58368, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 152, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_104_0_0_conv2d_106_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_106_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.029464116320014}}, .zeropoint = {1, (const int16_t[1]){2}}, .name = "conv2d_106_0_0_eltwise_108_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.029880983754992485}}, .zeropoint = {1, (const int16_t[1]){1}}, .name = "eltwise_108_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.01472543366253376}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_112_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 120, 96}}, .scale = {1, (const float[1]){0.06278161704540253}}, .zeropoint = {1, (const int16_t[1]){-29}}, .name = "eltwise_114_output" },
   { .size_bytes = 17664, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 184, 1, 96}}, .scale = {1, (const float[1]){0.06278161704540253}}, .zeropoint = {1, (const int16_t[1]){-29}}, .name = "pad_119_output" },
   { .size_bytes = 70656, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 184, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_119_0_0_conv2d_121_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_121_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.10135211050510406}}, .zeropoint = {1, (const int16_t[1]){16}}, .name = "conv2d_121_0_0_eltwise_123_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.07416337728500366}}, .zeropoint = {1, (const int16_t[1]){69}}, .name = "eltwise_123_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.016977179795503616}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_127_output" },
   { .size_bytes = 17664, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 184, 1, 96}}, .scale = {1, (const float[1]){0.016977179795503616}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_133_output" },
   { .size_bytes = 70656, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 184, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_133_0_0_conv2d_135_conversion_output" },
   { .size_bytes = 46080, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_135_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.01921113394200802}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "conv2d_135_0_0_eltwise_137_conversion_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.01973133161664009}}, .zeropoint = {1, (const int16_t[1]){7}}, .name = "eltwise_137_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 96}}, .scale = {1, (const float[1]){0.009253321215510368}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_141_output" },
   { .size_bytes = 11520, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 120, 96}}, .scale = {1, (const float[1]){0.07079243659973145}}, .zeropoint = {1, (const int16_t[1]){-41}}, .name = "eltwise_143_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 128}}, .scale = {1, (const float[1]){0.08966033160686493}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "conv2d_148_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 128}}, .scale = {1, (const float[1]){0.04055215045809746}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_152_output" },
   { .size_bytes = 120, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 120, 1, 1}}, .scale = {1, (const float[1]){0.0050134118646383286}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_158_output" }
  },
  .nodes = (const stai_node_details[63]){
    {.id = 22, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv2d_22 */
    {.id = 4, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* pad_4 */
    {.id = 5, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_5 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* nl_9 */
    {.id = 15, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* pad_15 */
    {.id = 16, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* conv2d_16 */
    {.id = 20, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* nl_20 */
    {.id = 27, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){7, 1}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* eltwise_27 */
    {.id = 32, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* pad_32 */
    {.id = 32, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* pad_32_0_conversion */
    {.id = 35, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* conv2d_34 */
    {.id = 35, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* conv2d_34_0_0_eltwise_36_conversion */
    {.id = 36, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* eltwise_36 */
    {.id = 40, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* nl_40 */
    {.id = 46, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* pad_46 */
    {.id = 46, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* pad_46_0_0_conv2d_48_conversion */
    {.id = 49, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* conv2d_48 */
    {.id = 49, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* conv2d_48_0_0_eltwise_50_conversion */
    {.id = 50, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* eltwise_50 */
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* nl_54 */
    {.id = 56, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){20, 8}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* eltwise_56 */
    {.id = 61, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* pad_61 */
    {.id = 61, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* pad_61_0_0_conv2d_63_conversion */
    {.id = 64, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* conv2d_63 */
    {.id = 64, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* conv2d_63_0_0_eltwise_65_conversion */
    {.id = 65, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* eltwise_65 */
    {.id = 69, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* nl_69 */
    {.id = 75, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* pad_75 */
    {.id = 75, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* pad_75_0_0_conv2d_77_conversion */
    {.id = 78, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* conv2d_77 */
    {.id = 78, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* conv2d_77_0_0_eltwise_79_conversion */
    {.id = 79, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* eltwise_79 */
    {.id = 83, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* nl_83 */
    {.id = 85, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){33, 21}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* eltwise_85 */
    {.id = 90, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* pad_90 */
    {.id = 90, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* pad_90_0_0_conv2d_92_conversion */
    {.id = 93, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} }, /* conv2d_92 */
    {.id = 93, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){37}}, .output_tensors = {1, (const int32_t[1]){38}} }, /* conv2d_92_0_0_eltwise_94_conversion */
    {.id = 94, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){38}}, .output_tensors = {1, (const int32_t[1]){39}} }, /* eltwise_94 */
    {.id = 98, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){39}}, .output_tensors = {1, (const int32_t[1]){40}} }, /* nl_98 */
    {.id = 104, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){40}}, .output_tensors = {1, (const int32_t[1]){41}} }, /* pad_104 */
    {.id = 104, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){41}}, .output_tensors = {1, (const int32_t[1]){42}} }, /* pad_104_0_0_conv2d_106_conversion */
    {.id = 107, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){42}}, .output_tensors = {1, (const int32_t[1]){43}} }, /* conv2d_106 */
    {.id = 107, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){43}}, .output_tensors = {1, (const int32_t[1]){44}} }, /* conv2d_106_0_0_eltwise_108_conversion */
    {.id = 108, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){44}}, .output_tensors = {1, (const int32_t[1]){45}} }, /* eltwise_108 */
    {.id = 112, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){45}}, .output_tensors = {1, (const int32_t[1]){46}} }, /* nl_112 */
    {.id = 114, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){46, 34}}, .output_tensors = {1, (const int32_t[1]){47}} }, /* eltwise_114 */
    {.id = 119, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){47}}, .output_tensors = {1, (const int32_t[1]){48}} }, /* pad_119 */
    {.id = 119, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){48}}, .output_tensors = {1, (const int32_t[1]){49}} }, /* pad_119_0_0_conv2d_121_conversion */
    {.id = 122, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){49}}, .output_tensors = {1, (const int32_t[1]){50}} }, /* conv2d_121 */
    {.id = 122, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){50}}, .output_tensors = {1, (const int32_t[1]){51}} }, /* conv2d_121_0_0_eltwise_123_conversion */
    {.id = 123, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){51}}, .output_tensors = {1, (const int32_t[1]){52}} }, /* eltwise_123 */
    {.id = 127, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){52}}, .output_tensors = {1, (const int32_t[1]){53}} }, /* nl_127 */
    {.id = 133, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){53}}, .output_tensors = {1, (const int32_t[1]){54}} }, /* pad_133 */
    {.id = 133, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){54}}, .output_tensors = {1, (const int32_t[1]){55}} }, /* pad_133_0_0_conv2d_135_conversion */
    {.id = 136, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){55}}, .output_tensors = {1, (const int32_t[1]){56}} }, /* conv2d_135 */
    {.id = 136, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){56}}, .output_tensors = {1, (const int32_t[1]){57}} }, /* conv2d_135_0_0_eltwise_137_conversion */
    {.id = 137, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){57}}, .output_tensors = {1, (const int32_t[1]){58}} }, /* eltwise_137 */
    {.id = 141, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){58}}, .output_tensors = {1, (const int32_t[1]){59}} }, /* nl_141 */
    {.id = 143, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){59, 47}}, .output_tensors = {1, (const int32_t[1]){60}} }, /* eltwise_143 */
    {.id = 148, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){60}}, .output_tensors = {1, (const int32_t[1]){61}} }, /* conv2d_148 */
    {.id = 152, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){61}}, .output_tensors = {1, (const int32_t[1]){62}} }, /* nl_152 */
    {.id = 158, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){62}}, .output_tensors = {1, (const int32_t[1]){63}} } /* conv2d_158 */
  },
  .n_nodes = 63
};
#endif

