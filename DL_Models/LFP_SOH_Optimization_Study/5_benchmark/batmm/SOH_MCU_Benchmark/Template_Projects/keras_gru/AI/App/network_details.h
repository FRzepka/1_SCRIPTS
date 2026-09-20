/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-05-20T11:09:32+0000
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
  .tensors = (const stai_tensor[157]) {
   { .size_bytes = 20, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 20}}, .scale = {1, (const float[1]){0.05971222743391991}}, .zeropoint = {1, (const int16_t[1]){37}}, .name = "serving_default_input_x0_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.005003895610570908}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "serving_default_h_in_00_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.04996602237224579}}, .zeropoint = {1, (const int16_t[1]){13}}, .name = "serving_default_c_in_00_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.007734459359198809}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "serving_default_h_in_10_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.09746155887842178}}, .zeropoint = {1, (const int16_t[1]){19}}, .name = "serving_default_c_in_10_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.021409345790743828}}, .zeropoint = {1, (const int16_t[1]){-30}}, .name = "gemm_35_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.005983965005725622}}, .zeropoint = {1, (const int16_t[1]){6}}, .name = "gemm_18_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.03652653470635414}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "gemm_0_output" },
   { .size_bytes = 640, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_0_0_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_1_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_1_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.0015480673173442483}}, .zeropoint = {1, (const int16_t[1]){67}}, .name = "reduce_1_Mul_0_0_conversion_2_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_1_Mul_0_0_conversion_2_conversion_0_1_eltwise_5_conversion_output" },
   { .size_bytes = 640, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_5_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_6_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_6_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.013643942773342133}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "reduce_6_Mul_0_0_eltwise_7_conversion_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.013643981888890266}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_7_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.014291660860180855}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_8_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.014466660097241402}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_9_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.02564668282866478}}, .zeropoint = {1, (const int16_t[1]){-4}}, .name = "eltwise_12_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_2_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_3_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.0015480673173442483}}, .zeropoint = {1, (const int16_t[1]){-68}}, .name = "conversion_4_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.0016923407092690468}}, .zeropoint = {1, (const int16_t[1]){14}}, .name = "eltwise_10_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.0016923407092690468}}, .zeropoint = {1, (const int16_t[1]){14}}, .name = "eltwise_11_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.024792253971099854}}, .zeropoint = {1, (const int16_t[1]){-2}}, .name = "eltwise_13_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.013215373270213604}}, .zeropoint = {1, (const int16_t[1]){-115}}, .name = "nl_14_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.01941734179854393}}, .zeropoint = {1, (const int16_t[1]){-3}}, .name = "gemm_15_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.010531038045883179}}, .zeropoint = {1, (const int16_t[1]){-112}}, .name = "nl_16_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 160}}, .scale = {1, (const float[1]){0.010531038045883179}}, .zeropoint = {1, (const int16_t[1]){-112}}, .name = "unpack_17_output0" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.01144704781472683}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "gemm_20_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.019391372799873352}}, .zeropoint = {1, (const int16_t[1]){-49}}, .name = "eltwise_21_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.019391372799873352}}, .zeropoint = {1, (const int16_t[1]){-49}}, .name = "eltwise_22_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.019391372799873352}}, .zeropoint = {1, (const int16_t[1]){-49}}, .name = "split_23_output0" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.019391372799873352}}, .zeropoint = {1, (const int16_t[1]){-49}}, .name = "split_23_output1" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.019391372799873352}}, .zeropoint = {1, (const int16_t[1]){-49}}, .name = "split_23_output2" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.019391372799873352}}, .zeropoint = {1, (const int16_t[1]){-49}}, .name = "split_23_output3" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.0078125}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "nl_28_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_27_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_25_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.046362265944480896}}, .zeropoint = {1, (const int16_t[1]){14}}, .name = "eltwise_26_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_24_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.0045182667672634125}}, .zeropoint = {1, (const int16_t[1]){-17}}, .name = "eltwise_29_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.04996602237224579}}, .zeropoint = {1, (const int16_t[1]){13}}, .name = "eltwise_30_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.0078125}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "nl_31_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.005003895610570908}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "eltwise_32_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.005003895610570908}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "unpack_34_output0" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.027632519602775574}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "gemm_37_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.027896909043192863}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "eltwise_38_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 768}}, .scale = {1, (const float[1]){0.027896909043192863}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "eltwise_39_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.027896909043192863}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "split_40_output0" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.027896909043192863}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "split_40_output1" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.027896909043192863}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "split_40_output2" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.027896909043192863}}, .zeropoint = {1, (const int16_t[1]){-34}}, .name = "split_40_output3" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.0078125}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "nl_45_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_44_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_42_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.09379451721906662}}, .zeropoint = {1, (const int16_t[1]){19}}, .name = "eltwise_43_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.00390625}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_41_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.005379935726523399}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "eltwise_46_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.09746155142784119}}, .zeropoint = {1, (const int16_t[1]){19}}, .name = "eltwise_47_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.0078125}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "nl_48_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {1, (const float[1]){0.007734459359198809}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "eltwise_49_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_49_0_0_reduce_51_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_51_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_51_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.00011819493374787271}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "reduce_51_Mul_0_0_conversion_52_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_51_Mul_0_0_conversion_52_conversion_0_1_eltwise_55_conversion_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_55_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_56_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_56_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.0016360300360247493}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "reduce_56_Mul_0_0_eltwise_57_conversion_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.001636069267988205}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_57_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.21597149968147278}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_58_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.2293166071176529}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_59_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.018836040049791336}}, .zeropoint = {1, (const int16_t[1]){7}}, .name = "eltwise_62_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_52_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_53_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.00011819493374787271}}, .zeropoint = {1, (const int16_t[1]){127}}, .name = "conversion_54_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.0004493551969062537}}, .zeropoint = {1, (const int16_t[1]){127}}, .name = "eltwise_60_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.0004493551969062537}}, .zeropoint = {1, (const int16_t[1]){127}}, .name = "eltwise_61_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.01866452768445015}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "eltwise_63_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.017521385103464127}}, .zeropoint = {1, (const int16_t[1]){58}}, .name = "gemm_64_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.0048801652155816555}}, .zeropoint = {1, (const int16_t[1]){-93}}, .name = "nl_65_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.005223656538873911}}, .zeropoint = {1, (const int16_t[1]){8}}, .name = "gemm_66_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.019948186352849007}}, .zeropoint = {1, (const int16_t[1]){9}}, .name = "eltwise_68_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_68_0_0_reduce_69_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_69_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_69_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){2.9806222300976515e-05}}, .zeropoint = {1, (const int16_t[1]){-92}}, .name = "reduce_69_Mul_0_0_conversion_70_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_69_Mul_0_0_conversion_70_conversion_0_1_eltwise_73_conversion_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_73_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_74_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_74_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.0041339644230902195}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "reduce_74_Mul_0_0_eltwise_75_conversion_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.004134004004299641}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_75_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.0038783985655754805}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_76_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.004137359093874693}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_77_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.019546333700418472}}, .zeropoint = {1, (const int16_t[1]){9}}, .name = "eltwise_80_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_70_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_71_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){2.9806222300976515e-05}}, .zeropoint = {1, (const int16_t[1]){91}}, .name = "conversion_72_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.00023837077606003731}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "eltwise_78_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.00023837077606003731}}, .zeropoint = {1, (const int16_t[1]){5}}, .name = "eltwise_79_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.019546333700418472}}, .zeropoint = {1, (const int16_t[1]){9}}, .name = "eltwise_81_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.017427612096071243}}, .zeropoint = {1, (const int16_t[1]){48}}, .name = "gemm_82_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.005605463404208422}}, .zeropoint = {1, (const int16_t[1]){-98}}, .name = "nl_83_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.005172138102352619}}, .zeropoint = {1, (const int16_t[1]){-15}}, .name = "gemm_84_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.02173658460378647}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "eltwise_86_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_86_0_0_reduce_87_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_87_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_87_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){6.878512795083225e-05}}, .zeropoint = {1, (const int16_t[1]){-102}}, .name = "reduce_87_Mul_0_0_conversion_88_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_87_Mul_0_0_conversion_88_conversion_0_1_eltwise_91_conversion_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_91_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_92_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_92_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.004613934550434351}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "reduce_92_Mul_0_0_eltwise_93_conversion_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.004613974131643772}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_93_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.003864243859425187}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_94_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.004095065873116255}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_95_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.020018214359879494}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "eltwise_98_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_88_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_89_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){6.878512795083225e-05}}, .zeropoint = {1, (const int16_t[1]){101}}, .name = "conversion_90_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.0002874901983886957}}, .zeropoint = {1, (const int16_t[1]){26}}, .name = "eltwise_96_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.0002874901983886957}}, .zeropoint = {1, (const int16_t[1]){26}}, .name = "eltwise_97_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.019971251487731934}}, .zeropoint = {1, (const int16_t[1]){12}}, .name = "eltwise_99_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.016991404816508293}}, .zeropoint = {1, (const int16_t[1]){46}}, .name = "gemm_100_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.005642358213663101}}, .zeropoint = {1, (const int16_t[1]){-98}}, .name = "nl_101_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.005822161678224802}}, .zeropoint = {1, (const int16_t[1]){14}}, .name = "gemm_102_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.025078754872083664}}, .zeropoint = {1, (const int16_t[1]){17}}, .name = "eltwise_104_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_104_0_0_reduce_105_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_105_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_105_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){6.42166705802083e-05}}, .zeropoint = {1, (const int16_t[1]){119}}, .name = "reduce_105_Mul_0_0_conversion_106_conversion_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_105_Mul_0_0_conversion_106_conversion_0_1_eltwise_109_conversion_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_109_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_110_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "reduce_110_Mul_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.005092015955597162}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "reduce_110_Mul_0_0_eltwise_111_conversion_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.005092055071145296}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_111_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.0038652559742331505}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_112_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.003910123836249113}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "eltwise_113_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.021799402311444283}}, .zeropoint = {1, (const int16_t[1]){16}}, .name = "eltwise_116_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conversion_106_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_107_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){6.42166705802083e-05}}, .zeropoint = {1, (const int16_t[1]){-120}}, .name = "conversion_108_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.00015868947957642376}}, .zeropoint = {1, (const int16_t[1]){-36}}, .name = "eltwise_114_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.00015868947957642376}}, .zeropoint = {1, (const int16_t[1]){-36}}, .name = "eltwise_115_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.02175808697938919}}, .zeropoint = {1, (const int16_t[1]){16}}, .name = "eltwise_117_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.018802570179104805}}, .zeropoint = {1, (const int16_t[1]){65}}, .name = "gemm_118_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.004642080049961805}}, .zeropoint = {1, (const int16_t[1]){-91}}, .name = "nl_119_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.0056098573841154575}}, .zeropoint = {1, (const int16_t[1]){41}}, .name = "gemm_120_output" },
   { .size_bytes = 192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 192}}, .scale = {1, (const float[1]){0.001963462447747588}}, .zeropoint = {1, (const int16_t[1]){-41}}, .name = "nl_121_output" },
   { .size_bytes = 1, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 1, 1}}, .scale = {1, (const float[1]){0.004383135586977005}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "gemm_122_output" }
  },
  .nodes = (const stai_node_details[146]){
    {.id = 35, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* gemm_35 */
    {.id = 18, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* gemm_18 */
    {.id = 0, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* gemm_0 */
    {.id = 0, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* gemm_0_0_conversion */
    {.id = 1, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* reduce_1 */
    {.id = 1, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* reduce_1_Mul */
    {.id = 1, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* reduce_1_Mul_0_0_conversion_2_conversion */
    {.id = 1, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* reduce_1_Mul_0_0_conversion_2_conversion_0_1_eltwise_5_conversion */
    {.id = 5, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){8, 12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* eltwise_5 */
    {.id = 6, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* reduce_6 */
    {.id = 6, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* reduce_6_Mul */
    {.id = 6, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* reduce_6_Mul_0_0_eltwise_7_conversion */
    {.id = 7, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* eltwise_7 */
    {.id = 8, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* nl_8 */
    {.id = 9, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* eltwise_9 */
    {.id = 12, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){7, 19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* eltwise_12 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* conversion_2 */
    {.id = 3, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* nl_3 */
    {.id = 4, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* conversion_4 */
    {.id = 10, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){23, 19}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* eltwise_10 */
    {.id = 11, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* eltwise_11 */
    {.id = 13, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){20, 25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* eltwise_13 */
    {.id = 14, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* nl_14 */
    {.id = 15, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* gemm_15 */
    {.id = 16, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* nl_16 */
    {.id = 17, .type = AI_LAYER_UNPACK_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* unpack_17 */
    {.id = 20, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* gemm_20 */
    {.id = 21, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){31, 6}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* eltwise_21 */
    {.id = 22, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* eltwise_22 */
    {.id = 23, .type = AI_LAYER_SPLIT_TYPE, .input_tensors = {1, (const int32_t[1]){33}}, .output_tensors = {4, (const int32_t[4]){34, 35, 36, 37}} }, /* split_23 */
    {.id = 28, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){38}} }, /* nl_28 */
    {.id = 27, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){37}}, .output_tensors = {1, (const int32_t[1]){39}} }, /* nl_27 */
    {.id = 25, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){40}} }, /* nl_25 */
    {.id = 26, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){40, 2}}, .output_tensors = {1, (const int32_t[1]){41}} }, /* eltwise_26 */
    {.id = 24, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){42}} }, /* nl_24 */
    {.id = 29, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){42, 38}}, .output_tensors = {1, (const int32_t[1]){43}} }, /* eltwise_29 */
    {.id = 30, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){41, 43}}, .output_tensors = {1, (const int32_t[1]){44}} }, /* eltwise_30 */
    {.id = 31, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){44}}, .output_tensors = {1, (const int32_t[1]){45}} }, /* nl_31 */
    {.id = 32, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){39, 45}}, .output_tensors = {1, (const int32_t[1]){46}} }, /* eltwise_32 */
    {.id = 34, .type = AI_LAYER_UNPACK_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){47}} }, /* unpack_34 */
    {.id = 37, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){47}}, .output_tensors = {1, (const int32_t[1]){48}} }, /* gemm_37 */
    {.id = 38, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){48, 5}}, .output_tensors = {1, (const int32_t[1]){49}} }, /* eltwise_38 */
    {.id = 39, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){49}}, .output_tensors = {1, (const int32_t[1]){50}} }, /* eltwise_39 */
    {.id = 40, .type = AI_LAYER_SPLIT_TYPE, .input_tensors = {1, (const int32_t[1]){50}}, .output_tensors = {4, (const int32_t[4]){51, 52, 53, 54}} }, /* split_40 */
    {.id = 45, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){53}}, .output_tensors = {1, (const int32_t[1]){55}} }, /* nl_45 */
    {.id = 44, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){54}}, .output_tensors = {1, (const int32_t[1]){56}} }, /* nl_44 */
    {.id = 42, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){52}}, .output_tensors = {1, (const int32_t[1]){57}} }, /* nl_42 */
    {.id = 43, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){57, 4}}, .output_tensors = {1, (const int32_t[1]){58}} }, /* eltwise_43 */
    {.id = 41, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){51}}, .output_tensors = {1, (const int32_t[1]){59}} }, /* nl_41 */
    {.id = 46, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){59, 55}}, .output_tensors = {1, (const int32_t[1]){60}} }, /* eltwise_46 */
    {.id = 47, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){58, 60}}, .output_tensors = {1, (const int32_t[1]){61}} }, /* eltwise_47 */
    {.id = 48, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){61}}, .output_tensors = {1, (const int32_t[1]){62}} }, /* nl_48 */
    {.id = 49, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){56, 62}}, .output_tensors = {1, (const int32_t[1]){63}} }, /* eltwise_49 */
    {.id = 49, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){63}}, .output_tensors = {1, (const int32_t[1]){64}} }, /* eltwise_49_0_0_reduce_51_conversion */
    {.id = 51, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){64}}, .output_tensors = {1, (const int32_t[1]){65}} }, /* reduce_51 */
    {.id = 51, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){65}}, .output_tensors = {1, (const int32_t[1]){66}} }, /* reduce_51_Mul */
    {.id = 51, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){66}}, .output_tensors = {1, (const int32_t[1]){67}} }, /* reduce_51_Mul_0_0_conversion_52_conversion */
    {.id = 51, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){67}}, .output_tensors = {1, (const int32_t[1]){68}} }, /* reduce_51_Mul_0_0_conversion_52_conversion_0_1_eltwise_55_conversion */
    {.id = 55, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){64, 68}}, .output_tensors = {1, (const int32_t[1]){69}} }, /* eltwise_55 */
    {.id = 56, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){69}}, .output_tensors = {1, (const int32_t[1]){70}} }, /* reduce_56 */
    {.id = 56, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){70}}, .output_tensors = {1, (const int32_t[1]){71}} }, /* reduce_56_Mul */
    {.id = 56, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){71}}, .output_tensors = {1, (const int32_t[1]){72}} }, /* reduce_56_Mul_0_0_eltwise_57_conversion */
    {.id = 57, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){72}}, .output_tensors = {1, (const int32_t[1]){73}} }, /* eltwise_57 */
    {.id = 58, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){73}}, .output_tensors = {1, (const int32_t[1]){74}} }, /* nl_58 */
    {.id = 59, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){74}}, .output_tensors = {1, (const int32_t[1]){75}} }, /* eltwise_59 */
    {.id = 62, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){63, 75}}, .output_tensors = {1, (const int32_t[1]){76}} }, /* eltwise_62 */
    {.id = 52, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){67}}, .output_tensors = {1, (const int32_t[1]){77}} }, /* conversion_52 */
    {.id = 53, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){77}}, .output_tensors = {1, (const int32_t[1]){78}} }, /* nl_53 */
    {.id = 54, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){78}}, .output_tensors = {1, (const int32_t[1]){79}} }, /* conversion_54 */
    {.id = 60, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){79, 75}}, .output_tensors = {1, (const int32_t[1]){80}} }, /* eltwise_60 */
    {.id = 61, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){80}}, .output_tensors = {1, (const int32_t[1]){81}} }, /* eltwise_61 */
    {.id = 63, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){76, 81}}, .output_tensors = {1, (const int32_t[1]){82}} }, /* eltwise_63 */
    {.id = 64, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){82}}, .output_tensors = {1, (const int32_t[1]){83}} }, /* gemm_64 */
    {.id = 65, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){83}}, .output_tensors = {1, (const int32_t[1]){84}} }, /* nl_65 */
    {.id = 66, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){84}}, .output_tensors = {1, (const int32_t[1]){85}} }, /* gemm_66 */
    {.id = 68, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){82, 85}}, .output_tensors = {1, (const int32_t[1]){86}} }, /* eltwise_68 */
    {.id = 68, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){86}}, .output_tensors = {1, (const int32_t[1]){87}} }, /* eltwise_68_0_0_reduce_69_conversion */
    {.id = 69, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){87}}, .output_tensors = {1, (const int32_t[1]){88}} }, /* reduce_69 */
    {.id = 69, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){88}}, .output_tensors = {1, (const int32_t[1]){89}} }, /* reduce_69_Mul */
    {.id = 69, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){89}}, .output_tensors = {1, (const int32_t[1]){90}} }, /* reduce_69_Mul_0_0_conversion_70_conversion */
    {.id = 69, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){90}}, .output_tensors = {1, (const int32_t[1]){91}} }, /* reduce_69_Mul_0_0_conversion_70_conversion_0_1_eltwise_73_conversion */
    {.id = 73, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){87, 91}}, .output_tensors = {1, (const int32_t[1]){92}} }, /* eltwise_73 */
    {.id = 74, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){92}}, .output_tensors = {1, (const int32_t[1]){93}} }, /* reduce_74 */
    {.id = 74, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){93}}, .output_tensors = {1, (const int32_t[1]){94}} }, /* reduce_74_Mul */
    {.id = 74, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){94}}, .output_tensors = {1, (const int32_t[1]){95}} }, /* reduce_74_Mul_0_0_eltwise_75_conversion */
    {.id = 75, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){95}}, .output_tensors = {1, (const int32_t[1]){96}} }, /* eltwise_75 */
    {.id = 76, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){96}}, .output_tensors = {1, (const int32_t[1]){97}} }, /* nl_76 */
    {.id = 77, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){97}}, .output_tensors = {1, (const int32_t[1]){98}} }, /* eltwise_77 */
    {.id = 80, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){86, 98}}, .output_tensors = {1, (const int32_t[1]){99}} }, /* eltwise_80 */
    {.id = 70, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){90}}, .output_tensors = {1, (const int32_t[1]){100}} }, /* conversion_70 */
    {.id = 71, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){100}}, .output_tensors = {1, (const int32_t[1]){101}} }, /* nl_71 */
    {.id = 72, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){101}}, .output_tensors = {1, (const int32_t[1]){102}} }, /* conversion_72 */
    {.id = 78, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){102, 98}}, .output_tensors = {1, (const int32_t[1]){103}} }, /* eltwise_78 */
    {.id = 79, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){103}}, .output_tensors = {1, (const int32_t[1]){104}} }, /* eltwise_79 */
    {.id = 81, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){99, 104}}, .output_tensors = {1, (const int32_t[1]){105}} }, /* eltwise_81 */
    {.id = 82, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){105}}, .output_tensors = {1, (const int32_t[1]){106}} }, /* gemm_82 */
    {.id = 83, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){106}}, .output_tensors = {1, (const int32_t[1]){107}} }, /* nl_83 */
    {.id = 84, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){107}}, .output_tensors = {1, (const int32_t[1]){108}} }, /* gemm_84 */
    {.id = 86, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){105, 108}}, .output_tensors = {1, (const int32_t[1]){109}} }, /* eltwise_86 */
    {.id = 86, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){109}}, .output_tensors = {1, (const int32_t[1]){110}} }, /* eltwise_86_0_0_reduce_87_conversion */
    {.id = 87, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){110}}, .output_tensors = {1, (const int32_t[1]){111}} }, /* reduce_87 */
    {.id = 87, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){111}}, .output_tensors = {1, (const int32_t[1]){112}} }, /* reduce_87_Mul */
    {.id = 87, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){112}}, .output_tensors = {1, (const int32_t[1]){113}} }, /* reduce_87_Mul_0_0_conversion_88_conversion */
    {.id = 87, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){113}}, .output_tensors = {1, (const int32_t[1]){114}} }, /* reduce_87_Mul_0_0_conversion_88_conversion_0_1_eltwise_91_conversion */
    {.id = 91, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){110, 114}}, .output_tensors = {1, (const int32_t[1]){115}} }, /* eltwise_91 */
    {.id = 92, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){115}}, .output_tensors = {1, (const int32_t[1]){116}} }, /* reduce_92 */
    {.id = 92, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){116}}, .output_tensors = {1, (const int32_t[1]){117}} }, /* reduce_92_Mul */
    {.id = 92, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){117}}, .output_tensors = {1, (const int32_t[1]){118}} }, /* reduce_92_Mul_0_0_eltwise_93_conversion */
    {.id = 93, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){118}}, .output_tensors = {1, (const int32_t[1]){119}} }, /* eltwise_93 */
    {.id = 94, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){119}}, .output_tensors = {1, (const int32_t[1]){120}} }, /* nl_94 */
    {.id = 95, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){120}}, .output_tensors = {1, (const int32_t[1]){121}} }, /* eltwise_95 */
    {.id = 98, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){109, 121}}, .output_tensors = {1, (const int32_t[1]){122}} }, /* eltwise_98 */
    {.id = 88, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){113}}, .output_tensors = {1, (const int32_t[1]){123}} }, /* conversion_88 */
    {.id = 89, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){123}}, .output_tensors = {1, (const int32_t[1]){124}} }, /* nl_89 */
    {.id = 90, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){124}}, .output_tensors = {1, (const int32_t[1]){125}} }, /* conversion_90 */
    {.id = 96, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){125, 121}}, .output_tensors = {1, (const int32_t[1]){126}} }, /* eltwise_96 */
    {.id = 97, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){126}}, .output_tensors = {1, (const int32_t[1]){127}} }, /* eltwise_97 */
    {.id = 99, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){122, 127}}, .output_tensors = {1, (const int32_t[1]){128}} }, /* eltwise_99 */
    {.id = 100, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){128}}, .output_tensors = {1, (const int32_t[1]){129}} }, /* gemm_100 */
    {.id = 101, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){129}}, .output_tensors = {1, (const int32_t[1]){130}} }, /* nl_101 */
    {.id = 102, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){130}}, .output_tensors = {1, (const int32_t[1]){131}} }, /* gemm_102 */
    {.id = 104, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){128, 131}}, .output_tensors = {1, (const int32_t[1]){132}} }, /* eltwise_104 */
    {.id = 104, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){132}}, .output_tensors = {1, (const int32_t[1]){133}} }, /* eltwise_104_0_0_reduce_105_conversion */
    {.id = 105, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){133}}, .output_tensors = {1, (const int32_t[1]){134}} }, /* reduce_105 */
    {.id = 105, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){134}}, .output_tensors = {1, (const int32_t[1]){135}} }, /* reduce_105_Mul */
    {.id = 105, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){135}}, .output_tensors = {1, (const int32_t[1]){136}} }, /* reduce_105_Mul_0_0_conversion_106_conversion */
    {.id = 105, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){136}}, .output_tensors = {1, (const int32_t[1]){137}} }, /* reduce_105_Mul_0_0_conversion_106_conversion_0_1_eltwise_109_conversion */
    {.id = 109, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){133, 137}}, .output_tensors = {1, (const int32_t[1]){138}} }, /* eltwise_109 */
    {.id = 110, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){138}}, .output_tensors = {1, (const int32_t[1]){139}} }, /* reduce_110 */
    {.id = 110, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){139}}, .output_tensors = {1, (const int32_t[1]){140}} }, /* reduce_110_Mul */
    {.id = 110, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){140}}, .output_tensors = {1, (const int32_t[1]){141}} }, /* reduce_110_Mul_0_0_eltwise_111_conversion */
    {.id = 111, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){141}}, .output_tensors = {1, (const int32_t[1]){142}} }, /* eltwise_111 */
    {.id = 112, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){142}}, .output_tensors = {1, (const int32_t[1]){143}} }, /* nl_112 */
    {.id = 113, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){143}}, .output_tensors = {1, (const int32_t[1]){144}} }, /* eltwise_113 */
    {.id = 116, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){132, 144}}, .output_tensors = {1, (const int32_t[1]){145}} }, /* eltwise_116 */
    {.id = 106, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){136}}, .output_tensors = {1, (const int32_t[1]){146}} }, /* conversion_106 */
    {.id = 107, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){146}}, .output_tensors = {1, (const int32_t[1]){147}} }, /* nl_107 */
    {.id = 108, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){147}}, .output_tensors = {1, (const int32_t[1]){148}} }, /* conversion_108 */
    {.id = 114, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){148, 144}}, .output_tensors = {1, (const int32_t[1]){149}} }, /* eltwise_114 */
    {.id = 115, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){149}}, .output_tensors = {1, (const int32_t[1]){150}} }, /* eltwise_115 */
    {.id = 117, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){145, 150}}, .output_tensors = {1, (const int32_t[1]){151}} }, /* eltwise_117 */
    {.id = 118, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){151}}, .output_tensors = {1, (const int32_t[1]){152}} }, /* gemm_118 */
    {.id = 119, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){152}}, .output_tensors = {1, (const int32_t[1]){153}} }, /* nl_119 */
    {.id = 120, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){153}}, .output_tensors = {1, (const int32_t[1]){154}} }, /* gemm_120 */
    {.id = 121, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){154}}, .output_tensors = {1, (const int32_t[1]){155}} }, /* nl_121 */
    {.id = 122, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){155}}, .output_tensors = {1, (const int32_t[1]){156}} } /* gemm_122 */
  },
  .n_nodes = 146
};
#endif

