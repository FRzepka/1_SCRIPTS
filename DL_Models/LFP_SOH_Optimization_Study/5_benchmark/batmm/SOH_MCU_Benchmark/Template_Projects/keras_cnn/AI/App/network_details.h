/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-05-15T16:26:50+0000
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
  .tensors = (const stai_tensor[38]) {
   { .size_bytes = 1920, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 96, 20}}, .scale = {1, (const float[1]){0.07195434719324112}}, .zeropoint = {1, (const int16_t[1]){40}}, .name = "serving_default_input0_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.03005221113562584}}, .zeropoint = {1, (const int16_t[1]){2}}, .name = "conv2d_4_output" },
   { .size_bytes = 12800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 100, 1, 128}}, .scale = {1, (const float[1]){0.03005221113562584}}, .zeropoint = {1, (const int16_t[1]){2}}, .name = "pad_13_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.06683990359306335}}, .zeropoint = {1, (const int16_t[1]){66}}, .name = "conv2d_14_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.01600812003016472}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_18_output" },
   { .size_bytes = 12800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 100, 1, 128}}, .scale = {1, (const float[1]){0.01600812003016472}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_24_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.03006605990231037}}, .zeropoint = {1, (const int16_t[1]){62}}, .name = "conv2d_25_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.007697582710534334}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_29_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 96, 128}}, .scale = {1, (const float[1]){0.030483368784189224}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "eltwise_31_output" },
   { .size_bytes = 13312, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 104, 1, 128}}, .scale = {1, (const float[1]){0.030483368784189224}}, .zeropoint = {1, (const int16_t[1]){0}}, .name = "pad_36_output" },
   { .size_bytes = 53248, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 104, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_36_0_conversion_output" },
   { .size_bytes = 49152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_38_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.041260287165641785}}, .zeropoint = {1, (const int16_t[1]){77}}, .name = "conv2d_38_0_0_eltwise_40_conversion_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.04132267087697983}}, .zeropoint = {1, (const int16_t[1]){78}}, .name = "eltwise_40_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.007900108583271503}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_44_output" },
   { .size_bytes = 13312, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 104, 1, 128}}, .scale = {1, (const float[1]){0.007900108583271503}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_50_output" },
   { .size_bytes = 53248, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 104, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_50_0_0_conv2d_52_conversion_output" },
   { .size_bytes = 49152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_52_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.015926923602819443}}, .zeropoint = {1, (const int16_t[1]){43}}, .name = "conv2d_52_0_0_eltwise_54_conversion_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.015453035943210125}}, .zeropoint = {1, (const int16_t[1]){39}}, .name = "eltwise_54_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.005308276973664761}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_58_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 96, 128}}, .scale = {1, (const float[1]){0.030903693288564682}}, .zeropoint = {1, (const int16_t[1]){-2}}, .name = "eltwise_60_output" },
   { .size_bytes = 14336, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 112, 1, 128}}, .scale = {1, (const float[1]){0.030903693288564682}}, .zeropoint = {1, (const int16_t[1]){-2}}, .name = "pad_65_output" },
   { .size_bytes = 57344, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 112, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_65_0_0_conv2d_67_conversion_output" },
   { .size_bytes = 49152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_67_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.05199640244245529}}, .zeropoint = {1, (const int16_t[1]){77}}, .name = "conv2d_67_0_0_eltwise_69_conversion_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.05238831415772438}}, .zeropoint = {1, (const int16_t[1]){76}}, .name = "eltwise_69_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.01041225902736187}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_73_output" },
   { .size_bytes = 14336, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 112, 1, 128}}, .scale = {1, (const float[1]){0.01041225902736187}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "pad_79_output" },
   { .size_bytes = 57344, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 112, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pad_79_0_0_conv2d_81_conversion_output" },
   { .size_bytes = 49152, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv2d_81_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.02176525816321373}}, .zeropoint = {1, (const int16_t[1]){97}}, .name = "conv2d_81_0_0_eltwise_83_conversion_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.021538667380809784}}, .zeropoint = {1, (const int16_t[1]){94}}, .name = "eltwise_83_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 128}}, .scale = {1, (const float[1]){0.0027950801886618137}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_87_output" },
   { .size_bytes = 12288, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {3, (const int32_t[3]){1, 96, 128}}, .scale = {1, (const float[1]){0.03198061138391495}}, .zeropoint = {1, (const int16_t[1]){-6}}, .name = "eltwise_89_output" },
   { .size_bytes = 6144, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 64}}, .scale = {1, (const float[1]){0.042549774050712585}}, .zeropoint = {1, (const int16_t[1]){51}}, .name = "conv2d_94_output" },
   { .size_bytes = 6144, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 64}}, .scale = {1, (const float[1]){0.012757483869791031}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "nl_98_output" },
   { .size_bytes = 96, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 96, 1, 1}}, .scale = {1, (const float[1]){0.003846552222967148}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "conv2d_104_output" }
  },
  .nodes = (const stai_node_details[37]){
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv2d_4 */
    {.id = 13, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* pad_13 */
    {.id = 14, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv2d_14 */
    {.id = 18, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* nl_18 */
    {.id = 24, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* pad_24 */
    {.id = 25, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* conv2d_25 */
    {.id = 29, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* nl_29 */
    {.id = 31, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){7, 1}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* eltwise_31 */
    {.id = 36, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* pad_36 */
    {.id = 36, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* pad_36_0_conversion */
    {.id = 39, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* conv2d_38 */
    {.id = 39, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* conv2d_38_0_0_eltwise_40_conversion */
    {.id = 40, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* eltwise_40 */
    {.id = 44, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* nl_44 */
    {.id = 50, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* pad_50 */
    {.id = 50, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* pad_50_0_0_conv2d_52_conversion */
    {.id = 53, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* conv2d_52 */
    {.id = 53, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* conv2d_52_0_0_eltwise_54_conversion */
    {.id = 54, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* eltwise_54 */
    {.id = 58, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* nl_58 */
    {.id = 60, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){20, 8}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* eltwise_60 */
    {.id = 65, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* pad_65 */
    {.id = 65, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* pad_65_0_0_conv2d_67_conversion */
    {.id = 68, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* conv2d_67 */
    {.id = 68, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* conv2d_67_0_0_eltwise_69_conversion */
    {.id = 69, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* eltwise_69 */
    {.id = 73, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* nl_73 */
    {.id = 79, .type = AI_LAYER_PAD_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* pad_79 */
    {.id = 79, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* pad_79_0_0_conv2d_81_conversion */
    {.id = 82, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* conv2d_81 */
    {.id = 82, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* conv2d_81_0_0_eltwise_83_conversion */
    {.id = 83, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* eltwise_83 */
    {.id = 87, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* nl_87 */
    {.id = 89, .type = AI_LAYER_ELTWISE_INTEGER_TYPE, .input_tensors = {2, (const int32_t[2]){33, 21}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* eltwise_89 */
    {.id = 94, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* conv2d_94 */
    {.id = 98, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* nl_98 */
    {.id = 104, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} } /* conv2d_104 */
  },
  .n_nodes = 37
};
#endif

