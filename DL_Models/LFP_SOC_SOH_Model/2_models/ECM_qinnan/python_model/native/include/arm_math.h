#ifndef ARM_MATH_COMPAT_H
#define ARM_MATH_COMPAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float float32_t;

typedef struct {
    uint16_t numRows;
    uint16_t numCols;
    float32_t *pData;
} arm_matrix_instance_f32;

typedef int arm_status;
#define ARM_MATH_SUCCESS 0

static inline arm_status arm_mat_init_f32(arm_matrix_instance_f32 *S,
                                          uint16_t nRows,
                                          uint16_t nCols,
                                          float32_t *pData) {
    if (!S || !pData || nRows == 0 || nCols == 0) {
        return -1;
    }
    S->numRows = nRows;
    S->numCols = nCols;
    S->pData = pData;
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_trans_f32(const arm_matrix_instance_f32 *A,
                                           arm_matrix_instance_f32 *AT) {
    if (!A || !AT || !A->pData || !AT->pData) {
        return -1;
    }
    if (AT->numRows != A->numCols || AT->numCols != A->numRows) {
        return -1;
    }
    for (uint16_t r = 0; r < A->numRows; ++r) {
        for (uint16_t c = 0; c < A->numCols; ++c) {
            AT->pData[c * AT->numCols + r] = A->pData[r * A->numCols + c];
        }
    }
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_add_f32(const arm_matrix_instance_f32 *A,
                                         const arm_matrix_instance_f32 *B,
                                         arm_matrix_instance_f32 *C) {
    if (!A || !B || !C || !A->pData || !B->pData || !C->pData) {
        return -1;
    }
    if (A->numRows != B->numRows || A->numCols != B->numCols ||
        C->numRows != A->numRows || C->numCols != A->numCols) {
        return -1;
    }
    uint32_t n = (uint32_t)A->numRows * (uint32_t)A->numCols;
    for (uint32_t i = 0; i < n; ++i) {
        C->pData[i] = A->pData[i] + B->pData[i];
    }
    return ARM_MATH_SUCCESS;
}

static inline arm_status arm_mat_mult_f32(const arm_matrix_instance_f32 *A,
                                          const arm_matrix_instance_f32 *B,
                                          arm_matrix_instance_f32 *C) {
    if (!A || !B || !C || !A->pData || !B->pData || !C->pData) {
        return -1;
    }
    if (A->numCols != B->numRows || C->numRows != A->numRows || C->numCols != B->numCols) {
        return -1;
    }
    for (uint16_t r = 0; r < C->numRows; ++r) {
        for (uint16_t c = 0; c < C->numCols; ++c) {
            float32_t acc = 0.0f;
            for (uint16_t k = 0; k < A->numCols; ++k) {
                acc += A->pData[r * A->numCols + k] * B->pData[k * B->numCols + c];
            }
            C->pData[r * C->numCols + c] = acc;
        }
    }
    return ARM_MATH_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif
