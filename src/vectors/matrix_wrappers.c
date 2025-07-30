/**
 * @file matrix_wrappers.c
 * @brief C wrapper functions for assembly matrix operations
 * 
 * This file provides the C interface to high-performance assembly
 * matrix operations with proper memory management and error checking.
 */

#define _POSIX_C_SOURCE 200112L

#include "ml_assembly.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Assembly function declarations */
extern void ml_matrix_vector_mul_asm(const float* matrix, const float* vector,
                                    float* result, size_t rows, size_t cols);
extern void ml_matrix_mul_asm(const float* A, const float* B, float* C,
                             size_t rows_A, size_t cols_A, size_t cols_B);
extern void ml_matrix_transpose_asm(const float* input, float* output,
                                   size_t rows, size_t cols);
extern void ml_matrix_add_asm(const float* A, const float* B, float* C,
                             size_t rows, size_t cols);

/* Internal helper functions */
static void* aligned_malloc(size_t size, size_t alignment);
static void aligned_free(void* ptr);

/**
 * @brief Allocate aligned memory for optimal SIMD performance
 */
static void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    
    #ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return NULL;
        }
    #endif
    
    return ptr;
}

/**
 * @brief Free aligned memory
 */
static void aligned_free(void* ptr) {
    if (ptr == NULL) return;
    
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

/* ============================================================================
 * MATRIX MANAGEMENT FUNCTIONS
 * ============================================================================ */

ml_matrix_t* ml_matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }
    
    ml_matrix_t* matrix = malloc(sizeof(ml_matrix_t));
    if (!matrix) {
        return NULL;
    }
    
    /* Calculate aligned size for optimal SIMD performance */
    size_t total_elements = rows * cols;
    size_t aligned_size = ((total_elements * sizeof(ml_float_t)) + 31) & ~31;
    
    matrix->data = aligned_malloc(aligned_size, 32);
    if (!matrix->data) {
        free(matrix);
        return NULL;
    }
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->capacity = aligned_size / sizeof(ml_float_t);
    matrix->owns_data = true;
    
    /* Initialize to zero */
    memset(matrix->data, 0, aligned_size);
    
    return matrix;
}

ml_matrix_t* ml_matrix_from_data(const ml_float_t* data, size_t rows, size_t cols, bool copy_data) {
    if (!data || rows == 0 || cols == 0) {
        return NULL;
    }
    
    ml_matrix_t* matrix = malloc(sizeof(ml_matrix_t));
    if (!matrix) {
        return NULL;
    }
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->owns_data = copy_data;
    
    if (copy_data) {
        /* Create our own aligned copy */
        size_t total_elements = rows * cols;
        size_t aligned_size = ((total_elements * sizeof(ml_float_t)) + 31) & ~31;
        
        matrix->data = aligned_malloc(aligned_size, 32);
        if (!matrix->data) {
            free(matrix);
            return NULL;
        }
        matrix->capacity = aligned_size / sizeof(ml_float_t);
        memcpy(matrix->data, data, total_elements * sizeof(ml_float_t));
    } else {
        /* Use existing data */
        matrix->data = (ml_float_t*)data;
        matrix->capacity = rows * cols;
    }
    
    return matrix;
}

void ml_matrix_free(ml_matrix_t* matrix) {
    if (!matrix) return;
    
    if (matrix->owns_data && matrix->data) {
        aligned_free(matrix->data);
    }
    
    free(matrix);
}

/* ============================================================================
 * MATRIX OPERATIONS
 * ============================================================================ */

ml_error_t ml_matrix_vector_mul(const ml_matrix_t* matrix, 
                               const ml_vector_t* vector, 
                               ml_vector_t* result) {
    if (!matrix || !vector || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!matrix->data || !vector->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (matrix->cols != vector->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (matrix->rows != result->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_matrix_vector_mul_asm(matrix->data, vector->data, result->data,
                            matrix->rows, matrix->cols);
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_mul(const ml_matrix_t* a, const ml_matrix_t* b, ml_matrix_t* result) {
    if (!a || !b || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!a->data || !b->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (a->cols != b->rows) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (a->rows != result->rows || b->cols != result->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_matrix_mul_asm(a->data, b->data, result->data,
                     a->rows, a->cols, b->cols);
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_transpose(const ml_matrix_t* input, ml_matrix_t* output) {
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->rows != output->cols || input->cols != output->rows) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_matrix_transpose_asm(input->data, output->data, input->rows, input->cols);
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_add(const ml_matrix_t* a, const ml_matrix_t* b, ml_matrix_t* result) {
    if (!a || !b || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!a->data || !b->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (a->rows != b->rows || a->cols != b->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (a->rows != result->rows || a->cols != result->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_matrix_add_asm(a->data, b->data, result->data, a->rows, a->cols);
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_scale(const ml_matrix_t* matrix, ml_float_t scalar, ml_matrix_t* result) {
    if (!matrix || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!matrix->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (matrix->rows != result->rows || matrix->cols != result->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    /* Use vector scale operation on flattened matrix */
    size_t total_elements = matrix->rows * matrix->cols;
    for (size_t i = 0; i < total_elements; ++i) {
        result->data[i] = matrix->data[i] * scalar;
    }
    
    return ML_SUCCESS;
}

/* ============================================================================
 * MATRIX UTILITY FUNCTIONS
 * ============================================================================ */

ml_error_t ml_matrix_set(ml_matrix_t* matrix, size_t row, size_t col, ml_float_t value) {
    if (!matrix || !matrix->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (row >= matrix->rows || col >= matrix->cols) {
        return ML_ERROR_INVALID_INPUT;
    }
    
    matrix->data[row * matrix->cols + col] = value;
    return ML_SUCCESS;
}

ml_float_t ml_matrix_get(const ml_matrix_t* matrix, size_t row, size_t col) {
    if (!matrix || !matrix->data) {
        return 0.0f;
    }
    
    if (row >= matrix->rows || col >= matrix->cols) {
        return 0.0f;
    }
    
    return matrix->data[row * matrix->cols + col];
}

ml_error_t ml_matrix_fill(ml_matrix_t* matrix, ml_float_t value) {
    if (!matrix || !matrix->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    size_t total_elements = matrix->rows * matrix->cols;
    for (size_t i = 0; i < total_elements; ++i) {
        matrix->data[i] = value;
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_identity(ml_matrix_t* matrix) {
    if (!matrix || !matrix->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (matrix->rows != matrix->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    /* Zero out the matrix first */
    ml_matrix_fill(matrix, 0.0f);
    
    /* Set diagonal elements to 1 */
    for (size_t i = 0; i < matrix->rows; ++i) {
        matrix->data[i * matrix->cols + i] = 1.0f;
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_copy(const ml_matrix_t* src, ml_matrix_t* dst) {
    if (!src || !dst) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!src->data || !dst->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (src->rows != dst->rows || src->cols != dst->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    size_t total_elements = src->rows * src->cols;
    memcpy(dst->data, src->data, total_elements * sizeof(ml_float_t));
    
    return ML_SUCCESS;
}

ml_vector_t* ml_matrix_get_row(const ml_matrix_t* matrix, size_t row) {
    if (!matrix || !matrix->data || row >= matrix->rows) {
        return NULL;
    }
    
    /* Create vector from row data (without copying) */
    return ml_vector_from_data(&matrix->data[row * matrix->cols], matrix->cols, false);
}

ml_vector_t* ml_matrix_get_col(const ml_matrix_t* matrix, size_t col) {
    if (!matrix || !matrix->data || col >= matrix->cols) {
        return NULL;
    }
    
    /* Need to copy column data since it's not contiguous */
    ml_vector_t* column = ml_vector_create(matrix->rows);
    if (!column) {
        return NULL;
    }
    
    for (size_t i = 0; i < matrix->rows; ++i) {
        column->data[i] = matrix->data[i * matrix->cols + col];
    }
    
    return column;
}

ml_error_t ml_matrix_set_row(ml_matrix_t* matrix, size_t row, const ml_vector_t* vector) {
    if (!matrix || !vector || !matrix->data || !vector->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (row >= matrix->rows || vector->size != matrix->cols) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    memcpy(&matrix->data[row * matrix->cols], vector->data, 
           matrix->cols * sizeof(ml_float_t));
    
    return ML_SUCCESS;
}

ml_error_t ml_matrix_set_col(ml_matrix_t* matrix, size_t col, const ml_vector_t* vector) {
    if (!matrix || !vector || !matrix->data || !vector->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (col >= matrix->cols || vector->size != matrix->rows) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < matrix->rows; ++i) {
        matrix->data[i * matrix->cols + col] = vector->data[i];
    }
    
    return ML_SUCCESS;
}

/* ============================================================================
 * DEBUGGING AND UTILITY FUNCTIONS
 * ============================================================================ */

void ml_matrix_print(const ml_matrix_t* matrix, const char* name) {
    if (!matrix || !matrix->data) {
        printf("Matrix %s: NULL\n", name ? name : "");
        return;
    }
    
    printf("Matrix %s [%zux%zu]:\n", name ? name : "", matrix->rows, matrix->cols);
    for (size_t i = 0; i < matrix->rows; ++i) {
        printf("  [");
        for (size_t j = 0; j < matrix->cols; ++j) {
            printf("%8.4f", matrix->data[i * matrix->cols + j]);
            if (j < matrix->cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

bool ml_matrix_equals(const ml_matrix_t* a, const ml_matrix_t* b, ml_float_t tolerance) {
    if (!a || !b) return false;
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (!a->data || !b->data) return false;
    
    size_t total_elements = a->rows * a->cols;
    for (size_t i = 0; i < total_elements; ++i) {
        if (fabsf(a->data[i] - b->data[i]) > tolerance) {
            return false;
        }
    }
    
    return true;
}

ml_float_t ml_matrix_frobenius_norm(const ml_matrix_t* matrix) {
    if (!matrix || !matrix->data) {
        return 0.0f;
    }
    
    ml_float_t sum = 0.0f;
    size_t total_elements = matrix->rows * matrix->cols;
    
    for (size_t i = 0; i < total_elements; ++i) {
        sum += matrix->data[i] * matrix->data[i];
    }
    
    return sqrtf(sum);
}
