/**
 * @file vector_wrappers.c
 * @brief C wrapper functions for assembly vector operations
 * 
 * This file provides the C interface to the high-performance assembly
 * vector operations, including memory management and error checking.
 */

#define _POSIX_C_SOURCE 200112L

#include "ml_assembly.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <stdint.h>

/* Assembly function declarations */
extern float ml_vector_dot_asm(const float* a, const float* b, size_t size);
extern void ml_vector_add_asm(const float* a, const float* b, float* result, size_t size);
extern void ml_vector_scale_asm(const float* input, float scalar, float* result, size_t size);
extern void ml_vector_normalize_asm(const float* input, float* result, size_t size);
extern float ml_vector_sum_asm(const float* vector, size_t size);

/* Internal helper functions */
static void* aligned_malloc(size_t size, size_t alignment);
static void aligned_free(void* ptr);
static bool is_aligned(const void* ptr, size_t alignment);

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

/**
 * @brief Check if pointer is properly aligned
 */
__attribute__((unused))
static bool is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

/* ============================================================================
 * VECTOR MANAGEMENT FUNCTIONS
 * ============================================================================ */

ml_vector_t* ml_vector_create(size_t size) {
    if (size == 0) {
        return NULL;
    }
    
    ml_vector_t* vector = malloc(sizeof(ml_vector_t));
    if (!vector) {
        return NULL;
    }
    
    /* Allocate aligned memory for optimal SIMD performance */
    size_t aligned_size = ((size * sizeof(ml_float_t)) + 31) & ~31; /* Round up to 32-byte boundary */
    vector->data = aligned_malloc(aligned_size, 32);
    if (!vector->data) {
        free(vector);
        return NULL;
    }
    
    vector->size = size;
    vector->capacity = aligned_size / sizeof(ml_float_t);
    vector->owns_data = true;
    
    /* Initialize to zero */
    memset(vector->data, 0, aligned_size);
    
    return vector;
}

ml_vector_t* ml_vector_from_data(const ml_float_t* data, size_t size, bool copy_data) {
    if (!data || size == 0) {
        return NULL;
    }
    
    ml_vector_t* vector = malloc(sizeof(ml_vector_t));
    if (!vector) {
        return NULL;
    }
    
    vector->size = size;
    vector->owns_data = copy_data;
    
    if (copy_data) {
        /* Create our own aligned copy */
        size_t aligned_size = ((size * sizeof(ml_float_t)) + 31) & ~31;
        vector->data = aligned_malloc(aligned_size, 32);
        if (!vector->data) {
            free(vector);
            return NULL;
        }
        vector->capacity = aligned_size / sizeof(ml_float_t);
        memcpy(vector->data, data, size * sizeof(ml_float_t));
    } else {
        /* Use existing data (caller retains ownership) */
        vector->data = (ml_float_t*)data;
        vector->capacity = size;
    }
    
    return vector;
}

void ml_vector_free(ml_vector_t* vector) {
    if (!vector) return;
    
    if (vector->owns_data && vector->data) {
        aligned_free(vector->data);
    }
    
    free(vector);
}

/* ============================================================================
 * VECTOR OPERATIONS
 * ============================================================================ */

ml_float_t ml_vector_dot(const ml_vector_t* a, const ml_vector_t* b) {
    if (!a || !b || !a->data || !b->data) {
        return 0.0f; /* Return 0 for null vectors */
    }
    
    if (a->size != b->size) {
        return 0.0f; /* Size mismatch */
    }
    
    return ml_vector_dot_asm(a->data, b->data, a->size);
}

ml_error_t ml_vector_add(const ml_vector_t* a, const ml_vector_t* b, ml_vector_t* result) {
    if (!a || !b || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!a->data || !b->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (a->size != b->size || a->size != result->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_vector_add_asm(a->data, b->data, result->data, a->size);
    return ML_SUCCESS;
}

ml_error_t ml_vector_scale(const ml_vector_t* vector, ml_float_t scalar, ml_vector_t* result) {
    if (!vector || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!vector->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (vector->size != result->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_vector_scale_asm(vector->data, scalar, result->data, vector->size);
    return ML_SUCCESS;
}

ml_error_t ml_vector_normalize(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (input->size == 0) {
        return ML_ERROR_INVALID_INPUT;
    }
    
    ml_vector_normalize_asm(input->data, output->data, input->size);
    return ML_SUCCESS;
}

ml_float_t ml_vector_sum(const ml_vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        return 0.0f;
    }
    
    return ml_vector_sum_asm(vector->data, vector->size);
}

ml_float_t ml_vector_mean(const ml_vector_t* vector) {
    if (!vector || vector->size == 0) {
        return 0.0f;
    }
    
    return ml_vector_sum(vector) / (ml_float_t)vector->size;
}

ml_error_t ml_vector_copy(const ml_vector_t* src, ml_vector_t* dst) {
    if (!src || !dst) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!src->data || !dst->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (src->size != dst->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    memcpy(dst->data, src->data, src->size * sizeof(ml_float_t));
    return ML_SUCCESS;
}

ml_error_t ml_vector_fill(ml_vector_t* vector, ml_float_t value) {
    if (!vector || !vector->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    for (size_t i = 0; i < vector->size; ++i) {
        vector->data[i] = value;
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_vector_set(ml_vector_t* vector, size_t index, ml_float_t value) {
    if (!vector || !vector->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (index >= vector->size) {
        return ML_ERROR_INVALID_INPUT;
    }
    
    vector->data[index] = value;
    return ML_SUCCESS;
}

ml_float_t ml_vector_get(const ml_vector_t* vector, size_t index) {
    if (!vector || !vector->data || index >= vector->size) {
        return 0.0f;
    }
    
    return vector->data[index];
}

ml_error_t ml_vector_element_wise_mul(const ml_vector_t* a, const ml_vector_t* b, ml_vector_t* result) {
    if (!a || !b || !result) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!a->data || !b->data || !result->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (a->size != b->size || a->size != result->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    /* Element-wise multiplication */
    for (size_t i = 0; i < a->size; ++i) {
        result->data[i] = a->data[i] * b->data[i];
    }
    
    return ML_SUCCESS;
}

ml_float_t ml_vector_max(const ml_vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        return 0.0f;
    }
    
    ml_float_t max_val = vector->data[0];
    for (size_t i = 1; i < vector->size; ++i) {
        if (vector->data[i] > max_val) {
            max_val = vector->data[i];
        }
    }
    
    return max_val;
}

ml_float_t ml_vector_min(const ml_vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        return 0.0f;
    }
    
    ml_float_t min_val = vector->data[0];
    for (size_t i = 1; i < vector->size; ++i) {
        if (vector->data[i] < min_val) {
            min_val = vector->data[i];
        }
    }
    
    return min_val;
}

size_t ml_vector_argmax(const ml_vector_t* vector) {
    if (!vector || !vector->data || vector->size == 0) {
        return 0;
    }
    
    size_t max_idx = 0;
    ml_float_t max_val = vector->data[0];
    
    for (size_t i = 1; i < vector->size; ++i) {
        if (vector->data[i] > max_val) {
            max_val = vector->data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/* ============================================================================
 * DEBUGGING AND UTILITY FUNCTIONS
 * ============================================================================ */

void ml_vector_print(const ml_vector_t* vector, const char* name) {
    if (!vector || !vector->data) {
        printf("Vector %s: NULL\n", name ? name : "");
        return;
    }
    
    printf("Vector %s [%zu]: [", name ? name : "", vector->size);
    for (size_t i = 0; i < vector->size; ++i) {
        printf("%.6f", vector->data[i]);
        if (i < vector->size - 1) printf(", ");
    }
    printf("]\n");
}

bool ml_vector_equals(const ml_vector_t* a, const ml_vector_t* b, ml_float_t tolerance) {
    if (!a || !b) return false;
    if (a->size != b->size) return false;
    if (!a->data || !b->data) return false;
    
    for (size_t i = 0; i < a->size; ++i) {
        if (fabsf(a->data[i] - b->data[i]) > tolerance) {
            return false;
        }
    }
    
    return true;
}
