/**
 * @file activation_wrappers.c
 * @brief C wrapper functions for assembly activation functions
 * 
 * This file provides the C interface to high-performance assembly
 * activation functions with proper error checking and initialization.
 */

#include "ml_assembly.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Assembly function declarations */
extern void ml_activation_relu_asm(const float* input, float* output, size_t size);
extern void ml_activation_leaky_relu_asm(const float* input, float* output, size_t size, float alpha);
extern void ml_activation_sigmoid_asm(const float* input, float* output, size_t size);
extern void ml_activation_tanh_asm(const float* input, float* output, size_t size);
extern void ml_activation_softmax_asm(const float* input, float* output, size_t size);

/* Static lookup table initialization flag */
static bool lookup_tables_initialized = false;

/**
 * @brief Initialize lookup tables for fast activation functions
 */
static void init_lookup_tables(void) {
    if (lookup_tables_initialized) return;
    
    /* Initialize sigmoid lookup table */
    extern float sigmoid_lut[256];
    for (int i = 0; i < 256; ++i) {
        float x = (i - 128.0f) / 16.0f;  /* Map to [-8, 8] range */
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
    }
    
    lookup_tables_initialized = true;
}

/* ============================================================================
 * ACTIVATION FUNCTION IMPLEMENTATIONS
 * ============================================================================ */

ml_error_t ml_activation_apply(ml_activation_t activation,
                              const ml_vector_t* input,
                              ml_vector_t* output) {
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
        return ML_SUCCESS;
    }
    
    /* Initialize lookup tables if needed */
    init_lookup_tables();
    
    switch (activation) {
        case ML_ACTIVATION_LINEAR:
            /* Linear activation is just a copy */
            if (input != output) {
                memcpy(output->data, input->data, input->size * sizeof(ml_float_t));
            }
            break;
            
        case ML_ACTIVATION_RELU:
            ml_activation_relu_asm(input->data, output->data, input->size);
            break;
            
        case ML_ACTIVATION_SIGMOID:
            ml_activation_sigmoid_asm(input->data, output->data, input->size);
            break;
            
        case ML_ACTIVATION_TANH:
            ml_activation_tanh_asm(input->data, output->data, input->size);
            break;
            
        case ML_ACTIVATION_SOFTMAX:
            ml_activation_softmax_asm(input->data, output->data, input->size);
            break;
            
        case ML_ACTIVATION_LEAKY_RELU:
            ml_activation_leaky_relu_asm(input->data, output->data, input->size, 0.01f);
            break;
            
        default:
            return ML_ERROR_INVALID_INPUT;
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_activation_relu(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_activation_relu_asm(input->data, output->data, input->size);
    return ML_SUCCESS;
}

ml_error_t ml_activation_leaky_relu(const ml_vector_t* input, ml_vector_t* output, ml_float_t alpha) {
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_activation_leaky_relu_asm(input->data, output->data, input->size, alpha);
    return ML_SUCCESS;
}

ml_error_t ml_activation_sigmoid(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_activation_sigmoid_asm(input->data, output->data, input->size);
    return ML_SUCCESS;
}

ml_error_t ml_activation_tanh(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    ml_activation_tanh_asm(input->data, output->data, input->size);
    return ML_SUCCESS;
}

ml_error_t ml_activation_softmax(const ml_vector_t* input, ml_vector_t* output) {
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
        return ML_SUCCESS;
    }
    
    ml_activation_softmax_asm(input->data, output->data, input->size);
    return ML_SUCCESS;
}

/* ============================================================================
 * REFERENCE IMPLEMENTATIONS (for testing and fallback)
 * ============================================================================ */

ml_error_t ml_activation_relu_ref(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output || !input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < input->size; ++i) {
        output->data[i] = fmaxf(0.0f, input->data[i]);
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_activation_sigmoid_ref(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output || !input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < input->size; ++i) {
        /* Numerically stable sigmoid */
        ml_float_t x = input->data[i];
        if (x >= 0) {
            ml_float_t exp_neg_x = expf(-x);
            output->data[i] = 1.0f / (1.0f + exp_neg_x);
        } else {
            ml_float_t exp_x = expf(x);
            output->data[i] = exp_x / (1.0f + exp_x);
        }
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_activation_tanh_ref(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output || !input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < input->size; ++i) {
        output->data[i] = tanhf(input->data[i]);
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_activation_softmax_ref(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output || !input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (input->size == 0) {
        return ML_SUCCESS;
    }
    
    /* Find maximum for numerical stability */
    ml_float_t max_val = input->data[0];
    for (size_t i = 1; i < input->size; ++i) {
        if (input->data[i] > max_val) {
            max_val = input->data[i];
        }
    }
    
    /* Compute exp(x_i - max) and sum */
    ml_float_t sum = 0.0f;
    for (size_t i = 0; i < input->size; ++i) {
        output->data[i] = expf(input->data[i] - max_val);
        sum += output->data[i];
    }
    
    /* Normalize */
    if (sum > 0.0f) {
        for (size_t i = 0; i < input->size; ++i) {
            output->data[i] /= sum;
        }
    }
    
    return ML_SUCCESS;
}

/* ============================================================================
 * ACTIVATION DERIVATIVES (for backpropagation)
 * ============================================================================ */

ml_error_t ml_activation_relu_derivative(const ml_vector_t* input, ml_vector_t* output) {
    if (!input || !output || !input->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (input->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < input->size; ++i) {
        output->data[i] = (input->data[i] > 0.0f) ? 1.0f : 0.0f;
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_activation_sigmoid_derivative(const ml_vector_t* sigmoid_output, ml_vector_t* output) {
    if (!sigmoid_output || !output || !sigmoid_output->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (sigmoid_output->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < sigmoid_output->size; ++i) {
        ml_float_t s = sigmoid_output->data[i];
        output->data[i] = s * (1.0f - s);
    }
    
    return ML_SUCCESS;
}

ml_error_t ml_activation_tanh_derivative(const ml_vector_t* tanh_output, ml_vector_t* output) {
    if (!tanh_output || !output || !tanh_output->data || !output->data) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (tanh_output->size != output->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    for (size_t i = 0; i < tanh_output->size; ++i) {
        ml_float_t t = tanh_output->data[i];
        output->data[i] = 1.0f - t * t;
    }
    
    return ML_SUCCESS;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

const char* ml_activation_name(ml_activation_t activation) {
    switch (activation) {
        case ML_ACTIVATION_LINEAR:     return "Linear";
        case ML_ACTIVATION_RELU:       return "ReLU";
        case ML_ACTIVATION_SIGMOID:    return "Sigmoid";
        case ML_ACTIVATION_TANH:       return "Tanh";
        case ML_ACTIVATION_SOFTMAX:    return "Softmax";
        case ML_ACTIVATION_LEAKY_RELU: return "Leaky ReLU";
        default:                       return "Unknown";
    }
}

bool ml_activation_is_valid(ml_activation_t activation) {
    return activation >= 0 && activation < 6;
}
