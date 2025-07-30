/**
 * @file linear_regression.c
 * @brief Linear regression model implementation
 * 
 * High-performance linear regression inference using optimized
 * matrix-vector operations.
 */

#include "ml_assembly.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/**
 * @brief Linear regression model internal data structure
 */
typedef struct {
    ml_vector_t* weights;    /* Weight vector */
    ml_float_t bias;         /* Bias term */
    bool has_bias;           /* Whether model includes bias */
    size_t input_features;   /* Number of input features */
} linear_regression_data_t;

/* ============================================================================
 * MODEL CREATION AND MANAGEMENT
 * ============================================================================ */

ml_model_t* ml_linear_regression_create(size_t input_features, bool has_bias) {
    if (input_features == 0) {
        return NULL;
    }
    
    /* Create model structure */
    ml_model_t* model = malloc(sizeof(ml_model_t));
    if (!model) {
        return NULL;
    }
    
    /* Create internal data */
    linear_regression_data_t* lr_data = malloc(sizeof(linear_regression_data_t));
    if (!lr_data) {
        free(model);
        return NULL;
    }
    
    /* Initialize weights vector */
    lr_data->weights = ml_vector_create(input_features);
    if (!lr_data->weights) {
        free(lr_data);
        free(model);
        return NULL;
    }
    
    lr_data->bias = 0.0f;
    lr_data->has_bias = has_bias;
    lr_data->input_features = input_features;
    
    /* Set up model configuration */
    model->config.type = ML_LINEAR_REGRESSION;
    model->config.input_size = input_features;
    model->config.output_size = 1;
    model->config.layer_count = 1;
    model->config.activations = NULL;
    model->config.layer_sizes = NULL;
    model->config.model_data = lr_data;
    
    model->internal_data = lr_data;
    model->is_loaded = false;
    
    return model;
}

ml_error_t ml_linear_regression_set_weights(ml_model_t* model, 
                                           const ml_float_t* weights, 
                                           ml_float_t bias) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (!weights) {
        return ML_ERROR_NULL_POINTER;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data || !lr_data->weights) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    /* Copy weights */
    memcpy(lr_data->weights->data, weights, 
           lr_data->input_features * sizeof(ml_float_t));
    
    lr_data->bias = bias;
    model->is_loaded = true;
    
    return ML_SUCCESS;
}

ml_error_t ml_linear_regression_get_weights(const ml_model_t* model, 
                                           ml_float_t* weights, 
                                           ml_float_t* bias) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (!weights || !bias) {
        return ML_ERROR_NULL_POINTER;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data || !lr_data->weights) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    /* Copy weights */
    memcpy(weights, lr_data->weights->data, 
           lr_data->input_features * sizeof(ml_float_t));
    
    *bias = lr_data->bias;
    
    return ML_SUCCESS;
}

/* ============================================================================
 * INFERENCE FUNCTIONS
 * ============================================================================ */

ml_error_t ml_linear_regression_predict(const ml_model_t* model,
                                       const ml_float_t* input,
                                       ml_float_t* output) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (!input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!model->is_loaded) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data || !lr_data->weights) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    /* Create input vector wrapper (no copy) */
    ml_vector_t* input_vec = ml_vector_from_data(input, lr_data->input_features, false);
    if (!input_vec) {
        return ML_ERROR_OUT_OF_MEMORY;
    }
    
    /* Compute dot product: weights^T * input */
    ml_float_t result = ml_vector_dot(lr_data->weights, input_vec);
    
    /* Add bias if present */
    if (lr_data->has_bias) {
        result += lr_data->bias;
    }
    
    *output = result;
    
    /* Cleanup */
    ml_vector_free(input_vec);
    
    return ML_SUCCESS;
}

ml_error_t ml_linear_regression_predict_batch(const ml_model_t* model,
                                             const ml_matrix_t* inputs,
                                             ml_vector_t* outputs) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (!inputs || !outputs) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!model->is_loaded) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data || !lr_data->weights) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    /* Check dimensions */
    if (inputs->cols != lr_data->input_features) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (inputs->rows != outputs->size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    /* Perform matrix-vector multiplication: inputs * weights */
    ml_error_t error = ml_matrix_vector_mul(inputs, lr_data->weights, outputs);
    if (error != ML_SUCCESS) {
        return error;
    }
    
    /* Add bias to all predictions if present */
    if (lr_data->has_bias) {
        for (size_t i = 0; i < outputs->size; ++i) {
            outputs->data[i] += lr_data->bias;
        }
    }
    
    return ML_SUCCESS;
}

/* ============================================================================
 * MODEL PERSISTENCE
 * ============================================================================ */

ml_error_t ml_linear_regression_save(const ml_model_t* model, const char* filename) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (!filename) {
        return ML_ERROR_NULL_POINTER;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data || !lr_data->weights) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return ML_ERROR_FILE_NOT_FOUND;
    }
    
    /* Write model header */
    uint32_t magic = 0x4D4C4152; /* "MLAR" - ML Assembly Linear Regression */
    uint32_t version = 1;
    
    if (fwrite(&magic, sizeof(uint32_t), 1, file) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        return ML_ERROR_FILE_NOT_FOUND;
    }
    
    /* Write model parameters */
    if (fwrite(&lr_data->input_features, sizeof(size_t), 1, file) != 1 ||
        fwrite(&lr_data->has_bias, sizeof(bool), 1, file) != 1 ||
        fwrite(&lr_data->bias, sizeof(ml_float_t), 1, file) != 1) {
        fclose(file);
        return ML_ERROR_FILE_NOT_FOUND;
    }
    
    /* Write weights */
    if (fwrite(lr_data->weights->data, sizeof(ml_float_t), 
               lr_data->input_features, file) != lr_data->input_features) {
        fclose(file);
        return ML_ERROR_FILE_NOT_FOUND;
    }
    
    fclose(file);
    return ML_SUCCESS;
}

ml_model_t* ml_linear_regression_load(const char* filename) {
    if (!filename) {
        return NULL;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }
    
    /* Read and verify header */
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1 ||
        fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    if (magic != 0x4D4C4152 || version != 1) {
        fclose(file);
        return NULL;
    }
    
    /* Read model parameters */
    size_t input_features;
    bool has_bias;
    ml_float_t bias;
    
    if (fread(&input_features, sizeof(size_t), 1, file) != 1 ||
        fread(&has_bias, sizeof(bool), 1, file) != 1 ||
        fread(&bias, sizeof(ml_float_t), 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    
    /* Create model */
    ml_model_t* model = ml_linear_regression_create(input_features, has_bias);
    if (!model) {
        fclose(file);
        return NULL;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    
    /* Read weights */
    if (fread(lr_data->weights->data, sizeof(ml_float_t), 
              input_features, file) != input_features) {
        fclose(file);
        ml_free_model(model);
        return NULL;
    }
    
    lr_data->bias = bias;
    model->is_loaded = true;
    
    fclose(file);
    return model;
}

/* ============================================================================
 * TRAINING UTILITIES (Basic implementations)
 * ============================================================================ */

ml_error_t ml_linear_regression_train_normal_equation(ml_model_t* model,
                                                     const ml_matrix_t* X,
                                                     const ml_vector_t* y) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (!X || !y) {
        return ML_ERROR_NULL_POINTER;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (X->rows != y->size || X->cols != lr_data->input_features) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    /* This is a simplified implementation - in practice, you'd want to use
     * more sophisticated numerical methods like QR decomposition or SVD */
    
    /* For now, just compute simple least squares approximation */
    /* weights = (X^T X)^-1 X^T y */
    
    /* This would require matrix inversion which is complex to implement
     * efficiently in assembly. For now, we'll use gradient descent approach
     * or require pre-trained weights. */
    
    return ML_ERROR_UNSUPPORTED_MODEL; /* Not implemented yet */
}

/* ============================================================================
 * UTILITY AND DEBUG FUNCTIONS
 * ============================================================================ */

void ml_linear_regression_print_info(const ml_model_t* model) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        printf("Invalid linear regression model\n");
        return;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data) {
        printf("Linear regression model data is NULL\n");
        return;
    }
    
    printf("Linear Regression Model:\n");
    printf("  Input features: %zu\n", lr_data->input_features);
    printf("  Has bias: %s\n", lr_data->has_bias ? "Yes" : "No");
    printf("  Bias value: %.6f\n", lr_data->bias);
    printf("  Loaded: %s\n", model->is_loaded ? "Yes" : "No");
    
    if (lr_data->weights && model->is_loaded) {
        printf("  Weights: [");
        for (size_t i = 0; i < lr_data->input_features; ++i) {
            printf("%.6f", lr_data->weights->data[i]);
            if (i < lr_data->input_features - 1) printf(", ");
        }
        printf("]\n");
    }
}

size_t ml_linear_regression_memory_usage(const ml_model_t* model) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return 0;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (!lr_data) {
        return 0;
    }
    
    size_t total = sizeof(ml_model_t) + sizeof(linear_regression_data_t);
    
    if (lr_data->weights) {
        total += sizeof(ml_vector_t) + lr_data->weights->capacity * sizeof(ml_float_t);
    }
    
    return total;
}

void ml_linear_regression_free(ml_model_t* model) {
    if (!model || model->config.type != ML_LINEAR_REGRESSION) {
        return;
    }
    
    linear_regression_data_t* lr_data = (linear_regression_data_t*)model->internal_data;
    if (lr_data) {
        if (lr_data->weights) {
            ml_vector_free(lr_data->weights);
        }
        free(lr_data);
    }
    
    free(model);
}
