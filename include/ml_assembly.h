/**
 * @file ml_assembly.h
 * @brief High-Performance ML Inference Framework in x86-64 Assembly
 * 
 * This header provides the public API for the ML Assembly framework,
 * offering ultra-low latency machine learning inference capabilities.
 * 
 * @author ML Assembly Team
 * @version 1.0.0
 * @date 2025
 */

#ifndef ML_ASSEMBLY_H
#define ML_ASSEMBLY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define ML_ASSEMBLY_VERSION_MAJOR 1
#define ML_ASSEMBLY_VERSION_MINOR 0
#define ML_ASSEMBLY_VERSION_PATCH 0

/* Error codes */
typedef enum {
    ML_SUCCESS = 0,
    ML_ERROR_NULL_POINTER = -1,
    ML_ERROR_INVALID_MODEL = -2,
    ML_ERROR_INVALID_INPUT = -3,
    ML_ERROR_OUT_OF_MEMORY = -4,
    ML_ERROR_FILE_NOT_FOUND = -5,
    ML_ERROR_UNSUPPORTED_MODEL = -6,
    ML_ERROR_DIMENSION_MISMATCH = -7,
    ML_ERROR_COMPUTATION_FAILED = -8
} ml_error_t;

/* Model types */
typedef enum {
    ML_LINEAR_REGRESSION = 0,
    ML_LOGISTIC_REGRESSION = 1,
    ML_NEURAL_NETWORK = 2,
    ML_DECISION_TREE = 3,
    ML_RANDOM_FOREST = 4,
    ML_MODEL_TYPE_COUNT
} ml_model_type_t;

/* Activation functions */
typedef enum {
    ML_ACTIVATION_LINEAR = 0,
    ML_ACTIVATION_RELU = 1,
    ML_ACTIVATION_SIGMOID = 2,
    ML_ACTIVATION_TANH = 3,
    ML_ACTIVATION_SOFTMAX = 4,
    ML_ACTIVATION_LEAKY_RELU = 5
} ml_activation_t;

/* Data types */
typedef float ml_float_t;
typedef double ml_double_t;

/* Forward declarations */
typedef struct ml_model ml_model_t;
typedef struct ml_vector ml_vector_t;
typedef struct ml_matrix ml_matrix_t;

/**
 * @brief Vector structure for efficient SIMD operations
 */
struct ml_vector {
    ml_float_t* data;      /**< Data pointer (aligned to 32-byte boundary) */
    size_t size;           /**< Number of elements */
    size_t capacity;       /**< Allocated capacity */
    bool owns_data;        /**< Whether vector owns the data */
};

/**
 * @brief Matrix structure for linear algebra operations
 */
struct ml_matrix {
    ml_float_t* data;      /**< Data pointer (row-major order, aligned) */
    size_t rows;           /**< Number of rows */
    size_t cols;           /**< Number of columns */
    size_t capacity;       /**< Allocated capacity */
    bool owns_data;        /**< Whether matrix owns the data */
};

/**
 * @brief Model configuration structure
 */
typedef struct {
    ml_model_type_t type;         /**< Model type */
    size_t input_size;            /**< Input feature count */
    size_t output_size;           /**< Output size */
    size_t layer_count;           /**< Number of layers (for neural networks) */
    ml_activation_t* activations; /**< Activation functions per layer */
    size_t* layer_sizes;          /**< Neurons per layer */
    void* model_data;             /**< Model-specific data */
} ml_model_config_t;

/**
 * @brief Opaque model structure
 */
struct ml_model {
    ml_model_config_t config;
    void* internal_data;
    bool is_loaded;
};

/* ============================================================================
 * CORE API FUNCTIONS
 * ============================================================================ */

/**
 * @brief Initialize the ML Assembly framework
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_init(void);

/**
 * @brief Cleanup the ML Assembly framework
 */
void ml_cleanup(void);

/**
 * @brief Get version string
 * @return Version string in format "major.minor.patch"
 */
const char* ml_get_version(void);

/**
 * @brief Check if CPU supports required features
 * @return true if CPU is supported, false otherwise
 */
bool ml_check_cpu_support(void);

/* ============================================================================
 * MODEL MANAGEMENT
 * ============================================================================ */

/**
 * @brief Create a new model
 * @param config Model configuration
 * @return Pointer to new model or NULL on error
 */
ml_model_t* ml_create_model(const ml_model_config_t* config);

/**
 * @brief Load model from file
 * @param filename Path to model file
 * @param type Expected model type
 * @return Pointer to loaded model or NULL on error
 */
ml_model_t* ml_load_model(const char* filename, ml_model_type_t type);

/**
 * @brief Save model to file
 * @param model Model to save
 * @param filename Output file path
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_save_model(const ml_model_t* model, const char* filename);

/**
 * @brief Free model memory
 * @param model Model to free
 */
void ml_free_model(ml_model_t* model);

/* ============================================================================
 * INFERENCE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Single sample prediction
 * @param model Trained model
 * @param input Input features
 * @param output Output buffer (allocated by caller)
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_predict(const ml_model_t* model, 
                      const ml_float_t* input, 
                      ml_float_t* output);

/**
 * @brief Batch prediction for multiple samples
 * @param model Trained model
 * @param inputs Input matrix (samples x features)
 * @param outputs Output matrix (samples x outputs)
 * @param num_samples Number of samples to process
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_predict_batch(const ml_model_t* model,
                           const ml_matrix_t* inputs,
                           ml_matrix_t* outputs,
                           size_t num_samples);

/* ============================================================================
 * VECTOR OPERATIONS (SIMD-optimized)
 * ============================================================================ */

/**
 * @brief Create aligned vector
 * @param size Number of elements
 * @return Pointer to new vector or NULL on error
 */
ml_vector_t* ml_vector_create(size_t size);

/**
 * @brief Create vector from existing data
 * @param data Existing data pointer
 * @param size Number of elements
 * @param copy_data Whether to copy data or use reference
 * @return Pointer to new vector or NULL on error
 */
ml_vector_t* ml_vector_from_data(const ml_float_t* data, size_t size, bool copy_data);

/**
 * @brief Free vector memory
 * @param vector Vector to free
 */
void ml_vector_free(ml_vector_t* vector);

/**
 * @brief Vector dot product (SIMD-optimized)
 * @param a First vector
 * @param b Second vector
 * @return Dot product result
 */
ml_float_t ml_vector_dot(const ml_vector_t* a, const ml_vector_t* b);

/**
 * @brief Vector addition (SIMD-optimized)
 * @param a First vector
 * @param b Second vector
 * @param result Result vector (can be same as a or b)
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_vector_add(const ml_vector_t* a, const ml_vector_t* b, ml_vector_t* result);

/**
 * @brief Vector scalar multiplication (SIMD-optimized)
 * @param vector Input vector
 * @param scalar Scalar value
 * @param result Result vector (can be same as input)
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_vector_scale(const ml_vector_t* vector, ml_float_t scalar, ml_vector_t* result);

/**
 * @brief Set vector element
 * @param vector Target vector
 * @param index Element index
 * @param value Value to set
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_vector_set(ml_vector_t* vector, size_t index, ml_float_t value);

/**
 * @brief Get vector element
 * @param vector Source vector
 * @param index Element index
 * @return Element value or 0.0 on error
 */
ml_float_t ml_vector_get(const ml_vector_t* vector, size_t index);

/**
 * @brief Get vector sum
 * @param vector Input vector
 * @return Sum of all elements
 */
ml_float_t ml_vector_sum(const ml_vector_t* vector);

/**
 * @brief Get vector mean
 * @param vector Input vector
 * @return Mean of all elements
 */
ml_float_t ml_vector_mean(const ml_vector_t* vector);

/**
 * @brief Get vector maximum value
 * @param vector Input vector
 * @return Maximum value
 */
ml_float_t ml_vector_max(const ml_vector_t* vector);

/**
 * @brief Get vector minimum value
 * @param vector Input vector
 * @return Minimum value
 */
ml_float_t ml_vector_min(const ml_vector_t* vector);

/**
 * @brief Get index of maximum element
 * @param vector Input vector
 * @return Index of maximum element
 */
size_t ml_vector_argmax(const ml_vector_t* vector);

/**
 * @brief Print vector contents
 * @param vector Vector to print
 * @param name Optional name for display
 */
void ml_vector_print(const ml_vector_t* vector, const char* name);

/* ============================================================================
 * MATRIX OPERATIONS (SIMD-optimized)
 * ============================================================================ */

/**
 * @brief Create aligned matrix
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to new matrix or NULL on error
 */
ml_matrix_t* ml_matrix_create(size_t rows, size_t cols);

/**
 * @brief Free matrix memory
 * @param matrix Matrix to free
 */
void ml_matrix_free(ml_matrix_t* matrix);

/**
 * @brief Matrix-vector multiplication (SIMD-optimized)
 * @param matrix Input matrix
 * @param vector Input vector
 * @param result Result vector
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_matrix_vector_mul(const ml_matrix_t* matrix, 
                               const ml_vector_t* vector, 
                               ml_vector_t* result);

/**
 * @brief Matrix-matrix multiplication (SIMD-optimized)
 * @param a First matrix
 * @param b Second matrix
 * @param result Result matrix
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_matrix_mul(const ml_matrix_t* a, const ml_matrix_t* b, ml_matrix_t* result);

/**
 * @brief Set matrix element
 * @param matrix Target matrix
 * @param row Row index
 * @param col Column index
 * @param value Value to set
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_matrix_set(ml_matrix_t* matrix, size_t row, size_t col, ml_float_t value);

/**
 * @brief Get matrix element
 * @param matrix Source matrix
 * @param row Row index
 * @param col Column index
 * @return Element value or 0.0 on error
 */
ml_float_t ml_matrix_get(const ml_matrix_t* matrix, size_t row, size_t col);

/**
 * @brief Matrix transpose
 * @param input Input matrix
 * @param output Output matrix (transposed)
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_matrix_transpose(const ml_matrix_t* input, ml_matrix_t* output);

/**
 * @brief Print matrix contents
 * @param matrix Matrix to print
 * @param name Optional name for display
 */
void ml_matrix_print(const ml_matrix_t* matrix, const char* name);

/* ============================================================================
 * ACTIVATION FUNCTIONS (SIMD-optimized)
 * ============================================================================ */

/**
 * @brief Apply activation function to vector
 * @param activation Activation function type
 * @param input Input vector
 * @param output Output vector (can be same as input)
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_activation_apply(ml_activation_t activation,
                              const ml_vector_t* input,
                              ml_vector_t* output);

/**
 * @brief ReLU activation (SIMD-optimized)
 */
ml_error_t ml_activation_relu(const ml_vector_t* input, ml_vector_t* output);

/**
 * @brief Sigmoid activation (SIMD-optimized with lookup table)
 */
ml_error_t ml_activation_sigmoid(const ml_vector_t* input, ml_vector_t* output);

/**
 * @brief Tanh activation (SIMD-optimized)
 */
ml_error_t ml_activation_tanh(const ml_vector_t* input, ml_vector_t* output);

/**
 * @brief Softmax activation (numerically stable)
 */
ml_error_t ml_activation_softmax(const ml_vector_t* input, ml_vector_t* output);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Get error string for error code
 * @param error Error code
 * @return Human-readable error string
 */
const char* ml_error_string(ml_error_t error);

/**
 * @brief Print model information
 * @param model Model to inspect
 */
void ml_model_info(const ml_model_t* model);

/**
 * @brief Get model memory usage
 * @param model Model to inspect
 * @return Memory usage in bytes
 */
size_t ml_model_memory_usage(const ml_model_t* model);

/* ============================================================================
 * PERFORMANCE MONITORING
 * ============================================================================ */

/**
 * @brief Performance statistics structure
 */
typedef struct {
    uint64_t total_predictions;
    uint64_t total_time_ns;
    double avg_latency_us;
    double throughput_per_sec;
    uint64_t cache_hits;
    uint64_t cache_misses;
} ml_perf_stats_t;

/**
 * @brief Get performance statistics
 * @param stats Output statistics structure
 * @return ML_SUCCESS on success, error code otherwise
 */
ml_error_t ml_get_performance_stats(ml_perf_stats_t* stats);

/**
 * @brief Reset performance counters
 */
void ml_reset_performance_stats(void);

/**
 * @brief Print system information
 */
void ml_print_system_info(void);

/**
 * @brief Print performance statistics
 */
void ml_print_performance_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* ML_ASSEMBLY_H */
