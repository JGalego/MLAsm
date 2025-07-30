# API Reference

## Overview

The ML Assembly framework provides a high-performance C API for machine learning inference, implemented with optimized x86-64 assembly code for maximum performance.

## Core Types

### Error Handling

```c
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
```

### Model Types

```c
typedef enum {
    ML_LINEAR_REGRESSION = 0,
    ML_LOGISTIC_REGRESSION = 1,
    ML_NEURAL_NETWORK = 2,
    ML_DECISION_TREE = 3,
    ML_RANDOM_FOREST = 4
} ml_model_type_t;
```

### Data Structures

#### Vector

```c
typedef struct ml_vector {
    ml_float_t* data;      // Data pointer (32-byte aligned)
    size_t size;           // Number of elements
    size_t capacity;       // Allocated capacity
    bool owns_data;        // Memory ownership flag
} ml_vector_t;
```

#### Matrix

```c
typedef struct ml_matrix {
    ml_float_t* data;      // Data pointer (row-major, 32-byte aligned)
    size_t rows;           // Number of rows
    size_t cols;           // Number of columns
    size_t capacity;       // Allocated capacity
    bool owns_data;        // Memory ownership flag
} ml_matrix_t;
```

#### Model

```c
typedef struct ml_model {
    ml_model_config_t config;  // Model configuration
    void* internal_data;       // Model-specific data
    bool is_loaded;           // Load status
} ml_model_t;
```

## Framework Functions

### Initialization

#### `ml_init()`
```c
ml_error_t ml_init(void);
```
Initializes the ML Assembly framework. Must be called before using any other functions.

**Returns:** `ML_SUCCESS` on success, error code on failure.

#### `ml_cleanup()`
```c
void ml_cleanup(void);
```
Cleans up framework resources. Call when done using the framework.

#### `ml_check_cpu_support()`
```c
bool ml_check_cpu_support(void);
```
Checks if the CPU supports required features (AVX2 minimum).

**Returns:** `true` if CPU is supported, `false` otherwise.

#### `ml_get_version()`
```c
const char* ml_get_version(void);
```
Returns the framework version string.

**Returns:** Version string in "major.minor.patch" format.

## Vector Operations

### Creation and Management

#### `ml_vector_create()`
```c
ml_vector_t* ml_vector_create(size_t size);
```
Creates a new vector with specified size.

**Parameters:**
- `size`: Number of elements

**Returns:** Pointer to new vector or `NULL` on error.

#### `ml_vector_from_data()`
```c
ml_vector_t* ml_vector_from_data(const ml_float_t* data, size_t size, bool copy_data);
```
Creates a vector from existing data.

**Parameters:**
- `data`: Source data pointer
- `size`: Number of elements
- `copy_data`: Whether to copy data or use reference

**Returns:** Pointer to new vector or `NULL` on error.

#### `ml_vector_free()`
```c
void ml_vector_free(ml_vector_t* vector);
```
Frees vector memory.

**Parameters:**
- `vector`: Vector to free

### Arithmetic Operations

#### `ml_vector_dot()`
```c
ml_float_t ml_vector_dot(const ml_vector_t* a, const ml_vector_t* b);
```
Computes dot product of two vectors using SIMD optimization.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Dot product result.

**Performance:** ~0.23μs for 1000 elements (4.4M ops/sec) with AVX2 optimization.

#### `ml_vector_add()`
```c
ml_error_t ml_vector_add(const ml_vector_t* a, const ml_vector_t* b, ml_vector_t* result);
```
Element-wise vector addition with SIMD optimization.

**Parameters:**
- `a`: First vector
- `b`: Second vector
- `result`: Result vector (can be same as input)

**Returns:** `ML_SUCCESS` on success, error code on failure.

#### `ml_vector_scale()`
```c
ml_error_t ml_vector_scale(const ml_vector_t* vector, ml_float_t scalar, ml_vector_t* result);
```
Scales vector by scalar value using SIMD optimization.

**Parameters:**
- `vector`: Input vector
- `scalar`: Scalar multiplier
- `result`: Result vector (can be same as input)

**Returns:** `ML_SUCCESS` on success, error code on failure.

### Utility Functions

#### `ml_vector_sum()`
```c
ml_float_t ml_vector_sum(const ml_vector_t* vector);
```
Sums all elements in vector.

#### `ml_vector_mean()`
```c
ml_float_t ml_vector_mean(const ml_vector_t* vector);
```
Computes mean of vector elements.

#### `ml_vector_max()` / `ml_vector_min()`
```c
ml_float_t ml_vector_max(const ml_vector_t* vector);
ml_float_t ml_vector_min(const ml_vector_t* vector);
```
Finds maximum/minimum element.

#### `ml_vector_argmax()`
```c
size_t ml_vector_argmax(const ml_vector_t* vector);
```
Returns index of maximum element.

## Matrix Operations

### Creation and Management

#### `ml_matrix_create()`
```c
ml_matrix_t* ml_matrix_create(size_t rows, size_t cols);
```
Creates a new matrix with specified dimensions.

**Parameters:**
- `rows`: Number of rows
- `cols`: Number of columns

**Returns:** Pointer to new matrix or `NULL` on error.

#### `ml_matrix_free()`
```c
void ml_matrix_free(ml_matrix_t* matrix);
```
Frees matrix memory.

### Arithmetic Operations

#### `ml_matrix_vector_mul()`
```c
ml_error_t ml_matrix_vector_mul(const ml_matrix_t* matrix, 
                               const ml_vector_t* vector, 
                               ml_vector_t* result);
```
Matrix-vector multiplication with SIMD optimization.

**Parameters:**
- `matrix`: Input matrix (M×N)
- `vector`: Input vector (N×1)
- `result`: Result vector (M×1)

**Returns:** `ML_SUCCESS` on success, error code on failure.

**Performance:** Optimized for cache efficiency and vectorization.

#### `ml_matrix_mul()`
```c
ml_error_t ml_matrix_mul(const ml_matrix_t* a, const ml_matrix_t* b, ml_matrix_t* result);
```
Matrix-matrix multiplication.

**Parameters:**
- `a`: First matrix (M×K)
- `b`: Second matrix (K×N)  
- `result`: Result matrix (M×N)

**Returns:** `ML_SUCCESS` on success, error code on failure.

## Activation Functions

### Available Activations

#### `ml_activation_relu()`
```c
ml_error_t ml_activation_relu(const ml_vector_t* input, ml_vector_t* output);
```
ReLU activation: `f(x) = max(0, x)`

**Performance:** ~0.07μs for 1000 elements (14.8M ops/sec) with SIMD optimization.

#### `ml_activation_sigmoid()`
```c
ml_error_t ml_activation_sigmoid(const ml_vector_t* input, ml_vector_t* output);
```
Sigmoid activation: `f(x) = 1 / (1 + exp(-x))`

Uses fast approximation with lookup tables.

#### `ml_activation_tanh()`
```c
ml_error_t ml_activation_tanh(const ml_vector_t* input, ml_vector_t* output);
```
Hyperbolic tangent activation.

#### `ml_activation_softmax()`
```c
ml_error_t ml_activation_softmax(const ml_vector_t* input, ml_vector_t* output);
```
Softmax activation (numerically stable implementation).

### Generic Interface

#### `ml_activation_apply()`
```c
ml_error_t ml_activation_apply(ml_activation_t activation,
                              const ml_vector_t* input,
                              ml_vector_t* output);
```
Applies specified activation function.

**Parameters:**
- `activation`: Activation function type
- `input`: Input vector
- `output`: Output vector

## Model Management

### Creation and Loading

#### `ml_create_model()`
```c
ml_model_t* ml_create_model(const ml_model_config_t* config);
```
Creates a new model with specified configuration.

#### `ml_load_model()`
```c
ml_model_t* ml_load_model(const char* filename, ml_model_type_t type);
```
Loads a model from file.

#### `ml_save_model()`
```c
ml_error_t ml_save_model(const ml_model_t* model, const char* filename);
```
Saves a model to file.

#### `ml_free_model()`
```c
void ml_free_model(ml_model_t* model);
```
Frees model memory.

### Inference

#### `ml_predict()`
```c
ml_error_t ml_predict(const ml_model_t* model, 
                      const ml_float_t* input, 
                      ml_float_t* output);
```
Single sample prediction.

**Parameters:**
- `model`: Trained model
- `input`: Input features array
- `output`: Output buffer

**Returns:** `ML_SUCCESS` on success, error code on failure.

**Performance:** Linear regression: ~0.25μs for 100 features (3.9M predictions/sec), scales linearly with features.

#### `ml_predict_batch()`
```c
ml_error_t ml_predict_batch(const ml_model_t* model,
                           const ml_matrix_t* inputs,
                           ml_matrix_t* outputs,
                           size_t num_samples);
```
Batch prediction for multiple samples.

## Performance Monitoring

#### `ml_get_performance_stats()`
```c
ml_error_t ml_get_performance_stats(ml_perf_stats_t* stats);
```
Retrieves performance statistics.

#### `ml_reset_performance_stats()`
```c
void ml_reset_performance_stats(void);
```
Resets performance counters.

## Error Handling

#### `ml_error_string()`
```c
const char* ml_error_string(ml_error_t error);
```
Returns human-readable error message.

## Best Practices

### Memory Management
- Always call `ml_init()` before using the framework
- Free all allocated vectors, matrices, and models
- Check return values for allocation functions

### Performance Tips
- Use aligned memory for best SIMD performance
- Prefer batch operations when possible
- Reuse vectors/matrices to avoid allocation overhead
- Profile your application to identify bottlenecks

### Error Handling
- Always check return values
- Use `ml_error_string()` for debugging
- Handle dimension mismatches gracefully

## Example Usage

```c
#include "ml_assembly.h"

int main() {
    // Initialize framework
    if (ml_init() != ML_SUCCESS) {
        printf("Failed to initialize framework\n");
        return 1;
    }
    
    // Create model
    ml_model_config_t config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 4,
        .output_size = 1
    };
    
    ml_model_t* model = ml_create_model(&config);
    if (!model) {
        printf("Failed to create model\n");
        ml_cleanup();
        return 1;
    }
    
    // Set weights (would normally load from file)
    float weights[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ml_linear_regression_set_weights(model, weights, 0.5f);
    
    // Make prediction
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output;
    
    if (ml_predict(model, input, &output) == ML_SUCCESS) {
        printf("Prediction: %f\n", output);
    }
    
    // Cleanup
    ml_free_model(model);
    ml_cleanup();
    
    return 0;
}
```
