/**
 * @file unit_tests.c
 * @brief Comprehensive unit tests for ML Assembly framework
 * 
 * Focused unit tests for individual functions and components,
 * designed for fast execution and detailed error reporting.
 */

#include "ml_assembly.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

/* Test framework configuration */
#define UNIT_TEST_TOLERANCE 1e-6f
#define MAX_TEST_NAME_LENGTH 128

/* Test statistics */
static int unit_tests_run = 0;
static int unit_tests_passed = 0;
static int unit_tests_failed = 0;
static char current_suite[MAX_TEST_NAME_LENGTH] = "";

/* Test result macros */
#define UNIT_TEST_START_SUITE(name) do { \
    strncpy(current_suite, name, MAX_TEST_NAME_LENGTH - 1); \
    current_suite[MAX_TEST_NAME_LENGTH - 1] = '\0'; \
    printf("\n--- %s ---\n", current_suite); \
} while(0)

#define UNIT_TEST_ASSERT(condition, test_name) do { \
    unit_tests_run++; \
    if (condition) { \
        unit_tests_passed++; \
        printf("  ✓ %s\n", test_name); \
    } else { \
        unit_tests_failed++; \
        printf("  ✗ %s [FAILED]\n", test_name); \
        printf("    Condition: %s\n", #condition); \
        printf("    File: %s:%d\n", __FILE__, __LINE__); \
    } \
} while(0)

#define UNIT_TEST_ASSERT_EQ(actual, expected, test_name) do { \
    unit_tests_run++; \
    if ((actual) == (expected)) { \
        unit_tests_passed++; \
        printf("  ✓ %s\n", test_name); \
    } else { \
        unit_tests_failed++; \
        printf("  ✗ %s [FAILED]\n", test_name); \
        printf("    Expected: %ld, Got: %ld\n", (long)(expected), (long)(actual)); \
        printf("    File: %s:%d\n", __FILE__, __LINE__); \
    } \
} while(0)

#define UNIT_TEST_ASSERT_FLOAT_EQ(actual, expected, test_name) do { \
    unit_tests_run++; \
    float diff = fabsf((actual) - (expected)); \
    if (diff < UNIT_TEST_TOLERANCE) { \
        unit_tests_passed++; \
        printf("  ✓ %s\n", test_name); \
    } else { \
        unit_tests_failed++; \
        printf("  ✗ %s [FAILED]\n", test_name); \
        printf("    Expected: %.6f, Got: %.6f (diff: %.6f)\n", \
               (float)(expected), (float)(actual), diff); \
        printf("    File: %s:%d\n", __FILE__, __LINE__); \
    } \
} while(0)

#define UNIT_TEST_ASSERT_NULL(ptr, test_name) do { \
    unit_tests_run++; \
    if ((ptr) == NULL) { \
        unit_tests_passed++; \
        printf("  ✓ %s\n", test_name); \
    } else { \
        unit_tests_failed++; \
        printf("  ✗ %s [FAILED]\n", test_name); \
        printf("    Expected: NULL, Got: %p\n", (void*)(ptr)); \
        printf("    File: %s:%d\n", __FILE__, __LINE__); \
    } \
} while(0)

#define UNIT_TEST_ASSERT_NOT_NULL(ptr, test_name) do { \
    unit_tests_run++; \
    if ((ptr) != NULL) { \
        unit_tests_passed++; \
        printf("  ✓ %s\n", test_name); \
    } else { \
        unit_tests_failed++; \
        printf("  ✗ %s [FAILED]\n", test_name); \
        printf("    Expected: non-NULL, Got: NULL\n"); \
        printf("    File: %s:%d\n", __FILE__, __LINE__); \
    } \
} while(0)

/* ============================================================================
 * FRAMEWORK INITIALIZATION UNIT TESTS
 * ============================================================================ */

void test_framework_init_unit(void) {
    UNIT_TEST_START_SUITE("Framework Initialization");
    
    /* Test multiple initializations */
    ml_error_t result1 = ml_init();
    UNIT_TEST_ASSERT_EQ(result1, ML_SUCCESS, "First initialization");
    
    ml_error_t result2 = ml_init();
    UNIT_TEST_ASSERT_EQ(result2, ML_SUCCESS, "Second initialization (idempotent)");
    
    /* Test version string format */
    const char* version = ml_get_version();
    UNIT_TEST_ASSERT_NOT_NULL(version, "Version string exists");
    UNIT_TEST_ASSERT(strlen(version) > 0, "Version string not empty");
    UNIT_TEST_ASSERT(strchr(version, '.') != NULL, "Version contains dot separator");
    
    /* Test CPU support detection */
    bool cpu_support = ml_check_cpu_support();
    UNIT_TEST_ASSERT(cpu_support == true || cpu_support == false, "CPU support is boolean");
    
    /* Test cleanup and re-initialization */
    ml_cleanup();
    ml_error_t result3 = ml_init();
    UNIT_TEST_ASSERT_EQ(result3, ML_SUCCESS, "Re-initialization after cleanup");
}

/* ============================================================================
 * VECTOR UNIT TESTS
 * ============================================================================ */

void test_vector_creation_unit(void) {
    UNIT_TEST_START_SUITE("Vector Creation");
    
    /* Test normal creation */
    ml_vector_t* vec = ml_vector_create(10);
    UNIT_TEST_ASSERT_NOT_NULL(vec, "Normal vector creation");
    UNIT_TEST_ASSERT_EQ(vec->size, 10, "Vector size correct");
    UNIT_TEST_ASSERT_NOT_NULL(vec->data, "Vector data allocated");
    ml_vector_free(vec);
    
    /* Test edge cases */
    ml_vector_t* empty_vec = ml_vector_create(0);
    UNIT_TEST_ASSERT_NULL(empty_vec, "Zero size vector returns NULL");
    
    ml_vector_t* large_vec = ml_vector_create(1000000);
    UNIT_TEST_ASSERT_NOT_NULL(large_vec, "Large vector creation");
    if (large_vec) ml_vector_free(large_vec);
    
    /* Test vector from data */
    float test_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ml_vector_t* vec_from_data = ml_vector_from_data(test_data, 4, true);
    UNIT_TEST_ASSERT_NOT_NULL(vec_from_data, "Vector from data creation");
    UNIT_TEST_ASSERT_EQ(vec_from_data->size, 4, "Vector from data size");
    UNIT_TEST_ASSERT_FLOAT_EQ(vec_from_data->data[0], 1.0f, "Vector from data content [0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(vec_from_data->data[3], 4.0f, "Vector from data content [3]");
    ml_vector_free(vec_from_data);
    
    /* Test NULL data */
    ml_vector_t* null_data_vec = ml_vector_from_data(NULL, 4, true);
    UNIT_TEST_ASSERT_NULL(null_data_vec, "Vector from NULL data returns NULL");
}

void test_vector_operations_unit(void) {
    UNIT_TEST_START_SUITE("Vector Operations");
    
    ml_vector_t* vec1 = ml_vector_create(4);
    ml_vector_t* vec2 = ml_vector_create(4);
    ml_vector_t* result = ml_vector_create(4);
    
    /* Initialize test vectors */
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    for (int i = 0; i < 4; i++) {
        ml_vector_set(vec1, i, data1[i]);
        ml_vector_set(vec2, i, data2[i]);
    }
    
    /* Test get/set operations */
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(vec1, 0), 1.0f, "Vector get element 0");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(vec1, 3), 4.0f, "Vector get element 3");
    
    ml_vector_set(vec1, 0, 10.0f);
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(vec1, 0), 10.0f, "Vector set/get element 0");
    ml_vector_set(vec1, 0, 1.0f); // Reset
    
    /* Test dot product */
    float dot = ml_vector_dot(vec1, vec2);
    float expected_dot = 1*5 + 2*6 + 3*7 + 4*8; // = 70
    UNIT_TEST_ASSERT_FLOAT_EQ(dot, expected_dot, "Vector dot product");
    
    /* Test vector addition */
    ml_error_t add_result = ml_vector_add(vec1, vec2, result);
    UNIT_TEST_ASSERT_EQ(add_result, ML_SUCCESS, "Vector addition returns success");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(result, 0), 6.0f, "Vector addition result [0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(result, 3), 12.0f, "Vector addition result [3]");
    
    /* Test vector scaling */
    ml_error_t scale_result = ml_vector_scale(vec1, 2.0f, result);
    UNIT_TEST_ASSERT_EQ(scale_result, ML_SUCCESS, "Vector scaling returns success");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(result, 0), 2.0f, "Vector scaling result [0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(result, 3), 8.0f, "Vector scaling result [3]");
    
    /* Test vector sum */
    float sum = ml_vector_sum(vec1);
    UNIT_TEST_ASSERT_FLOAT_EQ(sum, 10.0f, "Vector sum");
    
    /* Test error conditions */
    ml_vector_t* wrong_size = ml_vector_create(3);
    ml_error_t error_result = ml_vector_add(vec1, wrong_size, result);
    UNIT_TEST_ASSERT_EQ(error_result, ML_ERROR_DIMENSION_MISMATCH, "Size mismatch error");
    
    ml_vector_free(vec1);
    ml_vector_free(vec2);
    ml_vector_free(result);
    ml_vector_free(wrong_size);
}

void test_vector_edge_cases_unit(void) {
    UNIT_TEST_START_SUITE("Vector Edge Cases");
    
    /* Test NULL pointer handling */
    float dot_null = ml_vector_dot(NULL, NULL);
    UNIT_TEST_ASSERT(isnan(dot_null) || dot_null == 0.0f, "Dot product with NULL vectors");
    
    ml_error_t add_null = ml_vector_add(NULL, NULL, NULL);
    UNIT_TEST_ASSERT_EQ(add_null, ML_ERROR_NULL_POINTER, "Addition with NULL pointers");
    
    /* Test boundary values */
    ml_vector_t* vec = ml_vector_create(1);
    ml_vector_set(vec, 0, 0.0f);
    float sum_zero = ml_vector_sum(vec);
    UNIT_TEST_ASSERT_FLOAT_EQ(sum_zero, 0.0f, "Sum of zero vector");
    
    ml_vector_set(vec, 0, -1.0f);
    float sum_negative = ml_vector_sum(vec);
    UNIT_TEST_ASSERT_FLOAT_EQ(sum_negative, -1.0f, "Sum with negative values");
    
    ml_vector_free(vec);
}

/* ============================================================================
 * MATRIX UNIT TESTS
 * ============================================================================ */

void test_matrix_creation_unit(void) {
    UNIT_TEST_START_SUITE("Matrix Creation");
    
    /* Test normal creation */
    ml_matrix_t* mat = ml_matrix_create(3, 4);
    UNIT_TEST_ASSERT_NOT_NULL(mat, "Normal matrix creation");
    UNIT_TEST_ASSERT_EQ(mat->rows, 3, "Matrix rows correct");
    UNIT_TEST_ASSERT_EQ(mat->cols, 4, "Matrix cols correct");
    UNIT_TEST_ASSERT_NOT_NULL(mat->data, "Matrix data allocated");
    ml_matrix_free(mat);
    
    /* Test edge cases */
    ml_matrix_t* zero_mat = ml_matrix_create(0, 5);
    UNIT_TEST_ASSERT_NULL(zero_mat, "Zero rows matrix returns NULL");
    
    ml_matrix_t* zero_col_mat = ml_matrix_create(5, 0);
    UNIT_TEST_ASSERT_NULL(zero_col_mat, "Zero cols matrix returns NULL");
    
    /* Test square matrix */
    ml_matrix_t* square = ml_matrix_create(5, 5);
    UNIT_TEST_ASSERT_NOT_NULL(square, "Square matrix creation");
    UNIT_TEST_ASSERT_EQ(square->rows, square->cols, "Square matrix dimensions");
    ml_matrix_free(square);
}

void test_matrix_operations_unit(void) {
    UNIT_TEST_START_SUITE("Matrix Operations");
    
    /* Create test matrices */
    ml_matrix_t* mat1 = ml_matrix_create(2, 3);
    ml_matrix_t* mat2 = ml_matrix_create(3, 2);
    ml_matrix_t* result = ml_matrix_create(2, 2);
    
    /* Initialize mat1 = [[1, 2, 3], [4, 5, 6]] */
    ml_matrix_set(mat1, 0, 0, 1.0f); ml_matrix_set(mat1, 0, 1, 2.0f); ml_matrix_set(mat1, 0, 2, 3.0f);
    ml_matrix_set(mat1, 1, 0, 4.0f); ml_matrix_set(mat1, 1, 1, 5.0f); ml_matrix_set(mat1, 1, 2, 6.0f);
    
    /* Initialize mat2 = [[1, 2], [3, 4], [5, 6]] */
    ml_matrix_set(mat2, 0, 0, 1.0f); ml_matrix_set(mat2, 0, 1, 2.0f);
    ml_matrix_set(mat2, 1, 0, 3.0f); ml_matrix_set(mat2, 1, 1, 4.0f);
    ml_matrix_set(mat2, 2, 0, 5.0f); ml_matrix_set(mat2, 2, 1, 6.0f);
    
    /* Test get/set operations */
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(mat1, 0, 0), 1.0f, "Matrix get [0,0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(mat1, 1, 2), 6.0f, "Matrix get [1,2]");
    
    /* Test matrix multiplication */
    ml_error_t mul_result = ml_matrix_mul(mat1, mat2, result);
    UNIT_TEST_ASSERT_EQ(mul_result, ML_SUCCESS, "Matrix multiplication returns success");
    
    /* Expected: [[22, 28], [49, 64]] */
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 0, 0), 22.0f, "Matrix mul result [0,0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 0, 1), 28.0f, "Matrix mul result [0,1]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 1, 0), 49.0f, "Matrix mul result [1,0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 1, 1), 64.0f, "Matrix mul result [1,1]");
    
    /* Test matrix-vector multiplication */
    ml_vector_t* vec = ml_vector_create(3);
    ml_vector_t* mv_result = ml_vector_create(2);
    ml_vector_set(vec, 0, 1.0f); ml_vector_set(vec, 1, 2.0f); ml_vector_set(vec, 2, 3.0f);
    
    ml_error_t mv_mul_result = ml_matrix_vector_mul(mat1, vec, mv_result);
    UNIT_TEST_ASSERT_EQ(mv_mul_result, ML_SUCCESS, "Matrix-vector multiplication returns success");
    
    /* Expected: [14, 32] */
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(mv_result, 0), 14.0f, "Matrix-vector result [0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(mv_result, 1), 32.0f, "Matrix-vector result [1]");
    
    /* Test transpose */
    ml_matrix_t* transposed = ml_matrix_create(3, 2);
    ml_error_t trans_result = ml_matrix_transpose(mat1, transposed);
    UNIT_TEST_ASSERT_EQ(trans_result, ML_SUCCESS, "Matrix transpose returns success");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 0, 0), 1.0f, "Transpose [0,0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 1, 0), 2.0f, "Transpose [1,0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 0, 1), 4.0f, "Transpose [0,1]");
    
    ml_matrix_free(mat1);
    ml_matrix_free(mat2);
    ml_matrix_free(result);
    ml_matrix_free(transposed);
    ml_vector_free(vec);
    ml_vector_free(mv_result);
}

void test_matrix_edge_cases_unit(void) {
    UNIT_TEST_START_SUITE("Matrix Edge Cases");
    
    /* Test dimension mismatch */
    ml_matrix_t* mat1 = ml_matrix_create(2, 3);
    ml_matrix_t* mat2 = ml_matrix_create(2, 3); // Wrong dimensions for multiplication
    ml_matrix_t* result = ml_matrix_create(2, 3);
    
    ml_error_t error_result = ml_matrix_mul(mat1, mat2, result);
    UNIT_TEST_ASSERT_EQ(error_result, ML_ERROR_DIMENSION_MISMATCH, "Dimension mismatch error");
    
    /* Test NULL pointer handling */
    ml_error_t null_result = ml_matrix_mul(NULL, NULL, NULL);
    UNIT_TEST_ASSERT_EQ(null_result, ML_ERROR_NULL_POINTER, "Null pointer error");
    
    ml_matrix_free(mat1);
    ml_matrix_free(mat2);
    ml_matrix_free(result);
}

/* ============================================================================
 * ACTIVATION FUNCTION UNIT TESTS
 * ============================================================================ */

void test_activation_relu_unit(void) {
    UNIT_TEST_START_SUITE("ReLU Activation");
    
    ml_vector_t* input = ml_vector_create(5);
    ml_vector_t* output = ml_vector_create(5);
    
    /* Test data: [-2, -1, 0, 1, 2] */
    float test_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 5; i++) {
        ml_vector_set(input, i, test_data[i]);
    }
    
    ml_error_t result = ml_activation_relu(input, output);
    UNIT_TEST_ASSERT_EQ(result, ML_SUCCESS, "ReLU execution");
    
    /* Expected: [0, 0, 0, 1, 2] */
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 0), 0.0f, "ReLU(-2) = 0");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 1), 0.0f, "ReLU(-1) = 0");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 2), 0.0f, "ReLU(0) = 0");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 3), 1.0f, "ReLU(1) = 1");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 4), 2.0f, "ReLU(2) = 2");
    
    ml_vector_free(input);
    ml_vector_free(output);
}

void test_activation_sigmoid_unit(void) {
    UNIT_TEST_START_SUITE("Sigmoid Activation");
    
    ml_vector_t* input = ml_vector_create(3);
    ml_vector_t* output = ml_vector_create(3);
    
    /* Test specific values */
    ml_vector_set(input, 0, 0.0f);    // sigmoid(0) should be 0.5
    ml_vector_set(input, 1, 1000.0f); // sigmoid(large) should be ~1
    ml_vector_set(input, 2, -1000.0f); // sigmoid(-large) should be ~0
    
    ml_error_t result = ml_activation_sigmoid(input, output);
    UNIT_TEST_ASSERT_EQ(result, ML_SUCCESS, "Sigmoid execution");
    
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 0), 0.5f, "Sigmoid(0) = 0.5");
    UNIT_TEST_ASSERT(ml_vector_get(output, 1) > 0.99f, "Sigmoid(large) ≈ 1");
    UNIT_TEST_ASSERT(ml_vector_get(output, 2) < 0.01f, "Sigmoid(-large) ≈ 0");
    
    /* Test range constraints */
    for (int i = 0; i < 3; i++) {
        float val = ml_vector_get(output, i);
        UNIT_TEST_ASSERT(val > 0.0f && val < 1.0f, "Sigmoid output in (0,1)");
    }
    
    ml_vector_free(input);
    ml_vector_free(output);
}

void test_activation_tanh_unit(void) {
    UNIT_TEST_START_SUITE("Tanh Activation");
    
    ml_vector_t* input = ml_vector_create(3);
    ml_vector_t* output = ml_vector_create(3);
    
    ml_vector_set(input, 0, 0.0f);
    ml_vector_set(input, 1, 1000.0f);
    ml_vector_set(input, 2, -1000.0f);
    
    ml_error_t result = ml_activation_tanh(input, output);
    UNIT_TEST_ASSERT_EQ(result, ML_SUCCESS, "Tanh execution");
    
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 0), 0.0f, "Tanh(0) = 0");
    UNIT_TEST_ASSERT(ml_vector_get(output, 1) > 0.99f, "Tanh(large) ≈ 1");
    UNIT_TEST_ASSERT(ml_vector_get(output, 2) < -0.99f, "Tanh(-large) ≈ -1");
    
    /* Test range constraints */
    for (int i = 0; i < 3; i++) {
        float val = ml_vector_get(output, i);
        UNIT_TEST_ASSERT(val > -1.0f && val < 1.0f, "Tanh output in (-1,1)");
    }
    
    ml_vector_free(input);
    ml_vector_free(output);
}

void test_activation_softmax_unit(void) {
    UNIT_TEST_START_SUITE("Softmax Activation");
    
    ml_vector_t* input = ml_vector_create(3);
    ml_vector_t* output = ml_vector_create(3);
    
    /* Test with equal inputs - should give equal outputs */
    ml_vector_set(input, 0, 1.0f);
    ml_vector_set(input, 1, 1.0f);
    ml_vector_set(input, 2, 1.0f);
    
    ml_error_t result = ml_activation_softmax(input, output);
    UNIT_TEST_ASSERT_EQ(result, ML_SUCCESS, "Softmax execution");
    
    float expected_equal = 1.0f / 3.0f;
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 0), expected_equal, "Softmax equal inputs [0]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 1), expected_equal, "Softmax equal inputs [1]");
    UNIT_TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, 2), expected_equal, "Softmax equal inputs [2]");
    
    /* Test sum to 1 property */
    float sum = ml_vector_get(output, 0) + ml_vector_get(output, 1) + ml_vector_get(output, 2);
    UNIT_TEST_ASSERT_FLOAT_EQ(sum, 1.0f, "Softmax outputs sum to 1");
    
    /* Test with different inputs */
    ml_vector_set(input, 0, 1.0f);
    ml_vector_set(input, 1, 2.0f);
    ml_vector_set(input, 2, 3.0f);
    
    ml_activation_softmax(input, output);
    
    /* Should be monotonic */
    UNIT_TEST_ASSERT(ml_vector_get(output, 0) < ml_vector_get(output, 1), "Softmax monotonic property 1");
    UNIT_TEST_ASSERT(ml_vector_get(output, 1) < ml_vector_get(output, 2), "Softmax monotonic property 2");
    
    ml_vector_free(input);
    ml_vector_free(output);
}

/* ============================================================================
 * MODEL UNIT TESTS
 * ============================================================================ */

void test_model_creation_unit(void) {
    UNIT_TEST_START_SUITE("Model Creation");
    
    /* Test linear regression model */
    ml_model_config_t config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 5,
        .output_size = 1,
        .layer_count = 1
    };
    
    ml_model_t* model = ml_create_model(&config);
    UNIT_TEST_ASSERT_NOT_NULL(model, "Linear regression model creation");
    UNIT_TEST_ASSERT_EQ(model->config.type, ML_LINEAR_REGRESSION, "Model type correct");
    UNIT_TEST_ASSERT_EQ(model->config.input_size, 5, "Model input size correct");
    UNIT_TEST_ASSERT_EQ(model->config.output_size, 1, "Model output size correct");
    
    ml_free_model(model);
    
    /* Test invalid configuration */
    ml_model_config_t invalid_config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 0,  // Invalid
        .output_size = 1,
        .layer_count = 1
    };
    
    ml_model_t* invalid_model = ml_create_model(&invalid_config);
    UNIT_TEST_ASSERT_NULL(invalid_model, "Invalid model configuration returns NULL");
    
    /* Test NULL configuration */
    ml_model_t* null_config_model = ml_create_model(NULL);
    UNIT_TEST_ASSERT_NULL(null_config_model, "NULL configuration returns NULL");
}

void test_model_prediction_unit(void) {
    UNIT_TEST_START_SUITE("Model Prediction");
    
    ml_model_config_t config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 3,
        .output_size = 1,
        .layer_count = 1
    };
    
    ml_model_t* model = ml_create_model(&config);
    if (!model) {
        printf("  Model creation failed, skipping prediction tests\n");
        return;
    }
    
    /* Set known weights */
    float weights[] = {1.0f, 2.0f, 3.0f};
    extern ml_error_t ml_linear_regression_set_weights(ml_model_t* model, const ml_float_t* weights, ml_float_t bias);
    ml_linear_regression_set_weights(model, weights, 0.0f);
    
    /* Test single prediction */
    float input[] = {1.0f, 1.0f, 1.0f};
    float output;
    ml_error_t result = ml_predict(model, input, &output);
    UNIT_TEST_ASSERT_EQ(result, ML_SUCCESS, "Single prediction execution");
    UNIT_TEST_ASSERT_FLOAT_EQ(output, 6.0f, "Single prediction result"); // 1*1 + 2*1 + 3*1 = 6
    
    /* Test with different input */
    float input2[] = {2.0f, 3.0f, 4.0f};
    ml_error_t result2 = ml_predict(model, input2, &output);
    UNIT_TEST_ASSERT_EQ(result2, ML_SUCCESS, "Second prediction execution");
    UNIT_TEST_ASSERT_FLOAT_EQ(output, 20.0f, "Second prediction result"); // 1*2 + 2*3 + 3*4 = 20
    
    /* Test NULL input handling */
    ml_error_t null_result = ml_predict(model, NULL, &output);
    UNIT_TEST_ASSERT_EQ(null_result, ML_ERROR_NULL_POINTER, "NULL input error");
    
    ml_error_t null_output = ml_predict(model, input, NULL);
    UNIT_TEST_ASSERT_EQ(null_output, ML_ERROR_NULL_POINTER, "NULL output error");
    
    ml_free_model(model);
}

/* ============================================================================
 * ERROR HANDLING UNIT TESTS
 * ============================================================================ */

void test_error_handling_unit(void) {
    UNIT_TEST_START_SUITE("Error Handling");
    
    /* Test error string function */
    const char* success_str = ml_error_string(ML_SUCCESS);
    UNIT_TEST_ASSERT_NOT_NULL(success_str, "Success error string exists");
    UNIT_TEST_ASSERT(strlen(success_str) > 0, "Success error string not empty");
    
    const char* null_ptr_str = ml_error_string(ML_ERROR_NULL_POINTER);
    UNIT_TEST_ASSERT_NOT_NULL(null_ptr_str, "Null pointer error string exists");
    UNIT_TEST_ASSERT(strstr(null_ptr_str, "null") != NULL || strstr(null_ptr_str, "Null") != NULL, 
                     "Null pointer error string mentions null");
    
    /* Test invalid error code */
    const char* invalid_str = ml_error_string((ml_error_t)999);
    UNIT_TEST_ASSERT_NOT_NULL(invalid_str, "Invalid error code returns string");
}

/* ============================================================================
 * PERFORMANCE STATISTICS UNIT TESTS
 * ============================================================================ */

void test_performance_stats_unit(void) {
    UNIT_TEST_START_SUITE("Performance Statistics");
    
    /* Reset and test initial state */
    ml_reset_performance_stats();
    ml_perf_stats_t stats;
    ml_get_performance_stats(&stats);
    
    UNIT_TEST_ASSERT_EQ(stats.total_predictions, 0, "Initial predictions count is zero");
    UNIT_TEST_ASSERT_FLOAT_EQ(stats.avg_latency_us, 0.0f, "Initial average latency is zero");
    UNIT_TEST_ASSERT_FLOAT_EQ(stats.throughput_per_sec, 0.0f, "Initial throughput is zero");
    
    /* Perform some predictions to generate stats */
    ml_model_config_t config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 3,
        .output_size = 1,
        .layer_count = 1
    };
    
    ml_model_t* model = ml_create_model(&config);
    if (model) {
        float weights[] = {1.0f, 1.0f, 1.0f};
        extern ml_error_t ml_linear_regression_set_weights(ml_model_t* model, const ml_float_t* weights, ml_float_t bias);
        ml_linear_regression_set_weights(model, weights, 0.0f);
        
        float input[] = {1.0f, 1.0f, 1.0f};
        float output;
        
        /* Make several predictions */
        for (int i = 0; i < 10; i++) {
            ml_predict(model, input, &output);
        }
        
        ml_get_performance_stats(&stats);
        UNIT_TEST_ASSERT_EQ(stats.total_predictions, 10, "Predictions count after 10 calls");
        UNIT_TEST_ASSERT(stats.avg_latency_us >= 0.0f, "Average latency is non-negative");
        UNIT_TEST_ASSERT(stats.throughput_per_sec >= 0.0f, "Throughput is non-negative");
        
        ml_free_model(model);
    }
}

/* ============================================================================
 * MAIN UNIT TEST RUNNER
 * ============================================================================ */

void print_unit_test_summary(void) {
    printf("\n================== UNIT TEST SUMMARY ==================\n");
    printf("Total Tests: %d\n", unit_tests_run);
    printf("Passed: %d\n", unit_tests_passed);
    printf("Failed: %d\n", unit_tests_failed);
    printf("Success Rate: %.1f%%\n", 
           unit_tests_run > 0 ? (float)unit_tests_passed / unit_tests_run * 100 : 0.0f);
    
    if (unit_tests_failed == 0) {
        printf("🎉 ALL UNIT TESTS PASSED! 🎉\n");
    } else {
        printf("❌ %d unit tests failed. See details above.\n", unit_tests_failed);
    }
    printf("=======================================================\n");
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv; /* Suppress unused parameter warnings */
    
    printf("ML Assembly Framework - Comprehensive Unit Tests\n");
    printf("=======================================================\n");
    
    /* Initialize framework */
    ml_error_t init_result = ml_init();
    if (init_result != ML_SUCCESS) {
        printf("Framework initialization failed: %d\n", init_result);
        return 1;
    }
    
    /* Run all unit test suites */
    test_framework_init_unit();
    test_vector_creation_unit();
    test_vector_operations_unit();
    test_vector_edge_cases_unit();
    test_matrix_creation_unit();
    test_matrix_operations_unit();
    test_matrix_edge_cases_unit();
    test_activation_relu_unit();
    test_activation_sigmoid_unit();
    test_activation_tanh_unit();
    test_activation_softmax_unit();
    test_model_creation_unit();
    test_model_prediction_unit();
    test_error_handling_unit();
    test_performance_stats_unit();
    
    /* Cleanup */
    ml_cleanup();
    
    /* Print summary and return appropriate exit code */
    print_unit_test_summary();
    return (unit_tests_failed == 0) ? 0 : 1;
}
