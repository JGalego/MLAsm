/**
 * @file test_main.c
 * @brief Main test runner for ML Assembly framework
 * 
 * Comprehensive test suite covering unit tests, integration tests,
 * and performance benchmarks.
 */

#include "ml_assembly.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

/* Test framework macros */
#define TEST_TOLERANCE 1e-6f
#define PERFORMANCE_ITERATIONS 100000

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/* Test result macros */
#define TEST_ASSERT(condition, message) do { \
    tests_run++; \
    if (condition) { \
        tests_passed++; \
        printf("✓ %s\n", message); \
    } else { \
        tests_failed++; \
        printf("✗ %s\n", message); \
    } \
} while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, message) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < TEST_TOLERANCE) { \
        tests_passed++; \
        printf("✓ %s (%.6f ≈ %.6f)\n", message, (float)(a), (float)(b)); \
    } else { \
        tests_failed++; \
        printf("✗ %s (%.6f ≠ %.6f, diff=%.6f)\n", message, (float)(a), (float)(b), fabsf((a)-(b))); \
    } \
} while(0)

/* Forward declarations */
void test_framework_init(void);
void test_vector_operations(void);
void test_matrix_operations(void);
void test_activation_functions(void);
void test_linear_regression(void);
void test_performance_benchmarks(void);
void print_test_summary(void);

/* ============================================================================
 * FRAMEWORK TESTS
 * ============================================================================ */

void test_framework_init(void) {
    printf("\n=== Framework Initialization Tests ===\n");
    
    /* Test initialization */
    ml_error_t init_result = ml_init();
    TEST_ASSERT(init_result == ML_SUCCESS, "Framework initialization");
    
    /* Test CPU support check */
    bool cpu_support = ml_check_cpu_support();
    printf("CPU Support: %s\n", cpu_support ? "Yes" : "No");
    TEST_ASSERT(true, "CPU support check (informational)");
    
    /* Test version string */
    const char* version = ml_get_version();
    TEST_ASSERT(version != NULL && strlen(version) > 0, "Version string retrieval");
    printf("Framework Version: %s\n", version);
    
    /* Test system info */
    printf("System Information:\n");
    ml_print_system_info();
}

/* ============================================================================
 * VECTOR OPERATION TESTS
 * ============================================================================ */

void test_vector_operations(void) {
    printf("\n=== Vector Operation Tests ===\n");
    
    /* Test vector creation */
    ml_vector_t* vec1 = ml_vector_create(4);
    ml_vector_t* vec2 = ml_vector_create(4);
    ml_vector_t* result = ml_vector_create(4);
    
    TEST_ASSERT(vec1 != NULL && vec2 != NULL && result != NULL, "Vector creation");
    
    if (!vec1 || !vec2 || !result) {
        printf("Vector creation failed, skipping vector tests\n");
        return;
    }
    
    /* Initialize test data */
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data2[] = {2.0f, 3.0f, 4.0f, 5.0f};
    
    for (int i = 0; i < 4; i++) {
        ml_vector_set(vec1, i, data1[i]);
        ml_vector_set(vec2, i, data2[i]);
    }
    
    /* Test dot product */
    float dot = ml_vector_dot(vec1, vec2);
    float expected_dot = 1*2 + 2*3 + 3*4 + 4*5; /* = 40 */
    TEST_ASSERT_FLOAT_EQ(dot, expected_dot, "Vector dot product");
    
    /* Test vector addition */
    ml_error_t add_result = ml_vector_add(vec1, vec2, result);
    TEST_ASSERT(add_result == ML_SUCCESS, "Vector addition execution");
    
    for (int i = 0; i < 4; i++) {
        float expected = data1[i] + data2[i];
        float actual = ml_vector_get(result, i);
        char msg[100];
        snprintf(msg, sizeof(msg), "Vector addition element [%d]", i);
        TEST_ASSERT_FLOAT_EQ(actual, expected, msg);
    }
    
    /* Test vector scaling */
    float scalar = 2.5f;
    ml_error_t scale_result = ml_vector_scale(vec1, scalar, result);
    TEST_ASSERT(scale_result == ML_SUCCESS, "Vector scaling execution");
    
    for (int i = 0; i < 4; i++) {
        float expected = data1[i] * scalar;
        float actual = ml_vector_get(result, i);
        char msg[100];
        snprintf(msg, sizeof(msg), "Vector scaling element [%d]", i);
        TEST_ASSERT_FLOAT_EQ(actual, expected, msg);
    }
    
    /* Test vector sum */
    float sum = ml_vector_sum(vec1);
    float expected_sum = 1 + 2 + 3 + 4; /* = 10 */
    TEST_ASSERT_FLOAT_EQ(sum, expected_sum, "Vector sum");
    
    /* Test vector scaling (instead of normalization which doesn't exist) */
    ml_vector_t* scaled = ml_vector_create(4);
    ml_error_t scale_result2 = ml_vector_scale(vec1, 2.0f, scaled);
    TEST_ASSERT(scale_result2 == ML_SUCCESS, "Vector scaling execution");
    
    /* Check that scaled vector has correct values */
    TEST_ASSERT_FLOAT_EQ(scaled->data[0], 2.0f, "Scaled vector element 0");
    TEST_ASSERT_FLOAT_EQ(scaled->data[1], 4.0f, "Scaled vector element 1");
    
    /* Test vector from data */
    ml_vector_t* vec_from_data = ml_vector_from_data(data1, 4, true);
    TEST_ASSERT(vec_from_data != NULL, "Vector from data creation");
    
    float dot_from_data = ml_vector_dot(vec1, vec_from_data);
    float expected_dot_from_data = 1*1 + 2*2 + 3*3 + 4*4; /* = 30 */
    TEST_ASSERT_FLOAT_EQ(dot_from_data, expected_dot_from_data, "Vector from data content");
    
    /* Cleanup */
    ml_vector_free(vec1);
    ml_vector_free(vec2);
    ml_vector_free(result);
    ml_vector_free(scaled);
    ml_vector_free(vec_from_data);
}

/* ============================================================================
 * MATRIX OPERATION TESTS
 * ============================================================================ */

void test_matrix_operations(void) {
    printf("\n=== Matrix Operation Tests ===\n");
    
    /* Test matrix creation */
    ml_matrix_t* mat1 = ml_matrix_create(2, 3);
    ml_matrix_t* mat2 = ml_matrix_create(3, 2);
    ml_matrix_t* result = ml_matrix_create(2, 2);
    ml_vector_t* vec = ml_vector_create(3);
    ml_vector_t* mat_vec_result = ml_vector_create(2);
    
    TEST_ASSERT(mat1 != NULL && mat2 != NULL && result != NULL, "Matrix creation");
    TEST_ASSERT(vec != NULL && mat_vec_result != NULL, "Vector creation for matrix ops");
    
    if (!mat1 || !mat2 || !result || !vec || !mat_vec_result) {
        printf("Matrix/vector creation failed, skipping matrix tests\n");
        return;
    }
    
    /* Initialize matrices */
    /* mat1 = [[1, 2, 3], [4, 5, 6]] */
    ml_matrix_set(mat1, 0, 0, 1.0f); ml_matrix_set(mat1, 0, 1, 2.0f); ml_matrix_set(mat1, 0, 2, 3.0f);
    ml_matrix_set(mat1, 1, 0, 4.0f); ml_matrix_set(mat1, 1, 1, 5.0f); ml_matrix_set(mat1, 1, 2, 6.0f);
    
    /* mat2 = [[1, 2], [3, 4], [5, 6]] */
    ml_matrix_set(mat2, 0, 0, 1.0f); ml_matrix_set(mat2, 0, 1, 2.0f);
    ml_matrix_set(mat2, 1, 0, 3.0f); ml_matrix_set(mat2, 1, 1, 4.0f);
    ml_matrix_set(mat2, 2, 0, 5.0f); ml_matrix_set(mat2, 2, 1, 6.0f);
    
    /* vec = [1, 2, 3] */
    ml_vector_set(vec, 0, 1.0f);
    ml_vector_set(vec, 1, 2.0f);
    ml_vector_set(vec, 2, 3.0f);
    
    /* Test matrix-vector multiplication */
    ml_error_t matvec_result = ml_matrix_vector_mul(mat1, vec, mat_vec_result);
    TEST_ASSERT(matvec_result == ML_SUCCESS, "Matrix-vector multiplication execution");
    
    /* Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32] */
    TEST_ASSERT_FLOAT_EQ(ml_vector_get(mat_vec_result, 0), 14.0f, "Matrix-vector mult result [0]");
    TEST_ASSERT_FLOAT_EQ(ml_vector_get(mat_vec_result, 1), 32.0f, "Matrix-vector mult result [1]");
    
    /* Test matrix-matrix multiplication */
    ml_error_t matmul_result = ml_matrix_mul(mat1, mat2, result);
    TEST_ASSERT(matmul_result == ML_SUCCESS, "Matrix-matrix multiplication execution");
    
    /* Expected result: [[22, 28], [49, 64]] */
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 0, 0), 22.0f, "Matrix-matrix mult result [0,0]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 0, 1), 28.0f, "Matrix-matrix mult result [0,1]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 1, 0), 49.0f, "Matrix-matrix mult result [1,0]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(result, 1, 1), 64.0f, "Matrix-matrix mult result [1,1]");
    
    /* Test matrix transpose */
    ml_matrix_t* transposed = ml_matrix_create(3, 2);
    ml_error_t transpose_result = ml_matrix_transpose(mat1, transposed);
    TEST_ASSERT(transpose_result == ML_SUCCESS, "Matrix transpose execution");
    
    /* Check transposed values */
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 0, 0), 1.0f, "Transpose [0,0]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 1, 0), 2.0f, "Transpose [1,0]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 2, 0), 3.0f, "Transpose [2,0]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(transposed, 0, 1), 4.0f, "Transpose [0,1]");
    
    /* Cleanup */
    ml_matrix_free(mat1);
    ml_matrix_free(mat2);
    ml_matrix_free(result);
    ml_matrix_free(transposed);
    ml_vector_free(vec);
    ml_vector_free(mat_vec_result);
}

/* ============================================================================
 * ACTIVATION FUNCTION TESTS
 * ============================================================================ */

void test_activation_functions(void) {
    printf("\n=== Activation Function Tests ===\n");
    
    ml_vector_t* input = ml_vector_create(5);
    ml_vector_t* output = ml_vector_create(5);
    
    TEST_ASSERT(input != NULL && output != NULL, "Activation test vector creation");
    
    if (!input || !output) {
        printf("Vector creation failed, skipping activation tests\n");
        return;
    }
    
    /* Test data: [-2, -1, 0, 1, 2] */
    float test_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 5; i++) {
        ml_vector_set(input, i, test_data[i]);
    }
    
    /* Test ReLU */
    ml_error_t relu_result = ml_activation_relu(input, output);
    TEST_ASSERT(relu_result == ML_SUCCESS, "ReLU activation execution");
    
    float expected_relu[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 5; i++) {
        char msg[100];
        snprintf(msg, sizeof(msg), "ReLU output [%d]", i);
        TEST_ASSERT_FLOAT_EQ(ml_vector_get(output, i), expected_relu[i], msg);
    }
    
    /* Test Sigmoid */
    ml_error_t sigmoid_result = ml_activation_sigmoid(input, output);
    TEST_ASSERT(sigmoid_result == ML_SUCCESS, "Sigmoid activation execution");
    
    /* Verify sigmoid properties: output in (0,1), monotonic */
    bool sigmoid_valid = true;
    float prev_val = -1.0f;
    for (int i = 0; i < 5; i++) {
        float val = ml_vector_get(output, i);
        if (val <= 0.0f || val >= 1.0f || val <= prev_val) {
            sigmoid_valid = false;
            break;
        }
        prev_val = val;
    }
    TEST_ASSERT(sigmoid_valid, "Sigmoid activation properties");
    
    /* Test that sigmoid(0) ≈ 0.5 */
    ml_vector_t* zero_input = ml_vector_create(1);
    ml_vector_t* zero_output = ml_vector_create(1);
    ml_vector_set(zero_input, 0, 0.0f);
    ml_activation_sigmoid(zero_input, zero_output);
    TEST_ASSERT_FLOAT_EQ(ml_vector_get(zero_output, 0), 0.5f, "Sigmoid(0) = 0.5");
    
    /* Test Tanh */
    ml_error_t tanh_result = ml_activation_tanh(input, output);
    TEST_ASSERT(tanh_result == ML_SUCCESS, "Tanh activation execution");
    
    /* Verify tanh properties: output in (-1,1), monotonic, tanh(0) = 0 */
    bool tanh_valid = true;
    prev_val = -2.0f;
    for (int i = 0; i < 5; i++) {
        float val = ml_vector_get(output, i);
        if (val <= -1.0f || val >= 1.0f || val <= prev_val) {
            tanh_valid = false;
            break;
        }
        prev_val = val;
    }
    TEST_ASSERT(tanh_valid, "Tanh activation properties");
    
    ml_vector_set(zero_input, 0, 0.0f);
    ml_activation_tanh(zero_input, zero_output);
    TEST_ASSERT_FLOAT_EQ(ml_vector_get(zero_output, 0), 0.0f, "Tanh(0) = 0");
    
    /* Test Softmax */
    ml_vector_t* softmax_input = ml_vector_create(3);
    ml_vector_t* softmax_output = ml_vector_create(3);
    
    /* Input: [1, 2, 3] */
    ml_vector_set(softmax_input, 0, 1.0f);
    ml_vector_set(softmax_input, 1, 2.0f);
    ml_vector_set(softmax_input, 2, 3.0f);
    
    ml_error_t softmax_result = ml_activation_softmax(softmax_input, softmax_output);
    TEST_ASSERT(softmax_result == ML_SUCCESS, "Softmax activation execution");
    
    /* Verify softmax properties: all positive, sum to 1 */
    float softmax_sum = 0.0f;
    bool all_positive = true;
    for (int i = 0; i < 3; i++) {
        float val = ml_vector_get(softmax_output, i);
        if (val <= 0.0f) all_positive = false;
        softmax_sum += val;
    }
    TEST_ASSERT(all_positive, "Softmax outputs are positive");
    TEST_ASSERT_FLOAT_EQ(softmax_sum, 1.0f, "Softmax outputs sum to 1");
    
    /* Cleanup */
    ml_vector_free(input);
    ml_vector_free(output);
    ml_vector_free(zero_input);
    ml_vector_free(zero_output);
    ml_vector_free(softmax_input);
    ml_vector_free(softmax_output);
}

/* ============================================================================
 * LINEAR REGRESSION TESTS
 * ============================================================================ */

void test_linear_regression(void) {
    printf("\n=== Linear Regression Tests ===\n");
    
    /* Create linear regression model */
    ml_model_config_t config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 3,
        .output_size = 1,
        .layer_count = 1,
        .activations = NULL,
        .layer_sizes = NULL,
        .model_data = NULL
    };
    
    ml_model_t* model = ml_create_model(&config);
    TEST_ASSERT(model != NULL, "Linear regression model creation");
    
    if (!model) {
        printf("Model creation failed, skipping linear regression tests\n");
        return;
    }
    
    /* Set model weights: w = [1, 2, 3], b = 0.5 */
    float weights[] = {1.0f, 2.0f, 3.0f};
    extern ml_error_t ml_linear_regression_set_weights(ml_model_t* model, const ml_float_t* weights, ml_float_t bias);
    ml_error_t set_weights_result = ml_linear_regression_set_weights(model, weights, 0.5f);
    TEST_ASSERT(set_weights_result == ML_SUCCESS, "Setting linear regression weights");
    
    /* Test single prediction */
    float input[] = {1.0f, 2.0f, 3.0f};
    float output;
    ml_error_t predict_result = ml_predict(model, input, &output);
    TEST_ASSERT(predict_result == ML_SUCCESS, "Linear regression prediction execution");
    
    /* Expected: 1*1 + 2*2 + 3*3 + 0.5 = 14.5 */
    TEST_ASSERT_FLOAT_EQ(output, 14.5f, "Linear regression prediction result");
    
    /* Test batch prediction */
    ml_matrix_t* batch_input = ml_matrix_create(2, 3);
    ml_matrix_t* batch_output = ml_matrix_create(2, 1);
    
    /* Input 1: [1, 2, 3] -> 14.5 */
    /* Input 2: [2, 3, 4] -> 2*1 + 3*2 + 4*3 + 0.5 = 20.5 */
    ml_matrix_set(batch_input, 0, 0, 1.0f); ml_matrix_set(batch_input, 0, 1, 2.0f); ml_matrix_set(batch_input, 0, 2, 3.0f);
    ml_matrix_set(batch_input, 1, 0, 2.0f); ml_matrix_set(batch_input, 1, 1, 3.0f); ml_matrix_set(batch_input, 1, 2, 4.0f);
    
    ml_error_t batch_result = ml_predict_batch(model, batch_input, batch_output, 2);
    TEST_ASSERT(batch_result == ML_SUCCESS, "Linear regression batch prediction execution");
    
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(batch_output, 0, 0), 14.5f, "Batch prediction result [0]");
    TEST_ASSERT_FLOAT_EQ(ml_matrix_get(batch_output, 1, 0), 20.5f, "Batch prediction result [1]");
    
    /* Test model info */
    printf("Model Information:\n");
    ml_model_info(model);
    
    /* Test memory usage */
    size_t memory_usage = ml_model_memory_usage(model);
    TEST_ASSERT(memory_usage > 0, "Model memory usage calculation");
    printf("Model memory usage: %zu bytes\n", memory_usage);
    
    /* Cleanup */
    ml_matrix_free(batch_input);
    ml_matrix_free(batch_output);
    ml_free_model(model);
}

/* ============================================================================
 * PERFORMANCE BENCHMARKS
 * ============================================================================ */

void test_performance_benchmarks(void) {
    printf("\n=== Performance Benchmarks ===\n");
    
    /* Vector operations benchmark */
    printf("Vector Operations Benchmark (%d iterations):\n", PERFORMANCE_ITERATIONS);
    
    ml_vector_t* vec1 = ml_vector_create(1000);
    ml_vector_t* vec2 = ml_vector_create(1000);
    
    /* Initialize with random data */
    srand(42);
    for (size_t i = 0; i < 1000; i++) {
        ml_vector_set(vec1, i, (float)rand() / RAND_MAX);
        ml_vector_set(vec2, i, (float)rand() / RAND_MAX);
    }
    
    /* Benchmark dot product */
    clock_t start = clock();
    for (int i = 0; i < PERFORMANCE_ITERATIONS; i++) {
        volatile float result = ml_vector_dot(vec1, vec2);
        (void)result; /* Suppress unused variable warning */
    }
    clock_t end = clock();
    
    double dot_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double dot_throughput = PERFORMANCE_ITERATIONS / dot_time;
    
    printf("  Dot Product: %.2f ops/sec (%.2f μs/op)\n", 
           dot_throughput, (dot_time / PERFORMANCE_ITERATIONS) * 1000000);
    
    /* Model prediction benchmark */
    ml_model_config_t config = {
        .type = ML_LINEAR_REGRESSION,
        .input_size = 100,
        .output_size = 1,
        .layer_count = 1
    };
    
    ml_model_t* model = ml_create_model(&config);
    if (model) {
        float* weights = malloc(100 * sizeof(float));
        for (int i = 0; i < 100; i++) {
            weights[i] = (float)rand() / RAND_MAX;
        }
        
        extern ml_error_t ml_linear_regression_set_weights(ml_model_t* model, const ml_float_t* weights, ml_float_t bias);
        ml_linear_regression_set_weights(model, weights, 0.0f);
        
        float* input = malloc(100 * sizeof(float));
        for (int i = 0; i < 100; i++) {
            input[i] = (float)rand() / RAND_MAX;
        }
        
        /* Reset performance stats */
        ml_reset_performance_stats();
        
        /* Benchmark predictions */
        start = clock();
        for (int i = 0; i < PERFORMANCE_ITERATIONS / 10; i++) {
            float output;
            ml_predict(model, input, &output);
        }
        end = clock();
        
        double pred_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        double pred_throughput = (PERFORMANCE_ITERATIONS / 10) / pred_time;
        
        printf("  Linear Regression Prediction: %.2f ops/sec (%.2f μs/op)\n",
               pred_throughput, (pred_time / (PERFORMANCE_ITERATIONS / 10)) * 1000000);
        
        /* Print performance statistics */
        ml_perf_stats_t stats;
        ml_get_performance_stats(&stats);
        printf("  Framework Statistics:\n");
        printf("    Total Predictions: %lu\n", stats.total_predictions);
        printf("    Average Latency: %.2f μs\n", stats.avg_latency_us);
        printf("    Throughput: %.0f predictions/sec\n", stats.throughput_per_sec);
        
        free(weights);
        free(input);
        ml_free_model(model);
    }
    
    ml_vector_free(vec1);
    ml_vector_free(vec2);
    
    TEST_ASSERT(true, "Performance benchmark completed");
}

/* ============================================================================
 * MAIN TEST RUNNER
 * ============================================================================ */

void print_test_summary(void) {
    printf("\n============================================================\n");
    printf("TEST SUMMARY\n");
    printf("============================================================\n");
    printf("Tests Run: %d\n", tests_run);
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("Success Rate: %.1f%%\n", (float)tests_passed / tests_run * 100);
    
    if (tests_failed == 0) {
        printf("🎉 ALL TESTS PASSED! 🎉\n");
    } else {
        printf("❌ Some tests failed. Please review the output above.\n");
    }
    printf("============================================================\n");
}

int main(int argc, char* argv[]) {
    bool run_unit = false;
    bool run_integration = false;
    bool run_performance = false;
    bool run_all = true;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--unit") == 0) {
            run_unit = true;
            run_all = false;
        } else if (strcmp(argv[i], "--integration") == 0) {
            run_integration = true;
            run_all = false;
        } else if (strcmp(argv[i], "--performance") == 0) {
            run_performance = true;
            run_all = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("ML Assembly Test Suite\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --unit         Run unit tests only\n");
            printf("  --integration  Run integration tests only\n");
            printf("  --performance  Run performance benchmarks only\n");
            printf("  --help         Show this help message\n");
            printf("  (no options)   Run all tests\n");
            return 0;
        }
    }
    
    printf("ML Assembly Framework Test Suite\n");
    printf("===============================\n");
    
    /* Run tests based on command line arguments */
    if (run_all || run_unit) {
        test_framework_init();
        test_vector_operations();
        test_matrix_operations();
        test_activation_functions();
    }
    
    if (run_all || run_integration) {
        test_linear_regression();
    }
    
    if (run_all || run_performance) {
        test_performance_benchmarks();
    }
    
    /* Cleanup framework */
    ml_cleanup();
    
    /* Print summary and return appropriate exit code */
    print_test_summary();
    return (tests_failed == 0) ? 0 : 1;
}
