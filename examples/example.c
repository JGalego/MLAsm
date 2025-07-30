/**
 * @file example.c
 * @brief Example usage of the ML Assembly framework
 * 
 * This example demonstrates basic usage of the framework including:
 * - Framework initialization
 * - Vector and matrix operations
 * - Linear regression model usage
 * - Performance monitoring
 */

#include "ml_assembly.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void demonstrate_vector_operations(void) {
    printf("\n=== Vector Operations Demo ===\n");
    
    // Create test vectors
    ml_vector_t* vec1 = ml_vector_create(5);
    ml_vector_t* vec2 = ml_vector_create(5);
    ml_vector_t* result = ml_vector_create(5);
    
    if (!vec1 || !vec2 || !result) {
        printf("Failed to create vectors\n");
        return;
    }
    
    // Initialize vectors with sample data
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float data2[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    for (int i = 0; i < 5; i++) {
        ml_vector_set(vec1, i, data1[i]);
        ml_vector_set(vec2, i, data2[i]);
    }
    
    printf("Vector 1: ");
    ml_vector_print(vec1, "vec1");
    
    printf("Vector 2: ");
    ml_vector_print(vec2, "vec2");
    
    // Dot product
    float dot = ml_vector_dot(vec1, vec2);
    printf("Dot product: %.2f\n", dot);
    
    // Vector addition
    ml_vector_add(vec1, vec2, result);
    printf("Addition result: ");
    ml_vector_print(result, "vec1 + vec2");
    
    // Vector scaling
    ml_vector_scale(vec1, 2.5f, result);
    printf("Scaling vec1 by 2.5: ");
    ml_vector_print(result, "2.5 * vec1");
    
    // Vector statistics
    printf("Vector 1 sum: %.2f\n", ml_vector_sum(vec1));
    printf("Vector 1 mean: %.2f\n", ml_vector_mean(vec1));
    printf("Vector 1 max: %.2f\n", ml_vector_max(vec1));
    printf("Vector 1 min: %.2f\n", ml_vector_min(vec1));
    printf("Vector 1 argmax: %zu\n", ml_vector_argmax(vec1));
    
    // Cleanup
    ml_vector_free(vec1);
    ml_vector_free(vec2);
    ml_vector_free(result);
}

void demonstrate_matrix_operations(void) {
    printf("\n=== Matrix Operations Demo ===\n");
    
    // Create test matrix and vector
    ml_matrix_t* matrix = ml_matrix_create(3, 4);
    ml_vector_t* vector = ml_vector_create(4);
    ml_vector_t* result = ml_vector_create(3);
    
    if (!matrix || !vector || !result) {
        printf("Failed to create matrix/vectors\n");
        return;
    }
    
    // Initialize matrix with sample data
    float matrix_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            ml_matrix_set(matrix, i, j, matrix_data[i * 4 + j]);
        }
    }
    
    // Initialize vector
    float vector_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; i++) {
        ml_vector_set(vector, i, vector_data[i]);
    }
    
    printf("Matrix:\n");
    ml_matrix_print(matrix, "matrix");
    
    printf("Vector: ");
    ml_vector_print(vector, "vector");
    
    // Matrix-vector multiplication
    ml_matrix_vector_mul(matrix, vector, result);
    printf("Matrix * Vector: ");
    ml_vector_print(result, "result");
    
    // Create a square matrix for additional operations
    ml_matrix_t* square_mat = ml_matrix_create(2, 2);
    ml_matrix_t* transposed = ml_matrix_create(2, 2);
    
    if (square_mat && transposed) {
        ml_matrix_set(square_mat, 0, 0, 1.0f); ml_matrix_set(square_mat, 0, 1, 2.0f);
        ml_matrix_set(square_mat, 1, 0, 3.0f); ml_matrix_set(square_mat, 1, 1, 4.0f);
        
        printf("Square matrix:\n");
        ml_matrix_print(square_mat, "square");
        
        // Transpose
        ml_matrix_transpose(square_mat, transposed);
        printf("Transposed:\n");
        ml_matrix_print(transposed, "transposed");
        
        ml_matrix_free(square_mat);
        ml_matrix_free(transposed);
    }
    
    // Cleanup
    ml_matrix_free(matrix);
    ml_vector_free(vector);
    ml_vector_free(result);
}

void demonstrate_activation_functions(void) {
    printf("\n=== Activation Functions Demo ===\n");
    
    ml_vector_t* input = ml_vector_create(7);
    ml_vector_t* output = ml_vector_create(7);
    
    if (!input || !output) {
        printf("Failed to create activation test vectors\n");
        return;
    }
    
    // Test data from -3 to 3
    float test_data[] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 7; i++) {
        ml_vector_set(input, i, test_data[i]);
    }
    
    printf("Input: ");
    ml_vector_print(input, "input");
    
    // ReLU
    ml_activation_relu(input, output);
    printf("ReLU: ");
    ml_vector_print(output, "relu");
    
    // Sigmoid
    ml_activation_sigmoid(input, output);
    printf("Sigmoid: ");
    ml_vector_print(output, "sigmoid");
    
    // Tanh
    ml_activation_tanh(input, output);
    printf("Tanh: ");
    ml_vector_print(output, "tanh");
    
    // Softmax (use positive values)
    ml_vector_t* positive_input = ml_vector_create(3);
    ml_vector_t* softmax_output = ml_vector_create(3);
    
    if (positive_input && softmax_output) {
        ml_vector_set(positive_input, 0, 1.0f);
        ml_vector_set(positive_input, 1, 2.0f);
        ml_vector_set(positive_input, 2, 3.0f);
        
        printf("Softmax input: ");
        ml_vector_print(positive_input, "softmax_in");
        
        ml_activation_softmax(positive_input, softmax_output);
        printf("Softmax: ");
        ml_vector_print(softmax_output, "softmax_out");
        
        printf("Softmax sum: %.6f (should be 1.0)\n", ml_vector_sum(softmax_output));
        
        ml_vector_free(positive_input);
        ml_vector_free(softmax_output);
    }
    
    // Cleanup
    ml_vector_free(input);
    ml_vector_free(output);
}

void demonstrate_linear_regression(void) {
    printf("\n=== Linear Regression Demo ===\n");
    
    // Create a simple linear regression model
    // Model: y = 2*x1 + 3*x2 + 1*x3 + 0.5
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
    if (!model) {
        printf("Failed to create linear regression model\n");
        return;
    }
    
    // Set model weights
    float weights[] = {2.0f, 3.0f, 1.0f};
    float bias = 0.5f;
    
    extern ml_error_t ml_linear_regression_set_weights(ml_model_t* model, const ml_float_t* weights, ml_float_t bias);
    ml_error_t result = ml_linear_regression_set_weights(model, weights, bias);
    if (result != ML_SUCCESS) {
        printf("Failed to set model weights: %s\n", ml_error_string(result));
        ml_free_model(model);
        return;
    }
    
    printf("Model: y = 2*x1 + 3*x2 + 1*x3 + 0.5\n");
    
    // Print model information
    ml_model_info(model);
    
    // Single prediction example
    float input1[] = {1.0f, 2.0f, 3.0f};  // Expected: 2*1 + 3*2 + 1*3 + 0.5 = 11.5
    float output1;
    
    result = ml_predict(model, input1, &output1);
    if (result == ML_SUCCESS) {
        printf("Prediction for [%.1f, %.1f, %.1f]: %.2f (expected: 11.5)\n", 
               input1[0], input1[1], input1[2], output1);
    } else {
        printf("Prediction failed: %s\n", ml_error_string(result));
    }
    
    // Batch prediction example
    ml_matrix_t* batch_input = ml_matrix_create(3, 3);
    ml_matrix_t* batch_output = ml_matrix_create(3, 1);
    
    if (batch_input && batch_output) {
        // Input samples:
        // Sample 1: [1, 2, 3] -> 11.5
        // Sample 2: [2, 1, 0] -> 7.5
        // Sample 3: [0, 0, 1] -> 1.5
        
        ml_matrix_set(batch_input, 0, 0, 1.0f); ml_matrix_set(batch_input, 0, 1, 2.0f); ml_matrix_set(batch_input, 0, 2, 3.0f);
        ml_matrix_set(batch_input, 1, 0, 2.0f); ml_matrix_set(batch_input, 1, 1, 1.0f); ml_matrix_set(batch_input, 1, 2, 0.0f);
        ml_matrix_set(batch_input, 2, 0, 0.0f); ml_matrix_set(batch_input, 2, 1, 0.0f); ml_matrix_set(batch_input, 2, 2, 1.0f);
        
        printf("\nBatch Input:\n");
        ml_matrix_print(batch_input, "batch_input");
        
        result = ml_predict_batch(model, batch_input, batch_output, 3);
        if (result == ML_SUCCESS) {
            printf("Batch Output:\n");
            ml_matrix_print(batch_output, "batch_output");
            printf("Expected outputs: [11.5, 7.5, 1.5]\n");
        } else {
            printf("Batch prediction failed: %s\n", ml_error_string(result));
        }
        
        ml_matrix_free(batch_input);
        ml_matrix_free(batch_output);
    }
    
    // Performance test
    printf("\nPerformance Test (10000 predictions):\n");
    ml_reset_performance_stats();
    
    for (int i = 0; i < 10000; i++) {
        float dummy_output;
        ml_predict(model, input1, &dummy_output);
    }
    
    ml_perf_stats_t perf_stats;
    ml_get_performance_stats(&perf_stats);
    printf("Total predictions: %lu\n", perf_stats.total_predictions);
    printf("Average latency: %.2f μs\n", perf_stats.avg_latency_us);
    printf("Throughput: %.0f predictions/sec\n", perf_stats.throughput_per_sec);
    
    // Memory usage
    size_t memory_usage = ml_model_memory_usage(model);
    printf("Model memory usage: %zu bytes\n", memory_usage);
    
    // Cleanup
    ml_free_model(model);
}

int main(void) {
    printf("ML Assembly Framework Example\n");
    printf("============================\n");
    
    // Initialize the framework
    ml_error_t init_result = ml_init();
    if (init_result != ML_SUCCESS) {
        printf("Failed to initialize ML Assembly framework: %s\n", ml_error_string(init_result));
        return 1;
    }
    
    printf("Framework initialized successfully!\n");
    printf("Version: %s\n", ml_get_version());
    printf("CPU Support: %s\n", ml_check_cpu_support() ? "Yes" : "No");
    
    // Print system information
    printf("\nSystem Information:\n");
    ml_print_system_info();
    
    // Run demonstrations
    demonstrate_vector_operations();
    demonstrate_matrix_operations();
    demonstrate_activation_functions();
    demonstrate_linear_regression();
    
    // Final performance statistics
    printf("\n=== Final Performance Statistics ===\n");
    ml_print_performance_stats();
    
    // Cleanup
    ml_cleanup();
    printf("\nFramework cleanup completed.\n");
    printf("Example completed successfully!\n");
    
    return 0;
}
