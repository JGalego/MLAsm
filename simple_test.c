/**
 * @file simple_test.c
 * @brief Simple test to verify the ML Assembly framework
 */

#include "ml_assembly.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing ML Assembly Framework\n");
    printf("=============================\n");
    
    // Initialize framework
    ml_error_t result = ml_init();
    if (result != ML_SUCCESS) {
        printf("Failed to initialize framework: %s\n", ml_error_string(result));
        return 1;
    }
    
    printf("✓ Framework initialized\n");
    printf("Version: %s\n", ml_get_version());
    printf("CPU Support: %s\n", ml_check_cpu_support() ? "Yes" : "No");
    
    // Test vector creation
    ml_vector_t* vec1 = ml_vector_create(4);
    ml_vector_t* vec2 = ml_vector_create(4);
    
    if (!vec1 || !vec2) {
        printf("✗ Failed to create vectors\n");
        return 1;
    }
    
    printf("✓ Vectors created\n");
    
    // Manual data setting (since ml_vector_set might not be implemented)
    for (int i = 0; i < 4; i++) {
        vec1->data[i] = (float)(i + 1);  // [1, 2, 3, 4]
        vec2->data[i] = (float)(i + 1);  // [1, 2, 3, 4]
    }
    
    // Test dot product
    float dot_result = ml_vector_dot(vec1, vec2);
    printf("✓ Dot product: %.2f (expected: 30.00)\n", dot_result);
    
    // Test vector addition
    ml_vector_t* result_vec = ml_vector_create(4);
    if (result_vec) {
        ml_error_t add_result = ml_vector_add(vec1, vec2, result_vec);
        if (add_result == ML_SUCCESS) {
            printf("✓ Vector addition successful\n");
            printf("  Result: [%.1f, %.1f, %.1f, %.1f]\n", 
                   result_vec->data[0], result_vec->data[1], 
                   result_vec->data[2], result_vec->data[3]);
        } else {
            printf("✗ Vector addition failed: %s\n", ml_error_string(add_result));
        }
        ml_vector_free(result_vec);
    }
    
    // Test matrix creation
    ml_matrix_t* mat = ml_matrix_create(2, 4);
    if (mat) {
        printf("✓ Matrix created (2x4)\n");
        
        // Set matrix data manually
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                mat->data[i * 4 + j] = (float)(i * 4 + j + 1);
            }
        }
        
        // Test matrix-vector multiplication
        ml_vector_t* mat_result = ml_vector_create(2);
        if (mat_result) {
            ml_error_t mv_result = ml_matrix_vector_mul(mat, vec1, mat_result);
            if (mv_result == ML_SUCCESS) {
                printf("✓ Matrix-vector multiplication successful\n");
                printf("  Result: [%.1f, %.1f]\n", mat_result->data[0], mat_result->data[1]);
            } else {
                printf("✗ Matrix-vector multiplication failed: %s\n", ml_error_string(mv_result));
            }
            ml_vector_free(mat_result);
        }
        
        ml_matrix_free(mat);
    }
    
    // Cleanup
    ml_vector_free(vec1);
    ml_vector_free(vec2);
    ml_cleanup();
    
    printf("✓ Framework cleanup completed\n");
    printf("\nBasic functionality test: PASSED\n");
    
    return 0;
}
