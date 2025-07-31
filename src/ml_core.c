/**
 * @file ml_core.c
 * @brief Core ML Assembly framework functions
 * 
 * This file implements the main framework initialization, CPU feature
 * detection, error handling, and general model management functions.
 */

#define _POSIX_C_SOURCE 199309L

#include "ml_assembly.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#ifdef __linux__
#include <unistd.h>
#endif

/* Performance statistics */
static ml_perf_stats_t g_perf_stats = {0};
static bool g_framework_initialized = false;

/* CPU feature flags */
static struct {
    bool avx2_supported;
    bool avx512_supported;
    bool fma_supported;
} g_cpu_features = {false, false, false};

/* Forward declarations for model-specific functions */
extern ml_model_t* ml_linear_regression_create(size_t input_features, bool has_bias);
extern ml_model_t* ml_linear_regression_load(const char* filename);
extern ml_error_t ml_linear_regression_predict(const ml_model_t* model, const ml_float_t* input, ml_float_t* output);
extern void ml_linear_regression_free(ml_model_t* model);
extern void ml_linear_regression_print_info(const ml_model_t* model);
extern size_t ml_linear_regression_memory_usage(const ml_model_t* model);

/* ============================================================================
 * CPU FEATURE DETECTION
 * ============================================================================ */

/**
 * @brief Check if CPUID instruction is supported and execute it
 */
static bool check_cpu_feature(int leaf, int subleaf, int reg, int bit) {
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    
    /* Use GCC's builtin if available */
    #ifdef __GNUC__
    if (__builtin_cpu_supports("sse")) {
        __asm__ volatile (
            "cpuid"
            : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
            : "a" (leaf), "c" (subleaf)
        );
        
        uint32_t* regs[] = {&eax, &ebx, &ecx, &edx};
        return (*regs[reg] & (1U << bit)) != 0;
    }
    #endif
    
    return false;
}

/**
 * @brief Detect CPU features
 */
static void detect_cpu_features(void) {
    /* Check for AVX2 support (CPUID leaf 7, subleaf 0, EBX bit 5) */
    g_cpu_features.avx2_supported = check_cpu_feature(7, 0, 1, 5);
    
    /* Check for AVX-512F support (CPUID leaf 7, subleaf 0, EBX bit 16) */
    g_cpu_features.avx512_supported = check_cpu_feature(7, 0, 1, 16);
    
    /* Check for FMA support (CPUID leaf 1, subleaf 0, ECX bit 12) */
    g_cpu_features.fma_supported = check_cpu_feature(1, 0, 2, 12);
}

/* ============================================================================
 * FRAMEWORK INITIALIZATION
 * ============================================================================ */

ml_error_t ml_init(void) {
    if (g_framework_initialized) {
        return ML_SUCCESS;
    }
    
    /* Detect CPU features */
    detect_cpu_features();
    
    /* Initialize performance statistics */
    memset(&g_perf_stats, 0, sizeof(g_perf_stats));
    
    /* Initialize lookup tables for activation functions */
    /* This would be called from activation_wrappers.c */
    
    g_framework_initialized = true;
    return ML_SUCCESS;
}

void ml_cleanup(void) {
    if (!g_framework_initialized) {
        return;
    }
    
    /* Cleanup any global resources */
    memset(&g_perf_stats, 0, sizeof(g_perf_stats));
    
    g_framework_initialized = false;
}

const char* ml_get_version(void) {
    static char version_string[32];
    snprintf(version_string, sizeof(version_string), "%d.%d.%d", 
             ML_ASSEMBLY_VERSION_MAJOR, 
             ML_ASSEMBLY_VERSION_MINOR, 
             ML_ASSEMBLY_VERSION_PATCH);
    return version_string;
}

bool ml_check_cpu_support(void) {
    if (!g_framework_initialized) {
        detect_cpu_features();
    }
    
    /* Require at least AVX2 for optimal performance */
    return g_cpu_features.avx2_supported;
}

/* ============================================================================
 * MODEL MANAGEMENT
 * ============================================================================ */

ml_model_t* ml_create_model(const ml_model_config_t* config) {
    if (!config) {
        return NULL;
    }
    
    if (!g_framework_initialized) {
        ml_init();
    }
    
    switch (config->type) {
        case ML_LINEAR_REGRESSION:
            return ml_linear_regression_create(config->input_size, true);
            
        case ML_LOGISTIC_REGRESSION:
            /* TODO: Implement logistic regression */
            return NULL;
            
        case ML_NEURAL_NETWORK:
            /* TODO: Implement neural network */
            return NULL;
            
        case ML_DECISION_TREE:
            /* TODO: Implement decision tree */
            return NULL;
            
        case ML_RANDOM_FOREST:
            /* TODO: Implement random forest */
            return NULL;
            
        default:
            return NULL;
    }
}

ml_model_t* ml_load_model(const char* filename, ml_model_type_t type) {
    if (!filename) {
        return NULL;
    }
    
    if (!g_framework_initialized) {
        ml_init();
    }
    
    switch (type) {
        case ML_LINEAR_REGRESSION:
            return ml_linear_regression_load(filename);
            
        case ML_LOGISTIC_REGRESSION:
            /* TODO: Implement logistic regression loading */
            return NULL;
            
        case ML_NEURAL_NETWORK:
            /* TODO: Implement neural network loading */
            return NULL;
            
        case ML_DECISION_TREE:
            /* TODO: Implement decision tree loading */
            return NULL;
            
        case ML_RANDOM_FOREST:
            /* TODO: Implement random forest loading */
            return NULL;
            
        default:
            return NULL;
    }
}

ml_error_t ml_save_model(const ml_model_t* model, const char* filename) {
    if (!model || !filename) {
        return ML_ERROR_NULL_POINTER;
    }
    
    /* Model-specific save functions would be called here */
    switch (model->config.type) {
        case ML_LINEAR_REGRESSION:
            /* TODO: Call ml_linear_regression_save */
            return ML_ERROR_UNSUPPORTED_MODEL;
            
        default:
            return ML_ERROR_UNSUPPORTED_MODEL;
    }
}

void ml_free_model(ml_model_t* model) {
    if (!model) {
        return;
    }
    
    switch (model->config.type) {
        case ML_LINEAR_REGRESSION:
            ml_linear_regression_free(model);
            break;
            
        default:
            /* Generic cleanup */
            if (model->config.activations) {
                free(model->config.activations);
            }
            if (model->config.layer_sizes) {
                free(model->config.layer_sizes);
            }
            free(model);
            break;
    }
}

/* ============================================================================
 * INFERENCE FUNCTIONS
 * ============================================================================ */

ml_error_t ml_predict(const ml_model_t* model, 
                      const ml_float_t* input, 
                      ml_float_t* output) {
    if (!model || !input || !output) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!model->is_loaded) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    /* Start timing for performance statistics */
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    ml_error_t result;
    
    switch (model->config.type) {
        case ML_LINEAR_REGRESSION:
            result = ml_linear_regression_predict(model, input, output);
            break;
            
        default:
            result = ML_ERROR_UNSUPPORTED_MODEL;
            break;
    }
    
    /* Update performance statistics */
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    uint64_t elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000LL +
                         (end_time.tv_nsec - start_time.tv_nsec);
    
    g_perf_stats.total_predictions++;
    g_perf_stats.total_time_ns += elapsed_ns;
    g_perf_stats.avg_latency_us = (double)g_perf_stats.total_time_ns / 
                                 (double)g_perf_stats.total_predictions / 1000.0;
    g_perf_stats.throughput_per_sec = 1000000000.0 / g_perf_stats.avg_latency_us * 1000.0;
    
    return result;
}

ml_error_t ml_predict_batch(const ml_model_t* model,
                           const ml_matrix_t* inputs,
                           ml_matrix_t* outputs,
                           size_t num_samples) {
    if (!model || !inputs || !outputs) {
        return ML_ERROR_NULL_POINTER;
    }
    
    if (!model->is_loaded) {
        return ML_ERROR_INVALID_MODEL;
    }
    
    if (inputs->rows < num_samples || outputs->rows < num_samples) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    if (inputs->cols != model->config.input_size || 
        outputs->cols != model->config.output_size) {
        return ML_ERROR_DIMENSION_MISMATCH;
    }
    
    /* Process each sample individually for now */
    /* TODO: Optimize for true batch processing */
    for (size_t i = 0; i < num_samples; ++i) {
        ml_float_t* input_row = &inputs->data[i * inputs->cols];
        ml_float_t* output_row = &outputs->data[i * outputs->cols];
        
        ml_error_t result = ml_predict(model, input_row, output_row);
        if (result != ML_SUCCESS) {
            return result;
        }
    }
    
    return ML_SUCCESS;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

const char* ml_error_string(ml_error_t error) {
    switch (error) {
        case ML_SUCCESS:                return "Success";
        case ML_ERROR_NULL_POINTER:     return "Null pointer error";
        case ML_ERROR_INVALID_MODEL:    return "Invalid model error";
        case ML_ERROR_INVALID_INPUT:    return "Invalid input error";
        case ML_ERROR_OUT_OF_MEMORY:    return "Out of memory error";
        case ML_ERROR_FILE_NOT_FOUND:   return "File not found error";
        case ML_ERROR_UNSUPPORTED_MODEL: return "Unsupported model error";
        case ML_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case ML_ERROR_COMPUTATION_FAILED: return "Computation failed";
        default:                        return "Unknown error";
    }
}

void ml_model_info(const ml_model_t* model) {
    if (!model) {
        printf("Model: NULL\n");
        return;
    }
    
    printf("Model Information:\n");
    printf("  Type: ");
    switch (model->config.type) {
        case ML_LINEAR_REGRESSION:  printf("Linear Regression\n"); break;
        case ML_LOGISTIC_REGRESSION: printf("Logistic Regression\n"); break;
        case ML_NEURAL_NETWORK:     printf("Neural Network\n"); break;
        case ML_DECISION_TREE:      printf("Decision Tree\n"); break;
        case ML_RANDOM_FOREST:      printf("Random Forest\n"); break;
        default:                    printf("Unknown\n"); break;
    }
    
    printf("  Input size: %zu\n", model->config.input_size);
    printf("  Output size: %zu\n", model->config.output_size);
    printf("  Layer count: %zu\n", model->config.layer_count);
    printf("  Loaded: %s\n", model->is_loaded ? "Yes" : "No");
    
    /* Call model-specific info function */
    switch (model->config.type) {
        case ML_LINEAR_REGRESSION:
            ml_linear_regression_print_info(model);
            break;
        default:
            break;
    }
}

size_t ml_model_memory_usage(const ml_model_t* model) {
    if (!model) {
        return 0;
    }
    
    switch (model->config.type) {
        case ML_LINEAR_REGRESSION:
            return ml_linear_regression_memory_usage(model);
        default:
            return sizeof(ml_model_t);
    }
}

/* ============================================================================
 * PERFORMANCE MONITORING
 * ============================================================================ */

ml_error_t ml_get_performance_stats(ml_perf_stats_t* stats) {
    if (!stats) {
        return ML_ERROR_NULL_POINTER;
    }
    
    *stats = g_perf_stats;
    return ML_SUCCESS;
}

void ml_reset_performance_stats(void) {
    memset(&g_perf_stats, 0, sizeof(g_perf_stats));
}

/* ============================================================================
 * DEBUG AND SYSTEM INFO
 * ============================================================================ */

void ml_print_system_info(void) {
    if (!g_framework_initialized) {
        detect_cpu_features();
    }
    
    printf("ML Assembly Framework v%s\n", ml_get_version());
    printf("System Information:\n");
    printf("  AVX2 Support: %s\n", g_cpu_features.avx2_supported ? "Yes" : "No");
    printf("  AVX-512 Support: %s\n", g_cpu_features.avx512_supported ? "Yes" : "No");
    printf("  FMA Support: %s\n", g_cpu_features.fma_supported ? "Yes" : "No");
    printf("  CPU Compatible: %s\n", ml_check_cpu_support() ? "Yes" : "No");
    printf("  Framework Initialized: %s\n", g_framework_initialized ? "Yes" : "No");
}

void ml_print_performance_stats(void) {
    printf("Performance Statistics:\n");
    printf("  Total Predictions: %lu\n", g_perf_stats.total_predictions);
    printf("  Total Time: %.2f ms\n", g_perf_stats.total_time_ns / 1000000.0);
    printf("  Average Latency: %.2f μs\n", g_perf_stats.avg_latency_us);
    printf("  Throughput: %.0f predictions/sec\n", g_perf_stats.throughput_per_sec);
    printf("  Cache Hits: %lu\n", g_perf_stats.cache_hits);
    printf("  Cache Misses: %lu\n", g_perf_stats.cache_misses);
}
