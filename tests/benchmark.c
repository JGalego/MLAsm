/**
 * @file benchmark.c
 * @brief Performance benchmark suite for ML Assembly framework
 * 
 * Comprehensive performance testing to measure throughput, latency,
 * and system resource utilization across different model types and sizes.
 */

#define _POSIX_C_SOURCE 199309L

#include "ml_assembly.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

/* Benchmark configuration */
#define WARMUP_ITERATIONS 1000
#define BENCHMARK_ITERATIONS 10000
#define LARGE_BENCHMARK_ITERATIONS 100000

/* Timing utilities */
typedef struct {
    double min_time;
    double max_time;
    double avg_time;
    double total_time;
    uint64_t iterations;
} benchmark_stats_t;

static double get_time_microseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

static void init_benchmark_stats(benchmark_stats_t* stats) {
    stats->min_time = INFINITY;
    stats->max_time = 0.0;
    stats->avg_time = 0.0;
    stats->total_time = 0.0;
    stats->iterations = 0;
}

static void update_benchmark_stats(benchmark_stats_t* stats, double time) {
    stats->iterations++;
    stats->total_time += time;
    
    if (time < stats->min_time) stats->min_time = time;
    if (time > stats->max_time) stats->max_time = time;
    
    stats->avg_time = stats->total_time / stats->iterations;
}

static void print_benchmark_stats(const char* name, benchmark_stats_t* stats) {
    printf("%-30s | %10.2f | %10.2f | %10.2f | %12.0f | %10lu\n",
           name,
           stats->min_time,
           stats->avg_time,
           stats->max_time,
           stats->iterations / (stats->total_time / 1000000.0),
           stats->iterations);
}

/* ============================================================================
 * VECTOR OPERATION BENCHMARKS
 * ============================================================================ */

void benchmark_vector_operations(void) {
    printf("\n=== Vector Operations Benchmark ===\n");
    printf("%-30s | %10s | %10s | %10s | %12s | %10s\n",
           "Operation", "Min (μs)", "Avg (μs)", "Max (μs)", "Ops/sec", "Samples");
    printf("-%s\n", "-----------------------------+----------+----------+----------+------------+----------");
    
    /* Test different vector sizes */
    size_t sizes[] = {100, 1000, 10000};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (size_t s = 0; s < num_sizes; s++) {
        size_t size = sizes[s];
        
        /* Create test vectors */
        ml_vector_t* vec1 = ml_vector_create(size);
        ml_vector_t* vec2 = ml_vector_create(size);
        ml_vector_t* result = ml_vector_create(size);
        
        if (!vec1 || !vec2 || !result) {
            printf("Failed to create vectors of size %zu\n", size);
            continue;
        }
        
        /* Initialize with random data */
        srand(42);
        for (size_t i = 0; i < size; i++) {
            ml_vector_set(vec1, i, (float)rand() / RAND_MAX);
            ml_vector_set(vec2, i, (float)rand() / RAND_MAX);
        }
        
        /* Benchmark dot product */
        {
            benchmark_stats_t stats;
            init_benchmark_stats(&stats);
            
            /* Warmup */
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                volatile float dot_result = ml_vector_dot(vec1, vec2);
                (void)dot_result;
            }
            
            /* Actual benchmark */
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                double start = get_time_microseconds();
                volatile float dot_result = ml_vector_dot(vec1, vec2);
                double end = get_time_microseconds();
                (void)dot_result;
                
                update_benchmark_stats(&stats, end - start);
            }
            
            char name[100];
            snprintf(name, sizeof(name), "Dot Product (size %zu)", size);
            print_benchmark_stats(name, &stats);
        }
        
        /* Benchmark vector addition */
        {
            benchmark_stats_t stats;
            init_benchmark_stats(&stats);
            
            /* Warmup */
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ml_vector_add(vec1, vec2, result);
            }
            
            /* Actual benchmark */
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                double start = get_time_microseconds();
                ml_vector_add(vec1, vec2, result);
                double end = get_time_microseconds();
                
                update_benchmark_stats(&stats, end - start);
            }
            
            char name[100];
            snprintf(name, sizeof(name), "Vector Add (size %zu)", size);
            print_benchmark_stats(name, &stats);
        }
        
        /* Benchmark vector scaling */
        {
            benchmark_stats_t stats;
            init_benchmark_stats(&stats);
            
            /* Warmup */
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ml_vector_scale(vec1, 2.5f, result);
            }
            
            /* Actual benchmark */
            for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
                double start = get_time_microseconds();
                ml_vector_scale(vec1, 2.5f, result);
                double end = get_time_microseconds();
                
                update_benchmark_stats(&stats, end - start);
            }
            
            char name[100];
            snprintf(name, sizeof(name), "Vector Scale (size %zu)", size);
            print_benchmark_stats(name, &stats);
        }
        
        ml_vector_free(vec1);
        ml_vector_free(vec2);
        ml_vector_free(result);
    }
}

/* ============================================================================
 * MATRIX OPERATION BENCHMARKS
 * ============================================================================ */

void benchmark_matrix_operations(void) {
    printf("\n=== Matrix Operations Benchmark ===\n");
    printf("%-30s | %10s | %10s | %10s | %12s | %10s\n",
           "Operation", "Min (μs)", "Avg (μs)", "Max (μs)", "Ops/sec", "Samples");
    printf("-%s\n", "-----------------------------+----------+----------+----------+------------+----------");
    
    /* Test different matrix sizes */
    struct {
        size_t rows, cols;
    } sizes[] = {
        {100, 100},
        {500, 500},
        {1000, 100},
        {100, 1000}
    };
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (size_t s = 0; s < num_sizes; s++) {
        size_t rows = sizes[s].rows;
        size_t cols = sizes[s].cols;
        
        /* Create test matrices */
        ml_matrix_t* mat = ml_matrix_create(rows, cols);
        ml_vector_t* vec = ml_vector_create(cols);
        ml_vector_t* result = ml_vector_create(rows);
        
        if (!mat || !vec || !result) {
            printf("Failed to create matrix %zux%zu\n", rows, cols);
            continue;
        }
        
        /* Initialize with random data */
        srand(42);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                ml_matrix_set(mat, i, j, (float)rand() / RAND_MAX);
            }
        }
        for (size_t i = 0; i < cols; i++) {
            ml_vector_set(vec, i, (float)rand() / RAND_MAX);
        }
        
        /* Benchmark matrix-vector multiplication */
        {
            benchmark_stats_t stats;
            init_benchmark_stats(&stats);
            
            /* Warmup */
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                ml_matrix_vector_mul(mat, vec, result);
            }
            
            /* Actual benchmark */
            int iterations = (rows * cols > 100000) ? BENCHMARK_ITERATIONS / 10 : BENCHMARK_ITERATIONS;
            for (int i = 0; i < iterations; i++) {
                double start = get_time_microseconds();
                ml_matrix_vector_mul(mat, vec, result);
                double end = get_time_microseconds();
                
                update_benchmark_stats(&stats, end - start);
            }
            
            char name[100];
            snprintf(name, sizeof(name), "MatVec (%zux%zu)", rows, cols);
            print_benchmark_stats(name, &stats);
        }
        
        ml_matrix_free(mat);
        ml_vector_free(vec);
        ml_vector_free(result);
    }
}

/* ============================================================================
 * ACTIVATION FUNCTION BENCHMARKS
 * ============================================================================ */

void benchmark_activation_functions(void) {
    printf("\n=== Activation Functions Benchmark ===\n");
    printf("%-30s | %10s | %10s | %10s | %12s | %10s\n",
           "Operation", "Min (μs)", "Avg (μs)", "Max (μs)", "Ops/sec", "Samples");
    printf("-%s\n", "-----------------------------+----------+----------+----------+------------+----------");
    
    size_t size = 1000;
    ml_vector_t* input = ml_vector_create(size);
    ml_vector_t* output = ml_vector_create(size);
    
    if (!input || !output) {
        printf("Failed to create activation test vectors\n");
        return;
    }
    
    /* Initialize with random data in range [-5, 5] */
    srand(42);
    for (size_t i = 0; i < size; i++) {
        float val = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        ml_vector_set(input, i, val);
    }
    
    /* Benchmark ReLU */
    {
        benchmark_stats_t stats;
        init_benchmark_stats(&stats);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            ml_activation_relu(input, output);
        }
        
        /* Actual benchmark */
        for (int i = 0; i < LARGE_BENCHMARK_ITERATIONS; i++) {
            double start = get_time_microseconds();
            ml_activation_relu(input, output);
            double end = get_time_microseconds();
            
            update_benchmark_stats(&stats, end - start);
        }
        
        print_benchmark_stats("ReLU", &stats);
    }
    
    /* Benchmark Sigmoid */
    {
        benchmark_stats_t stats;
        init_benchmark_stats(&stats);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            ml_activation_sigmoid(input, output);
        }
        
        /* Actual benchmark */
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            double start = get_time_microseconds();
            ml_activation_sigmoid(input, output);
            double end = get_time_microseconds();
            
            update_benchmark_stats(&stats, end - start);
        }
        
        print_benchmark_stats("Sigmoid", &stats);
    }
    
    /* Benchmark Tanh */
    {
        benchmark_stats_t stats;
        init_benchmark_stats(&stats);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            ml_activation_tanh(input, output);
        }
        
        /* Actual benchmark */
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            double start = get_time_microseconds();
            ml_activation_tanh(input, output);
            double end = get_time_microseconds();
            
            update_benchmark_stats(&stats, end - start);
        }
        
        print_benchmark_stats("Tanh", &stats);
    }
    
    /* Benchmark Softmax */
    {
        benchmark_stats_t stats;
        init_benchmark_stats(&stats);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            ml_activation_softmax(input, output);
        }
        
        /* Actual benchmark */
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            double start = get_time_microseconds();
            ml_activation_softmax(input, output);
            double end = get_time_microseconds();
            
            update_benchmark_stats(&stats, end - start);
        }
        
        print_benchmark_stats("Softmax", &stats);
    }
    
    ml_vector_free(input);
    ml_vector_free(output);
}

/* ============================================================================
 * MODEL INFERENCE BENCHMARKS
 * ============================================================================ */

void benchmark_model_inference(void) {
    printf("\n=== Model Inference Benchmark ===\n");
    printf("%-30s | %10s | %10s | %10s | %12s | %10s\n",
           "Model", "Min (μs)", "Avg (μs)", "Max (μs)", "Ops/sec", "Samples");
    printf("-%s\n", "-----------------------------+----------+----------+----------+------------+----------");
    
    /* Test different model sizes */
    size_t input_sizes[] = {10, 100, 1000};
    size_t num_sizes = sizeof(input_sizes) / sizeof(input_sizes[0]);
    
    for (size_t s = 0; s < num_sizes; s++) {
        size_t input_size = input_sizes[s];
        
        /* Create linear regression model */
        ml_model_config_t config = {
            .type = ML_LINEAR_REGRESSION,
            .input_size = input_size,
            .output_size = 1,
            .layer_count = 1
        };
        
        ml_model_t* model = ml_create_model(&config);
        if (!model) {
            printf("Failed to create model with input size %zu\n", input_size);
            continue;
        }
        
        /* Set random weights */
        float* weights = malloc(input_size * sizeof(float));
        if (!weights) {
            ml_free_model(model);
            continue;
        }
        
        srand(42);
        for (size_t i = 0; i < input_size; i++) {
            weights[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        extern ml_error_t ml_linear_regression_set_weights(ml_model_t* model, const ml_float_t* weights, ml_float_t bias);
        ml_linear_regression_set_weights(model, weights, 0.0f);
        
        /* Create test input */
        float* input = malloc(input_size * sizeof(float));
        if (!input) {
            free(weights);
            ml_free_model(model);
            continue;
        }
        
        for (size_t i = 0; i < input_size; i++) {
            input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        /* Benchmark single prediction */
        {
            benchmark_stats_t stats;
            init_benchmark_stats(&stats);
            
            /* Warmup */
            for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                float output;
                ml_predict(model, input, &output);
            }
            
            /* Reset performance stats */
            ml_reset_performance_stats();
            
            /* Actual benchmark */
            int iterations = (input_size > 500) ? BENCHMARK_ITERATIONS / 10 : BENCHMARK_ITERATIONS;
            for (int i = 0; i < iterations; i++) {
                double start = get_time_microseconds();
                float output;
                ml_predict(model, input, &output);
                double end = get_time_microseconds();
                
                update_benchmark_stats(&stats, end - start);
            }
            
            char name[100];
            snprintf(name, sizeof(name), "Linear Reg (input %zu)", input_size);
            print_benchmark_stats(name, &stats);
        }
        
        /* Benchmark batch prediction */
        {
            size_t batch_size = 100;
            ml_matrix_t* batch_input = ml_matrix_create(batch_size, input_size);
            ml_matrix_t* batch_output = ml_matrix_create(batch_size, 1);
            
            if (batch_input && batch_output) {
                /* Initialize batch input */
                for (size_t i = 0; i < batch_size; i++) {
                    for (size_t j = 0; j < input_size; j++) {
                        float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                        ml_matrix_set(batch_input, i, j, val);
                    }
                }
                
                benchmark_stats_t stats;
                init_benchmark_stats(&stats);
                
                /* Warmup */
                for (int i = 0; i < WARMUP_ITERATIONS / 10; i++) {
                    ml_predict_batch(model, batch_input, batch_output, batch_size);
                }
                
                /* Actual benchmark */
                int iterations = (input_size > 500) ? BENCHMARK_ITERATIONS / 100 : BENCHMARK_ITERATIONS / 10;
                for (int i = 0; i < iterations; i++) {
                    double start = get_time_microseconds();
                    ml_predict_batch(model, batch_input, batch_output, batch_size);
                    double end = get_time_microseconds();
                    
                    update_benchmark_stats(&stats, end - start);
                }
                
                char name[100];
                snprintf(name, sizeof(name), "Linear Reg Batch (%zu)", input_size);
                print_benchmark_stats(name, &stats);
                
                ml_matrix_free(batch_input);
                ml_matrix_free(batch_output);
            }
        }
        
        free(input);
        free(weights);
        ml_free_model(model);
    }
}

/* ============================================================================
 * MEMORY USAGE ANALYSIS
 * ============================================================================ */

void analyze_memory_usage(void) {
    printf("\n=== Memory Usage Analysis ===\n");
    printf("%-30s | %15s | %15s\n", "Component", "Size (bytes)", "Alignment");
    printf("-%s\n", "-----------------------------+---------------+---------------");
    
    /* Analyze vector memory usage */
    for (size_t size = 100; size <= 10000; size *= 10) {
        ml_vector_t* vec = ml_vector_create(size);
        if (vec) {
            size_t memory = sizeof(ml_vector_t) + vec->capacity * sizeof(ml_float_t);
            char name[100];
            snprintf(name, sizeof(name), "Vector (size %zu)", size);
            printf("%-30s | %15zu | %15s\n", name, memory, "32-byte");
            ml_vector_free(vec);
        }
    }
    
    /* Analyze matrix memory usage */
    for (size_t size = 100; size <= 1000; size *= 10) {
        ml_matrix_t* mat = ml_matrix_create(size, size);
        if (mat) {
            size_t memory = sizeof(ml_matrix_t) + mat->capacity * sizeof(ml_float_t);
            char name[100];
            snprintf(name, sizeof(name), "Matrix (%zux%zu)", size, size);
            printf("%-30s | %15zu | %15s\n", name, memory, "32-byte");
            ml_matrix_free(mat);
        }
    }
    
    /* Analyze model memory usage */
    for (size_t input_size = 10; input_size <= 1000; input_size *= 10) {
        ml_model_config_t config = {
            .type = ML_LINEAR_REGRESSION,
            .input_size = input_size,
            .output_size = 1
        };
        
        ml_model_t* model = ml_create_model(&config);
        if (model) {
            size_t memory = ml_model_memory_usage(model);
            char name[100];
            snprintf(name, sizeof(name), "Linear Model (input %zu)", input_size);
            printf("%-30s | %15zu | %15s\n", name, memory, "32-byte");
            ml_free_model(model);
        }
    }
}

/* ============================================================================
 * MAIN BENCHMARK RUNNER
 * ============================================================================ */

int main(int argc, char* argv[]) {
    bool run_vector = false;
    bool run_matrix = false;
    bool run_activation = false;
    bool run_model = false;
    bool run_memory = false;
    bool run_all = true;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--vector") == 0) {
            run_vector = true;
            run_all = false;
        } else if (strcmp(argv[i], "--matrix") == 0) {
            run_matrix = true;
            run_all = false;
        } else if (strcmp(argv[i], "--activation") == 0) {
            run_activation = true;
            run_all = false;
        } else if (strcmp(argv[i], "--model") == 0) {
            run_model = true;
            run_all = false;
        } else if (strcmp(argv[i], "--memory") == 0) {
            run_memory = true;
            run_all = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("ML Assembly Performance Benchmark Suite\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --vector      Benchmark vector operations\n");
            printf("  --matrix      Benchmark matrix operations\n");
            printf("  --activation  Benchmark activation functions\n");
            printf("  --model       Benchmark model inference\n");
            printf("  --memory      Analyze memory usage\n");
            printf("  --help        Show this help message\n");
            printf("  (no options)  Run all benchmarks\n");
            return 0;
        }
    }
    
    /* Initialize framework */
    ml_error_t init_result = ml_init();
    if (init_result != ML_SUCCESS) {
        printf("Failed to initialize ML Assembly framework\n");
        return 1;
    }
    
    printf("ML Assembly Performance Benchmark Suite\n");
    printf("=======================================\n");
    
    /* Print system information */
    ml_print_system_info();
    
    /* Run benchmarks */
    if (run_all || run_vector) {
        benchmark_vector_operations();
    }
    
    if (run_all || run_matrix) {
        benchmark_matrix_operations();
    }
    
    if (run_all || run_activation) {
        benchmark_activation_functions();
    }
    
    if (run_all || run_model) {
        benchmark_model_inference();
    }
    
    if (run_all || run_memory) {
        analyze_memory_usage();
    }
    
    /* Print final performance statistics */
    printf("\n=== Framework Performance Statistics ===\n");
    ml_print_performance_stats();
    
    /* Cleanup */
    ml_cleanup();
    
    printf("\nBenchmark completed successfully!\n");
    return 0;
}
