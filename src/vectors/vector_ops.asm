; ==============================================================================
; Vector Operations - SIMD-optimized linear algebra primitives
; File: src/vectors/vector_ops.asm
; 
; This module implements high-performance vector operations using AVX2/AVX-512
; instructions for maximum throughput and minimum latency.
; ==============================================================================

section .text

; External C functions we'll call
extern malloc, free, memcpy, memset

; Export our functions
global ml_vector_dot_asm
global ml_vector_add_asm
global ml_vector_scale_asm
global ml_vector_normalize_asm
global ml_vector_sum_asm

; ==============================================================================
; Constants and alignment macros
; ==============================================================================

%define ALIGN_32 32
%define ALIGN_64 64
%define VECTOR_ALIGN 32

; Check for minimum vector size for SIMD operations
%define MIN_SIMD_SIZE 8

; ==============================================================================
; ml_vector_dot_asm - Compute dot product of two vectors
; 
; Signature: float ml_vector_dot_asm(const float* a, const float* b, size_t size)
; 
; Parameters:
;   rdi: pointer to vector a
;   rsi: pointer to vector b  
;   rdx: size of vectors
;
; Returns:
;   xmm0: dot product result (single precision float)
; ==============================================================================

ml_vector_dot_asm:
    ; Save callee-saved registers
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers
    test rdi, rdi
    jz .error
    test rsi, rsi  
    jz .error
    
    ; Check for zero size
    test rdx, rdx
    jz .zero_result
    
    ; Initialize accumulator
    vxorps ymm0, ymm0, ymm0    ; Zero out 8 floats
    xor rax, rax               ; Index counter
    
    ; Check if we have enough elements for SIMD
    cmp rdx, MIN_SIMD_SIZE
    jl .scalar_loop
    
    ; Calculate number of SIMD iterations (8 floats per iteration)
    mov rcx, rdx
    shr rcx, 3                 ; Divide by 8
    shl rcx, 3                 ; Multiply by 8 to get aligned count
    
.simd_loop:
    cmp rax, rcx
    jge .handle_remainder
    
    ; Load 8 floats from each vector
    vmovups ymm1, [rdi + rax*4]    ; Load 8 floats from vector a
    vmovups ymm2, [rsi + rax*4]    ; Load 8 floats from vector b
    
    ; Multiply and accumulate
    vfmadd231ps ymm0, ymm1, ymm2   ; ymm0 += ymm1 * ymm2
    
    add rax, 8
    jmp .simd_loop

.handle_remainder:
    ; Handle remaining elements (less than 8)
    cmp rax, rdx
    jge .reduce_accumulator
    
.scalar_loop:
    ; Process remaining elements one by one
    movss xmm1, [rdi + rax*4]
    movss xmm2, [rsi + rax*4]
    mulss xmm1, xmm2
    addss xmm0, xmm1           ; Add to scalar accumulator (xmm0[0])
    
    inc rax
    cmp rax, rdx
    jl .scalar_loop

.reduce_accumulator:
    ; Reduce 8-element accumulator to single value
    ; Extract high 128 bits and add to low 128 bits
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    
    ; Horizontal add within 128-bit register
    vhaddps xmm0, xmm0, xmm0   ; Add adjacent pairs
    vhaddps xmm0, xmm0, xmm0   ; Add pairs of pairs
    
    jmp .cleanup

.zero_result:
    vxorps xmm0, xmm0, xmm0

.error:
    ; Return NaN on error
    mov eax, 0x7FC00000        ; NaN bit pattern
    movd xmm0, eax

.cleanup:
    ; Clean up YMM registers to avoid performance penalty
    vzeroupper
    
    ; Restore stack
    pop rbp
    ret

; ==============================================================================
; ml_vector_add_asm - Add two vectors element-wise
; 
; Signature: void ml_vector_add_asm(const float* a, const float* b, 
;                                   float* result, size_t size)
; 
; Parameters:
;   rdi: pointer to vector a
;   rsi: pointer to vector b
;   rdx: pointer to result vector
;   rcx: size of vectors
; ==============================================================================

ml_vector_add_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; Check for zero size
    test rcx, rcx
    jz .cleanup
    
    xor rax, rax               ; Index counter
    
    ; Check if we have enough elements for SIMD
    cmp rcx, MIN_SIMD_SIZE
    jl .scalar_add_loop
    
    ; Calculate SIMD iterations
    mov r8, rcx
    shr r8, 3                  ; Divide by 8
    shl r8, 3                  ; Multiply by 8 for aligned count

.simd_add_loop:
    cmp rax, r8
    jge .handle_add_remainder
    
    ; Load 8 floats from each vector
    vmovups ymm0, [rdi + rax*4]
    vmovups ymm1, [rsi + rax*4]
    
    ; Add vectors
    vaddps ymm2, ymm0, ymm1
    
    ; Store result
    vmovups [rdx + rax*4], ymm2
    
    add rax, 8
    jmp .simd_add_loop

.handle_add_remainder:
    cmp rax, rcx
    jge .cleanup

.scalar_add_loop:
    ; Process remaining elements
    movss xmm0, [rdi + rax*4]
    movss xmm1, [rsi + rax*4]
    addss xmm0, xmm1
    movss [rdx + rax*4], xmm0
    
    inc rax
    cmp rax, rcx
    jl .scalar_add_loop

.cleanup:
    vzeroupper
    pop rbp
    ret

; ==============================================================================
; ml_vector_scale_asm - Multiply vector by scalar
; 
; Signature: void ml_vector_scale_asm(const float* input, float scalar,
;                                     float* result, size_t size)
; 
; Parameters:
;   rdi: pointer to input vector
;   xmm0: scalar value (float)
;   rsi: pointer to result vector  
;   rdx: size of vector
; ==============================================================================

ml_vector_scale_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointers
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    
    ; Check for zero size
    test rdx, rdx
    jz .cleanup
    
    ; Broadcast scalar to all elements of YMM register
    vbroadcastss ymm1, xmm0
    
    xor rax, rax               ; Index counter
    
    ; Check if we have enough elements for SIMD
    cmp rdx, MIN_SIMD_SIZE
    jl .scalar_scale_loop
    
    ; Calculate SIMD iterations
    mov rcx, rdx
    shr rcx, 3                 ; Divide by 8
    shl rcx, 3                 ; Multiply by 8

.simd_scale_loop:
    cmp rax, rcx
    jge .handle_scale_remainder
    
    ; Load 8 floats
    vmovups ymm0, [rdi + rax*4]
    
    ; Multiply by scalar
    vmulps ymm2, ymm0, ymm1
    
    ; Store result
    vmovups [rsi + rax*4], ymm2
    
    add rax, 8
    jmp .simd_scale_loop

.handle_scale_remainder:
    cmp rax, rdx
    jge .cleanup

.scalar_scale_loop:
    ; Process remaining elements
    movss xmm0, [rdi + rax*4]
    mulss xmm0, xmm1           ; xmm1[0] contains the scalar
    movss [rsi + rax*4], xmm0
    
    inc rax
    cmp rax, rdx
    jl .scalar_scale_loop

.cleanup:
    vzeroupper
    pop rbp
    ret

; ==============================================================================
; ml_vector_normalize_asm - Normalize vector to unit length
; 
; Signature: void ml_vector_normalize_asm(const float* input, float* result, size_t size)
; 
; Parameters:
;   rdi: pointer to input vector
;   rsi: pointer to result vector
;   rdx: size of vector
; ==============================================================================

ml_vector_normalize_asm:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Check for null pointers and size
    test rdi, rdi
    jz .cleanup
    test rsi, rsi
    jz .cleanup
    test rdx, rdx
    jz .cleanup
    
    ; First compute the magnitude (using our dot product function)
    ; Save parameters
    mov rbx, rsi               ; Save result pointer
    mov rsi, rdi               ; Second vector = first vector for dot product
    call ml_vector_dot_asm     ; Result in xmm0
    
    ; Check for zero magnitude
    comiss xmm0, xmm0
    jp .cleanup                ; NaN check
    xorps xmm1, xmm1
    comiss xmm0, xmm1
    je .cleanup                ; Zero magnitude
    
    ; Compute 1/sqrt(magnitude) for normalization
    rsqrtss xmm0, xmm0         ; Fast reciprocal square root
    
    ; Now scale the vector by 1/magnitude
    mov rsi, rbx               ; Restore result pointer
    call ml_vector_scale_asm

.cleanup:
    pop rbx
    pop rbp
    ret

; ==============================================================================
; ml_vector_sum_asm - Sum all elements of a vector
; 
; Signature: float ml_vector_sum_asm(const float* vector, size_t size)
; 
; Parameters:
;   rdi: pointer to vector
;   rsi: size of vector
;
; Returns:
;   xmm0: sum of all elements
; ==============================================================================

ml_vector_sum_asm:
    push rbp
    mov rbp, rsp
    
    ; Check for null pointer and size
    test rdi, rdi
    jz .error
    test rsi, rsi
    jz .zero_result
    
    ; Initialize accumulator
    vxorps ymm0, ymm0, ymm0
    xor rax, rax
    
    ; Check if we have enough elements for SIMD
    cmp rsi, MIN_SIMD_SIZE
    jl .scalar_sum_loop
    
    ; Calculate SIMD iterations
    mov rcx, rsi
    shr rcx, 3
    shl rcx, 3

.simd_sum_loop:
    cmp rax, rcx
    jge .handle_sum_remainder
    
    ; Load and accumulate 8 floats
    vmovups ymm1, [rdi + rax*4]
    vaddps ymm0, ymm0, ymm1
    
    add rax, 8
    jmp .simd_sum_loop

.handle_sum_remainder:
    cmp rax, rsi
    jge .reduce_sum

.scalar_sum_loop:
    movss xmm1, [rdi + rax*4]
    addss xmm0, xmm1           ; Add to scalar part
    
    inc rax
    cmp rax, rsi
    jl .scalar_sum_loop

.reduce_sum:
    ; Reduce 8-element accumulator to single value
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    
    jmp .cleanup

.zero_result:
    vxorps xmm0, xmm0, xmm0
    jmp .cleanup

.error:
    mov eax, 0x7FC00000        ; NaN
    movd xmm0, eax

.cleanup:
    vzeroupper
    pop rbp
    ret

section .data
align 32
; Lookup table for fast math operations (if needed)
one_ps: dd 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

; Mark stack as non-executable
section .note.GNU-stack noalloc noexec nowrite progbits
